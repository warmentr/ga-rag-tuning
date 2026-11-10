"""Exhaustive grid scoring: every valid config, tuning and test nDCG -> lookup table.

Work is grouped by chunking. Per chunking: chunk the corpus, load/build the
embeddings and BM25 index; per question set: rank both indexes once to depth
50 and load/build the rerank scores; then score all configs sharing that
chunking from those rankings and scores.

Resumable: each chunking writes results/lookup_table/parts/<cs_ov>.json. A
part is reused only if its `run_key` (everything that affects scores) matches
the current run. Code changes are NOT part of the run_key; delete the parts
(or pass force=True) after changing pipeline code.

Outputs in out_dir:
  lookup_table.csv   one row per valid config: key, genes, tuning_ndcg, test_ndcg
  per_question.npz   keys, tuning/test (n_configs x n_questions) nDCG, question ids
  metadata.json      models, versions, settings, split, runtime, GPU energy
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .data import load_corpus, load_questions
from .energy import GpuPowerMeter
from .metrics import score_question
from .pipeline.chunking import chunk_corpus, load_tokenizer
from .pipeline.indexing import (
    BM25Index,
    DenseIndex,
    Embedder,
    embedding_cache_path,
    load_or_build_embeddings,
    load_or_build_rows,
    strings_fingerprint,
)
from .pipeline.rerank import Reranker, ScoreMaps, candidate_union, load_or_build_scores, rerank, rerank_cache_path
from .pipeline.retrieval import Rankings, first_stage, rank_questions
from .space import GENE_NAMES, Config, SearchSpace

QUESTION_SETS = ("tuning", "test")


def score_config(
    config: Config,
    rankings: Rankings,
    score_maps: ScoreMaps,
    chunk_article_ids: np.ndarray,
    labels: Sequence[Sequence[str]],
    k: int,
    rrf_k: int,
) -> list[float]:
    """Per-question article-level nDCG@k for one config on one question set."""
    lists = first_stage(rankings, config.retrieval_type, config.top_k, rrf_k)
    if config.reranker:
        lists = rerank(lists, score_maps)
    return [score_question(chunk_article_ids[l], rel, k) for l, rel in zip(lists, labels)]


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build_lookup_table(
    root: Path,
    out_dir: Path,
    only_chunkings: Sequence[tuple[int, int]] | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Score the grid (or a subset of chunkings) and write outputs. Returns the metadata dict.

    With `only_chunkings`, only those parts are built and the merged outputs
    are not written (a partial table would be misleading).
    """
    root, out_dir = Path(root), Path(out_dir)
    cfg = yaml.safe_load((root / "config" / "experiment.yaml").read_text(encoding="utf-8"))
    space = SearchSpace.from_yaml(root / "config" / "search_space.yaml")
    p, paths = cfg["pipeline"], cfg["paths"]
    raw_dir, cache_dir = root / paths["raw_dir"], root / paths["cache_dir"]
    k, rrf_k = cfg["fitness"]["k"], p["rrf_k"]
    depth = max(space.genes["top_k"])
    bm25_params = {key: v for key, v in p["bm25"].items() if key != "library"}
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    log("Loading corpus, questions, models ...")
    corpus = load_corpus(raw_dir)
    article_ids, texts = corpus["id"], corpus["contents"]
    qsets = {s: load_questions(raw_dir, root / paths["splits"], s) for s in QUESTION_SETS}
    questions = {s: [q["question"] for q in qs] for s, qs in qsets.items()}
    labels = {s: [q["article_ids"] for q in qs] for s, qs in qsets.items()}
    qids = {s: [q["id"] for q in qs] for s, qs in qsets.items()}

    tok = load_tokenizer(p["embedding_model"])
    emb = Embedder(p["embedding_model"], p["embedding_query_prefix"], p["embedding_dtype"], p["embedding_batch_size"])
    rr = Reranker(p["reranker_model"], p["reranker_dtype"], p["reranker_batch_size"],
                  max_length=p["reranker_max_length"])

    run_key_obj = {
        "embedder": emb.settings,
        "query_prefix": p["embedding_query_prefix"],
        "reranker": rr.settings,
        "bm25": bm25_params,
        "rrf_k": rrf_k,
        "k": k,
        "depth": depth,
        "questions": {s: strings_fingerprint(qids[s] + questions[s]) for s in QUESTION_SETS},
        "corpus": strings_fingerprint(list(article_ids)),
        "genes": {g: list(v) for g, v in space.genes.items()},
    }
    run_key = _hash(run_key_obj)

    q_emb = {}
    for s in QUESTION_SETS:
        path = cache_dir / "queries" / p["embedding_model"].replace("/", "__") / s
        q_emb[s] = load_or_build_rows(
            questions[s], strings_fingerprint(questions[s]), emb.embed_queries,
            {**emb.settings, "query_prefix": p["embedding_query_prefix"]}, path,
        )

    chunkings = list(only_chunkings) if only_chunkings else list(space.chunkings())
    configs_by_chunking: dict[tuple[int, int], list[Config]] = {}
    for c in space.valid_configs():
        configs_by_chunking.setdefault(c.chunking, []).append(c)

    for n, (size, ov) in enumerate(chunkings, 1):
        if (size, ov) not in configs_by_chunking:
            raise ValueError(f"chunking {(size, ov)} is not in the search space")
        part_path = parts_dir / f"cs{size}_ov{ov}.json"
        if part_path.exists() and not force:
            part = json.loads(part_path.read_text(encoding="utf-8"))
            if part.get("run_key") == run_key:
                log(f"[{n}/{len(chunkings)}] cs{size}_ov{ov}: done already, skipping")
                continue

        log(f"[{n}/{len(chunkings)}] cs{size}_ov{ov}: building ...")
        timings = {}
        with GpuPowerMeter() as meter:
            t = time.perf_counter()
            chunks = chunk_corpus(article_ids, texts, tok, size, ov)
            chunk_aids = np.array([c.article_id for c in chunks])
            E = load_or_build_embeddings(
                chunks, lambda x: emb.embed_passages(x, show_progress=False), emb.settings,
                embedding_cache_path(cache_dir, p["embedding_model"], size, ov),
            )
            dense = DenseIndex(E)
            bm25 = BM25Index([c.text for c in chunks], **bm25_params)
            timings["index_s"] = time.perf_counter() - t

            results = {c.key: {} for c in configs_by_chunking[(size, ov)]}
            for s in QUESTION_SETS:
                t = time.perf_counter()
                rankings = rank_questions(questions[s], q_emb[s], dense, bm25, depth)
                cands = candidate_union(rankings)
                maps = load_or_build_scores(
                    questions[s], chunks, cands, rr.score_pairs, rr.settings,
                    rerank_cache_path(cache_dir, p["reranker_model"], s, size, ov),
                )
                timings[f"{s}_rank_rerank_s"] = time.perf_counter() - t
                t = time.perf_counter()
                for c in configs_by_chunking[(size, ov)]:
                    results[c.key][s] = score_config(c, rankings, maps, chunk_aids, labels[s], k, rrf_k)
                timings[f"{s}_score_s"] = time.perf_counter() - t

        part = {
            "run_key": run_key,
            "chunking": [size, ov],
            "n_chunks": len(chunks),
            "seconds": meter.seconds,
            "gpu_wh": meter.wh,
            "timings": timings,
            "built_at": dt.datetime.now().isoformat(timespec="seconds"),
            "configs": [
                {**asdict(c), "key": c.key, "scores": results[c.key]}
                for c in configs_by_chunking[(size, ov)]
            ],
        }
        part_path.write_text(json.dumps(part) + "\n", encoding="utf-8")
        best = max(part["configs"], key=lambda r: np.mean(r["scores"]["tuning"]))
        wh = f"{meter.wh:.2f} Wh" if meter.wh is not None else "n/a"
        log(f"    {len(chunks)} chunks, {meter.seconds:.0f}s, GPU {wh}; "
            f"best tuning here {np.mean(best['scores']['tuning']):.3f} ({best['key']})")

    if only_chunkings:
        log("Subset run: parts written, merged outputs not written.")
        return {}
    return _merge(root, out_dir, space, cfg, run_key, run_key_obj, qids, emb, rr,
                  time.perf_counter() - t_start, log)


def _merge(root, out_dir, space, cfg, run_key, run_key_obj, qids, emb, rr, this_run_s, log) -> dict:
    parts = {}
    for size, ov in space.chunkings():
        part = json.loads((out_dir / "parts" / f"cs{size}_ov{ov}.json").read_text(encoding="utf-8"))
        if part["run_key"] != run_key:
            raise RuntimeError(f"part cs{size}_ov{ov} is from a different run_key")
        for row in part["configs"]:
            parts[row["key"]] = row
        parts[(size, ov)] = part

    configs = space.valid_configs()
    if any(c.key not in parts for c in configs):
        raise RuntimeError("missing configs in parts")

    tuning = np.array([parts[c.key]["scores"]["tuning"] for c in configs], dtype=np.float64)
    test = np.array([parts[c.key]["scores"]["test"] for c in configs], dtype=np.float64)

    with (out_dir / "lookup_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", *GENE_NAMES, "tuning_ndcg", "test_ndcg"])
        for c, tu, te in zip(configs, tuning.mean(1), test.mean(1)):
            w.writerow([c.key, *(getattr(c, g) for g in GENE_NAMES), f"{tu:.6f}", f"{te:.6f}"])

    np.savez_compressed(
        out_dir / "per_question.npz",
        keys=np.array([c.key for c in configs]),
        tuning=tuning, test=test,
        tuning_qids=np.array(qids["tuning"]), test_qids=np.array(qids["test"]),
    )

    chunk_parts = [parts[ch] for ch in space.chunkings()]
    whs = [cp["gpu_wh"] for cp in chunk_parts]
    import sentence_transformers, torch, bm25s, tokenizers  # noqa: E401

    best_i = int(tuning.mean(1).argmax())
    meta = {
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(root),
        "run_key": run_key,
        "n_configs": len(configs),
        "n_questions": {"tuning": tuning.shape[1], "test": test.shape[1]},
        "split": cfg["split"],
        "fitness": cfg["fitness"],
        "pipeline": cfg["pipeline"],
        "run_key_inputs": run_key_obj,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "bm25s": bm25s.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "compute": {
            "grid_seconds_total": sum(cp["seconds"] for cp in chunk_parts),
            "gpu_wh_total": sum(whs) if all(w is not None for w in whs) else None,
            "gpu_energy_note": "GPU board power only (nvidia-smi, 1 s sampling); excludes CPU/system.",
            "this_invocation_seconds": this_run_s,
            "per_chunking": {
                f"cs{cp['chunking'][0]}_ov{cp['chunking'][1]}": {
                    "n_chunks": cp["n_chunks"], "seconds": cp["seconds"], "gpu_wh": cp["gpu_wh"],
                    "timings": cp["timings"], "built_at": cp["built_at"],
                }
                for cp in chunk_parts
            },
        },
        "summary": {
            "best_tuning": {"key": configs[best_i].key, "tuning_ndcg": float(tuning.mean(1)[best_i]),
                            "test_ndcg": float(test.mean(1)[best_i])},
            "best_test_ndcg": float(test.mean(1).max()),
            "tuning_ndcg_range": [float(tuning.mean(1).min()), float(tuning.mean(1).max())],
            "n_distinct_tuning_scores": int(len(np.unique(np.round(tuning.mean(1), 12)))),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {out_dir / 'lookup_table.csv'} ({len(configs)} configs), per_question.npz, metadata.json")
    return meta
