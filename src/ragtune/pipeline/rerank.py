"""Cross-encoder reranking with pointwise score reuse.

The cross-encoder scores each (question, chunk) pair independently, so per
(chunking, question set) every pair that any config could rerank is scored
once: the union of the bm25 and dense depth-50 rankings. Every first-stage
list (any retrieval_type, any top_k <= depth) is a subset of that union,
hybrid included, since it fuses bm25 top_k and dense top_k.

Scores are raw logits (identity activation; a sigmoid would saturate and
create ties). Reranking sorts a first-stage list by descending score; ties
keep first-stage order. Pairs longer than the cross-encoder's 512-token
limit are truncated by its tokenizer (the longer side, i.e. the chunk).

GPU scores are not bit-identical across runs, so scores are cached per
(chunking, question set), keyed by fingerprints of the chunks, questions,
and candidates plus the model settings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .indexing import chunks_fingerprint, strings_fingerprint
from .retrieval import Rankings

ScoreMaps = list[dict[int, float]]  # per question: chunk index -> score


class Reranker:
    def __init__(
        self,
        model_name: str,
        dtype: str = "float32",
        batch_size: int = 64,
        device: str | None = None,
        max_length: int = 512,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            activation_fn=torch.nn.Identity(),
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )

    @property
    def settings(self) -> dict:
        import sentence_transformers
        import torch

        return {
            "model": self.model_name,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "activation": "identity",
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        }

    def score_pairs(self, pairs: Sequence[tuple[str, str]], show_progress: bool = False) -> np.ndarray:
        """Score (question, passage) pairs. Pairs are length-sorted internally to cut padding."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        order = np.argsort([len(q) + len(p) for q, p in pairs], kind="stable")
        sorted_scores = self.model.predict(
            [pairs[i] for i in order],
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)
        out = np.empty(len(pairs), dtype=np.float32)
        out[order] = sorted_scores
        return out


def candidate_union(rankings: Rankings) -> list[np.ndarray]:
    """Per question: sorted unique chunk indices in bm25 ∪ dense rankings."""
    return [np.union1d(b, d) for b, d in zip(rankings.bm25, rankings.dense)]


def _fingerprint_candidates(candidates: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for c in candidates:
        h.update(np.asarray(c, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def load_or_build_scores(
    questions: Sequence[str],
    chunks: Sequence[Chunk],
    candidates: Sequence[np.ndarray],
    score_fn: Callable[[list[tuple[str, str]]], np.ndarray],
    settings: dict,
    cache_path: Path,
) -> ScoreMaps:
    """Score every (question, candidate chunk) pair, using cache_path(.npz/.json) if it matches."""
    if len(questions) != len(candidates):
        raise ValueError("questions and candidates must align")
    cache_path = Path(cache_path)
    npz, meta_path = cache_path.with_suffix(".npz"), cache_path.with_suffix(".json")
    q_idx = np.concatenate([np.full(len(c), i, dtype=np.int64) for i, c in enumerate(candidates)])
    c_idx = np.concatenate([np.asarray(c, dtype=np.int64) for c in candidates])
    meta = {
        "chunks": chunks_fingerprint(chunks),
        "questions": strings_fingerprint(questions),
        "candidates": _fingerprint_candidates(candidates),
        "n_pairs": int(len(q_idx)),
        "settings": settings,
    }

    scores = None
    if npz.exists() and meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            data = np.load(npz)
            if np.array_equal(data["q_idx"], q_idx) and np.array_equal(data["c_idx"], c_idx):
                scores = data["score"]
    if scores is None:
        pairs = [(questions[q], chunks[c].text) for q, c in zip(q_idx, c_idx)]
        scores = np.asarray(score_fn(pairs), dtype=np.float32)
        if scores.shape != (len(pairs),):
            raise RuntimeError(f"score_fn returned shape {scores.shape} for {len(pairs)} pairs")
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, q_idx=q_idx, c_idx=c_idx, score=scores)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    maps: ScoreMaps = [{} for _ in questions]
    for q, c, s in zip(q_idx.tolist(), c_idx.tolist(), scores.tolist()):
        maps[q][c] = s
    return maps


def rerank(first_stage_lists: Sequence[Sequence[int]], score_maps: ScoreMaps) -> list[list[int]]:
    """Reorder each question's first-stage list by descending score; ties keep first-stage order."""
    if len(first_stage_lists) != len(score_maps):
        raise ValueError("first_stage_lists and score_maps must align")
    out = []
    for lst, scores in zip(first_stage_lists, score_maps):
        missing = [c for c in lst if c not in scores]
        if missing:
            raise KeyError(f"no rerank score for chunks {missing[:5]}; candidates must cover first stage")
        out.append(sorted(lst, key=lambda c: -scores[c]))  # sorted() is stable
    return out


def rerank_cache_path(
    cache_dir: Path, model_name: str, question_set: str, chunk_size: int, chunk_overlap: int
) -> Path:
    return (
        Path(cache_dir) / "rerank" / model_name.replace("/", "__") / question_set
        / f"cs{chunk_size}_ov{chunk_overlap}"
    )
