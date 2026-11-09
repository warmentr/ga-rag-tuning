"""WixQA loading and the tuning/test question split.

splits.json stores only question IDs (e.g. "expert_007"), not question text,
so it can be committed without redistributing WixQA content. Use
`load_questions` to join IDs back to questions and labels from data/raw.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk

HF_DATASET = "Wix/WixQA"
CORPUS_CONFIG = "wix_kb_corpus"
QUESTION_CONFIGS = {
    "expert": "wixqa_expertwritten",
    "simulated": "wixqa_simulated",
}
ALL_CONFIGS = [CORPUS_CONFIG, *QUESTION_CONFIGS.values()]


def download_raw(raw_dir: Path) -> None:
    """Download each WixQA config to raw_dir/<config>, skipping ones already present."""
    for config in ALL_CONFIGS:
        target = raw_dir / config
        if target.exists():
            print(f"[skip] {config} already at {target}")
            continue
        ds = load_dataset(HF_DATASET, config)
        ds.save_to_disk(str(target))
        print(f"[done] {config}: {ds}")


def load_corpus(raw_dir: Path):
    """Return the KB corpus as a datasets.Dataset (fields: id, title, contents, article_type, ...)."""
    return load_from_disk(str(raw_dir / CORPUS_CONFIG))["train"]


def load_all_questions(raw_dir: Path) -> dict[str, dict]:
    """Return {question_id: {id, source, question, article_ids}} for the pooled question set."""
    questions = {}
    for source, config in QUESTION_CONFIGS.items():
        ds = load_from_disk(str(raw_dir / config))["train"]
        for i, row in enumerate(ds):
            qid = f"{source}_{i:03d}"
            questions[qid] = {
                "id": qid,
                "source": source,
                "question": row["question"],
                "article_ids": list(row["article_ids"]),
            }
    return questions


def make_splits(question_ids_by_source: dict[str, list[str]], tuning_ratio: float, seed: int) -> dict:
    """Stratified split: within each source, shuffle with `seed` and send round(n * ratio) to tuning."""
    rng = random.Random(seed)
    tuning, test = [], []
    for source in sorted(question_ids_by_source):
        ids = sorted(question_ids_by_source[source])
        rng.shuffle(ids)
        n_tune = round(len(ids) * tuning_ratio)
        tuning += ids[:n_tune]
        test += ids[n_tune:]
    return {
        "seed": seed,
        "tuning_ratio": tuning_ratio,
        "stratified_by": "source",
        "counts": {"tuning": len(tuning), "test": len(test)},
        "tuning": sorted(tuning),
        "test": sorted(test),
    }


def build_splits(raw_dir: Path, tuning_ratio: float, seed: int) -> dict:
    questions = load_all_questions(raw_dir)
    by_source: dict[str, list[str]] = {}
    for q in questions.values():
        by_source.setdefault(q["source"], []).append(q["id"])
    return make_splits(by_source, tuning_ratio, seed)


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


def load_questions(raw_dir: Path, splits_path: Path, split: str) -> list[dict]:
    """Return the question records for split ("tuning" or "test"), in a stable order."""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    questions = load_all_questions(raw_dir)
    return [questions[qid] for qid in splits[split]]
