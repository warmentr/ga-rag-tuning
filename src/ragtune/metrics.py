"""Article-level nDCG@k (see `fitness` in config/experiment.yaml).

Per question:
  1. The pipeline returns a ranked list of chunks; each chunk carries the ID
     of the article it came from.
  2. Collapse to articles by first occurrence, dropping duplicates.
  3. Binary nDCG@k over that article list: an article is relevant iff it is
     in the question's labeled `article_ids`.
     DCG  = sum over ranks r = 1..k of rel_r / log2(r + 1)
     IDCG = the DCG with min(len(relevant), k) relevant articles at the top.

A list with fewer than k articles is scored as-is (missing ranks count as
non-relevant), so a small top_k that fills fewer than k articles is penalized.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def chunks_to_articles(chunk_article_ids: Iterable[str]) -> list[str]:
    """Ranked chunk -> ranked unique articles, by first occurrence."""
    return list(dict.fromkeys(chunk_article_ids))


def _discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2))


def ndcg_at_k(ranked_articles: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary nDCG@k for one ranked list of (unique) article IDs."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("relevant must be non-empty (every WixQA question has labels)")
    if len(set(ranked_articles)) != len(ranked_articles):
        raise ValueError("ranked_articles has duplicates; collapse with chunks_to_articles first")

    top = ranked_articles[:k]
    gains = np.array([a in relevant for a in top], dtype=float)
    dcg = float(gains @ _discounts(len(top)))
    idcg = float(_discounts(min(len(relevant), k)).sum())
    return dcg / idcg


def score_question(chunk_article_ids: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Fitness for one question: ranked chunks' article IDs -> article-level nDCG@k."""
    return ndcg_at_k(chunks_to_articles(chunk_article_ids), relevant, k)


def mean_score(
    retrieved: Sequence[Iterable[str]], relevant: Sequence[Iterable[str]], k: int
) -> float:
    """Mean article-level nDCG@k over a question set (aligned lists)."""
    if len(retrieved) != len(relevant):
        raise ValueError(f"{len(retrieved)} retrieval lists vs {len(relevant)} label sets")
    if not retrieved:
        raise ValueError("empty question set")
    return float(np.mean([score_question(r, l, k) for r, l in zip(retrieved, relevant)]))
