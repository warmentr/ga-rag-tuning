"""First-stage retrieval: bm25, dense, or hybrid (RRF), at a given top_k.

For a question set, each index is searched once to `depth` (the largest
top_k in the search space). Every (retrieval_type, top_k) is then derived
from those rankings without re-searching:
- bm25 / dense: the first top_k of the ranking (exact, since `topk_stable`
  makes smaller top-k lists prefixes of larger ones).
- hybrid: take the bm25 top_k and the dense top_k, fuse with reciprocal rank
  fusion, keep the top_k fused chunks.
      RRF(c) = sum over lists containing c of 1 / (rrf_k + rank_in_list),
      ranks starting at 1.
  Ties in fused score go to the lower chunk index (same rule as topk_stable).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .indexing import BM25Index, DenseIndex

RETRIEVAL_TYPES = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class Rankings:
    """Depth-limited chunk rankings for one question set on one chunking."""

    bm25: np.ndarray   # (n_questions, depth) chunk indices
    dense: np.ndarray  # (n_questions, depth) chunk indices

    def __post_init__(self):
        if self.bm25.shape != self.dense.shape:
            raise ValueError(f"shape mismatch: bm25 {self.bm25.shape} vs dense {self.dense.shape}")

    @property
    def depth(self) -> int:
        return self.bm25.shape[1]

    @property
    def n_questions(self) -> int:
        return self.bm25.shape[0]


def rank_questions(
    questions: Sequence[str],
    query_embeddings: np.ndarray,
    dense: DenseIndex,
    bm25: BM25Index,
    depth: int,
) -> Rankings:
    """Search both indexes once to `depth`."""
    if len(questions) != query_embeddings.shape[0]:
        raise ValueError("questions and query_embeddings must align")
    dense_idx, _ = dense.search(query_embeddings, depth)
    bm25_idx, _ = bm25.search(questions, depth)
    if dense_idx.shape[1] < depth:
        raise ValueError(f"index has only {dense_idx.shape[1]} chunks, need depth {depth}")
    return Rankings(bm25=bm25_idx, dense=dense_idx)


def rrf_fuse(rank_lists: Sequence[Sequence[int]], top_k: int, rrf_k: int) -> list[int]:
    """Reciprocal rank fusion of ranked chunk-index lists; returns the top_k fused."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, c in enumerate(ranking, start=1):
            scores[int(c)] = scores.get(int(c), 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda c: (-scores[c], c))[:top_k]


def first_stage(rankings: Rankings, retrieval_type: str, top_k: int, rrf_k: int) -> list[list[int]]:
    """Ranked chunk indices per question for one (retrieval_type, top_k)."""
    if retrieval_type not in RETRIEVAL_TYPES:
        raise ValueError(f"retrieval_type must be one of {RETRIEVAL_TYPES}, got {retrieval_type!r}")
    if not 1 <= top_k <= rankings.depth:
        raise ValueError(f"top_k must be in [1, {rankings.depth}], got {top_k}")
    if retrieval_type == "bm25":
        return rankings.bm25[:, :top_k].tolist()
    if retrieval_type == "dense":
        return rankings.dense[:, :top_k].tolist()
    return [
        rrf_fuse([b[:top_k], d[:top_k]], top_k, rrf_k)
        for b, d in zip(rankings.bm25, rankings.dense)
    ]
