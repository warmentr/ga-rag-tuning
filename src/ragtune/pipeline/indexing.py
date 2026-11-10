"""Indexes over one chunking: dense (bge embeddings, NumPy exact search) and BM25.

Determinism:
- GPU embedding is not bit-identical across runs (fp32 differs by ~1e-7), so
  passage embeddings are computed once per chunking and cached to disk. The
  cache is keyed by a fingerprint of the chunks plus the model settings, and
  is rebuilt if either changes. Re-scoring from the cache is exact.
- Top-k uses a stable sort on descending score, so ties always go to the
  lower chunk index, for both index types.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


# ---- shared top-k -----------------------------------------------------------

def topk_stable(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k of a (n_queries, n_chunks) score matrix.

    Returns (indices, scores), each (n_queries, min(k, n_chunks)), ordered by
    descending score with ties broken by lower chunk index.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, scores.shape[1])
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(scores, idx, axis=1)


# ---- dense ------------------------------------------------------------------

class Embedder:
    """bge passage/query embedder (sentence-transformers), L2-normalized float32 output."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str,
        dtype: str = "float32",
        batch_size: int = 64,
        device: str | None = None,
        max_seq_length: int = 512,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model = SentenceTransformer(
            model_name, device=device, model_kwargs={"torch_dtype": getattr(torch, dtype)}
        )
        self.model.max_seq_length = max_seq_length

    @property
    def settings(self) -> dict:
        """Everything that changes passage embeddings; part of the cache key."""
        import sentence_transformers
        import torch

        return {
            "model": self.model_name,
            "dtype": self.dtype,
            "max_seq_length": self.max_seq_length,
            "normalize": True,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        }

    def _encode(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return emb.astype(np.float32, copy=False)

    def embed_passages(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        return self._encode(texts, show_progress)

    def embed_queries(self, questions: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.query_prefix + q for q in questions], show_progress)


@dataclass
class DenseIndex:
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalized float32

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact inner-product (= cosine) search. Returns (chunk indices, scores)."""
        return topk_stable(query_embeddings @ self.embeddings.T, k)


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    """SHA-256 over every chunk's article ID and text, in order."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.article_id.encode())
        h.update(b"\x00")
        h.update(c.text.encode())
        h.update(b"\x01")
    return h.hexdigest()


def strings_fingerprint(items: Sequence[str]) -> str:
    """SHA-256 over a sequence of strings, in order."""
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode())
        h.update(b"\x00")
    return h.hexdigest()


def load_or_build_rows(
    texts: Sequence[str],
    fingerprint: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    settings: dict,
    cache_path: Path,
) -> np.ndarray:
    """Embed `texts` (one row each), using cache_path(.npy/.json) if fingerprint and settings match.

    Any mismatch triggers a rebuild that overwrites the cache.
    """
    cache_path = Path(cache_path)
    npy, meta_path = cache_path.with_suffix(".npy"), cache_path.with_suffix(".json")
    meta = {"fingerprint": fingerprint, "n_chunks": len(texts), "settings": settings}
    if npy.exists() and meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached == meta:
            emb = np.load(npy)
            if emb.shape[0] == len(texts):
                return emb
    emb = embed_fn(list(texts))
    if emb.shape[0] != len(texts):
        raise RuntimeError(f"embed_fn returned {emb.shape[0]} rows for {len(texts)} texts")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, emb)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return emb


def load_or_build_embeddings(
    chunks: Sequence[Chunk],
    embed_fn: Callable[[list[str]], np.ndarray],
    settings: dict,
    cache_path: Path,
) -> np.ndarray:
    """Passage embeddings for `chunks`, cached and keyed by the chunk fingerprint + settings."""
    return load_or_build_rows(
        [c.text for c in chunks], chunks_fingerprint(chunks), embed_fn, settings, cache_path
    )


def embedding_cache_path(cache_dir: Path, model_name: str, chunk_size: int, chunk_overlap: int) -> Path:
    return Path(cache_dir) / "embeddings" / model_name.replace("/", "__") / f"cs{chunk_size}_ov{chunk_overlap}"


# ---- BM25 -------------------------------------------------------------------

class BM25Index:
    """bm25s over chunk texts. Scores every chunk, then uses `topk_stable`."""

    def __init__(self, texts: Sequence[str], lowercase: bool = True, stopwords: str = "en",
                 k1: float = 1.5, b: float = 0.75, method: str = "lucene"):
        import bm25s

        self._bm25s = bm25s
        self.lowercase = lowercase
        self.stopwords = stopwords
        self.n_chunks = len(texts)
        self.retriever = bm25s.BM25(k1=k1, b=b, method=method)
        self.retriever.index(self._tokenize(texts), show_progress=False)

    def _tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        return self._bm25s.tokenize(
            list(texts), lower=self.lowercase, stopwords=self.stopwords,
            return_ids=False, show_progress=False,
        )

    def scores(self, questions: Sequence[str]) -> np.ndarray:
        """(n_questions, n_chunks) BM25 scores. A query with no terms scores all zeros."""
        out = np.zeros((len(questions), self.n_chunks), dtype=np.float32)
        for i, toks in enumerate(self._tokenize(questions)):
            if toks:
                out[i] = self.retriever.get_scores(toks)
        return out

    def search(self, questions: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        return topk_stable(self.scores(questions), k)
