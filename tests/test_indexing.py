import json

import numpy as np
import pytest

from ragtune.pipeline.chunking import Chunk
from ragtune.pipeline.indexing import (
    BM25Index,
    DenseIndex,
    chunks_fingerprint,
    load_or_build_embeddings,
    topk_stable,
)


def _chunks(texts, aid="a"):
    return [Chunk(aid, i, t, 0, len(t), 1) for i, t in enumerate(texts)]


# ---- topk_stable ------------------------------------------------------------

def test_topk_orders_descending():
    idx, sc = topk_stable(np.array([[0.1, 0.9, 0.5, 0.7]]), 3)
    assert idx.tolist() == [[1, 3, 2]]
    assert sc[0].tolist() == pytest.approx([0.9, 0.7, 0.5])


def test_topk_ties_go_to_lower_index():
    idx, _ = topk_stable(np.array([[0.5, 0.9, 0.5, 0.9, 0.5]]), 4)
    assert idx.tolist() == [[1, 3, 0, 2]]


def test_topk_k_larger_than_n():
    idx, _ = topk_stable(np.array([[0.2, 0.1], [0.0, 0.3]]), 50)
    assert idx.tolist() == [[0, 1], [1, 0]]


def test_topk_prefix_property():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, size=(20, 100)).astype(float)  # many ties
    big, _ = topk_stable(s, 50)
    for k in (5, 10, 20, 30):
        assert np.array_equal(topk_stable(s, k)[0], big[:, :k])


# ---- dense ------------------------------------------------------------------

def test_dense_search_exact():
    emb = np.eye(4, dtype=np.float32)
    q = np.array([[0.0, 0.6, 0.8, 0.0]], dtype=np.float32)
    idx, sc = DenseIndex(emb).search(q, 2)
    assert idx.tolist() == [[2, 1]]
    assert sc[0].tolist() == pytest.approx([0.8, 0.6])


# ---- embedding cache ----------------------------------------------------------

class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)


def test_cache_hit_and_invalidation(tmp_path):
    chunks = _chunks(["one", "two", "three"])
    emb_fn = CountingEmbedder()
    path = tmp_path / "e" / "cs128_ov0"
    settings = {"model": "m", "dtype": "float32"}

    a = load_or_build_embeddings(chunks, emb_fn, settings, path)
    b = load_or_build_embeddings(chunks, emb_fn, settings, path)
    assert emb_fn.calls == 1 and np.array_equal(a, b)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["n_chunks"] == 3 and meta["fingerprint"] == chunks_fingerprint(chunks)

    load_or_build_embeddings(chunks, emb_fn, {**settings, "dtype": "float16"}, path)
    assert emb_fn.calls == 2  # settings changed

    load_or_build_embeddings(_chunks(["one", "two", "THREE"]), emb_fn, {**settings, "dtype": "float16"}, path)
    assert emb_fn.calls == 3  # chunk text changed


def test_fingerprint_sensitive_to_article_and_boundaries():
    base = chunks_fingerprint(_chunks(["ab", "c"]))
    assert base != chunks_fingerprint(_chunks(["a", "bc"]))
    assert base != chunks_fingerprint(_chunks(["ab", "c"], aid="other"))
    assert base == chunks_fingerprint(_chunks(["ab", "c"]))


# ---- BM25 -------------------------------------------------------------------

DOCS = [
    "How to connect a custom domain to your Wix site",
    "Accepting payments with Wix Payments",
    "Payments payouts and refunds",
    "the and of",  # all stopwords -> empty document
]


def test_bm25_ranks_matching_docs():
    idx, sc = BM25Index(DOCS).search(["How do I connect my domain?", "refunds for payments"], 2)
    assert idx[0, 0] == 0
    assert idx[1, 0] == 2
    assert (sc[:, 0] > 0).all()


def test_bm25_empty_query_scores_zero():
    s = BM25Index(DOCS).scores(["the of and", "a"])
    assert s.shape == (2, 4) and not s.any()


def test_bm25_lowercases():
    bm = BM25Index(DOCS)
    assert np.array_equal(bm.scores(["PAYMENTS"]), bm.scores(["payments"]))


# ---- real embedder (GPU) --------------------------------------------------------

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_embedder_normalized_and_prefix_matters():
    from ragtune.pipeline.indexing import Embedder

    e = Embedder("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ")
    p = e.embed_passages(["Connect a custom domain to your Wix site."], show_progress=False)
    q = e.embed_queries(["How do I connect my domain?"])
    assert p.shape == (1, 384) and p.dtype == np.float32
    assert np.linalg.norm(p, axis=1) == pytest.approx([1.0], abs=1e-5)
    raw = e.embed_passages(["How do I connect my domain?"], show_progress=False)
    assert not np.allclose(q, raw)  # prefix was applied
    assert float((q @ p.T)[0, 0]) > 0.5
