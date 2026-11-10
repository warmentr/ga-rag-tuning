import numpy as np
import pytest

from ragtune.pipeline.chunking import Chunk
from ragtune.pipeline.rerank import candidate_union, load_or_build_scores, rerank
from ragtune.pipeline.retrieval import Rankings, first_stage


def _chunks(texts):
    return [Chunk("a", i, t, 0, len(t), 1) for i, t in enumerate(texts)]


def test_candidate_union_sorted_unique():
    r = Rankings(bm25=np.array([[3, 1, 2]]), dense=np.array([[2, 5, 1]]))
    assert candidate_union(r)[0].tolist() == [1, 2, 3, 5]


def test_union_covers_every_first_stage_list():
    rng = np.random.default_rng(0)
    bm25 = np.array([rng.permutation(200)[:50] for _ in range(30)])
    dense = np.array([rng.permutation(200)[:50] for _ in range(30)])
    r = Rankings(bm25=bm25, dense=dense)
    cands = [set(c.tolist()) for c in candidate_union(r)]
    for t in ("bm25", "dense", "hybrid"):
        for k in (5, 10, 20, 30, 50):
            for lst, cs in zip(first_stage(r, t, k, 60), cands):
                assert set(lst) <= cs


def test_rerank_orders_by_score_ties_keep_first_stage_order():
    maps = [{0: 0.1, 1: 0.9, 2: 0.5, 3: 0.5}]
    assert rerank([[0, 3, 2, 1]], maps) == [[1, 3, 2, 0]]
    assert rerank([[0, 2, 3, 1]], maps) == [[1, 2, 3, 0]]


def test_rerank_only_reorders_never_adds():
    maps = [{0: 0.1, 1: 0.9, 2: 5.0}]
    assert rerank([[0, 1]], maps) == [[1, 0]]  # chunk 2 not in first stage


def test_rerank_missing_score_raises():
    with pytest.raises(KeyError):
        rerank([[0, 7]], [{0: 1.0}])


class FakeScorer:
    """Scores a pair by word overlap; counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, pairs):
        self.calls += 1
        return np.array([len(set(q.split()) & set(p.split())) for q, p in pairs], dtype=np.float32)


def test_scores_cache_and_invalidation(tmp_path):
    chunks = _chunks(["a b c", "a x", "b c d", "z"])
    qs = ["a b", "c d"]
    cands = [np.array([0, 1, 3]), np.array([0, 2])]
    fn, path = FakeScorer(), tmp_path / "r" / "cs128_ov0"
    m1 = load_or_build_scores(qs, chunks, cands, fn, {"model": "m"}, path)
    assert m1 == [{0: 2.0, 1: 1.0, 3: 0.0}, {0: 1.0, 2: 2.0}]
    m2 = load_or_build_scores(qs, chunks, cands, fn, {"model": "m"}, path)
    assert fn.calls == 1 and m1 == m2

    load_or_build_scores(qs, chunks, [np.array([0, 1]), np.array([0, 2])], fn, {"model": "m"}, path)
    assert fn.calls == 2  # candidates changed
    load_or_build_scores(["a b", "c"], chunks, [np.array([0, 1]), np.array([0, 2])], fn, {"model": "m"}, path)
    assert fn.calls == 3  # questions changed
    load_or_build_scores(["a b", "c"], chunks, [np.array([0, 1]), np.array([0, 2])], fn, {"model": "m2"}, path)
    assert fn.calls == 4  # settings changed


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_reranker_prefers_relevant_and_keeps_order():
    from ragtune.pipeline.rerank import Reranker

    r = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [
        ("how do I connect my domain", "Refund a payment from your dashboard."),
        ("how do I connect my domain", "Connect a custom domain to your Wix site in a few steps."),
        ("how do I refund", "Refund a payment from your dashboard."),
    ]
    s = r.score_pairs(pairs)
    assert s.shape == (3,) and s.dtype == np.float32
    assert s[1] > s[0] and s[2] > s[0]
    # length sorting must not scramble the output order
    s_rev = r.score_pairs(pairs[::-1])
    assert np.allclose(s_rev[::-1], s, atol=1e-4)
