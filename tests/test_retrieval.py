import numpy as np
import pytest

from ragtune.pipeline.indexing import BM25Index, DenseIndex
from ragtune.pipeline.retrieval import Rankings, first_stage, rank_questions, rrf_fuse

A, B, C, D, E = range(5)


def test_rrf_hand_computed():
    # dense [A,B,C], bm25 [B,D,A], rrf_k=60
    # B = 1/62 + 1/61, A = 1/61 + 1/63, D = 1/62, C = 1/63
    assert rrf_fuse([[A, B, C], [B, D, A]], 4, 60) == [B, A, D, C]
    assert rrf_fuse([[A, B, C], [B, D, A]], 2, 60) == [B, A]


def test_rrf_ties_go_to_lower_index():
    # D only in list 1 at rank 1, B only in list 2 at rank 1 -> tie, B (1) before D (3)
    assert rrf_fuse([[D], [B]], 2, 60) == [B, D]


def test_rrf_identical_lists_preserve_order():
    assert rrf_fuse([[C, A, B], [C, A, B]], 3, 60) == [C, A, B]


def _rankings():
    bm25 = np.array([[1, 3, 0, 4, 2], [0, 1, 2, 3, 4]])
    dense = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    return Rankings(bm25=bm25, dense=dense)


def test_single_types_are_prefixes():
    r = _rankings()
    assert first_stage(r, "bm25", 3, 60) == [[1, 3, 0], [0, 1, 2]]
    assert first_stage(r, "dense", 2, 60) == [[0, 1], [4, 3]]


def test_hybrid_fuses_top_k_of_each():
    r = _rankings()
    # q0, top_k=2: bm25 [1,3], dense [0,1] -> 1 = 1/61+1/62 ; 0 = 1/61 ; 3 = 1/62
    assert first_stage(r, "hybrid", 2, 60)[0] == [1, 0]
    # q0, top_k=3: bm25 [1,3,0], dense [0,1,2]
    # 0 = 1/63+1/61, 1 = 1/61+1/62, 3 = 1/62, 2 = 1/63 -> 1 (1/61+1/62) > 0 (1/61+1/63)
    assert first_stage(r, "hybrid", 3, 60)[0] == [1, 0, 3]


def test_hybrid_uses_only_top_k_not_full_depth():
    # chunk 9 is deep in both lists; with top_k=2 it must not appear
    r = Rankings(bm25=np.array([[1, 2, 9]]), dense=np.array([[3, 4, 9]]))
    assert 9 not in first_stage(r, "hybrid", 2, 60)[0]
    assert len(first_stage(r, "hybrid", 2, 60)[0]) == 2


def test_output_length_equals_top_k():
    r = _rankings()
    for t in ("bm25", "dense", "hybrid"):
        for k in (1, 3, 5):
            assert all(len(x) == k for x in first_stage(r, t, k, 60))


@pytest.mark.parametrize("args", [("sparse", 3), ("bm25", 0), ("dense", 6)])
def test_first_stage_rejects_bad_args(args):
    with pytest.raises(ValueError):
        first_stage(_rankings(), args[0], args[1], 60)


def test_rankings_shape_mismatch():
    with pytest.raises(ValueError):
        Rankings(bm25=np.zeros((2, 5), int), dense=np.zeros((2, 4), int))


def test_rank_questions_end_to_end():
    texts = ["custom domain setup", "payments payouts", "refund a payment", "site seo tips"]
    emb = np.eye(4, dtype=np.float32)
    q_emb = np.array([[0, 0, 1, 0]], dtype=np.float32)  # dense prefers chunk 2
    r = rank_questions(["connect domain"], q_emb, DenseIndex(emb), BM25Index(texts), depth=3)
    assert r.depth == 3 and r.n_questions == 1
    assert r.dense[0, 0] == 2 and r.bm25[0, 0] == 0
    with pytest.raises(ValueError):
        rank_questions(["x"], q_emb, DenseIndex(emb), BM25Index(texts), depth=10)
