"""Hand-computed nDCG cases. log2(3) = 1.5849625, log2(4) = 2, log2(6) = 2.5849625."""

import math

import pytest

from ragtune.metrics import chunks_to_articles, mean_score, ndcg_at_k, score_question

L3 = math.log2(3)


def test_chunks_to_articles_first_occurrence():
    assert chunks_to_articles(["A", "A", "B", "A", "C", "B"]) == ["A", "B", "C"]
    assert chunks_to_articles([]) == []


def test_single_relevant_at_rank_1():
    assert ndcg_at_k(["A", "B", "C"], {"A"}, 5) == pytest.approx(1.0)


def test_single_relevant_at_rank_2():
    assert ndcg_at_k(["X", "A", "C"], {"A"}, 5) == pytest.approx(1 / L3)  # 0.6309


def test_single_relevant_at_rank_5_and_6():
    ranked = ["X1", "X2", "X3", "X4", "A", "X5"]
    assert ndcg_at_k(ranked, {"A"}, 5) == pytest.approx(1 / math.log2(6))  # 0.3869
    ranked = ["X1", "X2", "X3", "X4", "X5", "A"]
    assert ndcg_at_k(ranked, {"A"}, 5) == 0.0


def test_two_relevant_ranks_1_and_3():
    # DCG = 1 + 1/2 = 1.5 ; IDCG = 1 + 1/log2(3) = 1.6309 ; nDCG = 0.9197
    assert ndcg_at_k(["A", "X", "B"], {"A", "B"}, 5) == pytest.approx(1.5 / (1 + 1 / L3))


def test_partial_recall_ideal_uses_all_labels():
    # 3 labels, only one retrieved at rank 1: 1 / (1 + 1/log2(3) + 1/2) = 0.4693
    assert ndcg_at_k(["A", "X"], {"A", "B", "C"}, 5) == pytest.approx(1 / (1 + 1 / L3 + 0.5))


def test_ideal_capped_at_k():
    rel = {f"R{i}" for i in range(7)}
    assert ndcg_at_k(sorted(rel)[:5], rel, 5) == pytest.approx(1.0)


def test_short_and_empty_lists():
    assert ndcg_at_k(["A"], {"A"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k([], {"A"}, 5) == 0.0


def test_score_question_dedupes_chunks():
    # chunks A,A,B,C -> articles A,B,C ; B at rank 2
    assert score_question(["A", "A", "B", "C"], {"B"}, 5) == pytest.approx(1 / L3)


def test_dedupe_lets_deep_chunk_into_top_k():
    # six chunks but only two articles: B is article rank 2, not chunk rank 6
    assert score_question(["A"] * 5 + ["B"], {"B"}, 5) == pytest.approx(1 / L3)


def test_small_top_k_penalized_by_truncation():
    # with top_k=3 chunks all from A, B never appears
    assert score_question(["A", "A", "A"], {"B"}, 5) == 0.0


def test_mean_score():
    got = mean_score([["A"], ["X", "B"]], [{"A"}, {"B"}], 5)
    assert got == pytest.approx((1 + 1 / L3) / 2)


@pytest.mark.parametrize(
    "args",
    [
        (["A"], set(), 5),          # no labels
        (["A"], {"A"}, 0),          # bad k
        (["A", "A"], {"A"}, 5),     # duplicate articles passed directly
    ],
)
def test_ndcg_rejects_bad_input(args):
    with pytest.raises(ValueError):
        ndcg_at_k(*args)


def test_mean_score_rejects_misaligned():
    with pytest.raises(ValueError):
        mean_score([["A"]], [{"A"}, {"B"}], 5)
