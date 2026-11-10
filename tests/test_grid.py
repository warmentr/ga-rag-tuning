import numpy as np

from ragtune.grid import score_config
from ragtune.pipeline.retrieval import Rankings
from ragtune.space import Config


def test_score_config_all_paths():
    # 4 chunks: articles X, X, Y, Z ; one question with label Y
    aids = np.array(["X", "X", "Y", "Z"])
    r = Rankings(bm25=np.array([[0, 1, 2, 3]]), dense=np.array([[2, 3, 0, 1]]))
    maps = [{0: 0.0, 1: 0.0, 2: 9.0, 3: 1.0}]
    labels = [["Y"]]
    L3 = 1 / np.log2(3)

    s = lambda **kw: score_config(Config(128, 0, **kw), r, maps, aids, labels, 5, 60)[0]
    # bm25 top 2 = chunks 0,1 -> articles [X] -> Y missing
    assert s(retrieval_type="bm25", reranker=False, top_k=2) == 0.0
    # bm25 top 3 -> articles [X, Y] -> Y at rank 2
    assert np.isclose(s(retrieval_type="bm25", reranker=False, top_k=3), L3)
    # rerank of bm25 top 3 puts chunk 2 first
    assert np.isclose(s(retrieval_type="bm25", reranker=True, top_k=3), 1.0)
    # dense top 1 = chunk 2
    assert np.isclose(s(retrieval_type="dense", reranker=False, top_k=1), 1.0)
    # hybrid top 2: bm25 [0,1], dense [2,3] -> ties at rank1 (0 vs 2) -> 0 first, then 2
    assert np.isclose(s(retrieval_type="hybrid", reranker=False, top_k=2), L3)
