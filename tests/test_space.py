from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ragtune.space import Config, Rule, SearchSpace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return SearchSpace.from_yaml(ROOT / "config" / "search_space.yaml")


def test_sizes(space):
    assert space.n_total == 600
    assert space.n_valid == 570
    assert [space.budget(f) for f in (0.05, 0.10, 0.20)] == [28, 57, 114]
    assert len(space.chunkings()) == 19


def test_validity_rule(space):
    assert not space.is_valid(Config(128, 128, "bm25", False, 5))
    assert space.is_valid(Config(192, 128, "bm25", False, 5))
    assert all(c.chunk_overlap < c.chunk_size for c in space.valid_configs())


def test_encode_decode_roundtrip(space):
    for c in space.valid_configs():
        assert space.decode(space.encode(c)) == c


def test_keys_unique(space):
    keys = [c.key for c in space.valid_configs()]
    assert len(set(keys)) == len(keys)
    assert Config(256, 32, "hybrid", True, 20).key == "cs256_ov32_hybrid_rr1_k20"


def test_decode_rejects_bad_genomes(space):
    with pytest.raises(ValueError):
        space.decode((0, 0, 0, 0))
    with pytest.raises(ValueError):
        space.decode((5, 0, 0, 0, 0))


def test_repair_leaves_valid_unchanged(space):
    rng = np.random.default_rng(0)
    g = space.encode(Config(512, 64, "dense", True, 50))
    assert space.repair(g, rng) == g


def test_repair_only_touches_rule_genes(space):
    rng = np.random.default_rng(0)
    bad = space.encode(Config(128, 128, "hybrid", True, 30))
    for _ in range(200):
        fixed = space.repair(bad, rng)
        assert space.is_valid(space.decode(fixed))
        assert fixed[2:] == bad[2:]


def test_sample_is_deterministic(space):
    a = [space.sample(np.random.default_rng(42)) for _ in range(3)]
    b = [space.sample(np.random.default_rng(42)) for _ in range(3)]
    assert a == b


def test_sample_is_uniform_over_valid(space):
    rng = np.random.default_rng(1)
    n = 200_000
    counts = Counter(space.sample(rng).chunking for _ in range(n))
    assert set(counts) == set(space.chunkings())
    expected = n / len(space.chunkings())
    # 19 cells; each within 5% of uniform is far outside chance failure
    assert all(abs(v - expected) / expected < 0.05 for v in counts.values())


def test_rule_parsing():
    assert Rule.parse("chunk_overlap < chunk_size").genes == ("chunk_overlap", "chunk_size")
    assert Rule.parse("top_k >= 5").genes == ("top_k",)
    for bad in ["chunk_overlap<chunk_size", "foo < chunk_size", "top_k ~ 5"]:
        with pytest.raises(ValueError):
            Rule.parse(bad)
