"""Search space: Config dataclass, validity rules, encode/decode, repair.

Genome representation: a tuple of integer indices, one per gene, in the gene
order of config/search_space.yaml. Index i of gene g means the i-th listed
value of g. Both search methods use this representation.

Invalid configs are handled by `SearchSpace.repair`, shared by both methods:
the genes named in any violated rule are resampled jointly and uniformly
until every rule holds. Uniform sampling followed by repair is therefore
uniform over the valid configs (see `sample`).
"""

from __future__ import annotations

import itertools
import operator
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

Genome = tuple[int, ...]


@dataclass(frozen=True, order=True)
class Config:
    chunk_size: int
    chunk_overlap: int
    retrieval_type: str
    reranker: bool
    top_k: int

    @property
    def key(self) -> str:
        """Stable string ID, used as the lookup-table key."""
        return (
            f"cs{self.chunk_size}_ov{self.chunk_overlap}_{self.retrieval_type}"
            f"_rr{int(self.reranker)}_k{self.top_k}"
        )

    @property
    def chunking(self) -> tuple[int, int]:
        """(chunk_size, chunk_overlap): configs sharing this share an index."""
        return (self.chunk_size, self.chunk_overlap)


GENE_NAMES = tuple(f.name for f in fields(Config))

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Rule:
    """A validity rule of the form `<operand> <op> <operand>`.

    Each operand is a gene name or a numeric literal.
    """

    text: str
    left: str | float
    op: str
    right: str | float

    @classmethod
    def parse(cls, text: str) -> "Rule":
        parts = text.split()
        if len(parts) != 3 or parts[1] not in _OPS:
            raise ValueError(f"Rule must be '<operand> <op> <operand>', got: {text!r}")
        left, op, right = parts
        return cls(text, _operand(left), op, _operand(right))

    @property
    def genes(self) -> tuple[str, ...]:
        return tuple(x for x in (self.left, self.right) if isinstance(x, str))

    def holds(self, config: Config) -> bool:
        def val(x):
            return getattr(config, x) if isinstance(x, str) else x

        return _OPS[self.op](val(self.left), val(self.right))


def _operand(token: str) -> str | float:
    if token in GENE_NAMES:
        return token
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"Unknown gene or non-numeric literal in rule: {token!r}") from None


class SearchSpace:
    def __init__(self, genes: dict[str, list], rules: list[str]):
        if tuple(genes) != GENE_NAMES:
            raise ValueError(f"Genes must be {GENE_NAMES} in that order, got {tuple(genes)}")
        for name, values in genes.items():
            if len(values) == 0 or len(set(values)) != len(values):
                raise ValueError(f"Gene {name!r} needs distinct, non-empty values: {values}")
        self.genes = {name: tuple(values) for name, values in genes.items()}
        self.rules = tuple(Rule.parse(r) for r in rules)
        self._valid = tuple(c for c in self._all_configs() if self.is_valid(c))
        self._valid_set = frozenset(self._valid)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchSpace":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(cfg["genes"], cfg.get("validity") or [])

    # ---- sizes -------------------------------------------------------------

    @property
    def cardinalities(self) -> tuple[int, ...]:
        """Number of values per gene (genome index upper bounds, exclusive)."""
        return tuple(len(v) for v in self.genes.values())

    @property
    def n_total(self) -> int:
        return int(np.prod(self.cardinalities))

    @property
    def n_valid(self) -> int:
        return len(self._valid)

    def budget(self, fraction: float) -> int:
        """Unique-evaluation budget for a fraction of the valid space, rounded down."""
        return int(fraction * self.n_valid)

    # ---- enumeration -------------------------------------------------------

    def _all_configs(self):
        for values in itertools.product(*self.genes.values()):
            yield Config(*values)

    def valid_configs(self) -> tuple[Config, ...]:
        """All valid configs, in a fixed order (gene order, value order)."""
        return self._valid

    def chunkings(self) -> tuple[tuple[int, int], ...]:
        """Distinct valid (chunk_size, chunk_overlap) pairs, in order."""
        return tuple(dict.fromkeys(c.chunking for c in self._valid))

    # ---- validity ----------------------------------------------------------

    def violated(self, config: Config) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if not r.holds(config))

    def is_valid(self, config: Config) -> bool:
        return not self.violated(config)

    # ---- encode / decode ---------------------------------------------------

    def encode(self, config: Config) -> Genome:
        return tuple(self.genes[name].index(v) for name, v in zip(GENE_NAMES, astuple(config)))

    def decode(self, genome) -> Config:
        genome = tuple(int(i) for i in genome)
        if len(genome) != len(GENE_NAMES):
            raise ValueError(f"Genome length {len(genome)} != {len(GENE_NAMES)} genes")
        for i, (idx, n) in enumerate(zip(genome, self.cardinalities)):
            if not 0 <= idx < n:
                raise ValueError(f"Gene {GENE_NAMES[i]!r} index {idx} out of range [0, {n})")
        return Config(*(self.genes[name][i] for name, i in zip(GENE_NAMES, genome)))

    # ---- sampling and repair ----------------------------------------------

    def random_genome(self, rng: np.random.Generator) -> Genome:
        """Uniform over all genomes, valid or not."""
        return tuple(int(rng.integers(n)) for n in self.cardinalities)

    def repair(self, genome, rng: np.random.Generator) -> Genome:
        """Return a valid genome. Valid input is returned unchanged.

        While any rule is violated, the genes named in the violated rules are
        resampled jointly and uniformly; other genes are left alone.
        """
        genome = tuple(int(i) for i in genome)
        while True:
            broken = self.violated(self.decode(genome))
            if not broken:
                return genome
            to_resample = {GENE_NAMES.index(g) for r in broken for g in r.genes}
            g = list(genome)
            for i in sorted(to_resample):
                g[i] = int(rng.integers(self.cardinalities[i]))
            genome = tuple(g)

    def sample(self, rng: np.random.Generator) -> Config:
        """Uniform over valid configs: uniform genome, then repair."""
        return self.decode(self.repair(self.random_genome(rng), rng))
