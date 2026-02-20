"""Tree I/O, Newick parsing, and subtree extraction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Iterable, List, Sequence, Tuple
from weakref import WeakKeyDictionary

import numpy as np
import treeswift
import io

Taxon = str
Split = Tuple[Taxon, Taxon]
QuintetTopology = Tuple[Split, Split]

_LEAF_SET_CACHE: "WeakKeyDictionary[treeswift.Tree, set[Taxon]]" = WeakKeyDictionary()
_QUINTET_TOPOLOGY_CACHE: "WeakKeyDictionary[treeswift.Tree, dict[tuple[Taxon, Taxon, Taxon, Taxon, Taxon], QuintetTopology]]" = (
    WeakKeyDictionary()
)


@dataclass(frozen=True)
class QuintetObservation:
    taxa: Tuple[Taxon, Taxon, Taxon, Taxon, Taxon]
    topology: QuintetTopology
    weight: float = 1.0


def read_gene_trees(path: str) -> List[treeswift.Tree]:
    """Read Newick trees from a file (one per line)."""
    trees: List[treeswift.Tree] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if hasattr(treeswift, "read_tree_newick"):
                tree = treeswift.read_tree_newick(line)
            else:
                tree = treeswift.read_tree(io.StringIO(line), "newick")
            trees.append(tree)
    return trees


def get_leaf_set(tree: treeswift.Tree) -> set[Taxon]:
    cached = _LEAF_SET_CACHE.get(tree)
    if cached is not None:
        return set(cached)
    leaves = set()
    for node in tree.root.traverse_preorder():
        if node.label is None:
            continue
        if node.is_leaf():
            leaves.add(str(node.label))
    _LEAF_SET_CACHE[tree] = set(leaves)
    return leaves


def get_shared_taxa(trees: Sequence[treeswift.Tree]) -> set[Taxon]:
    if not trees:
        return set()
    shared = get_leaf_set(trees[0])
    for tree in trees[1:]:
        shared &= get_leaf_set(tree)
    return shared


def _leaf_sets_by_node(tree: treeswift.Tree) -> dict[treeswift.Node, set[Taxon]]:
    leaf_sets: dict[treeswift.Node, set[Taxon]] = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            leaf_sets[node] = {str(node.label)}
        else:
            merged: set[Taxon] = set()
            for child in node.children:
                merged |= leaf_sets[child]
            leaf_sets[node] = merged
    return leaf_sets


def _internal_splits(tree: treeswift.Tree, taxa: Tuple[Taxon, ...]) -> List[Split]:
    n = len(taxa)
    leaf_sets = _leaf_sets_by_node(tree)
    splits: List[Split] = []
    for node, leaf_set in leaf_sets.items():
        if node is tree.root:
            continue
        size = len(leaf_set)
        if size <= 1 or size >= n - 1:
            continue
        smaller = leaf_set if size <= n // 2 else set(taxa) - leaf_set
        if len(smaller) != 2:
            continue
        splits.append(tuple(sorted(smaller)))
    return splits


def extract_induced_subtree(tree: treeswift.Tree, taxa: Tuple[Taxon, ...]) -> treeswift.Tree:
    taxa_set = set(taxa)
    leaf_set = get_leaf_set(tree)
    if not taxa_set.issubset(leaf_set):
        missing = taxa_set - leaf_set
        raise ValueError(f"Missing taxa in tree: {sorted(missing)}")
    induced = tree.extract_tree_with(taxa_set, suppress_unifurcations=True)
    return induced


def canonicalize_quintet(tree: treeswift.Tree, taxa: Tuple[Taxon, Taxon, Taxon, Taxon, Taxon]) -> QuintetTopology:
    if len(taxa) != 5:
        raise ValueError("Quintet canonicalization requires exactly 5 taxa")
    taxa_sorted = tuple(sorted(taxa))
    per_tree_cache = _QUINTET_TOPOLOGY_CACHE.get(tree)
    if per_tree_cache is None:
        per_tree_cache = {}
        _QUINTET_TOPOLOGY_CACHE[tree] = per_tree_cache
    cached = per_tree_cache.get(taxa_sorted)
    if cached is not None:
        return cached

    induced = extract_induced_subtree(tree, taxa_sorted)
    # Ensure binary resolution for polytomies
    induced.resolve_polytomies()
    induced.suppress_unifurcations()

    splits = _internal_splits(induced, taxa_sorted)
    unique_splits = sorted(set(splits))
    if len(unique_splits) != 2:
        # For unresolved multifurcations, choose a deterministic compatible
        # binary resolution so results are reproducible.
        topologies = []
        all_topologies = _all_topologies_for_taxa(taxa_sorted)
        for topo in all_topologies:
            if all(split in topo for split in unique_splits):
                topologies.append(topo)
        if not topologies:
            raise ValueError(f"Expected 2 internal splits for quintet, got {len(unique_splits)}")
        out = min(topologies)
        per_tree_cache[taxa_sorted] = out
        return out
    out = (unique_splits[0], unique_splits[1])
    per_tree_cache[taxa_sorted] = out
    return out


def enumerate_quintet_topologies() -> List[QuintetTopology]:
    taxa = ("a", "b", "c", "d", "e")
    pairs = [tuple(sorted(p)) for p in combinations(taxa, 2)]
    topologies = set()
    for i, p1 in enumerate(pairs):
        set1 = set(p1)
        for p2 in pairs[i + 1 :]:
            if set1.isdisjoint(p2):
                topo = tuple(sorted((p1, p2)))
                topologies.add(topo)
    return sorted(topologies)


def sample_quintets(
    tree: treeswift.Tree,
    n_samples: int,
    rng: np.random.Generator,
    full_taxa: Sequence[Taxon] | None = None,
) -> List[QuintetObservation]:
    leaves = sorted(get_leaf_set(tree))
    if len(leaves) < 5:
        return []
    if full_taxa is None:
        sampling_pool = leaves
    else:
        sampling_pool = sorted(set(full_taxa))
        if len(sampling_pool) < 5:
            return []

    observations: List[QuintetObservation] = []
    max_attempts = max(n_samples * 20, 100)
    attempts = 0
    leaf_set = set(leaves)
    while len(observations) < n_samples and attempts < max_attempts:
        attempts += 1
        taxa = tuple(sorted(str(x) for x in rng.choice(sampling_pool, size=5, replace=False)))
        # Skip quintets containing taxa missing from this gene tree.
        if not set(taxa).issubset(leaf_set):
            continue
        topology = canonicalize_quintet(tree, taxa)
        observations.append(QuintetObservation(taxa=taxa, topology=topology))
    return observations


def _all_quintet_subsets(taxa: Iterable[Taxon]) -> List[Tuple[Taxon, Taxon, Taxon, Taxon, Taxon]]:
    return [tuple(sorted(c)) for c in combinations(sorted(taxa), 5)]


def sample_quintet_subsets(
    taxa: Sequence[Taxon],
    *,
    max_quintets: int | None,
    rng: np.random.Generator,
) -> List[Tuple[Taxon, Taxon, Taxon, Taxon, Taxon]]:
    """Sample up to `max_quintets` unique quintet subsets without full enumeration."""
    taxa_sorted = tuple(sorted(set(str(t) for t in taxa)))
    n = len(taxa_sorted)
    if n < 5:
        return []
    if max_quintets is None:
        return [tuple(c) for c in combinations(taxa_sorted, 5)]
    k = int(max_quintets)
    if k <= 0:
        return []
    total = comb(n, 5)
    if k >= total:
        return [tuple(c) for c in combinations(taxa_sorted, 5)]

    # Exact uniform sampling over [0, C(n,5)) without materializing all quintets.
    ranks = sorted(int(i) for i in rng.choice(total, size=k, replace=False))

    def unrank_combination(rank: int) -> tuple[int, int, int, int, int]:
        out: list[int] = []
        x = 0
        remaining_rank = int(rank)
        choose = 5
        for pos in range(5):
            for j in range(x, n):
                c = comb(n - j - 1, choose - pos - 1) if choose - pos - 1 >= 0 else 0
                if remaining_rank < c:
                    out.append(j)
                    x = j + 1
                    break
                remaining_rank -= c
        return (out[0], out[1], out[2], out[3], out[4])

    sampled: list[Tuple[Taxon, Taxon, Taxon, Taxon, Taxon]] = []
    for r in ranks:
        i0, i1, i2, i3, i4 = unrank_combination(r)
        sampled.append((taxa_sorted[i0], taxa_sorted[i1], taxa_sorted[i2], taxa_sorted[i3], taxa_sorted[i4]))
    return sampled


@lru_cache(maxsize=128)
def _all_topologies_for_taxa(taxa: Tuple[Taxon, Taxon, Taxon, Taxon, Taxon]) -> List[QuintetTopology]:
    pairs = [tuple(sorted(p)) for p in combinations(taxa, 2)]
    out = set()
    for i, p1 in enumerate(pairs):
        s1 = set(p1)
        for p2 in pairs[i + 1 :]:
            if s1.isdisjoint(p2):
                out.add(tuple(sorted((p1, tuple(sorted(p2))))))
    return sorted(out)
