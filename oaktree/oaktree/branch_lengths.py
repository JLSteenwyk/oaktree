"""Baseline branch-length estimation from quintet frequencies."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import treeswift
from scipy.optimize import minimize

from .trees import QuintetObservation


def _read_tree_from_newick(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _normalize_topology(obs_topology) -> tuple[tuple[str, str], tuple[str, str]]:
    p1, p2 = tuple(sorted(tuple(sorted(p)) for p in obs_topology))
    return (p1, p2)


def _topology_to_index(topology) -> int:
    # Map any labeled quintet topology to one of 15 label-invariant classes by
    # ranking labels seen in pairs and assigning a synthetic singleton label.
    p1, p2 = _normalize_topology(topology)
    labels = sorted(set(p1) | set(p2))
    # Include synthetic singleton to preserve the 5-taxon template mapping.
    labels = labels + ["__SINGLETON__"]
    relabel = {x: chr(ord("a") + i) for i, x in enumerate(labels)}
    r1 = tuple(sorted((relabel[p1[0]], relabel[p1[1]])))
    r2 = tuple(sorted((relabel[p2[0]], relabel[p2[1]])))
    canonical = tuple(sorted((r1, r2)))
    all_topologies = []
    base = ["a", "b", "c", "d", "e"]
    for i, pair1 in enumerate(combinations(base, 2)):
        s1 = set(pair1)
        for pair2 in combinations(base[i + 1 :], 2):
            pass
    # Enumerate 15 disjoint-pair topologies on a,b,c,d,e.
    for pair1 in combinations(base, 2):
        s1 = set(pair1)
        for pair2 in combinations(base, 2):
            s2 = set(pair2)
            if pair1 >= pair2:
                continue
            if s1.isdisjoint(s2):
                all_topologies.append(tuple(sorted((tuple(sorted(pair1)), tuple(sorted(pair2))))))
    all_topologies = sorted(set(all_topologies))
    return all_topologies.index(canonical)


def _leaf_sets_by_node(tree: treeswift.Tree) -> dict[treeswift.Node, set[str]]:
    leaf_sets: dict[treeswift.Node, set[str]] = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            leaf_sets[node] = {str(node.label)}
        else:
            merged: set[str] = set()
            for ch in node.children:
                merged |= leaf_sets[ch]
            leaf_sets[node] = merged
    return leaf_sets


def _resolve_edge_split(
    species_tree: treeswift.Tree,
    edge: object,
) -> tuple[set[str], set[str]]:
    all_taxa = {str(n.label) for n in species_tree.traverse_leaves()}
    if isinstance(edge, treeswift.Node):
        leaf_sets = _leaf_sets_by_node(species_tree)
        side = set(leaf_sets[edge])
    elif isinstance(edge, (set, frozenset, list, tuple)):
        side = set(str(x) for x in edge)
    else:
        raise ValueError("edge must be a treeswift.Node or iterable taxon side")
    if not side or side == all_taxa:
        raise ValueError("edge side must be non-empty proper subset")
    return side, all_taxa - side


def _informative_for_split(quintet_taxa: Iterable[str], side_a: set[str], side_b: set[str]) -> bool:
    q = set(quintet_taxa)
    a = len(q & side_a)
    b = len(q & side_b)
    # Baseline criterion: both sides represented.
    return a >= 1 and b >= 1


def count_quintet_frequencies(
    species_tree: treeswift.Tree,
    edge,
    quintet_observations: Sequence[QuintetObservation],
) -> np.ndarray:
    """Count observed quintet topologies informative for an edge."""
    side_a, side_b = _resolve_edge_split(species_tree, edge)
    counts = np.zeros(15, dtype=float)
    for obs in quintet_observations:
        if not _informative_for_split(obs.taxa, side_a, side_b):
            continue
        idx = _topology_to_index(obs.topology)
        counts[idx] += float(getattr(obs, "weight", 1.0))
    return counts


def estimate_branch_length(
    observed_frequencies: np.ndarray,
) -> float:
    """Method-of-moments baseline from matching-topology dominance.

    Uses max observed topology frequency as proxy for matching probability and
    inverts the star-to-resolved mixture:
      p_match = 1/15 + (14/15) * (1 - exp(-tau))
    """
    obs = np.asarray(observed_frequencies, dtype=float)
    if obs.ndim != 1 or len(obs) != 15:
        raise ValueError("observed_frequencies must be a length-15 vector")
    total = float(np.sum(obs))
    if total <= 0.0:
        return 0.0
    p = float(np.max(obs) / total)
    floor = 1.0 / 15.0
    if p <= floor + 1e-15:
        return 0.0
    residual = max(1e-12, 1.0 - (p - floor) * (15.0 / 14.0))
    tau = -float(np.log(residual))
    return max(0.0, tau)


def optimize_branch_lengths(
    species_tree: treeswift.Tree,
    quintet_observations: Sequence[QuintetObservation],
) -> treeswift.Tree:
    """Assign baseline internal branch lengths from edge-informative quintets."""
    optimized = _read_tree_from_newick(species_tree.newick())
    leaf_sets = _leaf_sets_by_node(optimized)
    all_taxa = {str(n.label) for n in optimized.traverse_leaves()}

    for node in optimized.root.traverse_preorder():
        if node is optimized.root or node.is_leaf():
            continue
        side_a = set(leaf_sets[node])
        side_b = all_taxa - side_a
        if not side_a or not side_b:
            continue
        freq = count_quintet_frequencies(optimized, side_a, quintet_observations)
        tau = estimate_branch_length(freq)
        node.edge_length = float(tau)
    return optimized


def _internal_nodes(tree: treeswift.Tree) -> list[treeswift.Node]:
    return [n for n in tree.root.traverse_preorder() if n is not tree.root and not n.is_leaf()]


def _edge_match_counts(
    species_tree: treeswift.Tree,
    quintet_observations: Sequence[QuintetObservation],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-edge (match_count, total_count) based on frequency vectors."""
    nodes = _internal_nodes(species_tree)
    match = np.zeros(len(nodes), dtype=float)
    total = np.zeros(len(nodes), dtype=float)
    leaf_sets = _leaf_sets_by_node(species_tree)
    all_taxa = {str(n.label) for n in species_tree.traverse_leaves()}
    for i, node in enumerate(nodes):
        side_a = set(leaf_sets[node])
        side_b = all_taxa - side_a
        freq = count_quintet_frequencies(species_tree, side_a, quintet_observations)
        total[i] = float(np.sum(freq))
        match[i] = float(np.max(freq)) if total[i] > 0 else 0.0
    return match, total


def branch_length_log_likelihood(
    taus: np.ndarray,
    match_counts: np.ndarray,
    total_counts: np.ndarray,
) -> float:
    """Pseudo log-likelihood over edges using 15-topology star/resolved mixture."""
    t = np.asarray(taus, dtype=float)
    m = np.asarray(match_counts, dtype=float)
    n = np.asarray(total_counts, dtype=float)
    if t.shape != m.shape or t.shape != n.shape:
        raise ValueError("taus, match_counts, total_counts must have same shape")

    p_match = (1.0 / 15.0) + (14.0 / 15.0) * (1.0 - np.exp(-t))
    p_match = np.clip(p_match, 1e-12, 1.0 - 1e-12)
    p_other = np.clip((1.0 - p_match) / 14.0, 1e-12, 1.0)
    non_match = np.maximum(0.0, n - m)
    ll = np.sum(m * np.log(p_match) + non_match * np.log(p_other))
    return float(ll)


def optimize_branch_lengths_ml(
    species_tree: treeswift.Tree,
    quintet_observations: Sequence[QuintetObservation],
    *,
    max_tau: float = 10.0,
) -> treeswift.Tree:
    """Joint ML-style refinement of internal branch lengths via L-BFGS-B."""
    if max_tau <= 0:
        raise ValueError("max_tau must be > 0")

    # Start from moment baseline.
    init_tree = optimize_branch_lengths(species_tree, quintet_observations)
    nodes = _internal_nodes(init_tree)
    if not nodes:
        return init_tree

    init = np.array([max(0.0, float(n.edge_length or 0.0)) for n in nodes], dtype=float)
    init = np.clip(init, 0.0, max_tau)
    match, total = _edge_match_counts(init_tree, quintet_observations)

    if float(np.sum(total)) <= 0.0:
        return init_tree

    def objective(x: np.ndarray) -> float:
        x = np.clip(np.asarray(x, dtype=float), 0.0, max_tau)
        return -branch_length_log_likelihood(x, match, total)

    res = minimize(
        objective,
        x0=init,
        method="L-BFGS-B",
        bounds=[(0.0, max_tau) for _ in range(len(nodes))],
    )
    x_opt = np.clip(np.asarray(res.x if res.success else init, dtype=float), 0.0, max_tau)
    for node, tau in zip(nodes, x_opt):
        node.edge_length = float(tau)
    return init_tree
