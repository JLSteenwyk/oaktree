"""Baseline branch-length estimation from quintet frequencies."""

from __future__ import annotations

from functools import lru_cache
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


_BASE_TOPOLOGIES: list[tuple[tuple[str, str], tuple[str, str]]] = sorted(
    {
        tuple(sorted((tuple(sorted(pair1)), tuple(sorted(pair2)))))
        for pair1 in combinations(("a", "b", "c", "d", "e"), 2)
        for pair2 in combinations(("a", "b", "c", "d", "e"), 2)
        if pair1 < pair2 and set(pair1).isdisjoint(pair2)
    }
)


@lru_cache(maxsize=1024)
def _topology_to_index_cached(
    normalized_topology: tuple[tuple[str, str], tuple[str, str]],
) -> int:
    # Map labels to a,b,c,d preserving relative sort order and assign
    # singleton to e (consistent with previous implementation semantics).
    p1, p2 = normalized_topology
    labels = sorted(set(p1) | set(p2))
    relabel = {x: chr(ord("a") + i) for i, x in enumerate(labels)}
    r1 = tuple(sorted((relabel[p1[0]], relabel[p1[1]])))
    r2 = tuple(sorted((relabel[p2[0]], relabel[p2[1]])))
    canonical = tuple(sorted((r1, r2)))
    return _BASE_TOPOLOGIES.index(canonical)


def _topology_to_index(topology) -> int:
    return _topology_to_index_cached(_normalize_topology(topology))


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


def _estimate_branch_length_from_match_total(match: float, total: float) -> float:
    """Fast equivalent of `estimate_branch_length` from max/total only."""
    if total <= 0.0:
        return 0.0
    p = float(match) / float(total)
    floor = 1.0 / 15.0
    if p <= floor + 1e-15:
        return 0.0
    residual = max(1e-12, 1.0 - (p - floor) * (15.0 / 14.0))
    tau = -float(np.log(residual))
    return max(0.0, tau)


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
    nodes = _internal_nodes(optimized)
    if not nodes:
        return optimized
    match, total = _edge_match_counts(optimized, quintet_observations)
    for i, node in enumerate(nodes):
        tau = _estimate_branch_length_from_match_total(float(match[i]), float(total[i]))
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
    if not nodes:
        return match, total

    leaf_sets = _leaf_sets_by_node(species_tree)
    all_taxa = {str(n.label) for n in species_tree.traverse_leaves()}
    side_as = [set(leaf_sets[node]) for node in nodes]
    side_bs = [all_taxa - side_a for side_a in side_as]
    topo_counts = [{} for _ in nodes]

    obs_rows = [
        (set(obs.taxa), _normalize_topology(obs.topology), float(getattr(obs, "weight", 1.0)))
        for obs in quintet_observations
    ]
    for q, topo, w in obs_rows:
        for i, (side_a, side_b) in enumerate(zip(side_as, side_bs)):
            if q.isdisjoint(side_a) or q.isdisjoint(side_b):
                continue
            total[i] += w
            prev = float(topo_counts[i].get(topo, 0.0))
            topo_counts[i][topo] = prev + w
            if topo_counts[i][topo] > match[i]:
                match[i] = topo_counts[i][topo]
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

    def objective_grad(x: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(x, dtype=float), 0.0, max_tau)
        m = match
        n = total
        p_match = (1.0 / 15.0) + (14.0 / 15.0) * (1.0 - np.exp(-t))
        p_match = np.clip(p_match, 1e-12, 1.0 - 1e-12)
        one_minus = np.clip(1.0 - p_match, 1e-12, 1.0)
        p_prime = (14.0 / 15.0) * np.exp(-t)
        d_ll = p_prime * (m / p_match - (np.maximum(0.0, n - m) / one_minus))
        return -d_ll

    res = minimize(
        objective,
        x0=init,
        method="L-BFGS-B",
        jac=objective_grad,
        bounds=[(0.0, max_tau) for _ in range(len(nodes))],
    )
    x_opt = np.clip(np.asarray(res.x if res.success else init, dtype=float), 0.0, max_tau)
    for node, tau in zip(nodes, x_opt):
        node.edge_length = float(tau)
    return init_tree
