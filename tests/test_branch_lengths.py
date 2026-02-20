"""Phase 3 baseline tests for branch-length estimation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import treeswift

from oaktree.branch_lengths import (
    _edge_match_counts,
    branch_length_log_likelihood,
    count_quintet_frequencies,
    estimate_branch_length,
    optimize_branch_lengths_ml,
    optimize_branch_lengths,
)
from oaktree.trees import QuintetObservation, canonicalize_quintet


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def test_count_quintet_frequencies_vector_shape_and_sum():
    species_tree = _read_tree("(((A:1,B:1):1,(C:1,D:1):1):1,(E:1,F:1):1);")
    obs = [
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "F"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "E", "F"), topology=(("A", "B"), ("E", "F"))),
    ]
    # Use split {A,B,C,D} | {E,F}.
    freq = count_quintet_frequencies(species_tree, {"A", "B", "C", "D"}, obs)
    assert isinstance(freq, np.ndarray)
    assert freq.shape == (15,)
    assert int(freq.sum()) >= 2


def test_estimate_branch_length_monotone_in_match_signal():
    low = np.ones(15)
    high = np.ones(15)
    high[0] = 10.0
    tau_low = estimate_branch_length(low)
    tau_high = estimate_branch_length(high)
    assert tau_low >= 0.0
    assert tau_high >= 0.0
    assert tau_high > tau_low


def test_estimate_branch_length_uniform_is_zero():
    uniform = np.ones(15)
    tau = estimate_branch_length(uniform)
    assert abs(tau - 0.0) < 1e-12


def test_optimize_branch_lengths_sets_internal_edges():
    species_tree = _read_tree("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);")
    taxa = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Build synthetic high-signal observations from the same topology.
    observations = []
    for q in combinations(taxa, 5):
        q = tuple(sorted(q))
        topo = canonicalize_quintet(species_tree, q)
        observations.append(QuintetObservation(taxa=q, topology=topo))

    optimized = optimize_branch_lengths(species_tree, observations)
    internal_lengths = []
    for node in optimized.root.traverse_preorder():
        if node is optimized.root or node.is_leaf():
            continue
        if node.edge_length is not None:
            internal_lengths.append(float(node.edge_length))
    assert len(internal_lengths) > 0
    assert all(x >= 0.0 for x in internal_lengths)
    assert any(x > 0.0 for x in internal_lengths)


def test_optimize_branch_lengths_ml_improves_likelihood():
    species_tree = _read_tree("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);")
    taxa = ["A", "B", "C", "D", "E", "F", "G", "H"]
    observations = []
    for q in combinations(taxa, 5):
        q = tuple(sorted(q))
        topo = canonicalize_quintet(species_tree, q)
        observations.append(QuintetObservation(taxa=q, topology=topo))

    baseline = optimize_branch_lengths(species_tree, observations)
    nodes = [n for n in baseline.root.traverse_preorder() if n is not baseline.root and not n.is_leaf()]
    match, total = _edge_match_counts(baseline, observations)
    x0 = np.array([float(n.edge_length or 0.0) for n in nodes], dtype=float)
    ll0 = branch_length_log_likelihood(np.clip(x0, 0.0, 8.0), match, total)

    refined = optimize_branch_lengths_ml(species_tree, observations, max_tau=8.0)
    nodes_r = [n for n in refined.root.traverse_preorder() if n is not refined.root and not n.is_leaf()]
    x1 = np.array([float(n.edge_length or 0.0) for n in nodes_r], dtype=float)
    ll1 = branch_length_log_likelihood(x1, match, total)

    assert ll1 >= ll0 - 1e-8
    assert np.all(x1 >= 0.0)
    assert np.all(x1 <= 8.0 + 1e-12)
