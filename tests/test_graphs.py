"""Phase 2 tests for taxon graph construction and max-cut partitioning."""

from __future__ import annotations

import networkx as nx

from oaktree.graphs import (
    build_taxon_graph,
    partition_tree_to_newick,
    project_observations_for_subproblem,
    quintet_bipartition_weight,
    recursive_partition,
    spectral_max_cut,
)
from oaktree.trees import QuintetObservation


def test_quintet_bipartition_weight_sign():
    taxa = ("A", "B", "C", "D", "E")
    gene_topology = (("A", "B"), ("C", "D"))
    side1 = {"A", "B", "E"}
    side2 = {"C", "D"}

    positive = quintet_bipartition_weight(
        quintet_taxa=taxa,
        gene_topology=gene_topology,
        species_topology=(("A", "B"), ("C", "D")),
        msc_likelihood=0.8,
        bipartition_side1=side1,
        bipartition_side2=side2,
    )
    negative = quintet_bipartition_weight(
        quintet_taxa=taxa,
        gene_topology=gene_topology,
        species_topology=(("A", "C"), ("B", "D")),
        msc_likelihood=0.8,
        bipartition_side1=side1,
        bipartition_side2=side2,
    )
    assert positive > 0
    assert negative < 0


def test_build_taxon_graph_signed_edges():
    taxa = ["A", "B", "C", "D", "E"]
    observations = [
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "E"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
    ]
    g = build_taxon_graph(taxa, observations, species_tree_estimate=None)

    # A-B repeatedly appear as a cherry => same-side preference => negative edge.
    assert g["A"]["B"]["weight"] < 0
    # A-C repeatedly appear across cherries => opposite-side preference => positive edge.
    assert g["A"]["C"]["weight"] > 0


def test_spectral_cut_planted_partition():
    # Positive edges prefer opposite sides, negative edges prefer same side.
    left = {"A", "B", "C"}
    right = {"D", "E", "F"}
    g = nx.Graph()
    for t in sorted(left | right):
        g.add_node(t)
    for u in g.nodes:
        for v in g.nodes:
            if u >= v:
                continue
            if (u in left and v in left) or (u in right and v in right):
                g.add_edge(u, v, weight=-3.0)
            else:
                g.add_edge(u, v, weight=3.0)

    a, b = spectral_max_cut(g)
    # Compare up to side swap.
    assert (a == left and b == right) or (a == right and b == left)


def test_project_observations_n2_normalization():
    observations = [
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
    ]
    subset = {"A", "B", "C", "D"}
    represented = {"E", "F", "G"}

    n2_obs = project_observations_for_subproblem(
        observations,
        real_taxa_subset=subset,
        artificial_taxon="__ART__",
        represented_taxa=represented,
        n2_normalization=True,
    )
    n0_obs = project_observations_for_subproblem(
        observations,
        real_taxa_subset=subset,
        artificial_taxon="__ART__",
        represented_taxa=represented,
        n2_normalization=False,
    )
    assert len(n2_obs) == 1
    assert len(n0_obs) == 1
    assert abs(n2_obs[0].weight - (1.0 / 3.0)) < 1e-12
    assert abs(n0_obs[0].weight - 1.0) < 1e-12
    assert "__ART__" in n2_obs[0].taxa


def test_recursive_partition_artificial_bookkeeping():
    taxa = ["A", "B", "C", "D"]
    observations = [
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "F"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "G"), topology=(("A", "B"), ("C", "D"))),
    ]
    tree, artificial_map = recursive_partition(
        taxa=taxa,
        quintet_observations=observations,
        species_tree_estimate=None,
        n2_normalization=True,
    )

    assert set(tree["taxa"]) == set(taxa)
    # First split on 4 taxa should create two artificial complement representatives.
    assert tree["artificial_for_left"] in artificial_map
    assert tree["artificial_for_right"] in artificial_map
    left_rep = set(artificial_map[tree["artificial_for_right"]])
    right_rep = set(artificial_map[tree["artificial_for_left"]])
    # Side complements should be disjoint and cover all real taxa.
    assert left_rep.isdisjoint(right_rep)
    assert left_rep | right_rep == set(taxa)


def test_recursive_partition_low_signal_unresolved():
    taxa = ["A", "B", "C", "D", "E", "F"]
    # No observations => no informative signal.
    tree, _ = recursive_partition(
        taxa=taxa,
        quintet_observations=[],
        species_tree_estimate=None,
        n2_normalization=True,
        low_signal_threshold=0.05,
    )
    assert tree["left"] is None
    assert tree["right"] is None
    assert tree["unresolved"] is True


def test_recursive_partition_resolves_when_signal_strong():
    taxa = ["A", "B", "C", "D", "E"]
    observations = [
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
        QuintetObservation(taxa=("A", "B", "C", "D", "E"), topology=(("A", "B"), ("C", "D"))),
    ]
    tree, _ = recursive_partition(
        taxa=taxa,
        quintet_observations=observations,
        species_tree_estimate=None,
        low_signal_threshold=0.05,
    )
    assert tree["left"] is not None
    assert tree["right"] is not None
    assert tree["unresolved"] is False


def test_recursive_partition_adaptive_mode_unresolved_on_zero_signal():
    taxa = ["A", "B", "C", "D", "E", "F"]
    tree_fixed, _ = recursive_partition(
        taxa=taxa,
        quintet_observations=[],
        species_tree_estimate=None,
        low_signal_threshold=0.0,
        low_signal_mode="fixed",
    )
    tree_adapt, _ = recursive_partition(
        taxa=taxa,
        quintet_observations=[],
        species_tree_estimate=None,
        low_signal_mode="adaptive",
    )
    assert tree_fixed["left"] is not None and tree_fixed["right"] is not None
    assert tree_adapt["left"] is None and tree_adapt["right"] is None
    assert tree_adapt["unresolved"] is True


def test_partition_tree_to_newick_polytomy_for_unresolved():
    tree = {
        "taxa": ("A", "B", "C", "D"),
        "left": None,
        "right": None,
        "artificial_for_left": None,
        "artificial_for_right": None,
        "unresolved": True,
    }
    nwk = partition_tree_to_newick(tree)
    assert nwk == "(A,B,C,D);"
