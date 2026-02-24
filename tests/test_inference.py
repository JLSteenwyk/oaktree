"""Integration tests for high-level inference pipeline."""

from __future__ import annotations

from itertools import combinations

import msprime
import numpy as np
import treeswift

from oaktree.graphs import partition_tree_to_newick
from oaktree.inference import (
    _compute_robust_gene_weights,
    _generate_nni_neighbors_newicks,
    extract_quintet_observations_from_gene_trees,
    extract_projected_quintet_observations_from_higher_order,
    infer_species_tree_newick_phase2,
    infer_species_tree_newick_phase4_em,
    shrink_quintet_observations_by_confidence,
)
from oaktree.trees import QuintetObservation
from oaktree.validation import apply_missing_and_label_noise


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _canonical_split(split: set[str], all_taxa: set[str]) -> tuple[str, ...]:
    other = all_taxa - split
    side = split if len(split) <= len(other) else other
    return tuple(sorted(side))


def _unrooted_splits(tree: treeswift.Tree, taxa: set[str]) -> set[tuple[str, ...]]:
    parent = {tree.root: None}
    for node in tree.root.traverse_preorder():
        for ch in node.children:
            parent[ch] = node

    leaf_sets = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            leaf_sets[node] = {str(node.label)}
        else:
            s = set()
            for ch in node.children:
                s |= leaf_sets[ch]
            leaf_sets[node] = s

    splits = set()
    for node, subset in leaf_sets.items():
        if node is tree.root:
            continue
        if len(subset) <= 1 or len(subset) >= len(taxa) - 1:
            continue
        splits.add(_canonical_split(set(subset), taxa))
    return splits


def _balanced_8_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...]]:
    taxa = ("A", "B", "C", "D", "E", "F", "G", "H")
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in ("AB", "CD", "EF", "GH", "ABCD", "EFGH", "ROOT"):
        dem.add_population(name=p, initial_size=1)

    # Long branches to reduce ILS and make recovery robust.
    dem.add_population_split(time=0.5, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.5, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.5, derived=["E", "F"], ancestral="EF")
    dem.add_population_split(time=0.5, derived=["G", "H"], ancestral="GH")
    dem.add_population_split(time=2.0, derived=["AB", "CD"], ancestral="ABCD")
    dem.add_population_split(time=2.0, derived=["EF", "GH"], ancestral="EFGH")
    dem.add_population_split(time=4.0, derived=["ABCD", "EFGH"], ancestral="ROOT")
    return dem, taxa


def _asymmetric_8_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...]]:
    taxa = ("A", "B", "C", "D", "E", "F", "G", "H")
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in ("AB", "CD", "CDE", "FG", "FGH", "REST", "ROOT"):
        dem.add_population(name=p, initial_size=1)

    # (((A,B),((C,D),E)),((F,G),H))
    dem.add_population_split(time=0.5, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.5, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.5, derived=["F", "G"], ancestral="FG")
    dem.add_population_split(time=1.5, derived=["CD", "E"], ancestral="CDE")
    dem.add_population_split(time=1.5, derived=["FG", "H"], ancestral="FGH")
    dem.add_population_split(time=3.0, derived=["AB", "CDE"], ancestral="REST")
    dem.add_population_split(time=5.0, derived=["REST", "FGH"], ancestral="ROOT")
    dem.sort_events()
    return dem, taxa


def test_partition_tree_to_newick_builder():
    partition_tree = {
        "taxa": ("A", "B", "C", "D"),
        "left": {"taxa": ("A", "B"), "left": None, "right": None, "artificial_for_left": None, "artificial_for_right": None},
        "right": {"taxa": ("C", "D"), "left": None, "right": None, "artificial_for_left": None, "artificial_for_right": None},
        "artificial_for_left": "__ART_0",
        "artificial_for_right": "__ART_1",
    }
    nwk = partition_tree_to_newick(partition_tree)
    tr = _read_tree(nwk)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == ["A", "B", "C", "D"]


def test_phase2_end_to_end_recovery_8_taxa():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=300,
        random_seed=11,
    ):
        nwk = ts.first().as_newick(node_labels=labels)
        gene_trees.append(_read_tree(nwk))

    inferred_newick = infer_species_tree_newick_phase2(gene_trees, taxa=taxa)
    inferred = _read_tree(inferred_newick)
    true_tree = _read_tree("(((A,B),(C,D)),((E,F),(G,H)));")

    all_taxa = set(taxa)
    inferred_splits = _unrooted_splits(inferred, all_taxa)
    true_splits = _unrooted_splits(true_tree, all_taxa)
    # Require full split recovery for this easy high-signal case.
    assert inferred_splits == true_splits


def test_phase2_asymmetric_8_taxa_key_clades():
    dem, taxa = _asymmetric_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=400,
        random_seed=12,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))

    inferred = _read_tree(infer_species_tree_newick_phase2(gene_trees, taxa=taxa))
    splits = _unrooted_splits(inferred, set(taxa))
    # Core asymmetric clades should be present.
    assert ("A", "B") in splits
    assert ("C", "D", "E") in splits
    assert ("F", "G", "H") in splits


def test_phase2_missing_taxa_with_explicit_taxa_runs():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=120,
        random_seed=7,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))

    rng = np.random.default_rng(9)
    pruned = []
    for tr in gene_trees:
        leaves = [str(n.label) for n in tr.traverse_leaves()]
        drop = leaves[int(rng.integers(0, len(leaves)))]
        keep = set(leaves) - {drop}
        pruned.append(tr.extract_tree_with(keep, suppress_unifurcations=True))

    inferred = _read_tree(infer_species_tree_newick_phase2(pruned, taxa=taxa))
    leaves = sorted(str(n.label) for n in inferred.traverse_leaves())
    assert leaves == sorted(taxa)


def test_phase4_em_wrapper_runs_from_phase2_init():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=120,
        random_seed=21,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))

    out = infer_species_tree_newick_phase4_em(
        gene_trees,
        taxa=taxa,
        n_iterations=2,
        max_quintets_per_tree=40,
        rng=np.random.default_rng(1),
    )
    tr = _read_tree(out)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == sorted(taxa)


def test_phase2_default_policy_matches_explicit_adaptive_clean():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=80,
        random_seed=31,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))

    a = infer_species_tree_newick_phase2(gene_trees, taxa=taxa)
    b = infer_species_tree_newick_phase2(
        gene_trees,
        taxa=taxa,
        low_signal_mode="adaptive",
        low_signal_threshold=0.5,
    )
    ta = _read_tree(a)
    tb = _read_tree(b)
    assert _unrooted_splits(ta, set(taxa)) == _unrooted_splits(tb, set(taxa))


def test_phase2_default_policy_matches_explicit_adaptive_noisy():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=80,
        random_seed=32,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))
    noisy = apply_missing_and_label_noise(
        gene_trees,
        missing_fraction=0.5,
        label_noise_fraction=0.35,
        rng=np.random.default_rng(99),
    )

    a = infer_species_tree_newick_phase2(noisy, taxa=taxa)
    b = infer_species_tree_newick_phase2(
        noisy,
        taxa=taxa,
        low_signal_mode="adaptive",
        low_signal_threshold=0.5,
    )
    ta = _read_tree(a)
    tb = _read_tree(b)
    assert _unrooted_splits(ta, set(taxa)) == _unrooted_splits(tb, set(taxa))
    tr = ta
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == sorted(taxa)


def test_extract_quintet_observations_supports_gene_weights():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=12,
        random_seed=41,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))
    weights = np.full(len(gene_trees), 0.25, dtype=float)
    obs = extract_quintet_observations_from_gene_trees(
        gene_trees,
        taxa=taxa,
        max_quintets_per_tree=10,
        rng=np.random.default_rng(2),
        gene_weights=weights,
    )
    assert len(obs) > 0
    assert all(abs(float(o.weight) - 0.25) < 1e-12 for o in obs)


def test_shrink_quintet_observations_confidence_behavior():
    taxa = ("A", "B", "C", "D", "E")
    topo1 = (("A", "B"), ("C", "D"))
    topo2 = (("A", "C"), ("B", "D"))
    obs = [
        QuintetObservation(taxa=taxa, topology=topo1, weight=5.0),
        QuintetObservation(taxa=taxa, topology=topo2, weight=4.0),
    ]
    shrunk = shrink_quintet_observations_by_confidence(obs, high_confidence_passthrough=0.9)
    assert len(shrunk) == 1
    assert tuple(sorted(shrunk[0].topology)) == tuple(sorted(topo1))
    assert abs(float(shrunk[0].weight) - 1.0) < 1e-12


def test_projected_higher_order_observations_smoke():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    gene_trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=6,
        random_seed=77,
    ):
        gene_trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))
    obs = extract_projected_quintet_observations_from_higher_order(
        gene_trees,
        taxa,
        subset_sizes=[6, 7],
        subsets_per_tree=2,
        quintets_per_subset=4,
        base_weight=1.5,
        rng=np.random.default_rng(3),
    )
    assert len(obs) > 0
    assert all(float(o.weight) > 0.0 for o in obs)


def test_robust_gene_weights_downweight_low_coverage_outliers():
    dem, taxa = _balanced_8_taxon_demography()
    labels = {i: taxa[i] for i in range(8)}
    genes = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=16,
        random_seed=91,
    ):
        genes.append(_read_tree(ts.first().as_newick(node_labels=labels)))
    rng = np.random.default_rng(123)
    mixed = []
    for i, tr in enumerate(genes):
        if i < 8:
            mixed.append(tr)
            continue
        leaves = [str(n.label) for n in tr.traverse_leaves()]
        drop = set(rng.choice(leaves, size=3, replace=False))
        keep = set(leaves) - drop
        pruned = tr.extract_tree_with(keep, suppress_unifurcations=True)
        mixed.append(pruned)

    w = _compute_robust_gene_weights(
        mixed,
        taxa=taxa,
        max_quintets_per_tree=20,
        rng=np.random.default_rng(7),
    )
    assert len(w) == len(mixed)
    high_cov = float(np.mean(w[:8]))
    low_cov = float(np.mean(w[8:]))
    assert high_cov > low_cov


def test_generate_nni_neighbors_preserves_taxa_and_changes_topology():
    base = "((((A,B),(C,D)),((E,F),(G,H))),((I,J),(K,L)));"
    neigh = _generate_nni_neighbors_newicks(base, max_neighbors=8)
    assert len(neigh) > 0
    base_tree = _read_tree(base)
    base_leaves = sorted(str(n.label) for n in base_tree.traverse_leaves())
    assert any(nw != base for nw in neigh)
    for nw in neigh:
        tr = _read_tree(nw)
        leaves = sorted(str(n.label) for n in tr.traverse_leaves())
        assert leaves == base_leaves
