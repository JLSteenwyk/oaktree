"""Phase 4 tests for gene-tree weighting and EM entry loop."""

from __future__ import annotations

import msprime
import numpy as np
import treeswift

from oaktree.inference import infer_species_tree_newick_phase2
from oaktree.weights import compute_gene_tree_weights, em_refine_species_tree_newick


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _balanced_demography():
    taxa = ("A", "B", "C", "D", "E", "F", "G", "H")
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in ("AB", "CD", "EF", "GH", "ABCD", "EFGH", "ROOT"):
        dem.add_population(name=p, initial_size=1)
    dem.add_population_split(time=0.5, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.5, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.5, derived=["E", "F"], ancestral="EF")
    dem.add_population_split(time=0.5, derived=["G", "H"], ancestral="GH")
    dem.add_population_split(time=2.0, derived=["AB", "CD"], ancestral="ABCD")
    dem.add_population_split(time=2.0, derived=["EF", "GH"], ancestral="EFGH")
    dem.add_population_split(time=4.0, derived=["ABCD", "EFGH"], ancestral="ROOT")
    return dem, taxa


def _short_branch_demography():
    taxa = ("A", "B", "C", "D", "E", "F", "G", "H")
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in ("AB", "CD", "EF", "GH", "ABCD", "EFGH", "ROOT"):
        dem.add_population(name=p, initial_size=1)
    # Very short internal branches => high ILS / noisy signal.
    dem.add_population_split(time=0.4, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.4, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.4, derived=["E", "F"], ancestral="EF")
    dem.add_population_split(time=0.4, derived=["G", "H"], ancestral="GH")
    dem.add_population_split(time=0.5, derived=["AB", "CD"], ancestral="ABCD")
    dem.add_population_split(time=0.5, derived=["EF", "GH"], ancestral="EFGH")
    dem.add_population_split(time=0.6, derived=["ABCD", "EFGH"], ancestral="ROOT")
    return dem, taxa


def _simulate_gene_trees(n_reps: int, seed: int):
    dem, taxa = _balanced_demography()
    labels = {i: taxa[i] for i in range(8)}
    trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=n_reps,
        random_seed=seed,
    ):
        trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))
    return trees, taxa


def test_compute_gene_tree_weights_normalized():
    gene_trees, taxa = _simulate_gene_trees(50, 14)
    init = infer_species_tree_newick_phase2(gene_trees, taxa=taxa, max_quintets_per_tree=40)
    w = compute_gene_tree_weights(
        gene_trees,
        init,
        taxa,
        temperature=1.0,
        max_quintets_per_tree=40,
        rng=np.random.default_rng(2),
    )
    assert len(w) == len(gene_trees)
    assert abs(float(np.sum(w)) - 1.0) < 1e-12
    assert np.all(w > 0)


def test_em_refine_species_tree_newick_history():
    gene_trees, taxa = _simulate_gene_trees(80, 22)
    final, history = em_refine_species_tree_newick(
        gene_trees,
        taxa=taxa,
        n_iterations=2,
        max_quintets_per_tree=35,
        rng=np.random.default_rng(3),
    )
    assert len(history) >= 1
    assert history[0].iteration == 0
    assert history[-1].species_tree_newick == final
    tr = _read_tree(final)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == sorted(taxa)


def test_em_convergence_criteria_stops_early():
    gene_trees, taxa = _simulate_gene_trees(60, 101)
    final, history = em_refine_species_tree_newick(
        gene_trees,
        taxa=taxa,
        n_iterations=10,
        max_quintets_per_tree=35,
        rng=np.random.default_rng(5),
        rf_tolerance=0,
        branch_length_tolerance=1.0,
        min_iterations_before_stop=1,
    )
    assert len(history) <= 3
    assert history[-1].species_tree_newick == final
    for rec in history[1:]:
        assert rec.rf_distance is not None
        assert rec.branch_length_delta is not None


def test_em_noisy_short_branch_validation_runs():
    dem, taxa = _short_branch_demography()
    labels = {i: taxa[i] for i in range(8)}
    trees = []
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=100,
        random_seed=202,
    ):
        trees.append(_read_tree(ts.first().as_newick(node_labels=labels)))

    # Inject light label noise in a subset of trees.
    noisy = []
    rng = np.random.default_rng(8)
    for i, tr in enumerate(trees):
        if i % 7 != 0:
            noisy.append(tr)
            continue
        a = tr.newick()
        # Swap two labels in string representation for a rough noise model.
        a = a.replace("A", "__TMP__", 1).replace("B", "A", 1).replace("__TMP__", "B", 1)
        noisy.append(_read_tree(a))

    final, history = em_refine_species_tree_newick(
        noisy,
        taxa=taxa,
        n_iterations=3,
        max_quintets_per_tree=30,
        rng=rng,
        rf_tolerance=0,
        branch_length_tolerance=5e-2,
    )
    tr = _read_tree(final)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == sorted(taxa)
    assert len(history) >= 2


def test_weight_separation_clean_vs_outlier_mixture():
    gene_trees, taxa = _simulate_gene_trees(80, 55)
    true_newick = "(((A,B),(C,D)),((E,F),(G,H)));"

    # Controlled outliers: deterministic comb trees with permuted labels.
    outliers = []
    perms = [
        ("A", "C", "E", "G", "B", "D", "F", "H"),
        ("H", "F", "D", "B", "G", "E", "C", "A"),
        ("A", "D", "G", "B", "E", "H", "C", "F"),
        ("C", "F", "A", "D", "G", "B", "E", "H"),
    ]
    for i in range(40):
        p = perms[i % len(perms)]
        cur = p[0]
        for lbl in p[1:]:
            cur = f"({cur},{lbl})"
        nwk = f"{cur};"
        outliers.append(_read_tree(nwk))

    mixed = gene_trees[:40] + outliers
    w = compute_gene_tree_weights(
        mixed,
        true_newick,
        taxa,
        temperature=1.0,
        max_quintets_per_tree=50,
        rng=np.random.default_rng(13),
    )
    clean_w = w[:40]
    out_w = w[40:]
    assert float(np.median(clean_w)) > float(np.median(out_w))
    assert float(np.mean(clean_w)) > float(np.mean(out_w))
