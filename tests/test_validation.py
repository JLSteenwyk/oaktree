"""Phase 6 validation/benchmark helpers tests."""

from __future__ import annotations

import numpy as np
import treeswift

from oaktree.validation import (
    BenchmarkResult,
    apply_missing_and_label_noise,
    balanced_16_taxon_demography,
    balanced_8_taxon_demography,
    calibrate_low_signal_threshold,
    nj_distance_baseline_newick,
    run_expanded_benchmark,
    run_baseline_benchmark,
    simulate_gene_trees,
    summarize_rf_replicates,
    upgma_distance_baseline_newick,
)


def test_simulate_gene_trees_count():
    dem, taxa, _ = balanced_8_taxon_demography()
    trees = simulate_gene_trees(dem, taxa, n_replicates=12, seed=5)
    assert len(trees) == 12


def test_simulate_gene_trees_count_16_taxa():
    dem, taxa, _ = balanced_16_taxon_demography()
    trees = simulate_gene_trees(dem, taxa, n_replicates=6, seed=12)
    assert len(trees) == 6
    for tr in trees:
        leaves = sorted(str(n.label) for n in tr.traverse_leaves())
        assert leaves == sorted(taxa)


def test_run_baseline_benchmark_structure():
    results = run_baseline_benchmark(n_gene_trees=40, seed=1)
    assert len(results) == 2
    names = {r.dataset for r in results}
    assert names == {"balanced8", "asymmetric8"}
    for r in results:
        assert r.phase2_rf >= 0
        assert r.phase4_rf >= 0
        assert r.consensus_rf >= 0
        assert r.strict_consensus_rf >= 0
        assert r.nj_rf >= 0
        assert r.upgma_rf >= 0
        assert (r.astral_rf is None) or (r.astral_rf >= 0)
        assert r.n_gene_trees == 40


def test_distance_baselines_cover_all_taxa():
    dem, taxa, _ = balanced_8_taxon_demography()
    trees = simulate_gene_trees(dem, taxa, n_replicates=20, seed=9)
    nj = nj_distance_baseline_newick(trees, taxa)
    up = upgma_distance_baseline_newick(trees, taxa)
    for nwk in (nj, up):
        tr = treeswift.read_tree_newick(nwk)
        leaves = sorted(str(n.label) for n in tr.traverse_leaves())
        assert leaves == sorted(taxa)


def test_run_expanded_benchmark_structure():
    results = run_expanded_benchmark(n_gene_trees=30, seed=2)
    names = {r.dataset for r in results}
    assert names == {"balanced8", "asymmetric8", "shortbranch8", "balanced8_missing"}
    for r in results:
        assert r.phase2_rf >= 0
        assert r.phase4_rf >= 0
        assert r.consensus_rf >= 0
        assert r.strict_consensus_rf >= 0
        assert r.nj_rf >= 0
        assert r.upgma_rf >= 0
        assert (r.astral_rf is None) or (r.astral_rf >= 0)


def test_run_expanded_benchmark_core_mode_structure():
    results = run_expanded_benchmark(n_gene_trees=20, seed=4, baseline_guardrail=False)
    names = {r.dataset for r in results}
    assert names == {"balanced8", "asymmetric8", "shortbranch8", "balanced8_missing"}
    for r in results:
        assert r.phase2_rf >= 0
        assert r.phase4_rf >= 0


def test_summarize_rf_replicates_stats():
    r1 = BenchmarkResult(
        dataset="balanced8",
        n_gene_trees=20,
        phase2_rf=0,
        phase4_rf=2,
        consensus_rf=3,
        strict_consensus_rf=4,
        nj_rf=1,
        upgma_rf=1,
        astral_rf=None,
        phase2_newick="(A,B);",
        phase4_newick="(A,B);",
        consensus_newick="(A,B);",
        strict_consensus_newick="(A,B);",
        nj_newick="(A,B);",
        upgma_newick="(A,B);",
        astral_newick=None,
        true_newick="(A,B);",
    )
    r2 = BenchmarkResult(
        dataset="balanced8",
        n_gene_trees=20,
        phase2_rf=2,
        phase4_rf=4,
        consensus_rf=5,
        strict_consensus_rf=6,
        nj_rf=3,
        upgma_rf=3,
        astral_rf=None,
        phase2_newick="(A,B);",
        phase4_newick="(A,B);",
        consensus_newick="(A,B);",
        strict_consensus_newick="(A,B);",
        nj_newick="(A,B);",
        upgma_newick="(A,B);",
        astral_newick=None,
        true_newick="(A,B);",
    )
    agg = summarize_rf_replicates([[r1], [r2]])
    s = agg["balanced8"]["phase2_rf"]
    assert s["mean"] == 1.0
    assert abs(s["std"] - 1.4142135623730951) < 1e-12
    assert s["ci95_low"] <= s["mean"] <= s["ci95_high"]


def test_calibrate_low_signal_threshold_structure():
    rows = calibrate_low_signal_threshold(
        thresholds=[0.0, 0.05],
        n_gene_trees=20,
        seed=3,
        n_replicates=1,
        datasets=["balanced8", "balanced8_noisy"],
        low_signal_mode="adaptive",
    )
    assert len(rows) == 4
    for row in rows:
        assert row.dataset in {"balanced8", "balanced8_noisy"}
        assert row.threshold in (0.0, 0.05)
        assert row.mean_rf >= 0.0
        assert row.std_rf >= 0.0
        assert row.ci95_low <= row.mean_rf <= row.ci95_high


def test_apply_missing_and_label_noise_preserves_label_set_under_noise_only():
    dem, taxa, _ = balanced_8_taxon_demography()
    trees = simulate_gene_trees(dem, taxa, n_replicates=8, seed=11)
    noisy = apply_missing_and_label_noise(
        trees,
        missing_fraction=0.0,
        label_noise_fraction=1.0,
        rng=np.random.default_rng(2),
    )
    for tr in noisy:
        leaves = sorted(str(n.label) for n in tr.traverse_leaves())
        assert leaves == sorted(taxa)
