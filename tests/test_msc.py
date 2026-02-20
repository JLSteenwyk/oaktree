"""Phase 1 tests for core MSC probability calculations."""

from __future__ import annotations

import math
from itertools import combinations
from collections import Counter
import time

import numpy as np
import treeswift
import msprime

from oaktree.msc import (
    coalescent_probability,
    enumerate_coalescent_histories,
    lookup_quintet_probability,
    precompute_quintet_tables,
    quartet_anomaly_zone_threshold,
    quintet_probability,
    rooted_gene_tree_distribution,
)
from oaktree.trees import canonicalize_quintet


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _all_quintet_topologies_for_taxa(taxa: tuple[str, str, str, str, str]):
    pairs = [tuple(sorted(p)) for p in combinations(taxa, 2)]
    topologies = set()
    for i, p1 in enumerate(pairs):
        s1 = set(p1)
        for p2 in pairs[i + 1 :]:
            if s1.isdisjoint(p2):
                topologies.add(tuple(sorted((p1, tuple(sorted(p2))))))
    return sorted(topologies)


def _simulate_quintet_topology_distribution_msprime(
    species_tree_demography: msprime.Demography,
    taxa: tuple[str, str, str, str, str],
    n_replicates: int,
    random_seed: int,
):
    node_labels = {i: taxa[i] for i in range(5)}
    counts = Counter()
    for ts in msprime.sim_ancestry(
        samples={t: 1 for t in taxa},
        demography=species_tree_demography,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=n_replicates,
        random_seed=random_seed,
    ):
        t = _read_tree(ts.first().as_newick(node_labels=node_labels))
        topo = tuple(sorted(canonicalize_quintet(t, taxa)))
        counts[topo] += 1
    return {k: v / n_replicates for k, v in counts.items()}


def _rooted_signature(node) -> str:
    if node.is_leaf():
        return str(node.label)
    kids = sorted(_rooted_signature(c) for c in node.children)
    return "(" + ",".join(kids) + ")"


def test_coalescent_probability_tau_zero_boundary():
    for k in range(2, 7):
        assert coalescent_probability(k, k, 0.0) == 1.0
        for j in range(1, k):
            assert coalescent_probability(k, j, 0.0) == 0.0


def test_single_branch_two_lineages_formula():
    taus = [0.0, 0.1, 0.5, 1.0, 3.0]
    for tau in taus:
        expected = 1.0 - math.exp(-tau)
        actual = coalescent_probability(2, 1, tau)
        assert abs(actual - expected) < 1e-12


def test_coalescent_probability_long_time_to_one_lineage():
    for k in range(2, 7):
        p = coalescent_probability(k, 1, 100.0)
        assert p > 1.0 - 1e-10


def test_coalescent_probability_distribution_sums_to_one():
    rng = np.random.default_rng(2026)
    for _ in range(20):
        k = int(rng.integers(2, 8))
        tau = float(rng.uniform(0.0, 5.0))
        total = sum(coalescent_probability(k, j, tau) for j in range(1, k + 1))
        assert abs(total - 1.0) < 1e-10


def test_coalescent_probability_published_exact_values():
    # Exact values for n=3 at tau=ln(2), from standard closed forms
    # (Tavare/Kingman transition table equivalents):
    # P(3->3,t)=e^{-3t}, P(3->2,t)=3/2(e^{-t}-e^{-3t}),
    # P(3->1,t)=1-P(3->2,t)-P(3->3,t).
    tau = math.log(2.0)
    p33_expected = 1.0 / 8.0
    p32_expected = 9.0 / 16.0
    p31_expected = 5.0 / 16.0

    assert abs(coalescent_probability(3, 3, tau) - p33_expected) < 1e-12
    assert abs(coalescent_probability(3, 2, tau) - p32_expected) < 1e-12
    assert abs(coalescent_probability(3, 1, tau) - p31_expected) < 1e-12


def test_enumerate_coalescent_histories_nonempty():
    taxa = ("A", "B", "C", "D", "E")
    species_tree = _read_tree("((A:1,B:1):1,((C:1,D:1):1,E:1):1);")
    gene_topology = (("A", "B"), ("C", "D"))
    histories = enumerate_coalescent_histories(species_tree, gene_topology, taxa)
    assert len(histories) > 0


def test_quintet_probability_sum_to_one_over_15():
    taxa = ("A", "B", "C", "D", "E")
    species_tree = _read_tree("((A:1,B:1):1,((C:1,D:1):1,E:1):1);")
    topologies = _all_quintet_topologies_for_taxa(taxa)
    total = sum(quintet_probability(species_tree, topo, taxa) for topo in topologies)
    assert abs(total - 1.0) < 1e-10


def test_quintet_probability_star_tree_uniform():
    taxa = ("A", "B", "C", "D", "E")
    species_tree = _read_tree("((A:1,B:1):0,((C:1,D:1):0,E:1):0);")
    topologies = _all_quintet_topologies_for_taxa(taxa)
    probs = [quintet_probability(species_tree, topo, taxa) for topo in topologies]
    assert max(abs(p - (1.0 / 15.0)) for p in probs) < 1e-12


def test_quintet_probability_long_branches_matching_is_largest():
    taxa = ("A", "B", "C", "D", "E")
    species_tree = _read_tree("((A:1,B:1):10,((C:1,D:1):10,E:1):10);")
    topologies = _all_quintet_topologies_for_taxa(taxa)
    matching = canonicalize_quintet(species_tree, taxa)
    probs = {topo: quintet_probability(species_tree, topo, taxa) for topo in topologies}
    p_match = probs[tuple(sorted(matching))]
    assert p_match > 0.99
    assert all(p_match > p for topo, p in probs.items() if topo != tuple(sorted(matching)))


def test_coalescent_probability_matches_msprime_two_lineages():
    reps = 5000
    tau = 1.3
    merged = 0
    for ts in msprime.sim_ancestry(
        samples=2,
        population_size=1,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=reps,
        random_seed=19,
    ):
        tr = ts.first()
        t_mrca = tr.time(tr.root)
        if t_mrca <= tau:
            merged += 1
    sim_p = merged / reps
    exact_p = coalescent_probability(2, 1, tau)
    assert abs(sim_p - exact_p) < 0.03


def test_quintet_probability_matches_msprime_reference_distribution():
    taxa = ("A", "B", "C", "D", "E")
    species_tree = _read_tree("((A:1,B:1):2,((C:1,D:1):1,E:2):1);")

    # With ploidy=1 and N=1, msprime times equal coalescent units.
    dem = msprime.Demography()
    for pop in taxa:
        dem.add_population(name=pop, initial_size=1)
    for pop in ("AB", "CD", "CDE", "ROOT"):
        dem.add_population(name=pop, initial_size=1)
    dem.add_population_split(time=1.0, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=1.0, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=2.0, derived=["CD", "E"], ancestral="CDE")
    dem.add_population_split(time=3.0, derived=["AB", "CDE"], ancestral="ROOT")

    sim = _simulate_quintet_topology_distribution_msprime(
        species_tree_demography=dem,
        taxa=taxa,
        n_replicates=3000,
        random_seed=23,
    )
    topologies = _all_quintet_topologies_for_taxa(taxa)
    pred = {topo: quintet_probability(species_tree, topo, taxa) for topo in topologies}

    l1 = sum(abs(pred[topo] - sim.get(topo, 0.0)) for topo in topologies)
    assert l1 < 0.06


def test_lookup_table_precompute_and_normalization():
    grid = np.array([0.0, 0.5, 1.0], dtype=float)
    table = precompute_quintet_tables(grid, species_topology_ids=[0, 1])
    probs = table["probs"]

    # Subset IDs are precomputed and should be finite + normalized.
    for sid in [0, 1]:
        for i in range(len(grid)):
            for j in range(len(grid)):
                col = probs[sid, :, i, j]
                assert np.all(np.isfinite(col))
                assert abs(float(np.sum(col)) - 1.0) < 1e-10

    # Non-precomputed species IDs remain NaN.
    assert np.isnan(probs[2, :, 0, 0]).all()


def test_lookup_quintet_probability_interpolation_error_bound():
    taxa = ("a", "b", "c", "d", "e")
    topologies = _all_quintet_topologies_for_taxa(taxa)
    sid = 0
    gid = 0

    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
    table = precompute_quintet_tables(grid, species_topology_ids=[sid])
    species_topology = topologies[sid]
    gene_topology = topologies[gid]

    rng = np.random.default_rng(20260219)
    errs = []
    for _ in range(20):
        tau1 = float(rng.uniform(grid[0], grid[-1]))
        tau2 = float(rng.uniform(grid[0], grid[-1]))
        pred = lookup_quintet_probability(sid, gid, (tau1, tau2), table)
        species_tree = _read_tree(
            f"(({species_topology[0][0]}:0,{species_topology[0][1]}:0):{tau1},"
            f"(({species_topology[1][0]}:0,{species_topology[1][1]}:0):{tau2},"
            f"{(set(taxa)-set(species_topology[0])-set(species_topology[1])).pop()}:0):0);"
        )
        exact = quintet_probability(species_tree, gene_topology, taxa)
        errs.append(abs(pred - exact))

    assert max(errs) < 0.06


def test_anomaly_zone_quartet_subcase_msprime():
    # Rooted species tree (((A,B),C),D) with short internal branches can
    # produce an anomalous most-probable rooted gene tree.
    t1, t2, t3 = 0.005, 0.025, 0.225

    dem = msprime.Demography()
    for pop in ("A", "B", "C", "D", "AB", "ABC", "ROOT"):
        dem.add_population(name=pop, initial_size=1)
    dem.add_population_split(time=t1, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=t2, derived=["AB", "C"], ancestral="ABC")
    dem.add_population_split(time=t3, derived=["ABC", "D"], ancestral="ROOT")

    species_tree = _read_tree(f"(((A:{t1},B:{t1}):{t2 - t1},C:{t2}):{t3 - t2},D:{t3});")
    species_tree.suppress_unifurcations()
    species_sig = _rooted_signature(species_tree.root)

    reps = 3000
    counts = Counter()
    labels = {0: "A", 1: "B", 2: "C", 3: "D"}
    for ts in msprime.sim_ancestry(
        samples={"A": 1, "B": 1, "C": 1, "D": 1},
        demography=dem,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=reps,
        random_seed=31,
    ):
        gt = _read_tree(ts.first().as_newick(node_labels=labels))
        gt.suppress_unifurcations()
        counts[_rooted_signature(gt.root)] += 1

    p_match = counts[species_sig] / reps
    p_best_nonmatch = max(v / reps for k, v in counts.items() if k != species_sig)
    assert p_best_nonmatch > p_match + 0.01


def test_anomaly_zone_quartet_subcase_exact_distribution():
    t1, t2, t3 = 0.005, 0.025, 0.225
    taxa = ("A", "B", "C", "D")
    species_tree = _read_tree(f"(((A:{t1},B:{t1}):{t2 - t1},C:{t2}):{t3 - t2},D:{t3});")
    species_tree.suppress_unifurcations()
    species_sig = _rooted_signature(species_tree.root)

    dist = rooted_gene_tree_distribution(species_tree, taxa)
    assert abs(sum(dist.values()) - 1.0) < 1e-12
    p_match = dist[species_sig]
    p_best_nonmatch = max(v for k, v in dist.items() if k != species_sig)
    assert p_best_nonmatch > p_match + 0.01


def test_lookup_table_larger_grid_smoke():
    grid = np.linspace(0.0, 2.0, 9)
    table = precompute_quintet_tables(grid, species_topology_ids=[0, 1, 2])
    probs = table["probs"]
    assert probs.shape == (15, 15, 9, 9)
    assert np.isfinite(probs[0]).all()
    assert np.isfinite(probs[1]).all()
    assert np.isfinite(probs[2]).all()


def test_anomaly_zone_quartet_threshold_formula_exact():
    taxa = ("A", "B", "C", "D")
    t1 = 0.005
    x = 0.2  # older internal branch
    y_star = quartet_anomaly_zone_threshold(x)

    def p_match_and_best_nonmatch(y: float) -> tuple[float, float]:
        t2 = t1 + y
        t3 = t2 + x
        sp = _read_tree(f"(((A:{t1},B:{t1}):{y},C:{t2}):{x},D:{t3});")
        sp.suppress_unifurcations()
        sig = _rooted_signature(sp.root)
        dist = rooted_gene_tree_distribution(sp, taxa)
        p_match = dist[sig]
        p_non = max(v for k, v in dist.items() if k != sig)
        return p_match, p_non

    p_match_low, p_non_low = p_match_and_best_nonmatch(y_star - 0.02)
    p_match_high, p_non_high = p_match_and_best_nonmatch(y_star + 0.04)

    assert p_non_low > p_match_low
    assert p_match_high > p_non_high


def test_lookup_table_performance_guard_baseline():
    # Guard rails derived from recorded baseline (docs/MEMORY.md):
    # 9x9 ~1.04s, 13x13 ~1.97s, 17x17 ~3.62s for all 15 species topologies.
    species_ids = list(range(15))
    budgets = {9: 4.0, 13: 7.0, 17: 12.0}
    for grid_n, max_seconds in budgets.items():
        grid = np.linspace(0.0, 2.0, grid_n)
        t0 = time.perf_counter()
        table = precompute_quintet_tables(grid, species_topology_ids=species_ids)
        dt = time.perf_counter() - t0
        assert dt < max_seconds
        assert np.isfinite(table["probs"]).sum() == (15 * 15 * grid_n * grid_n)
