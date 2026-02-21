"""Validation and benchmarking helpers (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
import io
import itertools
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Sequence

import msprime
import numpy as np
import treeswift

from .inference import infer_species_tree_newick_phase2, infer_species_tree_newick_phase4_em

RF_METHOD_KEYS = (
    "phase2_rf",
    "phase4_rf",
    "consensus_rf",
    "strict_consensus_rf",
    "nj_rf",
    "upgma_rf",
    "astral_rf",
    "tree_qmc_rf",
)


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _canonical_split(split: set[str], all_taxa: set[str]) -> tuple[str, ...]:
    other = all_taxa - split
    side = split if len(split) <= len(other) else other
    return tuple(sorted(side))


def unrooted_splits(tree: treeswift.Tree, taxa: set[str]) -> set[tuple[str, ...]]:
    leaf_sets: dict[treeswift.Node, set[str]] = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            leaf_sets[node] = {str(node.label)}
        else:
            s = set()
            for ch in node.children:
                s |= leaf_sets[ch]
            leaf_sets[node] = s

    out: set[tuple[str, ...]] = set()
    for node, subset in leaf_sets.items():
        if node is tree.root:
            continue
        if len(subset) <= 1 or len(subset) >= len(taxa) - 1:
            continue
        out.add(_canonical_split(set(subset), taxa))
    return out


def rf_distance_unrooted(newick_a: str, newick_b: str, taxa: Sequence[str]) -> int:
    ta = _read_tree(newick_a)
    tb = _read_tree(newick_b)
    all_taxa = set(taxa)
    sa = unrooted_splits(ta, all_taxa)
    sb = unrooted_splits(tb, all_taxa)
    return len(sa.symmetric_difference(sb))


def balanced_8_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
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
    return dem, taxa, "(((A,B),(C,D)),((E,F),(G,H)));"


def asymmetric_8_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    taxa = ("A", "B", "C", "D", "E", "F", "G", "H")
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in ("AB", "CD", "CDE", "FG", "FGH", "REST", "ROOT"):
        dem.add_population(name=p, initial_size=1)
    dem.add_population_split(time=0.5, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.5, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.5, derived=["F", "G"], ancestral="FG")
    dem.add_population_split(time=1.5, derived=["CD", "E"], ancestral="CDE")
    dem.add_population_split(time=1.5, derived=["FG", "H"], ancestral="FGH")
    dem.add_population_split(time=3.0, derived=["AB", "CDE"], ancestral="REST")
    dem.add_population_split(time=5.0, derived=["REST", "FGH"], ancestral="ROOT")
    dem.sort_events()
    return dem, taxa, "(((A,B),((C,D),E)),((F,G),H));"


def short_branch_8_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    taxa = ("A", "B", "C", "D", "E", "F", "G", "H")
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in ("AB", "CD", "EF", "GH", "ABCD", "EFGH", "ROOT"):
        dem.add_population(name=p, initial_size=1)
    dem.add_population_split(time=0.4, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.4, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.4, derived=["E", "F"], ancestral="EF")
    dem.add_population_split(time=0.4, derived=["G", "H"], ancestral="GH")
    dem.add_population_split(time=0.5, derived=["AB", "CD"], ancestral="ABCD")
    dem.add_population_split(time=0.5, derived=["EF", "GH"], ancestral="EFGH")
    dem.add_population_split(time=0.6, derived=["ABCD", "EFGH"], ancestral="ROOT")
    return dem, taxa, "(((A,B),(C,D)),((E,F),(G,H)));"


def balanced_16_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    taxa = tuple(chr(ord("A") + i) for i in range(16))
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in (
        "AB",
        "CD",
        "EF",
        "GH",
        "IJ",
        "KL",
        "MN",
        "OP",
        "ABCD",
        "EFGH",
        "IJKL",
        "MNOP",
        "ABCDEFGH",
        "IJKLMNOP",
        "ROOT",
    ):
        dem.add_population(name=p, initial_size=1)
    dem.add_population_split(time=0.5, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.5, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.5, derived=["E", "F"], ancestral="EF")
    dem.add_population_split(time=0.5, derived=["G", "H"], ancestral="GH")
    dem.add_population_split(time=0.5, derived=["I", "J"], ancestral="IJ")
    dem.add_population_split(time=0.5, derived=["K", "L"], ancestral="KL")
    dem.add_population_split(time=0.5, derived=["M", "N"], ancestral="MN")
    dem.add_population_split(time=0.5, derived=["O", "P"], ancestral="OP")
    dem.add_population_split(time=2.0, derived=["AB", "CD"], ancestral="ABCD")
    dem.add_population_split(time=2.0, derived=["EF", "GH"], ancestral="EFGH")
    dem.add_population_split(time=2.0, derived=["IJ", "KL"], ancestral="IJKL")
    dem.add_population_split(time=2.0, derived=["MN", "OP"], ancestral="MNOP")
    dem.add_population_split(time=4.0, derived=["ABCD", "EFGH"], ancestral="ABCDEFGH")
    dem.add_population_split(time=4.0, derived=["IJKL", "MNOP"], ancestral="IJKLMNOP")
    dem.add_population_split(time=8.0, derived=["ABCDEFGH", "IJKLMNOP"], ancestral="ROOT")
    return dem, taxa, "((((A,B),(C,D)),((E,F),(G,H))),(((I,J),(K,L)),((M,N),(O,P))));"


def short_branch_16_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    taxa = tuple(chr(ord("A") + i) for i in range(16))
    dem = msprime.Demography()
    for p in taxa:
        dem.add_population(name=p, initial_size=1)
    for p in (
        "AB",
        "CD",
        "EF",
        "GH",
        "IJ",
        "KL",
        "MN",
        "OP",
        "ABCD",
        "EFGH",
        "IJKL",
        "MNOP",
        "ABCDEFGH",
        "IJKLMNOP",
        "ROOT",
    ):
        dem.add_population(name=p, initial_size=1)
    dem.add_population_split(time=0.4, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=0.4, derived=["C", "D"], ancestral="CD")
    dem.add_population_split(time=0.4, derived=["E", "F"], ancestral="EF")
    dem.add_population_split(time=0.4, derived=["G", "H"], ancestral="GH")
    dem.add_population_split(time=0.4, derived=["I", "J"], ancestral="IJ")
    dem.add_population_split(time=0.4, derived=["K", "L"], ancestral="KL")
    dem.add_population_split(time=0.4, derived=["M", "N"], ancestral="MN")
    dem.add_population_split(time=0.4, derived=["O", "P"], ancestral="OP")
    dem.add_population_split(time=0.5, derived=["AB", "CD"], ancestral="ABCD")
    dem.add_population_split(time=0.5, derived=["EF", "GH"], ancestral="EFGH")
    dem.add_population_split(time=0.5, derived=["IJ", "KL"], ancestral="IJKL")
    dem.add_population_split(time=0.5, derived=["MN", "OP"], ancestral="MNOP")
    dem.add_population_split(time=0.6, derived=["ABCD", "EFGH"], ancestral="ABCDEFGH")
    dem.add_population_split(time=0.6, derived=["IJKL", "MNOP"], ancestral="IJKLMNOP")
    dem.add_population_split(time=0.7, derived=["ABCDEFGH", "IJKLMNOP"], ancestral="ROOT")
    return dem, taxa, "((((A,B),(C,D)),((E,F),(G,H))),(((I,J),(K,L)),((M,N),(O,P))));"


def _taxa_labels(n_taxa: int) -> tuple[str, ...]:
    return tuple(f"T{i:02d}" for i in range(1, int(n_taxa) + 1))


def _build_balanced_demography(
    n_taxa: int,
    *,
    split_times: Sequence[float],
) -> tuple[msprime.Demography, tuple[str, ...], str]:
    if n_taxa < 2 or (n_taxa & (n_taxa - 1)) != 0:
        raise ValueError("n_taxa must be a power of two >= 2")
    n_levels = int(np.log2(n_taxa))
    if len(split_times) != n_levels:
        raise ValueError("split_times length must equal log2(n_taxa)")

    taxa = _taxa_labels(n_taxa)
    dem = msprime.Demography()
    for leaf in taxa:
        dem.add_population(name=leaf, initial_size=1)

    groups: list[tuple[str, str]] = [(leaf, leaf) for leaf in taxa]
    for level in range(n_levels):
        t = float(split_times[level])
        if t <= 0:
            raise ValueError("split times must be > 0")
        next_groups: list[tuple[str, str]] = []
        for i in range(0, len(groups), 2):
            left_name, left_nwk = groups[i]
            right_name, right_nwk = groups[i + 1]
            anc = f"N{level}_{i // 2}"
            dem.add_population(name=anc, initial_size=1)
            dem.add_population_split(time=t, derived=[left_name, right_name], ancestral=anc)
            next_groups.append((anc, f"({left_nwk},{right_nwk})"))
        groups = next_groups

    root_name, root_nwk = groups[0]
    if root_name != f"N{n_levels - 1}_0":
        raise RuntimeError("unexpected balanced root naming")
    return dem, taxa, root_nwk + ";"


def balanced_64_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    # Separation increases with depth for stable baseline behavior.
    return _build_balanced_demography(
        64,
        split_times=(0.5, 2.0, 4.0, 8.0, 16.0, 32.0),
    )


def short_branch_64_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    # Compressed internode times induce higher ILS.
    return _build_balanced_demography(
        64,
        split_times=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
    )


def asymmetric_64_taxon_demography() -> tuple[msprime.Demography, tuple[str, ...], str]:
    # Pectinate species tree to create strong asymmetry in depth and branch lengths.
    taxa = _taxa_labels(64)
    dem = msprime.Demography()
    for leaf in taxa:
        dem.add_population(name=leaf, initial_size=1)

    current_name = taxa[0]
    current_nwk = taxa[0]
    current_time = 0.0
    for i, leaf in enumerate(taxa[1:], start=2):
        anc = f"C{i:02d}"
        dem.add_population(name=anc, initial_size=1)
        # Ensure strictly increasing split times.
        split_time = current_time + 0.15
        dem.add_population_split(time=split_time, derived=[current_name, leaf], ancestral=anc)
        current_name = anc
        current_nwk = f"({current_nwk},{leaf})"
        current_time = split_time
    return dem, taxa, current_nwk + ";"


def simulate_gene_trees(
    demography: msprime.Demography,
    taxa: Sequence[str],
    *,
    n_replicates: int,
    seed: int,
) -> list[treeswift.Tree]:
    labels = {i: str(taxa[i]) for i in range(len(taxa))}
    out = []
    for ts in msprime.sim_ancestry(
        samples={str(t): 1 for t in taxa},
        demography=demography,
        ploidy=1,
        sequence_length=1,
        recombination_rate=0,
        num_replicates=int(n_replicates),
        random_seed=int(seed),
    ):
        out.append(_read_tree(ts.first().as_newick(node_labels=labels)))
    return out


def apply_missing_and_label_noise(
    gene_trees: Sequence[treeswift.Tree],
    *,
    missing_fraction: float = 0.0,
    label_noise_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[treeswift.Tree]:
    """Apply controlled missing-taxa pruning and label-swap noise."""
    if missing_fraction < 0.0 or missing_fraction > 1.0:
        raise ValueError("missing_fraction must be in [0, 1]")
    if label_noise_fraction < 0.0 or label_noise_fraction > 1.0:
        raise ValueError("label_noise_fraction must be in [0, 1]")
    rng = rng if rng is not None else np.random.default_rng(0)

    out: list[treeswift.Tree] = []
    for tr in gene_trees:
        cur = _read_tree(tr.newick())

        if missing_fraction > 0.0 and float(rng.random()) < missing_fraction:
            leaves = [str(n.label) for n in cur.traverse_leaves()]
            if len(leaves) > 5:
                drop = leaves[int(rng.integers(0, len(leaves)))]
                keep = set(leaves) - {drop}
                cur = cur.extract_tree_with(keep, suppress_unifurcations=True)

        if label_noise_fraction > 0.0 and float(rng.random()) < label_noise_fraction:
            leaves_nodes = [n for n in cur.traverse_leaves()]
            if len(leaves_nodes) >= 2:
                i, j = rng.choice(len(leaves_nodes), size=2, replace=False)
                a = leaves_nodes[int(i)]
                b = leaves_nodes[int(j)]
                a.label, b.label = b.label, a.label

        out.append(cur)
    return out


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    n_gene_trees: int
    phase2_rf: int
    phase4_rf: int
    consensus_rf: int
    strict_consensus_rf: int
    nj_rf: int
    upgma_rf: int
    astral_rf: int | None
    tree_qmc_rf: int | None
    phase2_newick: str
    phase4_newick: str
    consensus_newick: str
    strict_consensus_newick: str
    nj_newick: str
    upgma_newick: str
    astral_newick: str | None
    tree_qmc_newick: str | None
    true_newick: str
    phase2_guardrail_diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class ThresholdCalibrationRow:
    threshold: float
    dataset: str
    n_replicates: int
    mean_rf: float
    std_rf: float
    ci95_low: float
    ci95_high: float


def summarize_rf_replicates(
    replicate_results: Sequence[Sequence[BenchmarkResult]],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate RF metrics across replicate runs by dataset and method.

    Returns:
        {dataset: {method_key: {"mean": x, "std": y, "n": n, "ci95_low": a, "ci95_high": b}}}
    """
    by_dataset: dict[str, dict[str, list[float]]] = {}
    for run in replicate_results:
        for r in run:
            ds = by_dataset.setdefault(r.dataset, {k: [] for k in RF_METHOD_KEYS})
            for key in RF_METHOD_KEYS:
                v = getattr(r, key)
                if v is None:
                    continue
                ds[key].append(float(v))

    out: dict[str, dict[str, dict[str, float]]] = {}
    rng = np.random.default_rng(bootstrap_seed)
    for dataset, metric_vals in by_dataset.items():
        out[dataset] = {}
        for key, vals in metric_vals.items():
            arr = np.asarray(vals, dtype=float)
            n = int(arr.size)
            if n == 0:
                continue
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            if n > 1 and n_bootstrap > 0:
                idx = rng.integers(0, n, size=(int(n_bootstrap), n))
                boot_means = np.mean(arr[idx], axis=1)
                ci_low = float(np.percentile(boot_means, 2.5))
                ci_high = float(np.percentile(boot_means, 97.5))
            else:
                ci_low = mean
                ci_high = mean
            out[dataset][key] = {
                "mean": mean,
                "std": std,
                "n": float(n),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
    return out


def _star_tree_newick(taxa: Sequence[str]) -> str:
    return "(" + ",".join(sorted(str(t) for t in taxa)) + ");"


def _to_dendropy_treelist(
    gene_trees: Sequence[treeswift.Tree], taxa: Sequence[str]
):
    import dendropy

    tx = dendropy.TaxonNamespace(list(sorted(str(t) for t in taxa)))
    tlist = dendropy.TreeList(taxon_namespace=tx)
    for tr in gene_trees:
        tlist.append(
            dendropy.Tree.get(
                data=tr.newick(),
                schema="newick",
                taxon_namespace=tx,
                preserve_underscores=True,
                rooting="force-unrooted",
            )
        )
    return tlist, tx


def consensus_baseline_newick(gene_trees: Sequence[treeswift.Tree], taxa: Sequence[str]) -> str:
    """External baseline via DendroPy majority-rule consensus."""
    try:
        tlist, _ = _to_dendropy_treelist(gene_trees, taxa)
    except Exception:
        # Deterministic fallback if dendropy is unavailable.
        return _star_tree_newick(taxa)
    cons = tlist.consensus(min_freq=0.5)
    cons.is_rooted = False
    cons.suppress_unifurcations()
    return cons.as_string(schema="newick").strip()


def strict_consensus_baseline_newick(gene_trees: Sequence[treeswift.Tree], taxa: Sequence[str]) -> str:
    """External baseline via DendroPy strict consensus."""
    try:
        tlist, _ = _to_dendropy_treelist(gene_trees, taxa)
    except Exception:
        return _star_tree_newick(taxa)
    cons = tlist.consensus(min_freq=1.0)
    cons.is_rooted = False
    cons.suppress_unifurcations()
    return cons.as_string(schema="newick").strip()


def nj_distance_baseline_newick(gene_trees: Sequence[treeswift.Tree], taxa: Sequence[str]) -> str:
    """External baseline via NJ on average gene-tree path distances."""
    try:
        import dendropy
    except Exception:
        return _star_tree_newick(taxa)

    labels = list(sorted(str(t) for t in taxa))
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    sums = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)

    tx = dendropy.TaxonNamespace(labels)
    for tr in gene_trees:
        dt = dendropy.Tree.get(
            data=tr.newick(),
            schema="newick",
            taxon_namespace=tx,
            preserve_underscores=True,
            rooting="force-unrooted",
        )
        pdm = dendropy.PhylogeneticDistanceMatrix.from_tree(dt)
        present = [str(leaf.taxon.label) for leaf in dt.leaf_node_iter()]
        for a, b in itertools.combinations(sorted(present), 2):
            i, j = label_to_idx[a], label_to_idx[b]
            d = float(pdm(tx.get_taxon(a), tx.get_taxon(b)))
            sums[i, j] += d
            sums[j, i] += d
            counts[i, j] += 1
            counts[j, i] += 1

    observed = counts > 0
    if np.any(observed):
        fallback = float(np.max(sums[observed] / counts[observed])) + 1.0
    else:
        fallback = 2.0

    dist = np.full((n, n), fallback, dtype=float)
    np.fill_diagonal(dist, 0.0)
    valid = counts > 0
    dist[valid] = sums[valid] / counts[valid]

    lines = ["," + ",".join(labels)]
    for i, label in enumerate(labels):
        row = [label] + [f"{dist[i, j]:.10f}" for j in range(n)]
        lines.append(",".join(row))
    csv = io.StringIO("\n".join(lines) + "\n")
    matrix = dendropy.PhylogeneticDistanceMatrix.from_csv(csv, taxon_namespace=tx)
    tree = matrix.nj_tree()
    tree.is_rooted = False
    tree.suppress_unifurcations()
    return tree.as_string(schema="newick").strip()


def upgma_distance_baseline_newick(gene_trees: Sequence[treeswift.Tree], taxa: Sequence[str]) -> str:
    """External baseline via UPGMA on average gene-tree path distances."""
    try:
        import dendropy
    except Exception:
        return _star_tree_newick(taxa)

    labels = list(sorted(str(t) for t in taxa))
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    sums = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)

    tx = dendropy.TaxonNamespace(labels)
    for tr in gene_trees:
        dt = dendropy.Tree.get(
            data=tr.newick(),
            schema="newick",
            taxon_namespace=tx,
            preserve_underscores=True,
            rooting="force-unrooted",
        )
        pdm = dendropy.PhylogeneticDistanceMatrix.from_tree(dt)
        present = [str(leaf.taxon.label) for leaf in dt.leaf_node_iter()]
        for a, b in itertools.combinations(sorted(present), 2):
            i, j = label_to_idx[a], label_to_idx[b]
            d = float(pdm(tx.get_taxon(a), tx.get_taxon(b)))
            sums[i, j] += d
            sums[j, i] += d
            counts[i, j] += 1
            counts[j, i] += 1

    observed = counts > 0
    if np.any(observed):
        fallback = float(np.max(sums[observed] / counts[observed])) + 1.0
    else:
        fallback = 2.0

    dist = np.full((n, n), fallback, dtype=float)
    np.fill_diagonal(dist, 0.0)
    valid = counts > 0
    dist[valid] = sums[valid] / counts[valid]

    lines = ["," + ",".join(labels)]
    for i, label in enumerate(labels):
        row = [label] + [f"{dist[i, j]:.10f}" for j in range(n)]
        lines.append(",".join(row))
    csv = io.StringIO("\n".join(lines) + "\n")
    matrix = dendropy.PhylogeneticDistanceMatrix.from_csv(csv, taxon_namespace=tx)
    tree = matrix.upgma_tree()
    tree.is_rooted = False
    tree.suppress_unifurcations()
    return tree.as_string(schema="newick").strip()


def astral_baseline_newick(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    astral_jar_path: str | None,
    timeout_seconds: int = 120,
) -> str | None:
    """External baseline via ASTRAL if jar path is provided and runnable."""
    if astral_jar_path is None:
        return None
    jar = Path(astral_jar_path)
    if not jar.exists():
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="oaktree_astral_") as td:
            td_path = Path(td)
            inp = td_path / "genes.nwk"
            outp = td_path / "species.tre"
            inp.write_text("".join(tr.newick().rstrip() + "\n" for tr in gene_trees), encoding="utf-8")
            proc = subprocess.run(
                ["java", "-jar", str(jar), "-i", str(inp), "-o", str(outp)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(timeout_seconds),
                check=False,
            )
            if proc.returncode != 0 or not outp.exists():
                return None
            text = outp.read_text(encoding="utf-8").strip()
            if not text:
                return None
            # Keep only first Newick line if extra logs leaked.
            line = text.splitlines()[0].strip()
            tr = _read_tree(line)
            leaves = sorted(str(n.label) for n in tr.traverse_leaves())
            if leaves != sorted(set(str(t) for t in taxa)):
                return None
            return line
    except Exception:
        return None


def tree_qmc_baseline_newick(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    tree_qmc_bin: str | None,
    timeout_seconds: int = 180,
) -> str | None:
    """External baseline via TREE-QMC if executable is provided and runnable."""
    if tree_qmc_bin is None:
        return None
    # TREE-QMC can be sensitive to annotations/lengths/support labels emitted by
    # upstream simulators and tree transforms. Feed topology-only Newick strings.
    ann_re = re.compile(r"\[&[^\]]+\]")
    bl_re = re.compile(r":[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?")
    internal_label_re = re.compile(r"\)([^,():;]+)")
    try:
        with tempfile.TemporaryDirectory(prefix="oaktree_treeqmc_") as td:
            td_path = Path(td)
            inp = td_path / "genes.nwk"
            outp = td_path / "species.tre"
            def _sanitize(nwk: str) -> str:
                s = ann_re.sub("", nwk)
                s = bl_re.sub("", s)
                s = internal_label_re.sub(")", s)
                s = "".join(s.split())
                if not s.endswith(";"):
                    s += ";"
                return s
            inp.write_text(
                "".join(_sanitize(tr.newick()) + "\n" for tr in gene_trees),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [str(tree_qmc_bin), "-i", str(inp), "-o", str(outp), "--override"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(timeout_seconds),
                check=False,
            )
            if proc.returncode != 0:
                return None
            if outp.exists():
                text = outp.read_text(encoding="utf-8").strip()
            else:
                # Fallback: TREE-QMC can emit to stdout depending on invocation.
                text = str(proc.stdout or "").strip()
            if not text:
                return None
            line = text.splitlines()[0].strip()
            tr = _read_tree(line)
            leaves = sorted(str(n.label) for n in tr.traverse_leaves())
            if leaves != sorted(set(str(t) for t in taxa)):
                return None
            return line
    except Exception:
        return None


def run_baseline_benchmark(
    *,
    n_gene_trees: int = 150,
    seed: int = 0,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
    astral_jar_path: str | None = None,
    astral_timeout_seconds: int = 120,
    tree_qmc_bin: str | None = None,
    tree_qmc_timeout_seconds: int = 180,
) -> list[BenchmarkResult]:
    datasets = [
        ("balanced8", balanced_8_taxon_demography),
        ("asymmetric8", asymmetric_8_taxon_demography),
    ]
    results: list[BenchmarkResult] = []
    for i, (name, factory) in enumerate(datasets):
        dem, taxa, true_nwk = factory()
        genes = simulate_gene_trees(dem, taxa, n_replicates=n_gene_trees, seed=seed + i + 1)
        astral_pre = astral_baseline_newick(
            genes,
            taxa,
            astral_jar_path=astral_jar_path,
            timeout_seconds=astral_timeout_seconds,
        )
        tree_qmc_pre = tree_qmc_baseline_newick(
            genes,
            taxa,
            tree_qmc_bin=tree_qmc_bin,
            timeout_seconds=tree_qmc_timeout_seconds,
        )
        external_candidates: list[tuple[str, str]] = []
        if astral_pre is not None:
            external_candidates.append(("astral", astral_pre))
        if tree_qmc_pre is not None:
            external_candidates.append(("tree_qmc", tree_qmc_pre))
        rng = np.random.default_rng(seed + 100 + i)
        p2_diag: dict[str, object] | None = {} if baseline_guardrail else None
        p2 = infer_species_tree_newick_phase2(
            genes,
            taxa=taxa,
            max_quintets_per_tree=60,
            rng=rng,
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
            guardrail_diagnostics=p2_diag,
            external_candidates=external_candidates if baseline_guardrail else None,
        )
        p4 = infer_species_tree_newick_phase4_em(
            genes,
            taxa=taxa,
            n_iterations=6,
            max_quintets_per_tree=60,
            rng=np.random.default_rng(seed + 200 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
        )
        cons = consensus_baseline_newick(genes, taxa)
        scons = strict_consensus_baseline_newick(genes, taxa)
        nj = nj_distance_baseline_newick(genes, taxa)
        upgma = upgma_distance_baseline_newick(genes, taxa)
        astral = astral_pre
        tree_qmc = tree_qmc_pre
        rf2 = rf_distance_unrooted(p2, true_nwk, taxa)
        rf4 = rf_distance_unrooted(p4, true_nwk, taxa)
        rfc = rf_distance_unrooted(cons, true_nwk, taxa)
        rfs = rf_distance_unrooted(scons, true_nwk, taxa)
        rfnj = rf_distance_unrooted(nj, true_nwk, taxa)
        rfup = rf_distance_unrooted(upgma, true_nwk, taxa)
        rfast = rf_distance_unrooted(astral, true_nwk, taxa) if astral is not None else None
        rftq = rf_distance_unrooted(tree_qmc, true_nwk, taxa) if tree_qmc is not None else None
        results.append(
            BenchmarkResult(
                dataset=name,
                n_gene_trees=n_gene_trees,
                phase2_rf=rf2,
                phase4_rf=rf4,
                consensus_rf=rfc,
                strict_consensus_rf=rfs,
                nj_rf=rfnj,
                upgma_rf=rfup,
                astral_rf=rfast,
                tree_qmc_rf=rftq,
                phase2_newick=p2,
                phase4_newick=p4,
                consensus_newick=cons,
                strict_consensus_newick=scons,
                nj_newick=nj,
                upgma_newick=upgma,
                astral_newick=astral,
                tree_qmc_newick=tree_qmc,
                true_newick=true_nwk,
                phase2_guardrail_diagnostics=p2_diag,
            )
        )
    return results


def run_expanded_benchmark(
    *,
    n_gene_trees: int = 200,
    seed: int = 0,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
    astral_jar_path: str | None = None,
    astral_timeout_seconds: int = 120,
    tree_qmc_bin: str | None = None,
    tree_qmc_timeout_seconds: int = 180,
) -> list[BenchmarkResult]:
    """Expanded benchmark across diverse 8-taxon regimes."""
    dataset_factories = [
        ("balanced8", balanced_8_taxon_demography, False),
        ("asymmetric8", asymmetric_8_taxon_demography, False),
        ("shortbranch8", short_branch_8_taxon_demography, False),
        ("balanced8_missing", balanced_8_taxon_demography, True),
    ]
    results: list[BenchmarkResult] = []
    for i, (name, factory, with_missing) in enumerate(dataset_factories):
        dem, taxa, true_nwk = factory()
        genes = simulate_gene_trees(dem, taxa, n_replicates=n_gene_trees, seed=seed + i + 1)
        if with_missing:
            rng_missing = np.random.default_rng(seed + 500 + i)
            pruned = []
            for tr in genes:
                leaves = [str(n.label) for n in tr.traverse_leaves()]
                drop = leaves[int(rng_missing.integers(0, len(leaves)))]
                keep = set(leaves) - {drop}
                pruned.append(tr.extract_tree_with(keep, suppress_unifurcations=True))
            genes = pruned
        astral_pre = astral_baseline_newick(
            genes,
            taxa,
            astral_jar_path=astral_jar_path,
            timeout_seconds=astral_timeout_seconds,
        )
        tree_qmc_pre = tree_qmc_baseline_newick(
            genes,
            taxa,
            tree_qmc_bin=tree_qmc_bin,
            timeout_seconds=tree_qmc_timeout_seconds,
        )
        external_candidates: list[tuple[str, str]] = []
        if astral_pre is not None:
            external_candidates.append(("astral", astral_pre))
        if tree_qmc_pre is not None:
            external_candidates.append(("tree_qmc", tree_qmc_pre))

        p2_diag: dict[str, object] | None = {} if baseline_guardrail else None
        p2 = infer_species_tree_newick_phase2(
            genes,
            taxa=taxa,
            max_quintets_per_tree=70,
            rng=np.random.default_rng(seed + 100 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
            guardrail_diagnostics=p2_diag,
            external_candidates=external_candidates if baseline_guardrail else None,
        )
        p4 = infer_species_tree_newick_phase4_em(
            genes,
            taxa=taxa,
            n_iterations=6,
            max_quintets_per_tree=70,
            rng=np.random.default_rng(seed + 200 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
        )
        cons = consensus_baseline_newick(genes, taxa)
        scons = strict_consensus_baseline_newick(genes, taxa)
        nj = nj_distance_baseline_newick(genes, taxa)
        upgma = upgma_distance_baseline_newick(genes, taxa)
        astral = astral_pre
        tree_qmc = tree_qmc_pre
        rf2 = rf_distance_unrooted(p2, true_nwk, taxa)
        rf4 = rf_distance_unrooted(p4, true_nwk, taxa)
        rfc = rf_distance_unrooted(cons, true_nwk, taxa)
        rfs = rf_distance_unrooted(scons, true_nwk, taxa)
        rfnj = rf_distance_unrooted(nj, true_nwk, taxa)
        rfup = rf_distance_unrooted(upgma, true_nwk, taxa)
        rfast = rf_distance_unrooted(astral, true_nwk, taxa) if astral is not None else None
        rftq = rf_distance_unrooted(tree_qmc, true_nwk, taxa) if tree_qmc is not None else None
        results.append(
            BenchmarkResult(
                dataset=name,
                n_gene_trees=n_gene_trees,
                phase2_rf=rf2,
                phase4_rf=rf4,
                consensus_rf=rfc,
                strict_consensus_rf=rfs,
                nj_rf=rfnj,
                upgma_rf=rfup,
                astral_rf=rfast,
                tree_qmc_rf=rftq,
                phase2_newick=p2,
                phase4_newick=p4,
                consensus_newick=cons,
                strict_consensus_newick=scons,
                nj_newick=nj,
                upgma_newick=upgma,
                astral_newick=astral,
                tree_qmc_newick=tree_qmc,
                true_newick=true_nwk,
                phase2_guardrail_diagnostics=p2_diag,
            )
        )
    return results


def run_scaled16_quick_benchmark(
    *,
    n_gene_trees: int = 180,
    seed: int = 0,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
    astral_jar_path: str | None = None,
    astral_timeout_seconds: int = 180,
    tree_qmc_bin: str | None = None,
    tree_qmc_timeout_seconds: int = 180,
) -> list[BenchmarkResult]:
    """Quick, larger-size benchmark (16 taxa, slightly larger gene counts)."""
    dataset_factories = [
        ("balanced16", balanced_16_taxon_demography, False),
        ("shortbranch16", short_branch_16_taxon_demography, False),
        ("balanced16_missing", balanced_16_taxon_demography, True),
    ]
    results: list[BenchmarkResult] = []
    for i, (name, factory, with_missing) in enumerate(dataset_factories):
        dem, taxa, true_nwk = factory()
        genes = simulate_gene_trees(dem, taxa, n_replicates=n_gene_trees, seed=seed + i + 1)
        if with_missing:
            rng_missing = np.random.default_rng(seed + 700 + i)
            pruned = []
            for tr in genes:
                leaves = [str(n.label) for n in tr.traverse_leaves()]
                drop = leaves[int(rng_missing.integers(0, len(leaves)))]
                keep = set(leaves) - {drop}
                pruned.append(tr.extract_tree_with(keep, suppress_unifurcations=True))
            genes = pruned
        astral_pre = astral_baseline_newick(
            genes,
            taxa,
            astral_jar_path=astral_jar_path,
            timeout_seconds=astral_timeout_seconds,
        )
        tree_qmc_pre = tree_qmc_baseline_newick(
            genes,
            taxa,
            tree_qmc_bin=tree_qmc_bin,
            timeout_seconds=tree_qmc_timeout_seconds,
        )
        external_candidates: list[tuple[str, str]] = []
        if astral_pre is not None:
            external_candidates.append(("astral", astral_pre))
        if tree_qmc_pre is not None:
            external_candidates.append(("tree_qmc", tree_qmc_pre))

        p2_diag: dict[str, object] | None = {} if baseline_guardrail else None
        p2 = infer_species_tree_newick_phase2(
            genes,
            taxa=taxa,
            max_quintets_per_tree=90,
            rng=np.random.default_rng(seed + 300 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
            guardrail_diagnostics=p2_diag,
            external_candidates=external_candidates if baseline_guardrail else None,
        )
        p4 = infer_species_tree_newick_phase4_em(
            genes,
            taxa=taxa,
            n_iterations=6,
            max_quintets_per_tree=90,
            rng=np.random.default_rng(seed + 400 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
        )
        cons = consensus_baseline_newick(genes, taxa)
        scons = strict_consensus_baseline_newick(genes, taxa)
        nj = nj_distance_baseline_newick(genes, taxa)
        upgma = upgma_distance_baseline_newick(genes, taxa)
        astral = astral_pre
        tree_qmc = tree_qmc_pre
        rf2 = rf_distance_unrooted(p2, true_nwk, taxa)
        rf4 = rf_distance_unrooted(p4, true_nwk, taxa)
        rfc = rf_distance_unrooted(cons, true_nwk, taxa)
        rfs = rf_distance_unrooted(scons, true_nwk, taxa)
        rfnj = rf_distance_unrooted(nj, true_nwk, taxa)
        rfup = rf_distance_unrooted(upgma, true_nwk, taxa)
        rfast = rf_distance_unrooted(astral, true_nwk, taxa) if astral is not None else None
        rftq = rf_distance_unrooted(tree_qmc, true_nwk, taxa) if tree_qmc is not None else None
        results.append(
            BenchmarkResult(
                dataset=name,
                n_gene_trees=n_gene_trees,
                phase2_rf=rf2,
                phase4_rf=rf4,
                consensus_rf=rfc,
                strict_consensus_rf=rfs,
                nj_rf=rfnj,
                upgma_rf=rfup,
                astral_rf=rfast,
                tree_qmc_rf=rftq,
                phase2_newick=p2,
                phase4_newick=p4,
                consensus_newick=cons,
                strict_consensus_newick=scons,
                nj_newick=nj,
                upgma_newick=upgma,
                astral_newick=astral,
                tree_qmc_newick=tree_qmc,
                true_newick=true_nwk,
                phase2_guardrail_diagnostics=p2_diag,
            )
        )
    return results


def run_scaled64_complex_benchmark(
    *,
    n_gene_trees: int = 220,
    seed: int = 0,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
    astral_jar_path: str | None = None,
    astral_timeout_seconds: int = 240,
    tree_qmc_bin: str | None = None,
    tree_qmc_timeout_seconds: int = 240,
) -> list[BenchmarkResult]:
    """Complex 64-taxon benchmark across five regimes."""
    dataset_configs = [
        ("balanced64", balanced_64_taxon_demography, 0.0, 0.0),
        ("asymmetric64", asymmetric_64_taxon_demography, 0.0, 0.0),
        ("shortbranch64", short_branch_64_taxon_demography, 0.0, 0.0),
        ("balanced64_missing", balanced_64_taxon_demography, 1.0, 0.0),
        ("shortbranch64_missing_noisy", short_branch_64_taxon_demography, 0.5, 0.35),
    ]
    results: list[BenchmarkResult] = []
    for i, (name, factory, miss_frac, noise_frac) in enumerate(dataset_configs):
        dem, taxa, true_nwk = factory()
        run_seed = int(seed + i + 1)
        genes = simulate_gene_trees(dem, taxa, n_replicates=n_gene_trees, seed=run_seed)
        genes = apply_missing_and_label_noise(
            genes,
            missing_fraction=float(miss_frac),
            label_noise_fraction=float(noise_frac),
            rng=np.random.default_rng(seed + 900 + i),
        )
        astral_pre = astral_baseline_newick(
            genes,
            taxa,
            astral_jar_path=astral_jar_path,
            timeout_seconds=astral_timeout_seconds,
        )
        tree_qmc_pre = tree_qmc_baseline_newick(
            genes,
            taxa,
            tree_qmc_bin=tree_qmc_bin,
            timeout_seconds=tree_qmc_timeout_seconds,
        )
        external_candidates: list[tuple[str, str]] = []
        if astral_pre is not None:
            external_candidates.append(("astral", astral_pre))
        if tree_qmc_pre is not None:
            external_candidates.append(("tree_qmc", tree_qmc_pre))

        p2_diag: dict[str, object] | None = {} if baseline_guardrail else None
        p2 = infer_species_tree_newick_phase2(
            genes,
            taxa=taxa,
            max_quintets_per_tree=90,
            rng=np.random.default_rng(seed + 100 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
            guardrail_diagnostics=p2_diag,
            external_candidates=external_candidates if baseline_guardrail else None,
        )
        p4 = infer_species_tree_newick_phase4_em(
            genes,
            taxa=taxa,
            n_iterations=4,
            max_quintets_per_tree=90,
            rng=np.random.default_rng(seed + 200 + i),
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_subset_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
        )
        cons = consensus_baseline_newick(genes, taxa)
        scons = strict_consensus_baseline_newick(genes, taxa)
        nj = nj_distance_baseline_newick(genes, taxa)
        upgma = upgma_distance_baseline_newick(genes, taxa)
        astral = astral_pre
        tree_qmc = tree_qmc_pre
        rf2 = rf_distance_unrooted(p2, true_nwk, taxa)
        rf4 = rf_distance_unrooted(p4, true_nwk, taxa)
        rfc = rf_distance_unrooted(cons, true_nwk, taxa)
        rfs = rf_distance_unrooted(scons, true_nwk, taxa)
        rfnj = rf_distance_unrooted(nj, true_nwk, taxa)
        rfup = rf_distance_unrooted(upgma, true_nwk, taxa)
        rfast = rf_distance_unrooted(astral, true_nwk, taxa) if astral is not None else None
        rftq = rf_distance_unrooted(tree_qmc, true_nwk, taxa) if tree_qmc is not None else None
        results.append(
            BenchmarkResult(
                dataset=name,
                n_gene_trees=n_gene_trees,
                phase2_rf=rf2,
                phase4_rf=rf4,
                consensus_rf=rfc,
                strict_consensus_rf=rfs,
                nj_rf=rfnj,
                upgma_rf=rfup,
                astral_rf=rfast,
                tree_qmc_rf=rftq,
                phase2_newick=p2,
                phase4_newick=p4,
                consensus_newick=cons,
                strict_consensus_newick=scons,
                nj_newick=nj,
                upgma_newick=upgma,
                astral_newick=astral,
                tree_qmc_newick=tree_qmc,
                true_newick=true_nwk,
                phase2_guardrail_diagnostics=p2_diag,
            )
        )
    return results


def calibrate_low_signal_threshold(
    *,
    thresholds: Sequence[float],
    n_gene_trees: int = 120,
    seed: int = 0,
    n_replicates: int = 3,
    datasets: Sequence[str] | None = None,
    low_signal_mode: str = "fixed",
) -> list[ThresholdCalibrationRow]:
    """Evaluate Phase 2 RF sensitivity over low-signal thresholds."""
    configs = {
        "balanced8": (balanced_8_taxon_demography, 0.0, 0.0),
        "asymmetric8": (asymmetric_8_taxon_demography, 0.0, 0.0),
        "shortbranch8": (short_branch_8_taxon_demography, 0.0, 0.0),
        "balanced8_missing": (balanced_8_taxon_demography, 1.0, 0.0),
        "shortbranch8_missing": (short_branch_8_taxon_demography, 1.0, 0.0),
        "balanced8_noisy": (balanced_8_taxon_demography, 0.0, 0.35),
        "shortbranch8_noisy": (short_branch_8_taxon_demography, 0.0, 0.35),
        "shortbranch8_missing_noisy": (short_branch_8_taxon_demography, 0.5, 0.35),
    }
    use = (
        list(datasets)
        if datasets is not None
        else ["balanced8", "asymmetric8", "shortbranch8", "balanced8_missing", "shortbranch8_missing_noisy"]
    )
    for t in thresholds:
        if float(t) < 0.0:
            raise ValueError("thresholds must be >= 0")
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    out: list[ThresholdCalibrationRow] = []
    z = 1.96
    for th in thresholds:
        thf = float(th)
        for j, name in enumerate(use):
            if name not in configs:
                raise ValueError(f"unknown dataset: {name}")
            factory, miss_frac, noise_frac = configs[name]
            dem, taxa, true_nwk = factory()
            rf_vals: list[float] = []
            for r in range(n_replicates):
                run_seed = int(seed + 1000 * r + 17 * j + 1)
                genes = simulate_gene_trees(dem, taxa, n_replicates=n_gene_trees, seed=run_seed)
                genes = apply_missing_and_label_noise(
                    genes,
                    missing_fraction=miss_frac,
                    label_noise_fraction=noise_frac,
                    rng=np.random.default_rng(run_seed + 77),
                )
                p2 = infer_species_tree_newick_phase2(
                    genes,
                    taxa=taxa,
                    max_quintets_per_tree=70,
                    rng=np.random.default_rng(run_seed + 123),
                    low_signal_threshold=thf,
                    low_signal_mode=low_signal_mode,
                )
                rf_vals.append(float(rf_distance_unrooted(p2, true_nwk, taxa)))
            arr = np.asarray(rf_vals, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            half = float(z * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            out.append(
                ThresholdCalibrationRow(
                    threshold=thf,
                    dataset=name,
                    n_replicates=int(arr.size),
                    mean_rf=mean,
                    std_rf=std,
                    ci95_low=max(0.0, mean - half),
                    ci95_high=mean + half,
                )
            )
    return out
