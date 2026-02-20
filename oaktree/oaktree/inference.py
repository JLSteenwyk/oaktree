"""High-level species tree inference entry points."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import treeswift

from .graphs import partition_tree_to_newick, recursive_partition
from .trees import (
    QuintetObservation,
    canonicalize_quintet,
    get_leaf_set,
    get_shared_taxa,
    sample_quintet_subsets,
)


def extract_quintet_observations_from_gene_trees(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str] | None = None,
    max_quintets_per_tree: int | None = None,
    rng: np.random.Generator | None = None,
    gene_weights: Sequence[float] | None = None,
) -> list[QuintetObservation]:
    """Extract quintet observations from gene trees for a target taxa set."""
    if taxa is None:
        target_taxa = sorted(get_shared_taxa(gene_trees))
    else:
        target_taxa = sorted(set(taxa))
    if len(target_taxa) < 5:
        return []

    rng = rng if rng is not None else np.random.default_rng(0)
    if gene_weights is not None and len(gene_weights) != len(gene_trees):
        raise ValueError("gene_weights must have same length as gene_trees")

    observations: list[QuintetObservation] = []
    for i, tree in enumerate(gene_trees):
        tree_weight = 1.0 if gene_weights is None else float(gene_weights[i])
        if tree_weight <= 0.0:
            continue
        present = [t for t in target_taxa if t in get_leaf_set(tree)]
        if len(present) < 5:
            continue
        usable = sample_quintet_subsets(
            present,
            max_quintets=max_quintets_per_tree,
            rng=rng,
        )
        for q in usable:
            topology = canonicalize_quintet(tree, q)
            observations.append(QuintetObservation(taxa=q, topology=topology, weight=tree_weight))
    return observations


def extract_projected_quintet_observations_from_higher_order(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    subset_sizes: Sequence[int],
    subsets_per_tree: int,
    quintets_per_subset: int,
    base_weight: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[QuintetObservation]:
    """Sample k-taxon subsets (k>5) and project their signal into weighted quintets."""
    if subsets_per_tree <= 0 or quintets_per_subset <= 0 or base_weight <= 0.0:
        return []
    sizes = sorted({int(k) for k in subset_sizes if int(k) >= 6})
    if not sizes:
        return []
    rng = rng if rng is not None else np.random.default_rng(0)
    taxa_use = sorted(set(str(t) for t in taxa))
    out: list[QuintetObservation] = []
    for tr in gene_trees:
        leaf_set = get_leaf_set(tr)
        present = [t for t in taxa_use if t in leaf_set]
        if len(present) < 6:
            continue
        for _ in range(subsets_per_tree):
            k_candidates = [k for k in sizes if k <= len(present)]
            if not k_candidates:
                break
            k = int(k_candidates[int(rng.integers(0, len(k_candidates)))])
            subset = tuple(sorted(str(x) for x in rng.choice(present, size=k, replace=False)))
            q_all = [tuple(c) for c in combinations(subset, 5)]
            if len(q_all) > quintets_per_subset:
                idx = rng.choice(len(q_all), size=quintets_per_subset, replace=False)
                q_use = [q_all[int(i)] for i in sorted(idx)]
            else:
                q_use = q_all
            scale = float(base_weight) * (float(k - 5) / 3.0)
            for q in q_use:
                topo = canonicalize_quintet(tr, q)
                out.append(QuintetObservation(taxa=q, topology=topo, weight=scale))
    return out


def shrink_quintet_observations_by_confidence(
    observations: Sequence[QuintetObservation],
    *,
    min_confidence: float = 0.0,
    high_confidence_passthrough: float = 0.55,
) -> list[QuintetObservation]:
    """Collapse per-gene observations into confidence-weighted quintet signals.

    For each quintet taxa set, keep only the top-supported topology and weight it
    by support margin `(top1 - top2)` so ambiguous quintets are down-weighted.
    """
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if high_confidence_passthrough < 0.0 or high_confidence_passthrough > 1.0:
        raise ValueError("high_confidence_passthrough must be in [0, 1]")
    by_taxa: dict[tuple[str, str, str, str, str], dict[tuple, float]] = {}
    for obs in observations:
        taxa = tuple(sorted(obs.taxa))
        d = by_taxa.setdefault(taxa, {})
        topo = tuple(sorted(tuple(sorted(p)) for p in obs.topology))
        d[topo] = d.get(topo, 0.0) + float(getattr(obs, "weight", 1.0))

    out: list[QuintetObservation] = []
    for taxa, counts in by_taxa.items():
        if not counts:
            continue
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top1_topo, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        total = float(sum(counts.values()))
        if total <= 0.0:
            continue
        confidence = max(0.0, (float(top1) - float(top2)) / total)
        if confidence < min_confidence:
            continue
        if confidence >= high_confidence_passthrough:
            for topo, cnt in ranked:
                if float(cnt) <= 0.0:
                    continue
                out.append(
                    QuintetObservation(
                        taxa=taxa,
                        topology=((tuple(topo[0])), (tuple(topo[1]))),
                        weight=float(cnt),
                    )
                )
        else:
            weight = max(0.0, float(top1) - float(top2))
            if weight <= 0.0:
                continue
            out.append(
                QuintetObservation(
                    taxa=taxa,
                    topology=((tuple(top1_topo[0])), (tuple(top1_topo[1]))),
                    weight=weight,
                )
            )
    return out


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _score_species_tree_newick(
    gene_trees: Sequence[treeswift.Tree],
    species_tree_newick: str,
    taxa: Sequence[str],
    *,
    max_quintets_per_tree: int | None,
    rng: np.random.Generator,
) -> float:
    taxa_use = sorted(set(taxa))
    sampled_subsets_by_gene = _prepare_guardrail_quintet_subsets(
        gene_trees=gene_trees,
        taxa=taxa_use,
        max_quintets_per_tree=max_quintets_per_tree,
        rng=rng,
    )
    gene_topology_cache = _prepare_guardrail_gene_topology_cache(
        gene_trees=gene_trees,
        sampled_subsets_by_gene=sampled_subsets_by_gene,
    )
    return _score_species_tree_newick_cached(
        gene_trees=gene_trees,
        species_tree_newick=species_tree_newick,
        sampled_subsets_by_gene=sampled_subsets_by_gene,
        gene_topology_cache=gene_topology_cache,
    )


def _prepare_guardrail_quintet_subsets(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    max_quintets_per_tree: int | None,
    rng: np.random.Generator,
) -> dict[int, list[tuple[str, str, str, str, str]]]:
    taxa_use = sorted(set(taxa))
    out: dict[int, list[tuple[str, str, str, str, str]]] = {}
    for gt in gene_trees:
        present = [t for t in taxa_use if t in get_leaf_set(gt)]
        if len(present) < 5:
            out[id(gt)] = []
            continue
        subsets = sample_quintet_subsets(
            present,
            max_quintets=max_quintets_per_tree,
            rng=rng,
        )
        out[id(gt)] = subsets
    return out


def _prepare_guardrail_gene_topology_cache(
    gene_trees: Sequence[treeswift.Tree],
    *,
    sampled_subsets_by_gene: dict[int, list[tuple[str, str, str, str, str]]],
) -> dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]]:
    cache: dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]] = {}
    for gt in gene_trees:
        gid = id(gt)
        for q in sampled_subsets_by_gene.get(gid, []):
            cache[(gid, q)] = canonicalize_quintet(gt, q)
    return cache


def _score_species_tree_newick_cached(
    gene_trees: Sequence[treeswift.Tree],
    species_tree_newick: str,
    *,
    sampled_subsets_by_gene: dict[int, list[tuple[str, str, str, str, str]]],
    gene_topology_cache: dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]],
) -> float:
    st = _read_tree(species_tree_newick)
    species_topology_cache: dict[tuple[str, str, str, str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    per_tree = []
    for gt in gene_trees:
        gid = id(gt)
        usable = sampled_subsets_by_gene.get(gid, [])
        if not usable:
            continue
        m = 0
        for q in usable:
            st_top = species_topology_cache.get(q)
            if st_top is None:
                st_top = canonicalize_quintet(st, q)
                species_topology_cache[q] = st_top
            if gene_topology_cache[(gid, q)] == st_top:
                m += 1
        per_tree.append(float(m) / float(len(usable)))
    if not per_tree:
        return 0.0
    return float(np.mean(per_tree))


def _distance_baseline_newick(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    method: str,
) -> str:
    try:
        import dendropy
    except Exception:
        return "(" + ",".join(sorted(set(taxa))) + ");"

    labels = list(sorted(set(str(t) for t in taxa)))
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
        )
        pdm = dendropy.PhylogeneticDistanceMatrix.from_tree(dt)
        present = [str(leaf.taxon.label) for leaf in dt.leaf_node_iter()]
        for a, b in combinations(sorted(present), 2):
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

    import io

    lines = ["," + ",".join(labels)]
    for i, lab in enumerate(labels):
        lines.append(",".join([lab] + [f"{dist[i, j]:.10f}" for j in range(n)]))
    matrix = dendropy.PhylogeneticDistanceMatrix.from_csv(io.StringIO("\n".join(lines) + "\n"), taxon_namespace=tx)
    if method == "nj":
        tree = matrix.nj_tree()
    elif method == "upgma":
        tree = matrix.upgma_tree()
    else:
        raise ValueError("unknown distance baseline method")
    tree.is_rooted = False
    tree.suppress_unifurcations()
    return tree.as_string(schema="newick").strip()


def infer_species_tree_newick_phase2(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str] | None = None,
    max_quintets_per_tree: int | None = None,
    rng: np.random.Generator | None = None,
    *,
    n2_normalization: bool = True,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    precomputed_observations: Sequence[QuintetObservation] | None = None,
    confidence_shrink: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
) -> str:
    """Infer species-tree topology Newick using current Phase 2 pipeline."""
    if taxa is None:
        all_taxa = sorted(get_shared_taxa(gene_trees))
    else:
        all_taxa = sorted(set(taxa))
    if not all_taxa:
        raise ValueError("No taxa available for inference")
    if len(all_taxa) <= 3:
        return "(" + ",".join(all_taxa) + ");" if len(all_taxa) > 1 else all_taxa[0] + ";"

    effective_max_quintets = max_quintets_per_tree
    if baseline_guardrail and effective_max_quintets is not None and len(all_taxa) >= 24:
        # Large taxa sets need a slightly larger construction sample budget for
        # stable core candidate quality under guardrailed selection.
        effective_max_quintets = max(int(effective_max_quintets), 180)

    observations = (
        list(precomputed_observations)
        if precomputed_observations is not None
        else extract_quintet_observations_from_gene_trees(
            gene_trees=gene_trees,
            taxa=all_taxa,
            max_quintets_per_tree=effective_max_quintets,
            rng=rng,
        )
    )

    if (
        precomputed_observations is None
        and higher_order_subset_sizes
        and higher_order_subsets_per_tree > 0
        and higher_order_quintets_per_subset > 0
    ):
        observations.extend(
            extract_projected_quintet_observations_from_higher_order(
                gene_trees,
                all_taxa,
                subset_sizes=higher_order_subset_sizes,
                subsets_per_tree=higher_order_subsets_per_tree,
                quintets_per_subset=higher_order_quintets_per_subset,
                base_weight=higher_order_weight,
                rng=rng,
            )
        )

    if confidence_shrink:
        observations = shrink_quintet_observations_by_confidence(observations)

    partition_tree, _ = recursive_partition(
        taxa=all_taxa,
        quintet_observations=observations,
        species_tree_estimate=None,
        n2_normalization=n2_normalization,
        low_signal_threshold=low_signal_threshold,
        low_signal_mode=low_signal_mode,
    )
    p2_newick = partition_tree_to_newick(partition_tree)
    if not baseline_guardrail:
        return p2_newick

    candidates = [
        ("phase2", p2_newick),
        ("nj", _distance_baseline_newick(gene_trees, all_taxa, method="nj")),
        ("upgma", _distance_baseline_newick(gene_trees, all_taxa, method="upgma")),
    ]
    rng_local = np.random.default_rng(20260219)
    guardrail_score_max_quintets = max_quintets_per_tree
    if guardrail_score_max_quintets is not None and len(all_taxa) >= 24:
        # Slightly larger scoring sample stabilizes guardrail choice on larger
        # taxa sets while remaining much faster than exhaustive enumeration.
        guardrail_score_max_quintets = max(int(guardrail_score_max_quintets), 180)
    sampled_subsets_by_gene = _prepare_guardrail_quintet_subsets(
        gene_trees=gene_trees,
        taxa=all_taxa,
        max_quintets_per_tree=guardrail_score_max_quintets,
        rng=rng_local,
    )
    gene_topology_cache = _prepare_guardrail_gene_topology_cache(
        gene_trees=gene_trees,
        sampled_subsets_by_gene=sampled_subsets_by_gene,
    )
    scored = [
        (
            _score_species_tree_newick_cached(
                gene_trees=gene_trees,
                species_tree_newick=nwk,
                sampled_subsets_by_gene=sampled_subsets_by_gene,
                gene_topology_cache=gene_topology_cache,
            ),
            1 if name == "phase2" else 0,
            nwk,
        )
        for name, nwk in candidates
    ]
    scored.sort(reverse=True)
    return str(scored[0][2])


def infer_species_tree_newick_phase4_em(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str] | None = None,
    *,
    n_iterations: int = 3,
    max_quintets_per_tree: int | None = 200,
    rng: np.random.Generator | None = None,
    n2_normalization: bool = True,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
) -> str:
    """Phase 4 wrapper: run EM refinement initialized from Phase 2."""
    # Local import avoids module import cycles.
    from .weights import em_refine_species_tree_newick

    final, _ = em_refine_species_tree_newick(
        gene_trees=gene_trees,
        taxa=taxa,
        n_iterations=n_iterations,
        max_quintets_per_tree=max_quintets_per_tree,
        rng=rng,
        n2_normalization=n2_normalization,
        low_signal_threshold=low_signal_threshold,
        low_signal_mode=low_signal_mode,
        baseline_guardrail=baseline_guardrail,
        higher_order_subset_sizes=higher_order_subset_sizes,
        higher_order_subsets_per_tree=higher_order_subsets_per_tree,
        higher_order_quintets_per_subset=higher_order_quintets_per_subset,
        higher_order_weight=higher_order_weight,
    )
    return final
