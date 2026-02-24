"""High-level species tree inference entry points."""

from __future__ import annotations

from itertools import combinations
from math import log
import time
from typing import Sequence

import numpy as np
import treeswift

from .branch_lengths import optimize_branch_lengths_ml
from .graphs import partition_tree_to_newick, recursive_partition
from .msc import _quintet_probabilities_for_newick
from .trees import (
    QuintetObservation,
    canonicalize_quintet,
    extract_induced_subtree,
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
        gene_weights_by_id=None,
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


def _compute_robust_gene_weights(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    max_quintets_per_tree: int | None,
    rng: np.random.Generator,
) -> list[float]:
    """Coverage-aware + outlier-robust per-gene weights for quintet extraction."""
    taxa_use = sorted(set(str(t) for t in taxa))
    n_taxa = len(taxa_use)
    if n_taxa < 5 or not gene_trees:
        return [1.0 for _ in gene_trees]

    coverage_weight: list[float] = []
    sampled_subsets_by_gene: dict[int, list[tuple[str, str, str, str, str]]] = {}
    gene_topology_cache: dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]] = {}

    for gt in gene_trees:
        present = [t for t in taxa_use if t in get_leaf_set(gt)]
        frac = float(len(present)) / float(n_taxa)
        # Strongly penalize low-coverage genes in missing-data regimes.
        coverage_weight.append(float(max(frac * frac, 0.0)))
        if len(present) < 5:
            sampled_subsets_by_gene[id(gt)] = []
            continue
        subsets = sample_quintet_subsets(
            present,
            max_quintets=max_quintets_per_tree,
            rng=rng,
        )
        sampled_subsets_by_gene[id(gt)] = subsets
        for q in subsets:
            gene_topology_cache[(id(gt), q)] = canonicalize_quintet(gt, q)

    # Build aggregate quintet signal with coverage-aware votes.
    vote_by_quintet: dict[
        tuple[str, str, str, str, str],
        dict[tuple[tuple[str, str], tuple[str, str]], float],
    ] = {}
    for i, gt in enumerate(gene_trees):
        w = float(coverage_weight[i])
        if w <= 0.0:
            continue
        for q in sampled_subsets_by_gene.get(id(gt), []):
            topo = gene_topology_cache[(id(gt), q)]
            d = vote_by_quintet.setdefault(q, {})
            d[topo] = d.get(topo, 0.0) + w

    best_topology: dict[tuple[str, str, str, str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    confidence_by_quintet: dict[tuple[str, str, str, str, str], float] = {}
    for q, d in vote_by_quintet.items():
        if not d:
            continue
        ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        top1_topo, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        total = float(sum(d.values()))
        conf = max(0.0, (float(top1) - float(top2)) / total) if total > 0.0 else 0.0
        best_topology[q] = top1_topo
        confidence_by_quintet[q] = conf

    agreement: list[float] = []
    for gt in gene_trees:
        usable = sampled_subsets_by_gene.get(id(gt), [])
        if not usable:
            agreement.append(0.0)
            continue
        m = 0.0
        z = 0.0
        for q in usable:
            conf = max(float(confidence_by_quintet.get(q, 0.0)), 1e-6)
            z += conf
            if gene_topology_cache[(id(gt), q)] == best_topology.get(q):
                m += conf
        agreement.append(float(m / z) if z > 0.0 else 0.0)

    raw = np.array(
        [
            float(coverage_weight[i]) * (0.20 + 0.80 * float(agreement[i]))
            for i in range(len(gene_trees))
        ],
        dtype=float,
    )
    mean_coverage = float(np.mean(coverage_weight)) if coverage_weight else 1.0
    median_agreement = float(np.median(agreement)) if agreement else 1.0
    apply_trim = bool(mean_coverage < 0.95 or median_agreement < 0.60)
    valid = raw > 0.0
    if apply_trim and int(np.count_nonzero(valid)) >= 12:
        cutoff = float(np.quantile(raw[valid], 0.20))
        # Keep outliers but suppress their impact instead of hard dropping.
        raw = np.where(raw < cutoff, raw * 0.20, raw)
    valid = raw > 0.0
    if int(np.count_nonzero(valid)) == 0:
        return [1.0 for _ in gene_trees]
    raw = raw / float(np.mean(raw[valid]))
    return [float(w) for w in raw]


def _score_species_tree_newick_cached(
    gene_trees: Sequence[treeswift.Tree],
    species_tree_newick: str,
    *,
    sampled_subsets_by_gene: dict[int, list[tuple[str, str, str, str, str]]],
    gene_topology_cache: dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]],
    gene_weights_by_id: dict[int, float] | None = None,
) -> float:
    st = _read_tree(species_tree_newick)
    species_topology_cache: dict[tuple[str, str, str, str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    per_tree_scores: list[float] = []
    per_tree_weights: list[float] = []
    for gt in gene_trees:
        gid = id(gt)
        gene_w = float(gene_weights_by_id.get(gid, 1.0)) if gene_weights_by_id is not None else 1.0
        if gene_w <= 0.0:
            continue
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
        per_tree_scores.append(float(m) / float(len(usable)))
        per_tree_weights.append(gene_w)
    if not per_tree_scores:
        return 0.0
    return float(np.average(np.asarray(per_tree_scores, dtype=float), weights=np.asarray(per_tree_weights, dtype=float)))


def _score_species_tree_newick_msc_cached(
    gene_trees: Sequence[treeswift.Tree],
    species_tree_newick: str,
    *,
    sampled_subsets_by_gene: dict[int, list[tuple[str, str, str, str, str]]],
    gene_topology_cache: dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]],
    gene_weights_by_id: dict[int, float] | None = None,
) -> float:
    """Mean per-gene MSC log-likelihood score on sampled quintets."""
    st = _read_tree(species_tree_newick)
    has_branch_lengths = any(
        (node is not st.root) and (not node.is_leaf()) and (float(node.edge_length or 0.0) > 0.0)
        for node in st.root.traverse_preorder()
    )
    if not has_branch_lengths:
        return _score_species_tree_newick_cached(
            gene_trees=gene_trees,
            species_tree_newick=species_tree_newick,
            sampled_subsets_by_gene=sampled_subsets_by_gene,
            gene_topology_cache=gene_topology_cache,
            gene_weights_by_id=gene_weights_by_id,
        )

    species_prob_cache: dict[
        tuple[str, str, str, str, str], dict[tuple[tuple[str, str], tuple[str, str]], float]
    ] = {}
    per_tree_scores: list[float] = []
    per_tree_weights: list[float] = []
    for gt in gene_trees:
        gid = id(gt)
        gene_w = float(gene_weights_by_id.get(gid, 1.0)) if gene_weights_by_id is not None else 1.0
        if gene_w <= 0.0:
            continue
        usable = sampled_subsets_by_gene.get(gid, [])
        if not usable:
            continue
        total = 0.0
        for q in usable:
            probs = species_prob_cache.get(q)
            if probs is None:
                induced = extract_induced_subtree(st, q)
                induced.resolve_polytomies()
                induced.suppress_unifurcations()
                probs = _quintet_probabilities_for_newick(induced.newick())
                species_prob_cache[q] = probs
            gt_top = gene_topology_cache[(gid, q)]
            p = max(float(probs.get(tuple(sorted(gt_top)), 0.0)), 1e-12)
            total += log(p)
        per_tree_scores.append(total / float(len(usable)))
        per_tree_weights.append(gene_w)
    if not per_tree_scores:
        return -float("inf")
    return float(np.average(np.asarray(per_tree_scores, dtype=float), weights=np.asarray(per_tree_weights, dtype=float)))


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


def _node_leaf_signatures(tree: treeswift.Tree) -> dict[treeswift.Node, tuple[str, ...]]:
    sig: dict[treeswift.Node, tuple[str, ...]] = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            sig[node] = (str(node.label),)
        else:
            leaves: list[str] = []
            for ch in node.children:
                leaves.extend(sig[ch])
            sig[node] = tuple(sorted(leaves))
    return sig


def _generate_nni_neighbors_newicks(base_newick: str, *, max_neighbors: int = 16) -> list[str]:
    """Generate bounded rooted-NNI neighbors via child/sibling swaps."""
    tr = _read_tree(base_newick)
    sig = _node_leaf_signatures(tr)
    seen: set[str] = set()
    out: list[str] = []
    for parent in tr.root.traverse_preorder():
        if parent.is_leaf() or len(parent.children) != 2:
            continue
        left, right = parent.children[0], parent.children[1]
        for child, sibling in ((left, right), (right, left)):
            if child.is_leaf() or len(child.children) != 2:
                continue
            c1, c2 = child.children[0], child.children[1]
            for sub in (c1, c2):
                t2 = _read_tree(base_newick)
                sig2 = _node_leaf_signatures(t2)
                # Resolve nodes in cloned tree by clade signature.
                by_sig = {v: k for k, v in sig2.items()}
                p2 = by_sig.get(sig[parent])
                c2_node = by_sig.get(sig[child])
                s2 = by_sig.get(sig[sibling])
                sub2 = by_sig.get(sig[sub])
                if p2 is None or c2_node is None or s2 is None or sub2 is None:
                    continue
                if sub2.parent is not c2_node:
                    continue
                if s2.parent is not p2:
                    continue
                # Swap one child of c2_node with its sibling branch at p2.
                c2_node.remove_child(sub2)
                p2.remove_child(s2)
                c2_node.add_child(s2)
                p2.add_child(sub2)
                t2.suppress_unifurcations()
                nwk = t2.newick()
                if nwk in seen or nwk == base_newick:
                    continue
                seen.add(nwk)
                out.append(nwk)
                if len(out) >= int(max_neighbors):
                    return out
    return out


def _consensus_baseline_newick(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
) -> str:
    """Majority-rule consensus baseline candidate."""
    try:
        import dendropy
    except Exception:
        return "(" + ",".join(sorted(set(taxa))) + ");"
    tx = dendropy.TaxonNamespace(list(sorted(set(str(t) for t in taxa))))
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
    cons = tlist.consensus(min_freq=0.5)
    cons.is_rooted = False
    cons.suppress_unifurcations()
    return cons.as_string(schema="newick").strip()


def _canonical_split(subset: set[str], taxa_set: set[str]) -> tuple[str, ...]:
    other = taxa_set - subset
    side = subset if len(subset) <= len(other) else other
    return tuple(sorted(side))


def _tree_internal_splits(tree: treeswift.Tree, taxa_set: set[str]) -> set[tuple[str, ...]]:
    parent: dict[treeswift.Node, treeswift.Node | None] = {tree.root: None}
    for node in tree.root.traverse_preorder():
        for ch in node.children:
            parent[ch] = node
    leaf_sets: dict[treeswift.Node, set[str]] = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            leaf_sets[node] = {str(node.label)}
        else:
            s: set[str] = set()
            for ch in node.children:
                s |= leaf_sets[ch]
            leaf_sets[node] = s
    splits: set[tuple[str, ...]] = set()
    n = len(taxa_set)
    for node, subset in leaf_sets.items():
        if node is tree.root:
            continue
        if len(subset) <= 1 or len(subset) >= n - 1:
            continue
        splits.add(_canonical_split(set(subset), taxa_set))
    return splits


def _prepare_gene_split_cache(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
) -> tuple[dict[int, set[tuple[str, ...]]], dict[int, set[str]]]:
    taxa_use = set(str(t) for t in taxa)
    split_cache: dict[int, set[tuple[str, ...]]] = {}
    present_cache: dict[int, set[str]] = {}
    for gt in gene_trees:
        present = set(str(t) for t in get_leaf_set(gt) if str(t) in taxa_use)
        present_cache[id(gt)] = present
        if len(present) < 4:
            split_cache[id(gt)] = set()
            continue
        induced = gt.extract_tree_with(present, suppress_unifurcations=True)
        split_cache[id(gt)] = _tree_internal_splits(induced, present)
    return split_cache, present_cache


def _split_support_score_species_tree(
    species_tree_newick: str,
    taxa: Sequence[str],
    gene_trees: Sequence[treeswift.Tree],
    *,
    gene_split_cache: dict[int, set[tuple[str, ...]]],
    gene_present_cache: dict[int, set[str]],
    gene_weights_by_id: dict[int, float] | None = None,
) -> float:
    taxa_set = set(str(t) for t in taxa)
    st = _read_tree(species_tree_newick)
    species_splits = _tree_internal_splits(st, taxa_set)
    if not species_splits:
        return 0.0
    split_scores: list[float] = []
    for split in species_splits:
        a = set(split)
        num = 0.0
        den = 0.0
        for gt in gene_trees:
            gid = id(gt)
            gw = float(gene_weights_by_id.get(gid, 1.0)) if gene_weights_by_id is not None else 1.0
            if gw <= 0.0:
                continue
            present = gene_present_cache.get(gid, set())
            if len(present) < 4:
                continue
            left = a.intersection(present)
            right = present - left
            if len(left) < 2 or len(right) < 2:
                continue
            proj = _canonical_split(set(left), set(present))
            den += gw
            if proj in gene_split_cache.get(gid, set()):
                num += gw
        if den > 0.0:
            split_scores.append(num / den)
    if not split_scores:
        return 0.0
    return float(np.mean(np.asarray(split_scores, dtype=float)))


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
    core_local_search: bool = False,
    guardrail_diagnostics: dict[str, object] | None = None,
    external_candidates: Sequence[tuple[str, str]] | None = None,
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

    robust_gene_weights = (
        _compute_robust_gene_weights(
            gene_trees=gene_trees,
            taxa=all_taxa,
            max_quintets_per_tree=effective_max_quintets,
            rng=np.random.default_rng(20260221),
        )
        if precomputed_observations is None and len(all_taxa) >= 24
        else None
    )

    observations = (
        list(precomputed_observations)
        if precomputed_observations is not None
        else extract_quintet_observations_from_gene_trees(
            gene_trees=gene_trees,
            taxa=all_taxa,
            max_quintets_per_tree=effective_max_quintets,
            rng=rng,
            gene_weights=robust_gene_weights,
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
        min_conf = 0.0
        high_conf = 0.55
        if len(all_taxa) >= 24 and not baseline_guardrail:
            # Large-taxa standalone runs are sensitive to noisy/missing gene
            # trees; tighten confidence filtering to suppress ambiguous quintets.
            min_conf = 0.03
            high_conf = 0.80
        observations = shrink_quintet_observations_by_confidence(
            observations,
            min_confidence=min_conf,
            high_confidence_passthrough=high_conf,
        )

    partition_tree, _ = recursive_partition(
        taxa=all_taxa,
        quintet_observations=observations,
        species_tree_estimate=None,
        n2_normalization=n2_normalization,
        low_signal_threshold=low_signal_threshold,
        low_signal_mode=low_signal_mode,
    )
    p2_newick = partition_tree_to_newick(partition_tree)

    candidates: list[dict[str, object]] = []
    seen_newicks: set[str] = set()

    def _append_candidate(name: str, newick: str, generation_runtime_s: float) -> None:
        nwk = str(newick)
        if nwk in seen_newicks:
            return
        seen_newicks.add(nwk)
        candidates.append({"name": str(name), "newick": nwk, "generation_runtime_s": float(generation_runtime_s)})

    _append_candidate("phase2", p2_newick, 0.0)
    t0 = time.perf_counter()
    _append_candidate(
        "nj",
        _distance_baseline_newick(gene_trees, all_taxa, method="nj"),
        float(time.perf_counter() - t0),
    )
    t0 = time.perf_counter()
    _append_candidate(
        "upgma",
        _distance_baseline_newick(gene_trees, all_taxa, method="upgma"),
        float(time.perf_counter() - t0),
    )
    if len(all_taxa) >= 24 and observations:
        # Candidate bank for standalone robustness: sweep Phase2 split policy
        # settings on the same quintet observations.
        phase2_variants = (
            ("phase2_lsig025", True, 0.25, "adaptive"),
            ("phase2_lsig035", True, 0.35, "adaptive"),
            ("phase2_lsig065", True, 0.65, "adaptive"),
            ("phase2_no_norm_lsig050", False, 0.50, "adaptive"),
            ("phase2_no_norm_lsig035", False, 0.35, "adaptive"),
        )
        for name, n2_norm, lsig_thr, lsig_mode in phase2_variants:
            t0 = time.perf_counter()
            part_var, _ = recursive_partition(
                taxa=all_taxa,
                quintet_observations=observations,
                species_tree_estimate=None,
                n2_normalization=n2_norm,
                low_signal_threshold=lsig_thr,
                low_signal_mode=lsig_mode,
            )
            _append_candidate(name, partition_tree_to_newick(part_var), float(time.perf_counter() - t0))
    if len(all_taxa) >= 24:
        # Multi-resolution candidate: larger quintet sample may stabilize hard regimes.
        if max_quintets_per_tree is not None:
            hi_budget = max(int(max_quintets_per_tree) * 2, 360)
            t0 = time.perf_counter()
            obs_hi = extract_quintet_observations_from_gene_trees(
                gene_trees=gene_trees,
                taxa=all_taxa,
                max_quintets_per_tree=hi_budget,
                rng=np.random.default_rng(20260220),
            )
            if confidence_shrink:
                obs_hi = shrink_quintet_observations_by_confidence(obs_hi)
            part_hi, _ = recursive_partition(
                taxa=all_taxa,
                quintet_observations=obs_hi,
                species_tree_estimate=None,
                n2_normalization=n2_normalization,
                low_signal_threshold=low_signal_threshold,
                low_signal_mode=low_signal_mode,
            )
            _append_candidate(
                f"phase2_hires_q{hi_budget}",
                partition_tree_to_newick(part_hi),
                float(time.perf_counter() - t0),
            )
        # Consensus-informed candidate often helps asymmetric/noisy regimes.
        t0 = time.perf_counter()
        _append_candidate(
            "consensus_majority",
            _consensus_baseline_newick(gene_trees, all_taxa),
            float(time.perf_counter() - t0),
        )
    if baseline_guardrail and external_candidates:
        for name, nwk in external_candidates:
            if not name or not nwk:
                continue
            _append_candidate(str(name), str(nwk), 0.0)
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
    gene_weights_by_id = (
        {id(gt): float(robust_gene_weights[i]) for i, gt in enumerate(gene_trees)}
        if robust_gene_weights is not None
        else None
    )
    split_support_cache_by_newick: dict[str, float] = {}
    gene_split_cache: dict[int, set[tuple[str, ...]]] | None = None
    gene_present_cache: dict[int, set[str]] | None = None
    if len(all_taxa) >= 24 and not baseline_guardrail:
        gene_split_cache, gene_present_cache = _prepare_gene_split_cache(gene_trees, all_taxa)
    has_no_missing = all(len(get_leaf_set(gt).intersection(all_taxa)) == len(all_taxa) for gt in gene_trees)
    scored: list[tuple[float, int, str, str, float, float]] = []
    for c in candidates:
        name = str(c["name"])
        nwk = str(c["newick"])
        t0 = time.perf_counter()
        score = _score_species_tree_newick_cached(
            gene_trees=gene_trees,
            species_tree_newick=nwk,
            sampled_subsets_by_gene=sampled_subsets_by_gene,
            gene_topology_cache=gene_topology_cache,
            gene_weights_by_id=gene_weights_by_id,
        )
        score_runtime = float(time.perf_counter() - t0)
        generation_runtime = float(c.get("generation_runtime_s", 0.0))
        scored.append((float(score), 1 if name == "phase2" else 0, nwk, name, generation_runtime, score_runtime))
    # Core-only local topology refinement: explore bounded NNI neighborhoods
    # from robust seed candidates with shallow hill-climbing.
    local_search_added = 0
    if core_local_search and len(all_taxa) >= 24 and not baseline_guardrail and observations:
        by_name_pre = {s[3]: s for s in scored}
        seed_list: list[tuple[float, int, str, str, float, float]] = []
        for preferred in ("nj", "upgma", "consensus_majority"):
            if preferred in by_name_pre:
                seed_list.append(by_name_pre[preferred])
        for s in sorted(scored, key=lambda x: (x[0], x[1], x[3]), reverse=True):
            if s not in seed_list:
                seed_list.append(s)
            if len(seed_list) >= 4:
                break
        scored_by_newick: dict[str, tuple[float, int, str, str, float, float]] = {s[2]: s for s in scored}
        max_neighbors = 10 if has_no_missing else 16
        max_depth = 1 if has_no_missing else 2
        for seed in seed_list:
            current_score, _pref, current_nwk, seed_name, _grt, _srt = seed
            for depth in range(max_depth):
                best_score = float(current_score)
                best_nwk = str(current_nwk)
                neighbors = _generate_nni_neighbors_newicks(current_nwk, max_neighbors=max_neighbors)
                for i, nnwk in enumerate(neighbors, start=1):
                    existing = scored_by_newick.get(nnwk)
                    if existing is not None:
                        nscore = float(existing[0])
                        if nscore > best_score + 1e-9:
                            best_score = nscore
                            best_nwk = nnwk
                        continue
                    nname = f"{seed_name}_nni_d{depth+1}_{i}"
                    t0 = time.perf_counter()
                    nscore = _score_species_tree_newick_cached(
                        gene_trees=gene_trees,
                        species_tree_newick=nnwk,
                        sampled_subsets_by_gene=sampled_subsets_by_gene,
                        gene_topology_cache=gene_topology_cache,
                        gene_weights_by_id=gene_weights_by_id,
                    )
                    nscore_runtime = float(time.perf_counter() - t0)
                    tup = (float(nscore), 0, nnwk, nname, 0.0, nscore_runtime)
                    scored.append(tup)
                    scored_by_newick[nnwk] = tup
                    local_search_added += 1
                    if float(nscore) > best_score + 1e-9:
                        best_score = float(nscore)
                        best_nwk = nnwk
                if best_nwk == current_nwk:
                    break
                current_nwk = best_nwk
                current_score = best_score
    best_score = max(s[0] for s in scored)
    top = [s for s in scored if s[0] >= best_score - 1e-12]
    def _name_priority(name: str) -> int:
        if name.startswith("phase2_hires_q"):
            return 50
        if name == "phase2":
            return 40
        if name == "consensus_majority":
            return 30
        if name == "upgma":
            return 20
        if name == "nj":
            return 10
        return 0
    # Deterministic quick-stage winner.
    winner = max(top, key=lambda s: (_name_priority(s[3]), s[3]))
    winner_trace: list[str] = [f"quick_stage:{winner[3]}"]

    split_support_by_name: dict[str, float] = {}
    split_support_runtime_by_name: dict[str, float] = {}
    if len(all_taxa) >= 24 and not baseline_guardrail and gene_split_cache is not None and gene_present_cache is not None:
        top_split = sorted(scored, key=lambda s: (s[0], _name_priority(s[3]), s[3]), reverse=True)[:8]
        for _qscore, _pref, nwk, name, _gen_rt, _q_rt in top_split:
            t0 = time.perf_counter()
            ss = _split_support_score_species_tree(
                nwk,
                all_taxa,
                gene_trees,
                gene_split_cache=gene_split_cache,
                gene_present_cache=gene_present_cache,
                gene_weights_by_id=gene_weights_by_id,
            )
            split_support_cache_by_newick[nwk] = float(ss)
            split_support_by_name[name] = float(ss)
            split_support_runtime_by_name[name] = float(time.perf_counter() - t0)
        if split_support_by_name:
            best_name = max(split_support_by_name, key=lambda n: (split_support_by_name[n], _name_priority(n), n))
            winner = next(s for s in scored if s[3] == best_name)
            winner_trace.append(f"split_support_stage:{winner[3]}")

    msc_by_name: dict[str, float] = {}
    msc_runtime_by_name: dict[str, float] = {}
    combined_by_name: dict[str, float] = {}
    # Large taxa: refine candidate selection with MSC likelihood scoring after
    # branch-length optimization on top quick-stage candidates. In standalone
    # core mode we score all internal candidates to reduce Phase2 overfitting.
    if len(all_taxa) >= 24 and observations:
        top_k = 2 if baseline_guardrail else min(len(scored), 6)
        top_quick = sorted(scored, key=lambda s: (s[0], _name_priority(s[3]), s[3]), reverse=True)[:top_k]
        for _qscore, _pref, nwk, name, _gen_rt, _q_rt in top_quick:
            t0 = time.perf_counter()
            ml_tree = optimize_branch_lengths_ml(_read_tree(nwk), observations)
            ml_newick = ml_tree.newick()
            msc_score = _score_species_tree_newick_msc_cached(
                gene_trees=gene_trees,
                species_tree_newick=ml_newick,
                sampled_subsets_by_gene=sampled_subsets_by_gene,
                gene_topology_cache=gene_topology_cache,
                gene_weights_by_id=gene_weights_by_id,
            )
            msc_by_name[name] = float(msc_score)
            msc_runtime_by_name[name] = float(time.perf_counter() - t0)
        if msc_by_name:
            # Core standalone: blend MSC with split-support to avoid objective
            # overfitting on short-branch/missing-data regimes.
            if not baseline_guardrail and split_support_by_name:
                names = list(msc_by_name.keys())
                mvals = np.asarray([float(msc_by_name[n]) for n in names], dtype=float)
                svals = np.asarray([float(split_support_by_name.get(n, 0.0)) for n in names], dtype=float)
                mmin, mmax = float(np.min(mvals)), float(np.max(mvals))
                smin, smax = float(np.min(svals)), float(np.max(svals))
                if mmax > mmin:
                    mnorm = (mvals - mmin) / (mmax - mmin)
                else:
                    mnorm = np.zeros_like(mvals)
                if smax > smin:
                    snorm = (svals - smin) / (smax - smin)
                else:
                    snorm = np.zeros_like(svals)
                # Favor MSC but retain substantial split-support influence.
                alpha = 0.65
                blend = alpha * mnorm + (1.0 - alpha) * snorm
                for i, n in enumerate(names):
                    combined_by_name[n] = float(blend[i])
                best_name = max(names, key=lambda n: (combined_by_name[n], _name_priority(n), n))
                winner_trace.append(f"combined_stage:{best_name}")
            else:
                best_name = max(msc_by_name, key=lambda n: (msc_by_name[n], _name_priority(n), n))
            winner = next(s for s in scored if s[3] == best_name)
            winner_trace.append(f"msc_stage:{winner[3]}")
    # External-candidate preference on full-taxa inputs:
    # avoid selecting consensus on near-ties when external methods are present.
    by_name = {s[3]: s for s in scored}
    external_present = [n for n in ("tree_qmc", "astral") if n in by_name]
    if baseline_guardrail and has_no_missing and winner[3] == "consensus_majority" and external_present:
        best_external = max(
            (by_name[n] for n in external_present),
            key=lambda s: (s[0], 1 if s[3] == "tree_qmc" else 0, s[3]),
        )
        # Consensus is brittle on some asymmetric/full-taxa regimes; if an
        # external candidate is close, prefer the external candidate.
        if (winner[0] - best_external[0]) <= 0.02:
            winner = best_external
            winner_trace.append(f"external_preference:{winner[3]}")
    # If ASTRAL narrowly beats TREE-QMC, prefer TREE-QMC as a conservative
    # tie-break on full-taxa inputs.
    if baseline_guardrail and has_no_missing and "astral" in by_name and "tree_qmc" in by_name:
        a = by_name["astral"]
        t = by_name["tree_qmc"]
        if winner[3] == "astral" and (a[0] - t[0]) <= 0.015:
            winner = t
            winner_trace.append("astral_to_tree_qmc_tiebreak")
    if guardrail_diagnostics is not None:
        guardrail_diagnostics.clear()
        guardrail_diagnostics["winner"] = {
            "name": winner[3],
            "score": winner[0],
            "msc_score": msc_by_name.get(winner[3]),
        }
        guardrail_diagnostics["winner_trace"] = list(winner_trace)
        guardrail_diagnostics["selection_context"] = {
            "baseline_guardrail": bool(baseline_guardrail),
            "core_local_search": bool(core_local_search),
            "core_local_search_added_candidates": int(local_search_added),
            "has_no_missing": bool(has_no_missing),
            "external_present": list(external_present),
            "n_candidates_scored": int(len(scored)),
        }
        guardrail_diagnostics["candidates"] = [
            {
                "name": name,
                "score": score,
                "split_support_score": split_support_by_name.get(name),
                "msc_score": msc_by_name.get(name),
                "combined_score": combined_by_name.get(name),
                "generation_runtime_s": gen_rt,
                "score_runtime_s": sc_rt,
                "split_support_runtime_s": split_support_runtime_by_name.get(name),
                "msc_runtime_s": msc_runtime_by_name.get(name),
                "total_runtime_s": gen_rt + sc_rt,
            }
            for score, _pref, _nwk, name, gen_rt, sc_rt in scored
        ]
        guardrail_diagnostics["n_taxa"] = len(all_taxa)
        guardrail_diagnostics["max_quintets_per_tree"] = max_quintets_per_tree
    return str(winner[2])


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
