"""Gene-tree weighting and Phase 4 EM-loop entry points."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from typing import Sequence

import numpy as np
import treeswift

from .branch_lengths import optimize_branch_lengths_ml
from .inference import infer_species_tree_newick_phase2
from .msc import _quintet_probabilities_for_newick, quintet_probability
from .trees import canonicalize_quintet, extract_induced_subtree, get_leaf_set, sample_quintet_subsets


@dataclass(frozen=True)
class EMIterationResult:
    iteration: int
    species_tree_newick: str
    mean_weight: float
    rf_distance: int | None = None
    branch_length_delta: float | None = None


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def _canonical_split(split: set[str], all_taxa: set[str]) -> tuple[str, ...]:
    other = all_taxa - split
    side = split if len(split) <= len(other) else other
    return tuple(sorted(side))


def _splits_with_lengths(tree: treeswift.Tree, taxa: set[str]) -> dict[tuple[str, ...], float]:
    leaf_sets: dict[treeswift.Node, set[str]] = {}
    for node in tree.root.traverse_postorder():
        if node.is_leaf():
            leaf_sets[node] = {str(node.label)}
        else:
            s = set()
            for ch in node.children:
                s |= leaf_sets[ch]
            leaf_sets[node] = s

    out: dict[tuple[str, ...], float] = {}
    for node, subset in leaf_sets.items():
        if node is tree.root:
            continue
        if len(subset) <= 1 or len(subset) >= len(taxa) - 1:
            continue
        key = _canonical_split(set(subset), taxa)
        out[key] = float(node.edge_length or 0.0)
    return out


def _rf_distance(newick_a: str, newick_b: str, taxa: Sequence[str]) -> int:
    ta = _read_tree(newick_a)
    tb = _read_tree(newick_b)
    taxa_set = set(taxa)
    sa = set(_splits_with_lengths(ta, taxa_set).keys())
    sb = set(_splits_with_lengths(tb, taxa_set).keys())
    return len(sa.symmetric_difference(sb))


def _branch_length_delta(newick_a: str, newick_b: str, taxa: Sequence[str]) -> float:
    ta = _read_tree(newick_a)
    tb = _read_tree(newick_b)
    taxa_set = set(taxa)
    la = _splits_with_lengths(ta, taxa_set)
    lb = _splits_with_lengths(tb, taxa_set)
    keys = set(la) | set(lb)
    if not keys:
        return 0.0
    diffs = [abs(float(la.get(k, 0.0)) - float(lb.get(k, 0.0))) for k in keys]
    return float(np.mean(diffs))


def _shared_quintet_subsets(
    gene_tree: treeswift.Tree,
    species_tree: treeswift.Tree,
    taxa: Sequence[str],
) -> list[tuple[str, str, str, str, str]]:
    leaf_set = get_leaf_set(gene_tree) & get_leaf_set(species_tree)
    target = [t for t in sorted(set(taxa)) if t in leaf_set]
    if len(target) < 5:
        return []
    return sample_quintet_subsets(target, max_quintets=None, rng=np.random.default_rng(0))


def score_gene_tree_against_species_tree(
    gene_tree: treeswift.Tree,
    species_tree_newick: str,
    taxa: Sequence[str],
    *,
    max_quintets: int | None = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean per-quintet MSC log-likelihood score."""
    rng = rng if rng is not None else np.random.default_rng(0)
    species_tree = _read_tree(species_tree_newick)
    has_branch_lengths = any(
        (node is not species_tree.root)
        and (not node.is_leaf())
        and (float(node.edge_length or 0.0) > 0.0)
        for node in species_tree.root.traverse_preorder()
    )
    subsets = _shared_quintet_subsets(gene_tree, species_tree, taxa)
    if not subsets:
        return -float("inf")
    if max_quintets is not None and len(subsets) > max_quintets:
        idx = rng.choice(len(subsets), size=max_quintets, replace=False)
        subsets = [subsets[int(i)] for i in sorted(idx)]
    total = 0.0
    match = 0
    for q in subsets:
        gt = canonicalize_quintet(gene_tree, q)
        if not has_branch_lengths:
            st = canonicalize_quintet(species_tree, q)
            if tuple(sorted(gt)) == tuple(sorted(st)):
                match += 1
            continue
        p = max(quintet_probability(species_tree, gt, q), 1e-12)
        total += log(p)
    if not has_branch_lengths:
        return float(match) / float(len(subsets))
    return float(total) / float(len(subsets))


def _prepare_sampled_quintets_by_gene(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str],
    *,
    max_quintets_per_tree: int | None,
    rng: np.random.Generator,
) -> dict[int, list[tuple[str, str, str, str, str]]]:
    """Sample quintet subsets once per gene tree for reuse across EM scoring."""
    out: dict[int, list[tuple[str, str, str, str, str]]] = {}
    taxa_sorted = sorted(set(taxa))
    for gt in gene_trees:
        leaf_set = get_leaf_set(gt)
        target = [t for t in taxa_sorted if t in leaf_set]
        if len(target) < 5:
            out[id(gt)] = []
            continue
        subsets = sample_quintet_subsets(
            target,
            max_quintets=max_quintets_per_tree,
            rng=rng,
        )
        out[id(gt)] = subsets
    return out


def _score_gene_tree_against_species_tree_cached(
    gene_tree: treeswift.Tree,
    species_tree: treeswift.Tree,
    taxa: Sequence[str],
    *,
    sampled_subsets_by_gene: dict[int, list[tuple[str, str, str, str, str]]],
    has_branch_lengths: bool,
    species_topology_cache: dict[tuple[str, str, str, str, str], tuple[tuple[str, str], tuple[str, str]]],
    species_prob_cache: dict[tuple[str, str, str, str, str], dict[tuple[tuple[str, str], tuple[str, str]], float]],
    gene_topology_cache: dict[tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]],
) -> float:
    """Fast path scoring with reusable quintet subset/topology/probability caches."""
    subsets = sampled_subsets_by_gene.get(id(gene_tree), [])
    if not subsets:
        return -float("inf")
    total = 0.0
    match = 0
    gid = id(gene_tree)
    for q in subsets:
        gk = (gid, q)
        gt = gene_topology_cache.get(gk)
        if gt is None:
            gt = canonicalize_quintet(gene_tree, q)
            gene_topology_cache[gk] = gt
        if not has_branch_lengths:
            st = species_topology_cache.get(q)
            if st is None:
                st = canonicalize_quintet(species_tree, q)
                species_topology_cache[q] = st
            if tuple(sorted(gt)) == tuple(sorted(st)):
                match += 1
            continue

        probs = species_prob_cache.get(q)
        if probs is None:
            induced = extract_induced_subtree(species_tree, q)
            induced.resolve_polytomies()
            induced.suppress_unifurcations()
            probs = _quintet_probabilities_for_newick(induced.newick())
            species_prob_cache[q] = probs
        p = max(float(probs.get(tuple(sorted(gt)), 0.0)), 1e-12)
        total += log(p)
    if not has_branch_lengths:
        return float(match) / float(len(subsets))
    return float(total) / float(len(subsets))


def compute_gene_tree_weights(
    gene_trees: Sequence[treeswift.Tree],
    species_tree_newick: str,
    taxa: Sequence[str],
    *,
    temperature: float = 1.0,
    max_quintets_per_tree: int | None = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute softmax-normalized per-gene weights from consistency scores.

    Larger `temperature` increases contrast (inverse-temperature/beta behavior).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    rng = rng if rng is not None else np.random.default_rng(0)
    species_tree = _read_tree(species_tree_newick)
    has_branch_lengths = any(
        (node is not species_tree.root)
        and (not node.is_leaf())
        and (float(node.edge_length or 0.0) > 0.0)
        for node in species_tree.root.traverse_preorder()
    )
    sampled_subsets_by_gene = _prepare_sampled_quintets_by_gene(
        gene_trees,
        taxa,
        max_quintets_per_tree=max_quintets_per_tree,
        rng=rng,
    )
    species_topology_cache: dict[tuple[str, str, str, str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    species_prob_cache: dict[
        tuple[str, str, str, str, str], dict[tuple[tuple[str, str], tuple[str, str]], float]
    ] = {}
    gene_topology_cache: dict[
        tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]
    ] = {}
    scores = np.array(
        [
            _score_gene_tree_against_species_tree_cached(
                gt,
                species_tree,
                taxa,
                sampled_subsets_by_gene=sampled_subsets_by_gene,
                has_branch_lengths=has_branch_lengths,
                species_topology_cache=species_topology_cache,
                species_prob_cache=species_prob_cache,
                gene_topology_cache=gene_topology_cache,
            )
            for gt in gene_trees
        ],
        dtype=float,
    )
    if len(scores) == 0:
        return np.array([], dtype=float)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.full(len(scores), 1.0 / len(scores), dtype=float)
    min_finite = float(np.min(scores[finite]))
    scores = np.where(finite, scores, min_finite - 100.0)
    logits = scores * float(temperature)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    s = float(np.sum(w))
    if s <= 0:
        return np.full(len(scores), 1.0 / len(scores), dtype=float)
    return w / s


def _mean_consistency_score(
    gene_trees: Sequence[treeswift.Tree],
    species_tree_newick: str,
    taxa: Sequence[str],
    *,
    max_quintets_per_tree: int | None,
) -> float:
    if not gene_trees:
        return -float("inf")
    # Fixed RNG seed keeps accept/reject decisions reproducible.
    rng = np.random.default_rng(20260219)
    species_tree = _read_tree(species_tree_newick)
    has_branch_lengths = any(
        (node is not species_tree.root)
        and (not node.is_leaf())
        and (float(node.edge_length or 0.0) > 0.0)
        for node in species_tree.root.traverse_preorder()
    )
    sampled_subsets_by_gene = _prepare_sampled_quintets_by_gene(
        gene_trees,
        taxa,
        max_quintets_per_tree=max_quintets_per_tree,
        rng=rng,
    )
    species_topology_cache: dict[tuple[str, str, str, str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    species_prob_cache: dict[
        tuple[str, str, str, str, str], dict[tuple[tuple[str, str], tuple[str, str]], float]
    ] = {}
    gene_topology_cache: dict[
        tuple[int, tuple[str, str, str, str, str]], tuple[tuple[str, str], tuple[str, str]]
    ] = {}
    vals = [
        _score_gene_tree_against_species_tree_cached(
            gt,
            species_tree,
            taxa,
            sampled_subsets_by_gene=sampled_subsets_by_gene,
            has_branch_lengths=has_branch_lengths,
            species_topology_cache=species_topology_cache,
            species_prob_cache=species_prob_cache,
            gene_topology_cache=gene_topology_cache,
        )
        for gt in gene_trees
    ]
    return float(np.mean(vals))


def em_refine_species_tree_newick(
    gene_trees: Sequence[treeswift.Tree],
    taxa: Sequence[str] | None = None,
    *,
    n_iterations: int = 3,
    max_quintets_per_tree: int | None = 200,
    rng: np.random.Generator | None = None,
    n2_normalization: bool = True,
    rf_tolerance: int = 0,
    branch_length_tolerance: float = 1e-3,
    min_iterations_before_stop: int = 1,
    low_signal_threshold: float = 0.5,
    low_signal_mode: str = "adaptive",
    baseline_guardrail: bool = True,
    higher_order_subset_sizes: Sequence[int] | None = None,
    higher_order_subsets_per_tree: int = 0,
    higher_order_quintets_per_subset: int = 0,
    higher_order_weight: float = 1.0,
) -> tuple[str, list[EMIterationResult]]:
    """Phase 4 EM entry: initialize from Phase 2, then reweight/reinfer."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")
    if rf_tolerance < 0:
        raise ValueError("rf_tolerance must be >= 0")
    if branch_length_tolerance < 0:
        raise ValueError("branch_length_tolerance must be >= 0")
    if min_iterations_before_stop < 0:
        raise ValueError("min_iterations_before_stop must be >= 0")
    rng = rng if rng is not None else np.random.default_rng(0)

    if taxa is None:
        taxa_set = set()
        for gt in gene_trees:
            taxa_set |= get_leaf_set(gt)
        taxa_use = sorted(taxa_set)
    else:
        taxa_use = sorted(set(taxa))

    current = infer_species_tree_newick_phase2(
        gene_trees,
        taxa=taxa_use,
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
    # Phase 4: refine branch lengths for the initialized topology.
    from .inference import extract_quintet_observations_from_gene_trees

    init_obs = extract_quintet_observations_from_gene_trees(
        gene_trees=gene_trees,
        taxa=taxa_use,
        max_quintets_per_tree=max_quintets_per_tree,
        rng=rng,
    )
    current_tree = optimize_branch_lengths_ml(_read_tree(current), init_obs)
    current = current_tree.newick()
    current_obj = _mean_consistency_score(
        gene_trees,
        current,
        taxa_use,
        max_quintets_per_tree=max_quintets_per_tree,
    )
    history: list[EMIterationResult] = [
        EMIterationResult(iteration=0, species_tree_newick=current, mean_weight=1.0, rf_distance=None, branch_length_delta=None)
    ]

    # Large-taxa guardrailed runs are more sensitive to noisy topology flips.
    # Require a modest objective gain before accepting topology-changing updates.
    min_topology_change_gain = 0.0
    if baseline_guardrail and len(taxa_use) >= 24:
        min_topology_change_gain = 0.01
    freeze_topology_updates = bool(baseline_guardrail and len(taxa_use) >= 24)

    for it in range(1, n_iterations + 1):
        # Increase weighting contrast over iterations to progressively
        # emphasize consistent genes while keeping iteration-1 conservative.
        beta = 1.0 + 0.75 * it
        weights = compute_gene_tree_weights(
            gene_trees,
            current,
            taxa_use,
            temperature=beta,
            max_quintets_per_tree=max_quintets_per_tree,
            rng=rng,
        )
        if len(weights) == 0:
            break
        reweighted = list(gene_trees)
        weighted_obs = extract_quintet_observations_from_gene_trees(
            gene_trees=gene_trees,
            taxa=taxa_use,
            max_quintets_per_tree=max_quintets_per_tree,
            rng=rng,
            gene_weights=weights,
        )
        # Reduce split conservativeness slightly through EM rounds to help
        # recover weak-but-consistent internal structure in short-branch data.
        iter_threshold = max(0.2, float(low_signal_threshold) * (1.0 - 0.15 * it))
        if freeze_topology_updates:
            updated = current
        else:
            updated = infer_species_tree_newick_phase2(
                reweighted,
                taxa=taxa_use,
                max_quintets_per_tree=max_quintets_per_tree,
                rng=rng,
                n2_normalization=n2_normalization,
                low_signal_threshold=iter_threshold,
                low_signal_mode=low_signal_mode,
                baseline_guardrail=baseline_guardrail,
                precomputed_observations=weighted_obs,
            )
        updated_obs = weighted_obs
        updated_tree = optimize_branch_lengths_ml(_read_tree(updated), updated_obs)
        updated = updated_tree.newick()
        updated_obj = _mean_consistency_score(
            gene_trees,
            updated,
            taxa_use,
            max_quintets_per_tree=max_quintets_per_tree,
        )
        proposed_rf = _rf_distance(current, updated, taxa_use)
        obj_gain = float(updated_obj - current_obj)
        should_accept = updated_obj + 1e-12 >= current_obj
        if should_accept and proposed_rf > 0 and obj_gain < min_topology_change_gain:
            should_accept = False

        if should_accept:
            accepted = updated
            current_obj = updated_obj
            rf = proposed_rf
        else:
            accepted = current
            rf = 0

        bl_delta = _branch_length_delta(current, accepted, taxa_use)
        history.append(
            EMIterationResult(
                iteration=it,
                species_tree_newick=accepted,
                mean_weight=float(np.mean(weights)),
                rf_distance=rf,
                branch_length_delta=bl_delta,
            )
        )
        current = accepted
        if it >= min_iterations_before_stop and rf <= rf_tolerance and bl_delta <= branch_length_tolerance:
            break

    return current, history
