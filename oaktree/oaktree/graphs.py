"""Taxon graph construction and signed max-cut partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import networkx as nx
import numpy as np

from .trees import QuintetObservation, QuintetTopology


@dataclass(frozen=True)
class WeightedQuintetObservation:
    taxa: tuple[str, str, str, str, str]
    topology: QuintetTopology
    weight: float = 1.0


def _normalize_topology(topology: QuintetTopology) -> QuintetTopology:
    p1, p2 = tuple(sorted(tuple(sorted(p)) for p in topology))
    return (p1, p2)


def _pair_signal_from_topology(topology: QuintetTopology, x: str, y: str) -> int:
    """Return signed pair signal from a quintet topology.

    +1: pair prefers opposite sides in the next cut.
    -1: pair prefers same side in the next cut.
     0: uninformative pair (involves singleton interaction only).
    """
    p1, p2 = _normalize_topology(topology)
    if {x, y} == set(p1) or {x, y} == set(p2):
        return -1
    if (x in p1 and y in p2) or (x in p2 and y in p1):
        return 1
    return 0


def _topology_supports_bipartition(
    topology: QuintetTopology,
    side1: set[str],
    side2: set[str],
) -> bool:
    p1, p2 = _normalize_topology(topology)
    p1_side = 1 if p1[0] in side1 else 2
    p2_side = 1 if p2[0] in side1 else 2
    # Each cherry must remain intact and cherries should separate.
    p1_intact = (p1[0] in side1 and p1[1] in side1) or (p1[0] in side2 and p1[1] in side2)
    p2_intact = (p2[0] in side1 and p2[1] in side1) or (p2[0] in side2 and p2[1] in side2)
    return p1_intact and p2_intact and (p1_side != p2_side)


def quintet_bipartition_weight(
    quintet_taxa: tuple,
    gene_topology: QuintetTopology,
    species_topology: QuintetTopology,
    msc_likelihood: float,
    bipartition_side1: set,
    bipartition_side2: set,
) -> float:
    """Signed contribution of one quintet to a candidate bipartition."""
    taxa_set = set(quintet_taxa)
    if not taxa_set.issubset(set(bipartition_side1) | set(bipartition_side2)):
        raise ValueError("Quintet taxa must be contained in bipartition sides")
    if set(bipartition_side1) & set(bipartition_side2):
        raise ValueError("Bipartition sides must be disjoint")

    gene_support = _topology_supports_bipartition(
        gene_topology,
        set(bipartition_side1),
        set(bipartition_side2),
    )
    species_support = _topology_supports_bipartition(
        species_topology,
        set(bipartition_side1),
        set(bipartition_side2),
    )
    w = float(msc_likelihood)
    if gene_support and species_support:
        return w
    if gene_support and not species_support:
        return -w
    if (not gene_support) and species_support:
        return -w
    return 0.0


def build_taxon_graph(
    taxa: list[str],
    quintet_observations: Sequence[QuintetObservation | WeightedQuintetObservation],
    species_tree_estimate,
) -> nx.Graph:
    """Build signed taxon graph from quintet observations.

    Positive edge weight => taxa prefer opposite sides.
    Negative edge weight => taxa prefer same side.
    """
    del species_tree_estimate  # consumed in later phases when weighting by estimate.

    g = nx.Graph()
    for t in taxa:
        g.add_node(t)

    for obs in quintet_observations:
        obs_weight = float(getattr(obs, "weight", 1.0))
        for x, y in combinations(obs.taxa, 2):
            if x not in g.nodes or y not in g.nodes:
                continue
            signal = _pair_signal_from_topology(obs.topology, x, y)
            if signal == 0:
                continue
            current = g[x][y]["weight"] if g.has_edge(x, y) else 0.0
            g.add_edge(x, y, weight=current + float(signal) * obs_weight)
    return g


def _cut_objective(graph: nx.Graph, left: set[str], right: set[str]) -> float:
    score = 0.0
    left = set(left)
    right = set(right)
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 0.0))
        opposite = (u in left and v in right) or (u in right and v in left)
        score += w if opposite else -w
    return score


def _ensure_nonempty_partition(nodes: list[str], vector: np.ndarray) -> tuple[set[str], set[str]]:
    left = {n for n, v in zip(nodes, vector) if v >= 0}
    right = set(nodes) - left
    if left and right:
        return left, right
    order = [n for _, n in sorted(zip(vector, nodes))]
    mid = len(order) // 2
    left = set(order[:mid])
    right = set(order[mid:])
    return left, right


def _local_refinement(graph: nx.Graph, left: set[str], right: set[str]) -> tuple[set[str], set[str]]:
    best_left = set(left)
    best_right = set(right)
    improved = True
    while improved:
        improved = False
        current = _cut_objective(graph, best_left, best_right)
        for node in list(graph.nodes):
            if node in best_left and len(best_left) > 1:
                cand_left = set(best_left)
                cand_right = set(best_right)
                cand_left.remove(node)
                cand_right.add(node)
            elif node in best_right and len(best_right) > 1:
                cand_left = set(best_left)
                cand_right = set(best_right)
                cand_right.remove(node)
                cand_left.add(node)
            else:
                continue

            cand = _cut_objective(graph, cand_left, cand_right)
            if cand > current + 1e-12:
                best_left, best_right = cand_left, cand_right
                improved = True
                break
    return best_left, best_right


def spectral_max_cut(graph: nx.Graph) -> tuple[set[str], set[str]]:
    """Spectral signed max-cut approximation with local refinement."""
    nodes = sorted(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return set(nodes), set()
    if n == 2:
        return {nodes[0]}, {nodes[1]}

    idx = {node: i for i, node in enumerate(nodes)}
    w = np.zeros((n, n), dtype=float)
    for u, v, data in graph.edges(data=True):
        i, j = idx[u], idx[v]
        wij = float(data.get("weight", 0.0))
        w[i, j] = wij
        w[j, i] = wij

    eigvals, eigvecs = np.linalg.eigh(w)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    left, right = _ensure_nonempty_partition(nodes, principal)
    return _local_refinement(graph, left, right)


def _replace_taxon_in_topology(topology: QuintetTopology, old: str, new: str) -> QuintetTopology:
    p1, p2 = _normalize_topology(topology)
    q1 = tuple(sorted(new if x == old else x for x in p1))
    q2 = tuple(sorted(new if x == old else x for x in p2))
    return _normalize_topology((q1, q2))


def project_observations_for_subproblem(
    all_observations: Sequence[QuintetObservation | WeightedQuintetObservation],
    real_taxa_subset: set[str],
    artificial_taxon: str | None,
    represented_taxa: set[str] | None,
    *,
    n2_normalization: bool = True,
) -> list[WeightedQuintetObservation]:
    """Project quintet observations to a recursive subproblem.

    Rules:
    - keep observations fully inside the real subset (weight unchanged)
    - when exactly 4 of 5 taxa are inside and the outside taxon belongs to the
      represented complement, replace outside taxon with artificial taxon
    - optionally apply n2 normalization by scaling artificial-observation
      weights by 1 / |represented_taxa|
    """
    out: list[WeightedQuintetObservation] = []
    subset = set(real_taxa_subset)
    rep = set(represented_taxa or set())
    for obs in all_observations:
        base_weight = float(getattr(obs, "weight", 1.0))
        in_subset = [t for t in obs.taxa if t in subset]
        out_subset = [t for t in obs.taxa if t not in subset]

        if len(in_subset) == 5:
            out.append(
                WeightedQuintetObservation(
                    taxa=tuple(sorted(obs.taxa)),
                    topology=_normalize_topology(obs.topology),
                    weight=base_weight,
                )
            )
            continue

        if artificial_taxon is None or len(in_subset) != 4 or len(out_subset) != 1:
            continue

        outside = out_subset[0]
        if rep and outside not in rep:
            continue
        new_taxa = tuple(sorted(tuple(in_subset) + (artificial_taxon,)))
        new_topology = _replace_taxon_in_topology(obs.topology, outside, artificial_taxon)
        scale = 1.0
        if n2_normalization:
            denom = max(1, len(rep))
            scale = 1.0 / float(denom)
        out.append(
            WeightedQuintetObservation(
                taxa=new_taxa,
                topology=new_topology,
                weight=base_weight * scale,
            )
        )
    return out


def _leaf_node(taxa: set[str]) -> dict[str, Any]:
    return {
        "taxa": tuple(sorted(taxa)),
        "left": None,
        "right": None,
        "artificial_for_left": None,
        "artificial_for_right": None,
        "unresolved": True,
    }


def _partition_strength(graph: nx.Graph, left: set[str], right: set[str]) -> float:
    total_abs = 0.0
    for _, _, data in graph.edges(data=True):
        total_abs += abs(float(data.get("weight", 0.0)))
    if total_abs <= 0.0:
        return 0.0
    score = abs(_cut_objective(graph, left, right))
    return float(score / total_abs)


def _edge_coherence(
    taxa: Sequence[str],
    observations: Sequence[QuintetObservation | WeightedQuintetObservation],
) -> float:
    """Mean pairwise sign coherence in [0, 1] over informative pairs."""
    net: dict[tuple[str, str], float] = {}
    abs_sum: dict[tuple[str, str], float] = {}
    taxon_set = set(taxa)
    for obs in observations:
        w0 = float(getattr(obs, "weight", 1.0))
        for x, y in combinations(obs.taxa, 2):
            if x not in taxon_set or y not in taxon_set:
                continue
            sig = _pair_signal_from_topology(obs.topology, x, y)
            if sig == 0:
                continue
            k = (x, y) if x <= y else (y, x)
            net[k] = net.get(k, 0.0) + float(sig) * w0
            abs_sum[k] = abs_sum.get(k, 0.0) + abs(float(sig) * w0)
    if not abs_sum:
        return 0.0
    vals = [abs(net[k]) / abs_sum[k] for k in abs_sum if abs_sum[k] > 0.0]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _adaptive_low_signal_threshold(
    taxa: Sequence[str],
    observations: Sequence[QuintetObservation | WeightedQuintetObservation],
    *,
    min_threshold: float = 0.0,
    max_threshold: float = 0.5,
) -> float:
    """Signal-aware threshold: low for coherent graphs, high for noisy/conflicted graphs."""
    c = _edge_coherence(taxa, observations)  # 1.0 clean, 0.0 conflicted
    t = float(min_threshold) + (1.0 - float(c)) * (float(max_threshold) - float(min_threshold))
    return float(np.clip(t, min_threshold, max_threshold))


def recursive_partition(
    taxa: Sequence[str],
    quintet_observations: Sequence[QuintetObservation | WeightedQuintetObservation],
    species_tree_estimate,
    *,
    n2_normalization: bool = True,
    low_signal_threshold: float = 0.0,
    low_signal_mode: str = "fixed",
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    """Recursively bipartition taxa with artificial-taxon bookkeeping.

    Returns:
    - partition tree as nested dicts
    - artificial taxon map: artificial label -> represented real taxa
    """
    del species_tree_estimate

    artificial_map: dict[str, tuple[str, ...]] = {}
    counter = 0

    def next_artificial(represented: set[str]) -> str:
        nonlocal counter
        label = f"__ARTIFICIAL_{counter}"
        counter += 1
        artificial_map[label] = tuple(sorted(represented))
        return label

    def recurse(
        real_taxa: set[str],
        obs_here: Sequence[QuintetObservation | WeightedQuintetObservation],
    ) -> dict[str, Any]:
        if len(real_taxa) <= 3:
            return _leaf_node(real_taxa)

        graph = build_taxon_graph(sorted(real_taxa), obs_here, species_tree_estimate=None)
        left, right = spectral_max_cut(graph)
        left &= set(real_taxa)
        right &= set(real_taxa)

        if not left or not right:
            items = sorted(real_taxa)
            mid = len(items) // 2
            left = set(items[:mid])
            right = set(items[mid:])
        if not left or not right:
            return _leaf_node(real_taxa)

        if low_signal_mode == "fixed":
            threshold_here = float(low_signal_threshold)
        elif low_signal_mode == "adaptive":
            max_t = float(low_signal_threshold) if float(low_signal_threshold) > 0.0 else 0.5
            threshold_here = _adaptive_low_signal_threshold(
                sorted(real_taxa),
                obs_here,
                min_threshold=0.0,
                max_threshold=max_t,
            )
        else:
            raise ValueError("low_signal_mode must be one of {'fixed', 'adaptive'}")

        strength = _partition_strength(graph, left, right)
        if strength < threshold_here:
            return _leaf_node(real_taxa)

        art_left = next_artificial(set(right))
        art_right = next_artificial(set(left))

        obs_left = project_observations_for_subproblem(
            obs_here,
            real_taxa_subset=set(left),
            artificial_taxon=art_left,
            represented_taxa=set(right),
            n2_normalization=n2_normalization,
        )
        obs_right = project_observations_for_subproblem(
            obs_here,
            real_taxa_subset=set(right),
            artificial_taxon=art_right,
            represented_taxa=set(left),
            n2_normalization=n2_normalization,
        )

        return {
            "taxa": tuple(sorted(real_taxa)),
            "left": recurse(set(left), obs_left),
            "right": recurse(set(right), obs_right),
            "artificial_for_left": art_left,
            "artificial_for_right": art_right,
            "unresolved": False,
        }

    root = recurse(set(taxa), list(quintet_observations))
    return root, artificial_map


def partition_tree_to_newick(partition_tree: dict[str, Any]) -> str:
    """Convert recursive_partition output to a deterministic Newick string."""

    def leaf_newick(taxa: tuple[str, ...]) -> str:
        labels = sorted(taxa)
        if len(labels) == 0:
            raise ValueError("Leaf with no taxa")
        if len(labels) == 1:
            return labels[0]
        # Represent unresolved groups as a polytomy.
        return "(" + ",".join(labels) + ")"

    def rec(node: dict[str, Any]) -> str:
        left = node.get("left")
        right = node.get("right")
        taxa = tuple(node.get("taxa", ()))
        if left is None or right is None:
            return leaf_newick(taxa)
        return f"({rec(left)},{rec(right)})"

    return rec(partition_tree) + ";"
