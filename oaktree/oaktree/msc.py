"""MSC likelihood computation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import treeswift
from scipy.linalg import expm

from .trees import (
    QuintetTopology,
    canonicalize_quintet,
    enumerate_quintet_topologies,
    extract_induced_subtree,
)


def _rising_factorial(x: int, n: int) -> float:
    if n <= 0:
        return 1.0
    out = 1.0
    for i in range(n):
        out *= float(x + i)
    return out


def _falling_factorial(x: int, n: int) -> float:
    if n <= 0:
        return 1.0
    out = 1.0
    for i in range(n):
        out *= float(x - i)
    return out


def coalescent_probability(k_in: int, k_out: int, tau: float) -> float:
    """Exact P(k_in -> k_out in time tau) for the standard coalescent.

    Uses the closed-form transition probability (Tavare 1984 / classic
    Kingman coalescent result):

    P_{n,m}(t) = sum_{i=m..n} exp(-i(i-1)t/2)
                 * (2i-1) * (-1)^(i-m)
                 * m^(i-1) * n_[i] / (m! (i-m)! n^(i))

    where m^(r) is a rising factorial and n_[r] is a falling factorial.
    """
    if k_in < 1:
        raise ValueError("k_in must be >= 1")
    if k_out < 1 or k_out > k_in:
        raise ValueError("k_out must satisfy 1 <= k_out <= k_in")
    if tau < 0:
        raise ValueError("tau must be >= 0")

    if tau == 0.0:
        return 1.0 if k_in == k_out else 0.0

    n = k_in
    m = k_out

    total = 0.0
    m_fact = math.factorial(m)
    for i in range(m, n + 1):
        sign = -1.0 if ((i - m) % 2) else 1.0
        exp_term = math.exp(-0.5 * i * (i - 1) * tau)
        numer = (2 * i - 1) * sign
        numer *= _rising_factorial(m, i - 1)
        numer *= _falling_factorial(n, i)
        denom = m_fact * math.factorial(i - m) * _rising_factorial(n, i)
        total += exp_term * (numer / denom)

    # Guard against tiny numerical over/undershoots.
    if total < 0.0 and abs(total) < 1e-15:
        total = 0.0
    if total > 1.0 and abs(total - 1.0) < 1e-15:
        total = 1.0
    return float(total)


@dataclass(frozen=True)
class CoalescentHistory:
    """A topological assignment of gene-tree coalescences to species nodes."""

    # (event_name, species_node_id)
    assignments: Tuple[Tuple[str, int], ...]


def _node_metadata(tree: treeswift.Tree) -> tuple[
    Dict[treeswift.Node, treeswift.Node | None],
    Dict[str, treeswift.Node],
    Dict[treeswift.Node, int],
]:
    parent: Dict[treeswift.Node, treeswift.Node | None] = {tree.root: None}
    leaves: Dict[str, treeswift.Node] = {}
    node_ids: Dict[treeswift.Node, int] = {}
    idx = 0
    for node in tree.root.traverse_preorder():
        node_ids[node] = idx
        idx += 1
        for child in node.children:
            parent[child] = node
        if node.is_leaf() and node.label is not None:
            leaves[str(node.label)] = node
    return parent, leaves, node_ids


def _ancestor_chain(
    node: treeswift.Node,
    parent: Dict[treeswift.Node, treeswift.Node | None],
) -> List[treeswift.Node]:
    chain: List[treeswift.Node] = []
    cur: treeswift.Node | None = node
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    return chain


def _lca(
    a: treeswift.Node,
    b: treeswift.Node,
    parent: Dict[treeswift.Node, treeswift.Node | None],
) -> treeswift.Node:
    ancestors_a = set(_ancestor_chain(a, parent))
    cur: treeswift.Node | None = b
    while cur is not None:
        if cur in ancestors_a:
            return cur
        cur = parent[cur]
    raise RuntimeError("No LCA found")


def _is_ancestor(
    anc: treeswift.Node,
    node: treeswift.Node,
    parent: Dict[treeswift.Node, treeswift.Node | None],
) -> bool:
    cur: treeswift.Node | None = node
    while cur is not None:
        if cur is anc:
            return True
        cur = parent[cur]
    return False


def _all_ancestors_including_self(
    node: treeswift.Node,
    parent: Dict[treeswift.Node, treeswift.Node | None],
) -> List[treeswift.Node]:
    return _ancestor_chain(node, parent)


def _quintet_events_from_topology(
    topology: QuintetTopology,
    taxa: Sequence[str],
) -> Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...]:
    # Quintet topology is represented as two disjoint cherries.
    pair1, pair2 = tuple(sorted(tuple(sorted(p)) for p in topology))
    s1 = set(pair1)
    s2 = set(pair2)
    singleton = tuple(sorted(set(taxa) - s1 - s2))
    if len(singleton) != 1:
        raise ValueError("Invalid quintet topology encoding")
    c1 = tuple(sorted(pair1))
    c2 = tuple(sorted(pair2))
    s = singleton
    # Rooted event partial order consistent with the unrooted quintet:
    # e1/e2 happen first, then e3 merges one cherry with singleton,
    # then e4 merges both remaining lineages.
    return (
        ("e1", c1[:1], c1[1:]),
        ("e2", c2[:1], c2[1:]),
        ("e3", c1, s),
        ("e4", tuple(sorted(set(c2) | set(s))), c1),
    )


def _all_topologies_for_taxa(
    taxa: Tuple[str, str, str, str, str],
) -> List[QuintetTopology]:
    pairs = [tuple(sorted(p)) for p in combinations(taxa, 2)]
    out = set()
    for i, p1 in enumerate(pairs):
        s1 = set(p1)
        for p2 in pairs[i + 1 :]:
            if s1.isdisjoint(p2):
                out.add(tuple(sorted((p1, tuple(sorted(p2))))))
    return sorted(out)


def _lineage_key(lineage: object) -> str:
    if isinstance(lineage, str):
        return lineage
    left, right = lineage
    return f"({_lineage_key(left)},{_lineage_key(right)})"


def _rooted_signature(lineage: object) -> str:
    if isinstance(lineage, str):
        return lineage
    left, right = lineage
    children = sorted((_rooted_signature(left), _rooted_signature(right)))
    return f"({children[0]},{children[1]})"


def _merge_lineages(a: object, b: object) -> tuple[object, object]:
    ka = _lineage_key(a)
    kb = _lineage_key(b)
    if ka <= kb:
        return (a, b)
    return (b, a)


def _canonical_forest(lineages: Sequence[object]) -> tuple[object, ...]:
    return tuple(sorted(lineages, key=_lineage_key))


def _forest_children_states(forest: tuple[object, ...]) -> List[tuple[tuple[object, ...], float]]:
    k = len(forest)
    if k <= 1:
        return []
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            merged = _merge_lineages(forest[i], forest[j])
            next_lineages = [forest[t] for t in range(k) if t not in (i, j)]
            next_lineages.append(merged)
            out.append((_canonical_forest(next_lineages), 1.0))
    return out


def _reachable_forests(start_forests: Sequence[tuple[object, ...]]) -> List[tuple[object, ...]]:
    seen = set(start_forests)
    queue = list(start_forests)
    while queue:
        cur = queue.pop()
        for nxt, _ in _forest_children_states(cur):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return sorted(seen, key=lambda s: (len(s), tuple(_lineage_key(x) for x in s)))


def _branch_transition(
    start_dist: Dict[tuple[object, ...], float],
    tau: float,
) -> Dict[tuple[object, ...], float]:
    if not start_dist:
        return {}
    if tau <= 0:
        return dict(start_dist)

    states = _reachable_forests(list(start_dist.keys()))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    q = [[0.0 for _ in range(n)] for _ in range(n)]

    for s in states:
        i = idx[s]
        children = _forest_children_states(s)
        out_rate = 0.0
        for nxt, rate in children:
            j = idx[nxt]
            q[i][j] += rate
            out_rate += rate
        q[i][i] = -out_rate

    p = expm([[q[r][c] * tau for c in range(n)] for r in range(n)])
    out = {s: 0.0 for s in states}
    for s, prob in start_dist.items():
        i = idx[s]
        if prob == 0.0:
            continue
        for j, st in enumerate(states):
            out[st] += prob * float(p[i][j])
    return {k: v for k, v in out.items() if v > 0.0}


@lru_cache(maxsize=4096)
def _complete_to_root(forest: tuple[object, ...]) -> Dict[object, float]:
    if len(forest) == 1:
        return {forest[0]: 1.0}
    k = len(forest)
    denom = float(k * (k - 1) // 2)
    out: Dict[object, float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            merged = _merge_lineages(forest[i], forest[j])
            next_lineages = [forest[t] for t in range(k) if t not in (i, j)]
            next_lineages.append(merged)
            nxt = _canonical_forest(next_lineages)
            subtree = _complete_to_root(nxt)
            for tree, p in subtree.items():
                out[tree] = out.get(tree, 0.0) + p / denom
    return out


def _collect_internal_clades(lineage: object, out: List[frozenset[str]]) -> frozenset[str]:
    if isinstance(lineage, str):
        return frozenset([lineage])
    left, right = lineage
    lset = _collect_internal_clades(left, out)
    rset = _collect_internal_clades(right, out)
    here = lset | rset
    out.append(here)
    return here


def _rooted_to_quintet_topology(
    rooted_tree: object,
    taxa: Tuple[str, str, str, str, str],
) -> QuintetTopology:
    clades: List[frozenset[str]] = []
    full = _collect_internal_clades(rooted_tree, clades)
    if set(full) != set(taxa):
        raise ValueError("Rooted lineage does not match taxa set")
    top_pairs = set()
    all_taxa = set(taxa)
    for clade in clades:
        if len(clade) == 5:
            continue
        if len(clade) == 2:
            top_pairs.add(tuple(sorted(clade)))
        elif len(clade) == 3:
            top_pairs.add(tuple(sorted(all_taxa - set(clade))))
    if len(top_pairs) != 2:
        raise ValueError("Failed to map rooted lineage to quintet topology")
    pair_list = sorted(top_pairs)
    return (pair_list[0], pair_list[1])


def _node_bottom_distribution(node: treeswift.Node) -> Dict[tuple[object, ...], float]:
    if node.is_leaf():
        if node.label is None:
            raise ValueError("Leaf without label")
        return {_canonical_forest((str(node.label),)): 1.0}

    child_dists = [_node_top_distribution(ch) for ch in node.children]
    out: Dict[tuple[object, ...], float] = {}
    cur = {tuple(): 1.0}
    for dist in child_dists:
        nxt: Dict[tuple[object, ...], float] = {}
        for forest_a, pa in cur.items():
            for forest_b, pb in dist.items():
                merged = _canonical_forest(tuple(forest_a) + tuple(forest_b))
                nxt[merged] = nxt.get(merged, 0.0) + pa * pb
        cur = nxt
    out.update(cur)
    return out


def _node_top_distribution(node: treeswift.Node) -> Dict[tuple[object, ...], float]:
    bottom = _node_bottom_distribution(node)
    tau = getattr(node, "edge_length", None)
    tau_val = 0.0 if tau is None else max(0.0, float(tau))
    return _branch_transition(bottom, tau_val)


@lru_cache(maxsize=2048)
def _rooted_distribution_for_newick(induced_newick: str) -> Dict[object, float]:
    induced = treeswift.read_tree_newick(induced_newick)
    root_bottom = _node_bottom_distribution(induced.root)
    rooted_dist: Dict[object, float] = {}
    for forest, p_forest in root_bottom.items():
        completion = _complete_to_root(forest)
        for rooted, p_rooted in completion.items():
            rooted_dist[rooted] = rooted_dist.get(rooted, 0.0) + p_forest * p_rooted
    total = sum(rooted_dist.values())
    if total <= 0.0:
        return {}
    return {tree: prob / total for tree, prob in rooted_dist.items()}


@lru_cache(maxsize=2048)
def _quintet_probabilities_for_newick(induced_newick: str) -> Dict[QuintetTopology, float]:
    induced = treeswift.read_tree_newick(induced_newick)
    taxa = tuple(sorted(str(leaf.label) for leaf in induced.traverse_leaves()))
    if len(taxa) != 5:
        raise ValueError("Expected 5 taxa in induced tree")

    rooted_dist = _rooted_distribution_for_newick(induced_newick)

    topo_dist: Dict[QuintetTopology, float] = {topo: 0.0 for topo in _all_topologies_for_taxa(taxa)}
    for rooted, p in rooted_dist.items():
        topo = _rooted_to_quintet_topology(rooted, taxa)
        topo_dist[topo] = topo_dist.get(topo, 0.0) + p

    total = sum(topo_dist.values())
    if total <= 0.0:
        return {topo: 1.0 / 15.0 for topo in _all_topologies_for_taxa(taxa)}
    return {topo: prob / total for topo, prob in topo_dist.items()}


def rooted_gene_tree_distribution(
    species_tree: treeswift.Tree,
    taxa: Sequence[str],
) -> Dict[str, float]:
    """Exact rooted gene-tree topology distribution for a given taxon subset."""
    taxa_sorted = tuple(sorted(str(t) for t in taxa))
    if len(taxa_sorted) < 2:
        raise ValueError("Need at least two taxa")
    induced = extract_induced_subtree(species_tree, taxa_sorted)
    induced.resolve_polytomies()
    induced.suppress_unifurcations()
    rooted_dist = _rooted_distribution_for_newick(induced.newick())
    sig_dist: Dict[str, float] = {}
    for rooted, p in rooted_dist.items():
        sig = _rooted_signature(rooted)
        sig_dist[sig] = sig_dist.get(sig, 0.0) + p
    total = sum(sig_dist.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sig_dist.items()}


def quartet_anomaly_zone_threshold(older_internal_branch: float) -> float:
    """Return anomaly-zone threshold for 4-taxon caterpillar species trees.

    For rooted species tree topology (((A,B),C),D) with internal branches:
    - older branch length x (between (ABC) and D),
    - younger branch length y (between (AB) and C),
    anomalous gene trees occur when y < a(x), where:

        a(x) = log(2/3 + (3 e^{2x} - 2) / (18 (e^{3x} - e^{2x})))

    (Degnan & Rosenberg 2006 style threshold expression).
    """
    x = float(older_internal_branch)
    if x <= 0.0:
        raise ValueError("older_internal_branch must be > 0")
    num = (3.0 * math.exp(2.0 * x)) - 2.0
    den = 18.0 * (math.exp(3.0 * x) - math.exp(2.0 * x))
    return math.log((2.0 / 3.0) + (num / den))


def enumerate_coalescent_histories(
    species_tree: treeswift.Tree,
    gene_tree_topology: QuintetTopology,
    taxa: Tuple[str, str, str, str, str],
) -> List[CoalescentHistory]:
    """Enumerate topology-consistent coalescent-history assignments on quintets.

    This is an initial topological enumerator used for scaffolding Phase 1:
    each gene-tree coalescence event is mapped to an ancestral species node
    while preserving ancestry constraints.
    """
    taxa_sorted = tuple(sorted(taxa))
    induced = extract_induced_subtree(species_tree, taxa_sorted)
    induced.resolve_polytomies()
    induced.suppress_unifurcations()

    parent, leaves, node_ids = _node_metadata(induced)
    events = _quintet_events_from_topology(gene_tree_topology, taxa_sorted)

    event_candidates: Dict[str, List[treeswift.Node]] = {}
    for event_name, left_desc, right_desc in events:
        left_node = leaves[left_desc[0]]
        right_node = leaves[right_desc[0]]
        # For multi-leaf descriptors, collapse them through pairwise LCAs.
        for lbl in left_desc[1:]:
            left_node = _lca(left_node, leaves[lbl], parent)
        for lbl in right_desc[1:]:
            right_node = _lca(right_node, leaves[lbl], parent)
        lca_node = _lca(left_node, right_node, parent)
        event_candidates[event_name] = _all_ancestors_including_self(lca_node, parent)

    # Partial order on events.
    predecessors = {"e1": set(), "e2": set(), "e3": {"e1"}, "e4": {"e2", "e3"}}
    event_order = ["e1", "e2", "e3", "e4"]

    histories: List[CoalescentHistory] = []
    for choices in product(*(event_candidates[e] for e in event_order)):
        assign = {e: n for e, n in zip(event_order, choices)}
        valid = True
        for e in event_order:
            for p in predecessors[e]:
                # parent event must occur at or above predecessor event
                if not _is_ancestor(assign[e], assign[p], parent):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        mapped = tuple((e, node_ids[assign[e]]) for e in event_order)
        histories.append(CoalescentHistory(assignments=mapped))
    return histories


def _internal_tau_total(induced: treeswift.Tree) -> float:
    total = 0.0
    for node in induced.root.traverse_preorder():
        if node is induced.root:
            continue
        if node.is_leaf():
            continue
        length = getattr(node, "edge_length", None)
        if length is not None:
            total += max(0.0, float(length))
    return total


def _history_weight(
    induced: treeswift.Tree,
    history: CoalescentHistory,
) -> float:
    node_by_id = {}
    for i, node in enumerate(induced.root.traverse_preorder()):
        node_by_id[i] = node

    event_count_by_node_id: Dict[int, int] = {}
    for _, node_id in history.assignments:
        event_count_by_node_id[node_id] = event_count_by_node_id.get(node_id, 0) + 1

    k_out: Dict[treeswift.Node, int] = {}
    total_prob = 1.0

    for node in induced.root.traverse_postorder():
        if node.is_leaf():
            k_in = 1
        else:
            k_in = sum(k_out[ch] for ch in node.children)

        node_id = None
        for nid, n in node_by_id.items():
            if n is node:
                node_id = nid
                break
        if node_id is None:
            return 0.0
        c = event_count_by_node_id.get(node_id, 0)

        if c < 0 or c > (k_in - 1):
            return 0.0
        k_leave = k_in - c
        if k_leave < 1:
            return 0.0
        k_out[node] = k_leave

        if node is induced.root:
            # Above-root branch is effectively infinite; valid histories must
            # end with a single lineage.
            if k_leave != 1:
                return 0.0
            continue

        tau = getattr(node, "edge_length", None)
        tau_val = 0.0 if tau is None else max(0.0, float(tau))
        p = coalescent_probability(k_in, k_leave, tau_val)
        total_prob *= p

    return total_prob


def quintet_probability(
    species_tree: treeswift.Tree,
    gene_tree_topology: QuintetTopology,
    taxa: Tuple[str, str, str, str, str],
) -> float:
    """Return MSC quintet topology probability for one-sample-per-species case."""
    taxa_sorted = tuple(sorted(taxa))
    induced = extract_induced_subtree(species_tree, taxa_sorted)
    induced.resolve_polytomies()
    induced.suppress_unifurcations()

    query_topology = tuple(sorted(tuple(sorted(p)) for p in gene_tree_topology))
    if len(query_topology) != 2:
        raise ValueError("Invalid quintet topology")
    probs = _quintet_probabilities_for_newick(induced.newick())
    return probs.get(query_topology, 0.0)


def _species_tree_for_lookup(
    species_topology: QuintetTopology,
    tau1: float,
    tau2: float,
) -> treeswift.Tree:
    pair1, pair2 = tuple(sorted(tuple(sorted(p)) for p in species_topology))
    singleton = sorted(set(("a", "b", "c", "d", "e")) - set(pair1) - set(pair2))
    if len(singleton) != 1:
        raise ValueError("Invalid species topology for lookup table")
    s = singleton[0]
    tau1 = max(0.0, float(tau1))
    tau2 = max(0.0, float(tau2))
    # Rooted template where the two quintet internal branches are tau1/tau2.
    newick = (
        f"(({pair1[0]}:0,{pair1[1]}:0):{tau1},"
        f"(({pair2[0]}:0,{pair2[1]}:0):{tau2},{s}:0):0);"
    )
    return treeswift.read_tree_newick(newick)


def precompute_quintet_tables(
    branch_length_grid: np.ndarray,
    species_topology_ids: Sequence[int] | None = None,
) -> dict:
    """Precompute quintet probability table on a 2D branch-length grid.

    The table stores P(gene_topology | species_topology, tau1, tau2), where
    tau1 and tau2 are the two internal branches in a canonical rooted template
    for a quintet species topology.
    """
    grid = np.asarray(branch_length_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("branch_length_grid must be a 1D array with at least 2 points")
    if np.any(grid < 0):
        raise ValueError("branch_length_grid values must be >= 0")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("branch_length_grid must be strictly increasing")

    topologies = enumerate_quintet_topologies()
    n_topo = len(topologies)
    if n_topo != 15:
        raise ValueError("Expected exactly 15 quintet topologies")

    if species_topology_ids is None:
        species_ids = list(range(n_topo))
    else:
        species_ids = sorted(set(int(i) for i in species_topology_ids))
        if not species_ids:
            raise ValueError("species_topology_ids cannot be empty")
        for sid in species_ids:
            if sid < 0 or sid >= n_topo:
                raise ValueError(f"species_topology_id out of range: {sid}")

    probs = np.full((n_topo, n_topo, len(grid), len(grid)), np.nan, dtype=float)
    taxa = ("a", "b", "c", "d", "e")

    for sid in species_ids:
        species_topology = topologies[sid]
        for i, tau1 in enumerate(grid):
            for j, tau2 in enumerate(grid):
                sp_tree = _species_tree_for_lookup(species_topology, float(tau1), float(tau2))
                dist = _quintet_probabilities_for_newick(sp_tree.newick())
                for gid, gene_topology in enumerate(topologies):
                    probs[sid, gid, i, j] = float(dist.get(gene_topology, 0.0))
                # Guarantee per-point normalization even after numerical noise.
                col = probs[sid, :, i, j]
                s = float(np.nansum(col))
                if s > 0:
                    probs[sid, :, i, j] = col / s

    return {
        "grid": grid,
        "topologies": topologies,
        "probs": probs,
        "species_topology_ids": species_ids,
    }


def lookup_quintet_probability(
    species_topology_id: int,
    gene_topology_id: int,
    branch_lengths: tuple[float, float],
    table: dict,
) -> float:
    """Lookup interpolated quintet probability from precomputed table."""
    grid = np.asarray(table["grid"], dtype=float)
    probs = np.asarray(table["probs"], dtype=float)
    sid = int(species_topology_id)
    gid = int(gene_topology_id)
    if sid < 0 or sid >= probs.shape[0]:
        raise ValueError("species_topology_id out of range")
    if gid < 0 or gid >= probs.shape[1]:
        raise ValueError("gene_topology_id out of range")

    tau1 = max(float(branch_lengths[0]), float(grid[0]))
    tau2 = max(float(branch_lengths[1]), float(grid[0]))
    tau1 = min(tau1, float(grid[-1]))
    tau2 = min(tau2, float(grid[-1]))

    i_hi = int(np.searchsorted(grid, tau1, side="right"))
    j_hi = int(np.searchsorted(grid, tau2, side="right"))
    i_lo = max(0, i_hi - 1)
    j_lo = max(0, j_hi - 1)
    i_hi = min(i_hi, len(grid) - 1)
    j_hi = min(j_hi, len(grid) - 1)

    x0, x1 = float(grid[i_lo]), float(grid[i_hi])
    y0, y1 = float(grid[j_lo]), float(grid[j_hi])
    q00 = float(probs[sid, gid, i_lo, j_lo])
    q01 = float(probs[sid, gid, i_lo, j_hi])
    q10 = float(probs[sid, gid, i_hi, j_lo])
    q11 = float(probs[sid, gid, i_hi, j_hi])

    if math.isnan(q00) or math.isnan(q01) or math.isnan(q10) or math.isnan(q11):
        raise ValueError("Requested species_topology_id was not precomputed in table")

    if x1 == x0 and y1 == y0:
        return q00
    if x1 == x0:
        wy = 0.0 if y1 == y0 else (tau2 - y0) / (y1 - y0)
        return (1.0 - wy) * q00 + wy * q01
    if y1 == y0:
        wx = (tau1 - x0) / (x1 - x0)
        return (1.0 - wx) * q00 + wx * q10

    wx = (tau1 - x0) / (x1 - x0)
    wy = (tau2 - y0) / (y1 - y0)
    return (
        (1.0 - wx) * (1.0 - wy) * q00
        + (1.0 - wx) * wy * q01
        + wx * (1.0 - wy) * q10
        + wx * wy * q11
    )
