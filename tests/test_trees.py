"""Phase 0 tests for tree I/O and quintet handling."""

from __future__ import annotations

import numpy as np
import treeswift

from oaktree.trees import (
    canonicalize_quintet,
    enumerate_quintet_topologies,
    extract_induced_subtree,
    get_leaf_set,
    get_shared_taxa,
    read_gene_trees,
    sample_quintets,
)


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    # Fallback for older API
    import io

    return treeswift.read_tree(io.StringIO(newick), "newick")


def test_read_gene_trees(tmp_path):
    src = tmp_path / "trees.nwk"
    src.write_text("(A,B,C);\n(D,E,F);\n", encoding="utf-8")
    trees = read_gene_trees(str(src))
    assert len(trees) == 2
    assert get_leaf_set(trees[0]) == {"A", "B", "C"}
    assert get_leaf_set(trees[1]) == {"D", "E", "F"}


def test_get_shared_taxa():
    t1 = _read_tree("(A,B,(C,D));")
    t2 = _read_tree("((B,C),D);")
    assert get_shared_taxa([t1, t2]) == {"B", "C", "D"}


def test_enumerate_quintet_topologies():
    topologies = enumerate_quintet_topologies()
    assert len(topologies) == 15
    assert len(set(topologies)) == 15


def test_canonicalize_quintet_permutations():
    tree = _read_tree("((((A,B),(C,D)),((E,F),(G,H))),(I,J));")
    taxa1 = ("A", "B", "C", "D", "E")
    taxa2 = ("E", "D", "C", "B", "A")
    topo1 = canonicalize_quintet(tree, taxa1)
    topo2 = canonicalize_quintet(tree, taxa2)
    assert topo1 == topo2


def test_extract_induced_subtree():
    tree = _read_tree("((((A,B),(C,D)),((E,F),(G,H))),(I,J));")
    cases = [
        (("A", "B", "C", "D", "E"), (("A", "B"), ("C", "D"))),
        (("A", "B", "E", "F", "G"), (("A", "B"), ("E", "F"))),
        (("A", "C", "E", "G", "I"), (("A", "C"), ("E", "G"))),
        (("B", "D", "F", "H", "J"), (("B", "D"), ("F", "H"))),
        (("A", "B", "I", "J", "E"), (("A", "B"), ("I", "J"))),
    ]
    for taxa, expected in cases:
        induced = extract_induced_subtree(tree, taxa)
        topo = canonicalize_quintet(induced, taxa)
        assert topo == expected


def test_sample_reproducibility():
    tree = _read_tree("((((A,B),(C,D)),((E,F),(G,H))),(I,J));")
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    obs1 = sample_quintets(tree, n_samples=5, rng=rng1)
    obs2 = sample_quintets(tree, n_samples=5, rng=rng2)
    assert [(o.taxa, o.topology) for o in obs1] == [(o.taxa, o.topology) for o in obs2]


def test_sample_quintets_too_few_taxa():
    tree = _read_tree("(A,(B,C));")
    rng = np.random.default_rng(0)
    assert sample_quintets(tree, n_samples=5, rng=rng) == []


def test_sample_quintets_skips_missing_taxa_with_full_set():
    tree = _read_tree("((A,B),(C,D),E);")
    rng = np.random.default_rng(1)
    full_taxa = ["A", "B", "C", "D", "E", "F", "G"]
    obs = sample_quintets(tree, n_samples=20, rng=rng, full_taxa=full_taxa)
    assert len(obs) == 20
    for o in obs:
        assert set(o.taxa).issubset({"A", "B", "C", "D", "E"})


def test_extract_induced_subtree_matches_dendropy():
    try:
        import dendropy
    except ImportError:
        return

    tree = _read_tree("((((A,B),(C,D)),((E,F),(G,H))),(I,J));")
    taxa = ("A", "B", "C", "D", "E")
    induced = extract_induced_subtree(tree, taxa)

    # DendroPy comparison
    dp_tree = dendropy.Tree.get(data="((((A,B),(C,D)),((E,F),(G,H))),(I,J));", schema="newick")
    dp_tree.is_rooted = False
    dp_tree.retain_taxa_with_labels(set(taxa))
    dp_tree.suppress_unifurcations()
    dp_topo = canonicalize_quintet(_read_tree(dp_tree.as_string(schema="newick")), taxa)

    assert canonicalize_quintet(induced, taxa) == dp_topo


def _comb_newick(taxa: list[str]) -> str:
    tree = taxa[0]
    for label in taxa[1:]:
        tree = f"({tree},{label})"
    return tree + ";"


def test_extract_induced_subtree_random_dendropy_agreement():
    try:
        import dendropy
    except ImportError:
        return

    taxa = [f"T{i}" for i in range(1, 51)]
    newick = _comb_newick(taxa)
    tree = _read_tree(newick)

    rng = np.random.default_rng(7)
    for _ in range(20):
        subset = tuple(sorted(rng.choice(taxa, size=5, replace=False)))
        induced = extract_induced_subtree(tree, subset)

        dp_tree = dendropy.Tree.get(data=newick, schema="newick")
        dp_tree.is_rooted = False
        dp_tree.retain_taxa_with_labels(set(subset))
        dp_tree.suppress_unifurcations()
        dp_topo = canonicalize_quintet(_read_tree(dp_tree.as_string(schema="newick")), subset)

        assert canonicalize_quintet(induced, subset) == dp_topo
