"""CLI integration tests."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import treeswift

from oaktree import cli


def _read_tree(newick: str) -> treeswift.Tree:
    if hasattr(treeswift, "read_tree_newick"):
        return treeswift.read_tree_newick(newick)
    return treeswift.read_tree(io.StringIO(newick), "newick")


def _write_gene_trees(path, n: int = 40):
    tree = "(((A,B),(C,D)),((E,F),(G,H)));\n"
    path.write_text(tree * n, encoding="utf-8")


def test_cli_phase2_writes_output_file(tmp_path):
    inp = tmp_path / "genes.nwk"
    out = tmp_path / "species_out.nwk"
    _write_gene_trees(inp, n=30)

    code = cli.main(
        [
            str(inp),
            "--mode",
            "phase2",
            "--output",
            str(out),
            "--seed",
            "1",
            "--max-quintets-per-tree",
            "30",
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8").strip()
    tr = _read_tree(text)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_cli_phase4_stdout(tmp_path):
    inp = tmp_path / "genes2.nwk"
    _write_gene_trees(inp, n=20)

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(
            [
                str(inp),
                "--mode",
                "phase4",
                "--iterations",
                "2",
                "--seed",
                "2",
                "--max-quintets-per-tree",
                "25",
            ]
        )
    assert code == 0
    out = buf.getvalue().strip()
    tr = _read_tree(out)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_cli_phase2_adaptive_low_signal_mode(tmp_path):
    inp = tmp_path / "genes3.nwk"
    _write_gene_trees(inp, n=20)

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(
            [
                str(inp),
                "--mode",
                "phase2",
                "--low-signal-mode",
                "adaptive",
                "--low-signal-threshold",
                "0.2",
                "--seed",
                "3",
                "--max-quintets-per-tree",
                "25",
            ]
        )
    assert code == 0
    out = buf.getvalue().strip()
    tr = _read_tree(out)
    leaves = sorted(str(n.label) for n in tr.traverse_leaves())
    assert leaves == ["A", "B", "C", "D", "E", "F", "G", "H"]
