"""OAKTREE command-line interface."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .inference import infer_species_tree_newick_phase2, infer_species_tree_newick_phase4_em
from .trees import read_gene_trees


def _parse_taxa_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    taxa = [x.strip() for x in raw.split(",") if x.strip()]
    return sorted(set(taxa)) if taxa else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oaktree",
        description="Infer a species tree from input gene trees (Newick, one tree per line).",
    )
    parser.add_argument("input", help="Path to input file of gene trees (Newick, one per line).")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output path for species tree Newick. Defaults to stdout.",
    )
    parser.add_argument(
        "--mode",
        choices=["phase2", "phase4"],
        default="phase4",
        help="Inference mode: phase2 topology-only pipeline, or phase4 EM refinement.",
    )
    parser.add_argument(
        "--taxa",
        default=None,
        help="Optional comma-separated taxa list to enforce in inference.",
    )
    parser.add_argument(
        "--max-quintets-per-tree",
        type=int,
        default=200,
        help="Max quintets sampled per gene tree during inference/scoring.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="EM iterations for --mode phase4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic stochastic steps.",
    )
    parser.add_argument(
        "--no-n2-normalization",
        action="store_true",
        help="Disable n2 normalization for artificial taxa in recursive partitioning.",
    )
    parser.add_argument(
        "--low-signal-threshold",
        type=float,
        default=0.5,
        help="Phase 2 split-strength threshold below which clades remain unresolved.",
    )
    parser.add_argument(
        "--low-signal-mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="Low-signal handling mode: fixed threshold or signal-adaptive threshold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_quintets_per_tree is not None and args.max_quintets_per_tree <= 0:
        print("error: --max-quintets-per-tree must be > 0", file=sys.stderr)
        return 2
    if args.iterations < 1:
        print("error: --iterations must be >= 1", file=sys.stderr)
        return 2
    if args.low_signal_threshold < 0:
        print("error: --low-signal-threshold must be >= 0", file=sys.stderr)
        return 2

    taxa = _parse_taxa_arg(args.taxa)

    try:
        gene_trees = read_gene_trees(args.input)
    except Exception as exc:  # pragma: no cover - error path
        print(f"error: failed reading input gene trees: {exc}", file=sys.stderr)
        return 1
    if not gene_trees:
        print("error: no gene trees loaded from input file", file=sys.stderr)
        return 1

    import numpy as np

    rng = np.random.default_rng(args.seed)
    n2_normalization = not args.no_n2_normalization

    try:
        if args.mode == "phase2":
            species_newick = infer_species_tree_newick_phase2(
                gene_trees=gene_trees,
                taxa=taxa,
                max_quintets_per_tree=args.max_quintets_per_tree,
                rng=rng,
                n2_normalization=n2_normalization,
                low_signal_threshold=args.low_signal_threshold,
                low_signal_mode=args.low_signal_mode,
            )
        else:
            species_newick = infer_species_tree_newick_phase4_em(
                gene_trees=gene_trees,
                taxa=taxa,
                n_iterations=args.iterations,
                max_quintets_per_tree=args.max_quintets_per_tree,
                rng=rng,
                n2_normalization=n2_normalization,
                low_signal_threshold=args.low_signal_threshold,
                low_signal_mode=args.low_signal_mode,
            )
    except Exception as exc:  # pragma: no cover - error path
        print(f"error: inference failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(species_newick.rstrip() + "\n")
        except Exception as exc:  # pragma: no cover - error path
            print(f"error: failed writing output: {exc}", file=sys.stderr)
            return 1
    else:
        print(species_newick.rstrip())
    return 0
