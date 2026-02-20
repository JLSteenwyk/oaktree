#!/usr/bin/env python3
"""Generate simulated gene trees for validation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from oaktree.validation import (
    asymmetric_8_taxon_demography,
    balanced_8_taxon_demography,
    balanced_16_taxon_demography,
    short_branch_16_taxon_demography,
    simulate_gene_trees,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["balanced8", "asymmetric8", "balanced16", "shortbranch16"],
        default="balanced8",
    )
    parser.add_argument("--n-gene-trees", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="Output Newick file path.")
    args = parser.parse_args()

    if args.dataset == "balanced8":
        dem, taxa, _ = balanced_8_taxon_demography()
    elif args.dataset == "asymmetric8":
        dem, taxa, _ = asymmetric_8_taxon_demography()
    elif args.dataset == "balanced16":
        dem, taxa, _ = balanced_16_taxon_demography()
    else:
        dem, taxa, _ = short_branch_16_taxon_demography()

    trees = simulate_gene_trees(dem, taxa, n_replicates=args.n_gene_trees, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for tr in trees:
            handle.write(tr.newick().rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
