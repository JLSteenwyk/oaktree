#!/usr/bin/env python3
"""Profile quintet lookup-table precomputation on larger grids."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from oaktree.msc import precompute_quintet_tables


def run_profile(grid_sizes: list[int], max_tau: float, species_ids: list[int]) -> list[dict]:
    rows: list[dict] = []
    for n in grid_sizes:
        grid = np.linspace(0.0, max_tau, n)
        t0 = time.perf_counter()
        table = precompute_quintet_tables(grid, species_topology_ids=species_ids)
        dt = time.perf_counter() - t0
        probs = table["probs"]
        finite = np.isfinite(probs).sum()
        rows.append(
            {
                "grid_points": n,
                "grid_cells": int(n * n),
                "species_topologies_profiled": len(species_ids),
                "elapsed_seconds": dt,
                "table_shape": list(probs.shape),
                "n_finite_entries": int(finite),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[5, 9, 13],
        help="Grid point counts to profile.",
    )
    parser.add_argument(
        "--max-tau",
        type=float,
        default=2.0,
        help="Max branch length (min fixed at 0).",
    )
    parser.add_argument(
        "--species-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Species topology IDs to precompute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write profile results.",
    )
    args = parser.parse_args()

    rows = run_profile(args.grid_sizes, args.max_tau, args.species_ids)
    payload = {
        "grid_sizes": args.grid_sizes,
        "max_tau": args.max_tau,
        "species_ids": args.species_ids,
        "results": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
