#!/usr/bin/env python3
"""Profile end-to-end OAKTREE inference runtime on simulated datasets."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np

from oaktree.inference import infer_species_tree_newick_phase2, infer_species_tree_newick_phase4_em
from oaktree.validation import (
    asymmetric_8_taxon_demography,
    balanced_8_taxon_demography,
    balanced_16_taxon_demography,
    short_branch_8_taxon_demography,
    short_branch_16_taxon_demography,
    simulate_gene_trees,
)


DATASETS = {
    "balanced8": balanced_8_taxon_demography,
    "asymmetric8": asymmetric_8_taxon_demography,
    "shortbranch8": short_branch_8_taxon_demography,
    "balanced16": balanced_16_taxon_demography,
    "shortbranch16": short_branch_16_taxon_demography,
}


def _func_rows(stats: pstats.Stats, *, top_n: int, oaktree_only: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (filename, lineno, funcname), (cc, nc, tt, ct, _callers) in stats.stats.items():
        path = str(filename)
        if oaktree_only and "oaktree" not in path:
            continue
        rows.append(
            {
                "function": f"{Path(path).name}:{lineno}:{funcname}",
                "file": path,
                "line": int(lineno),
                "ccalls": int(cc),
                "ncalls": int(nc),
                "self_seconds": float(tt),
                "cum_seconds": float(ct),
            }
        )
    rows.sort(key=lambda r: r["cum_seconds"], reverse=True)
    return rows[: max(0, int(top_n))]


def run_profile(
    *,
    dataset: str,
    n_gene_trees: int,
    seed: int,
    mode: str,
    iterations: int,
    max_quintets_per_tree: int,
    low_signal_threshold: float,
    low_signal_mode: str,
    baseline_guardrail: bool,
    higher_order_sizes: list[int],
    higher_order_subsets_per_tree: int,
    higher_order_quintets_per_subset: int,
    higher_order_weight: float,
    top_n: int,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")

    dem, taxa, true_newick = DATASETS[dataset]()

    t0 = time.perf_counter()
    gene_trees = simulate_gene_trees(dem, taxa, n_replicates=n_gene_trees, seed=seed)
    sim_seconds = time.perf_counter() - t0

    rng = np.random.default_rng(seed + 101)
    profiler = cProfile.Profile()
    profiler.enable()
    t1 = time.perf_counter()
    if mode == "phase2":
        inferred_newick = infer_species_tree_newick_phase2(
            gene_trees=gene_trees,
            taxa=taxa,
            max_quintets_per_tree=max_quintets_per_tree,
            rng=rng,
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
        )
    else:
        inferred_newick = infer_species_tree_newick_phase4_em(
            gene_trees=gene_trees,
            taxa=taxa,
            n_iterations=iterations,
            max_quintets_per_tree=max_quintets_per_tree,
            rng=rng,
            low_signal_threshold=low_signal_threshold,
            low_signal_mode=low_signal_mode,
            baseline_guardrail=baseline_guardrail,
            higher_order_subset_sizes=higher_order_sizes,
            higher_order_subsets_per_tree=higher_order_subsets_per_tree,
            higher_order_quintets_per_subset=higher_order_quintets_per_subset,
            higher_order_weight=higher_order_weight,
        )
    infer_seconds = time.perf_counter() - t1
    profiler.disable()

    stats = pstats.Stats(profiler)
    top_oaktree = _func_rows(stats, top_n=top_n, oaktree_only=True)
    top_overall = _func_rows(stats, top_n=top_n, oaktree_only=False)

    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumtime").print_stats(top_n)

    payload: dict[str, Any] = {
        "dataset": dataset,
        "mode": mode,
        "n_gene_trees": int(n_gene_trees),
        "seed": int(seed),
        "iterations": int(iterations),
        "max_quintets_per_tree": int(max_quintets_per_tree),
        "low_signal_threshold": float(low_signal_threshold),
        "low_signal_mode": str(low_signal_mode),
        "baseline_guardrail": bool(baseline_guardrail),
        "higher_order_sizes": list(higher_order_sizes),
        "higher_order_subsets_per_tree": int(higher_order_subsets_per_tree),
        "higher_order_quintets_per_subset": int(higher_order_quintets_per_subset),
        "higher_order_weight": float(higher_order_weight),
        "timing_seconds": {
            "simulate_gene_trees": float(sim_seconds),
            "inference": float(infer_seconds),
            "total": float(sim_seconds + infer_seconds),
        },
        "inferred_newick": inferred_newick,
        "true_newick": true_newick,
        "top_functions_oaktree_cumtime": top_oaktree,
        "top_functions_overall_cumtime": top_overall,
        "raw_pstats_top_cumtime": s.getvalue(),
    }
    return payload


def _print_human_summary(payload: dict[str, Any], *, top_n: int) -> None:
    t = payload["timing_seconds"]
    print(
        f"Dataset={payload['dataset']} mode={payload['mode']} "
        f"genes={payload['n_gene_trees']} iterations={payload['iterations']}"
    )
    print(
        f"Timing (s): simulate={t['simulate_gene_trees']:.3f} "
        f"infer={t['inference']:.3f} total={t['total']:.3f}"
    )
    print(f"Top {top_n} OAKTREE hotspots by cumulative seconds:")
    for row in payload["top_functions_oaktree_cumtime"]:
        print(
            f"  {row['cum_seconds']:8.3f}s  {row['self_seconds']:8.3f}s self  "
            f"{row['ncalls']:7d} calls  {row['function']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="shortbranch8")
    parser.add_argument("--n-gene-trees", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["phase2", "phase4"], default="phase4")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--max-quintets-per-tree", type=int, default=80)
    parser.add_argument("--low-signal-threshold", type=float, default=0.2)
    parser.add_argument("--low-signal-mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument(
        "--guardrail-mode",
        choices=["core", "guardrailed"],
        default="guardrailed",
        help="Whether to enable distance-baseline guardrail candidate selection.",
    )
    parser.add_argument("--higher-order-sizes", default="")
    parser.add_argument("--higher-order-subsets-per-tree", type=int, default=0)
    parser.add_argument("--higher-order-quintets-per-subset", type=int, default=0)
    parser.add_argument("--higher-order-weight", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    args = parser.parse_args()

    if args.n_gene_trees < 1:
        raise ValueError("--n-gene-trees must be >= 1")
    if args.seed < 1:
        raise ValueError("--seed must be >= 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.max_quintets_per_tree < 1:
        raise ValueError("--max-quintets-per-tree must be >= 1")

    higher_sizes = [int(x.strip()) for x in str(args.higher_order_sizes).split(",") if x.strip()]
    payload = run_profile(
        dataset=args.dataset,
        n_gene_trees=args.n_gene_trees,
        seed=args.seed,
        mode=args.mode,
        iterations=args.iterations,
        max_quintets_per_tree=args.max_quintets_per_tree,
        low_signal_threshold=args.low_signal_threshold,
        low_signal_mode=args.low_signal_mode,
        baseline_guardrail=(args.guardrail_mode == "guardrailed"),
        higher_order_sizes=higher_sizes,
        higher_order_subsets_per_tree=args.higher_order_subsets_per_tree,
        higher_order_quintets_per_subset=args.higher_order_quintets_per_subset,
        higher_order_weight=args.higher_order_weight,
        top_n=args.top_n,
    )

    _print_human_summary(payload, top_n=args.top_n)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"Dataset={payload['dataset']} mode={payload['mode']} genes={payload['n_gene_trees']} iterations={payload['iterations']}",
            (
                f"Timing (s): simulate={payload['timing_seconds']['simulate_gene_trees']:.3f} "
                f"infer={payload['timing_seconds']['inference']:.3f} "
                f"total={payload['timing_seconds']['total']:.3f}"
            ),
            "",
            f"Top {args.top_n} OAKTREE hotspots by cumulative seconds:",
        ]
        for row in payload["top_functions_oaktree_cumtime"]:
            lines.append(
                f"{row['cum_seconds']:8.3f}s  {row['self_seconds']:8.3f}s self  {row['ncalls']:7d} calls  {row['function']}"
            )
        args.output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
