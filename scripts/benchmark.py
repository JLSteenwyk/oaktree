#!/usr/bin/env python3
"""Run baseline validation benchmark and emit JSON summary."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from oaktree.validation import (
    run_baseline_benchmark,
    run_expanded_benchmark,
    run_scaled16_quick_benchmark,
    run_scaled64_complex_benchmark,
    summarize_rf_replicates,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gene-trees", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of replicate runs with incremented seeds.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Seed increment between replicate runs.",
    )
    parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for replicate mean CI estimates.",
    )
    parser.add_argument(
        "--low-signal-threshold",
        type=float,
        default=0.5,
        help="Phase 2 low-signal split threshold.",
    )
    parser.add_argument(
        "--low-signal-mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="Low-signal handling mode for Phase 2 recursion.",
    )
    parser.add_argument(
        "--guardrail-mode",
        choices=["guardrailed", "core", "both"],
        default="guardrailed",
        help="Use baseline guardrail in Phase 2/4 (`core` disables it, `both` reports both).",
    )
    parser.add_argument(
        "--astral-jar",
        default=None,
        help="Optional path to ASTRAL jar for external baseline comparison.",
    )
    parser.add_argument(
        "--astral-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each ASTRAL invocation.",
    )
    parser.add_argument(
        "--tree-qmc-bin",
        default=None,
        help="Optional TREE-QMC executable path (or command name if on PATH).",
    )
    parser.add_argument(
        "--tree-qmc-timeout-seconds",
        type=int,
        default=180,
        help="Timeout for each TREE-QMC invocation.",
    )
    parser.add_argument(
        "--higher-order-sizes",
        default="",
        help="Comma-separated higher-order subset sizes to inject (e.g. 6,7,8).",
    )
    parser.add_argument(
        "--higher-order-subsets-per-tree",
        type=int,
        default=0,
        help="Sampled higher-order subsets per gene tree.",
    )
    parser.add_argument(
        "--higher-order-quintets-per-subset",
        type=int,
        default=0,
        help="Projected quintets sampled from each higher-order subset.",
    )
    parser.add_argument(
        "--higher-order-weight",
        type=float,
        default=1.0,
        help="Base weight multiplier for projected higher-order signals.",
    )
    parser.add_argument("--expanded", action="store_true", help="Run expanded multi-regime benchmark suite.")
    parser.add_argument(
        "--scaled16-quick",
        action="store_true",
        help="Run quick larger-size suite (16 taxa datasets).",
    )
    parser.add_argument(
        "--scaled64-complex",
        action="store_true",
        help=(
            "Run 64-taxon complex suite "
            "(balanced64, asymmetric64, shortbranch64, balanced64_missing, shortbranch64_missing_noisy)."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    higher_sizes = [int(x.strip()) for x in str(args.higher_order_sizes).split(",") if x.strip()]

    replicate_runs = []
    for i in range(args.replicates):
        run_seed = args.seed + i * args.seed_step
        mode_count = int(bool(args.expanded)) + int(bool(args.scaled16_quick)) + int(bool(args.scaled64_complex))
        if mode_count > 1:
            raise ValueError("--expanded, --scaled16-quick, and --scaled64-complex are mutually exclusive")
        modes = (
            [("guardrailed", True), ("core", False)]
            if args.guardrail_mode == "both"
            else [(args.guardrail_mode, args.guardrail_mode == "guardrailed")]
        )
        merged_results = []
        for mode_name, use_guardrail in modes:
            if args.scaled64_complex:
                results = run_scaled64_complex_benchmark(
                    n_gene_trees=args.n_gene_trees,
                    seed=run_seed,
                    low_signal_threshold=args.low_signal_threshold,
                    low_signal_mode=args.low_signal_mode,
                    baseline_guardrail=use_guardrail,
                    higher_order_subset_sizes=higher_sizes,
                    higher_order_subsets_per_tree=args.higher_order_subsets_per_tree,
                    higher_order_quintets_per_subset=args.higher_order_quintets_per_subset,
                    higher_order_weight=args.higher_order_weight,
                    astral_jar_path=args.astral_jar,
                    astral_timeout_seconds=args.astral_timeout_seconds,
                    tree_qmc_bin=args.tree_qmc_bin,
                    tree_qmc_timeout_seconds=args.tree_qmc_timeout_seconds,
                )
            elif args.scaled16_quick:
                results = run_scaled16_quick_benchmark(
                    n_gene_trees=args.n_gene_trees,
                    seed=run_seed,
                    low_signal_threshold=args.low_signal_threshold,
                    low_signal_mode=args.low_signal_mode,
                    baseline_guardrail=use_guardrail,
                    higher_order_subset_sizes=higher_sizes,
                    higher_order_subsets_per_tree=args.higher_order_subsets_per_tree,
                    higher_order_quintets_per_subset=args.higher_order_quintets_per_subset,
                    higher_order_weight=args.higher_order_weight,
                    astral_jar_path=args.astral_jar,
                    astral_timeout_seconds=args.astral_timeout_seconds,
                    tree_qmc_bin=args.tree_qmc_bin,
                    tree_qmc_timeout_seconds=args.tree_qmc_timeout_seconds,
                )
            elif args.expanded:
                results = run_expanded_benchmark(
                    n_gene_trees=args.n_gene_trees,
                    seed=run_seed,
                    low_signal_threshold=args.low_signal_threshold,
                    low_signal_mode=args.low_signal_mode,
                    baseline_guardrail=use_guardrail,
                    higher_order_subset_sizes=higher_sizes,
                    higher_order_subsets_per_tree=args.higher_order_subsets_per_tree,
                    higher_order_quintets_per_subset=args.higher_order_quintets_per_subset,
                    higher_order_weight=args.higher_order_weight,
                    astral_jar_path=args.astral_jar,
                    astral_timeout_seconds=args.astral_timeout_seconds,
                    tree_qmc_bin=args.tree_qmc_bin,
                    tree_qmc_timeout_seconds=args.tree_qmc_timeout_seconds,
                )
            else:
                results = run_baseline_benchmark(
                    n_gene_trees=args.n_gene_trees,
                    seed=run_seed,
                    low_signal_threshold=args.low_signal_threshold,
                    low_signal_mode=args.low_signal_mode,
                    baseline_guardrail=use_guardrail,
                    higher_order_subset_sizes=higher_sizes,
                    higher_order_subsets_per_tree=args.higher_order_subsets_per_tree,
                    higher_order_quintets_per_subset=args.higher_order_quintets_per_subset,
                    higher_order_weight=args.higher_order_weight,
                    astral_jar_path=args.astral_jar,
                    astral_timeout_seconds=args.astral_timeout_seconds,
                    tree_qmc_bin=args.tree_qmc_bin,
                    tree_qmc_timeout_seconds=args.tree_qmc_timeout_seconds,
                )
            if len(modes) > 1:
                results = [replace(r, dataset=f"{mode_name}:{r.dataset}") for r in results]
            merged_results.extend(results)
        replicate_runs.append((run_seed, merged_results))

    primary_results = replicate_runs[0][1]
    aggregate = summarize_rf_replicates(
        [r for _, r in replicate_runs],
        n_bootstrap=args.ci_bootstrap_samples,
        bootstrap_seed=args.seed + 9000,
    )
    payload = {
        "n_gene_trees": args.n_gene_trees,
        "seed": args.seed,
        "replicates": int(args.replicates),
        "seed_step": int(args.seed_step),
        "ci_method": "bootstrap_percentile",
        "ci_bootstrap_samples": int(args.ci_bootstrap_samples),
        "expanded": bool(args.expanded),
        "scaled16_quick": bool(args.scaled16_quick),
        "scaled64_complex": bool(args.scaled64_complex),
        "low_signal_threshold": float(args.low_signal_threshold),
        "low_signal_mode": str(args.low_signal_mode),
        "guardrail_mode": str(args.guardrail_mode),
        "astral_jar": args.astral_jar,
        "astral_timeout_seconds": int(args.astral_timeout_seconds),
        "tree_qmc_bin": args.tree_qmc_bin,
        "tree_qmc_timeout_seconds": int(args.tree_qmc_timeout_seconds),
        "higher_order_sizes": higher_sizes,
        "higher_order_subsets_per_tree": int(args.higher_order_subsets_per_tree),
        "higher_order_quintets_per_subset": int(args.higher_order_quintets_per_subset),
        "higher_order_weight": float(args.higher_order_weight),
        "results": [asdict(r) for r in primary_results],
        "replicate_results": [
            {"seed": s, "results": [asdict(r) for r in rs]} for s, rs in replicate_runs
        ],
        "aggregate_rf": aggregate,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
