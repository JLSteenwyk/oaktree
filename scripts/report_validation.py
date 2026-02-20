#!/usr/bin/env python3
"""Generate Phase 6 validation markdown table and RF comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("phase2_rf", "Phase 2"),
    ("phase4_rf", "Phase 4"),
    ("consensus_rf", "Consensus (Maj)"),
    ("strict_consensus_rf", "Consensus (Strict)"),
    ("nj_rf", "NJ"),
    ("upgma_rf", "UPGMA"),
    ("astral_rf", "ASTRAL"),
]


def _load_results(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, list(payload.get("results", []))


def _aggregate_from_replicates(payload: dict, results: list[dict]) -> dict:
    agg = payload.get("aggregate_rf")
    if isinstance(agg, dict) and agg:
        return agg
    # Fallback for legacy single-run payloads.
    out = {}
    for row in results:
        ds = str(row.get("dataset", "NA"))
        out[ds] = {}
        for key, _ in METHODS:
            if key in row:
                v = float(row[key])
                out[ds][key] = {
                    "mean": v,
                    "std": 0.0,
                    "n": 1.0,
                    "ci95_low": v,
                    "ci95_high": v,
                }
    return out


def _active_methods(payload: dict, results: list[dict]) -> list[tuple[str, str]]:
    agg = _aggregate_from_replicates(payload, results)
    active = []
    for key, label in METHODS:
        has_any = any(key in agg[ds] for ds in agg)
        if has_any:
            active.append((key, label))
    return active


def _write_markdown(payload: dict, results: list[dict], out_path: Path, figure_path: Path | None) -> None:
    agg = _aggregate_from_replicates(payload, results)
    methods = _active_methods(payload, results)
    reps = int(payload.get("replicates", 1))
    ci_method = str(payload.get("ci_method", "single-run"))
    ci_boot = payload.get("ci_bootstrap_samples", None)
    lines = [
        f"# Validation Summary ({payload.get('n_gene_trees', 'NA')} gene trees, seed={payload.get('seed', 'NA')}, replicates={reps})",
        "",
        "Values are `mean+-std [95% CI]` across replicate seeds.",
        f"CI method: `{ci_method}`" + (f" (samples={ci_boot})" if ci_boot is not None else ""),
        "",
        "| Dataset | " + " | ".join(label for _, label in methods) + " |",
        "|---|" + "|".join(["---"] * len(methods)) + "|",
    ]
    for ds in sorted(agg):
        vals = []
        for key, _ in methods:
            st = agg[ds].get(key)
            if st is None:
                vals.append("NA")
            else:
                vals.append(
                    f"{float(st['mean']):.3f}+-{float(st['std']):.3f} "
                    f"[{max(0.0, float(st['ci95_low'])):.3f}, {float(st['ci95_high']):.3f}]"
                )
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Mean RF Across Datasets")
    for key, label in methods:
        vals = [float(agg[ds][key]["mean"]) for ds in agg if key in agg[ds]]
        if vals:
            lines.append(f"- {label}: {np.mean(vals):.3f}")
    if figure_path is not None:
        lines.append("")
        lines.append(f"Figure: `{figure_path}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(payload: dict, results: list[dict], out_path: Path) -> None:
    agg = _aggregate_from_replicates(payload, results)
    methods = _active_methods(payload, results)
    datasets = sorted(agg.keys())
    x = np.arange(len(datasets), dtype=float)
    width = 0.12

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (key, label) in enumerate(methods):
        vals = []
        err = []
        for ds in datasets:
            st = agg[ds].get(key, None)
            if st is None:
                vals.append(np.nan)
                err.append(0.0)
            else:
                m = float(st["mean"])
                lo = float(st["ci95_low"])
                hi = float(st["ci95_high"])
                vals.append(m)
                err.append(max(m - lo, hi - m))
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            vals,
            width=width,
            yerr=err,
            capsize=2,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("RF Distance (Lower is Better)")
    ax.set_title("Phase 6 Benchmark: RF Mean with 95% CI")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Benchmark JSON input (from scripts/benchmark.py).")
    parser.add_argument("--output-md", required=True, help="Output markdown summary path.")
    parser.add_argument("--output-fig", default=None, help="Optional output figure path (.png).")
    args = parser.parse_args()

    payload, results = _load_results(Path(args.input))
    fig_path = Path(args.output_fig) if args.output_fig else None
    if fig_path is not None:
        _write_figure(payload, results, fig_path)
    _write_markdown(payload, results, Path(args.output_md), fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
