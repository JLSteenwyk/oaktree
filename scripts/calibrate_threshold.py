#!/usr/bin/env python3
"""Calibrate Phase 2 low-signal threshold across benchmark regimes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from oaktree.validation import calibrate_low_signal_threshold


def _best_thresholds(rows: list[dict]) -> dict[str, float]:
    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(str(r["dataset"]), []).append(r)
    best: dict[str, float] = {}
    for ds, vals in by_dataset.items():
        vals = sorted(vals, key=lambda x: (float(x["mean_rf"]), float(x["threshold"])))
        best[ds] = float(vals[0]["threshold"])
    return best


def _write_markdown(payload: dict, out_path: Path) -> None:
    rows = list(payload["rows"])
    lines = [
        "# Low-Signal Threshold Calibration",
        "",
        f"- n_gene_trees: `{payload['n_gene_trees']}`",
        f"- seed: `{payload['seed']}`",
        f"- n_replicates: `{payload['n_replicates']}`",
        "",
        "| Threshold | Dataset | Mean RF | Std RF | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (float(x["threshold"]), str(x["dataset"]))):
        lines.append(
            f"| {float(r['threshold']):.3f} | {r['dataset']} | {float(r['mean_rf']):.3f} | "
            f"{float(r['std_rf']):.3f} | [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}] |"
        )
    lines.append("")
    lines.append("## Best by Dataset")
    for ds, th in sorted(payload["best_threshold_by_dataset"].items()):
        lines.append(f"- {ds}: `{th:.3f}`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default="0.00,0.01,0.02,0.05,0.10")
    parser.add_argument("--n-gene-trees", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--datasets",
        default="balanced8,asymmetric8,shortbranch8,balanced8_missing,shortbranch8_missing_noisy",
        help=(
            "Comma-separated subset of: balanced8,asymmetric8,shortbranch8,"
            "balanced8_missing,shortbranch8_missing,balanced8_noisy,"
            "shortbranch8_noisy,shortbranch8_missing_noisy"
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--output-md", default=None, help="Optional markdown summary output path.")
    parser.add_argument(
        "--low-signal-mode",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Low-signal handling mode for calibration runs.",
    )
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows = calibrate_low_signal_threshold(
        thresholds=thresholds,
        n_gene_trees=args.n_gene_trees,
        seed=args.seed,
        n_replicates=args.replicates,
        datasets=datasets,
        low_signal_mode=args.low_signal_mode,
    )
    payload = {
        "thresholds": thresholds,
        "n_gene_trees": int(args.n_gene_trees),
        "seed": int(args.seed),
        "n_replicates": int(args.replicates),
        "datasets": datasets,
        "low_signal_mode": args.low_signal_mode,
        "rows": [asdict(r) for r in rows],
    }
    payload["best_threshold_by_dataset"] = _best_thresholds(payload["rows"])
    text = json.dumps(payload, indent=2)
    print(text)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(payload, Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
