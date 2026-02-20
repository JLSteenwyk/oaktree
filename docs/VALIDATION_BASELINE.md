# Phase 6 Baseline Results (2026-02-19)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --n-gene-trees 120 --seed 3 --output tables/validation_baseline.json
```

## Summary

- Dataset: `balanced8`
  - Phase 2 RF distance: `0`
  - Phase 4 RF distance: `0`
- Dataset: `asymmetric8`
  - Phase 2 RF distance: `2`
  - Phase 4 RF distance: `2`

Artifacts:
- JSON output: `tables/validation_baseline.json`
- Benchmark driver: `scripts/benchmark.py`
- Simulation driver: `scripts/simulate.py`

## Expanded Run (2026-02-19)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --expanded --n-gene-trees 120 --seed 4 --output tables/validation_expanded.json
```

Summary (RF to known true topology):

- `balanced8`: Phase 2 `0`, Phase 4 `0`, Consensus(Maj) `1`, Consensus(Strict) `6`, NJ `1`, UPGMA `0`
- `asymmetric8`: Phase 2 `2`, Phase 4 `2`, Consensus(Maj) `0`, Consensus(Strict) `5`, NJ `0`, UPGMA `0`
- `shortbranch8`: Phase 2 `4`, Phase 4 `4`, Consensus(Maj) `6`, Consensus(Strict) `6`, NJ `1`, UPGMA `0`
- `balanced8_missing`: Phase 2 `0`, Phase 4 `0`, Consensus(Maj) `2`, Consensus(Strict) `6`, NJ `1`, UPGMA `0`

Artifact:
- Expanded JSON output: `tables/validation_expanded.json`
- Expanded markdown table: `tables/validation_expanded_summary.md`
- Expanded RF figure: `tables/validation_expanded_rf.png`

Current recommendation:
- Use `tables/validation_expanded.json` as the active Phase 6 baseline artifact for follow-on figures and external-baseline additions.

Report generation command:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/report_validation.py --input tables/validation_expanded.json --output-md tables/validation_expanded_summary.md --output-fig tables/validation_expanded_rf.png
```

## Multi-Seed Replicate Run (2026-02-19)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --expanded --n-gene-trees 120 --seed 4 --replicates 3 --seed-step 1 --output tables/validation_expanded_replicates.json
```

Aggregate mean RF across datasets (from replicate summary):

- Phase 2: `1.500`
- Phase 4: `1.833`
- Consensus (Maj): `2.250`
- Consensus (Strict): `5.667`
- NJ: `0.917`
- UPGMA: `0.917`

Replicate reporting command:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/report_validation.py --input tables/validation_expanded_replicates.json --output-md tables/validation_expanded_replicates_summary.md --output-fig tables/validation_expanded_replicates_rf.png
```

Replicate artifacts:
- JSON output: `tables/validation_expanded_replicates.json`
- Markdown summary: `tables/validation_expanded_replicates_summary.md`
- RF figure with 95% CI bars: `tables/validation_expanded_replicates_rf.png`

## Phase 2 Threshold Calibration (2026-02-19)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/calibrate_threshold.py --thresholds 0.00,0.01,0.02,0.05,0.10 --n-gene-trees 80 --seed 7 --replicates 3 --datasets balanced8,asymmetric8,shortbranch8 --output tables/threshold_calibration.json --output-md tables/threshold_calibration.md
```

Observed result:
- Phase 2 mean RF was unchanged across thresholds `0.00` to `0.10` for all three datasets in this sweep.
- Best threshold selected by tie-break (lowest threshold) was `0.00` for each dataset.

Artifacts:
- `tables/threshold_calibration.json`
- `tables/threshold_calibration.md`

## Hard-Regime and Wide-Range Calibration (2026-02-19)

Commands run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/calibrate_threshold.py --thresholds 0.00,0.01,0.02,0.05,0.10 --n-gene-trees 80 --seed 7 --replicates 3 --datasets balanced8,asymmetric8,shortbranch8,balanced8_missing,shortbranch8_missing_noisy --output tables/threshold_calibration_hard.json --output-md tables/threshold_calibration_hard.md
PYTHONPATH=oaktree ./venv/bin/python scripts/calibrate_threshold.py --thresholds 0.00,0.05,0.10,0.20,0.30,0.50,0.80 --n-gene-trees 80 --seed 7 --replicates 3 --datasets balanced8,asymmetric8,shortbranch8,balanced8_missing,shortbranch8_missing_noisy --output tables/threshold_calibration_hard_wide.json --output-md tables/threshold_calibration_hard_wide.md
```

Observed behavior:
- Thresholds `0.00` to `0.10` remained flat in this setup.
- In the widened sweep, thresholds `0.20-0.50` improved short-branch and short-branch+missing+noisy mean RF without degrading balanced/asymmetric datasets.
- At `0.80`, performance degraded sharply on asymmetric and short-branch regimes.

Provisional recommendation:
- Use `low_signal_threshold ~= 0.50` for stress/noisy regimes.
- Keep `0.00` acceptable for cleaner regimes.

Artifacts:
- `tables/threshold_calibration_hard.json`
- `tables/threshold_calibration_hard.md`
- `tables/threshold_calibration_hard_wide.json`
- `tables/threshold_calibration_hard_wide.md`

## Adaptive-Mode Calibration (2026-02-19)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/calibrate_threshold.py --low-signal-mode adaptive --thresholds 0.10,0.20,0.30,0.50,0.80 --n-gene-trees 80 --seed 7 --replicates 3 --datasets balanced8,asymmetric8,shortbranch8,balanced8_missing,shortbranch8_missing_noisy --output tables/threshold_calibration_adaptive.json --output-md tables/threshold_calibration_adaptive.md
```

Observed behavior:
- Adaptive mode is stable on clean/easy datasets across caps.
- Increasing cap improves short-branch stress regimes (notably around `0.50` and above).
- Unlike fixed mode, high cap did not show the same asymmetric-dataset collapse in this run.

Practical recommendation:
- Use `--low-signal-mode adaptive` with cap `--low-signal-threshold 0.50` as a robust default for mixed regimes.

Artifacts:
- `tables/threshold_calibration_adaptive.json`
- `tables/threshold_calibration_adaptive.md`

## Recommended-Settings Benchmark (2026-02-19)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --expanded --n-gene-trees 80 --seed 4 --replicates 5 --seed-step 1 --low-signal-mode adaptive --low-signal-threshold 0.5 --ci-bootstrap-samples 2000 --output tables/validation_recommended.json
PYTHONPATH=oaktree ./venv/bin/python scripts/report_validation.py --input tables/validation_recommended.json --output-md tables/validation_recommended_summary.md --output-fig tables/validation_recommended_rf.png
```

Summary (mean RF across datasets, bootstrap-percentile CIs in report):

- Phase 2: `1.100`
- Phase 4: `2.400`
- Consensus (Maj): `2.100`
- Consensus (Strict): `5.500`
- NJ: `1.400`
- UPGMA: `1.250`

Artifacts:
- `tables/validation_recommended.json`
- `tables/validation_recommended_summary.md`
- `tables/validation_recommended_rf.png`

Note:
- `replicates=10` with larger `n_gene_trees` is supported but can be slow in-session; use the same command with `--replicates 10` for offline final packaging.

## Core vs Guardrailed Reporting Mode (2026-02-20)

To separate core OAKTREE behavior from baseline-assisted behavior, run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --expanded --n-gene-trees 20 --seed 1 --replicates 1 --guardrail-mode both --output /tmp/bench_guardrail_both.json
```

Behavior:
- Datasets are emitted as `guardrailed:<dataset>` and `core:<dataset>`.
- This allows side-by-side CI aggregation and plotting without changing downstream report tooling.

## ASTRAL Head-to-Head Snapshot (2026-02-20)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --expanded --n-gene-trees 60 --seed 6 --replicates 2 --seed-step 1 --ci-bootstrap-samples 200 --guardrail-mode both --astral-jar /mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/Astral/astral.5.7.8.jar --output tables/validation_astral_core_vs_guardrail.json
```

Mean RF across expanded 8-taxon datasets:
- Guardrailed OAKTREE Phase 2: `1.375`
- Guardrailed OAKTREE Phase 4: `1.375`
- Core OAKTREE Phase 2: `1.500`
- Core OAKTREE Phase 4: `1.500`
- ASTRAL: `0.750`

Artifact:
- `tables/validation_astral_core_vs_guardrail.json`

## ASTRAL Head-to-Head (Improved, 2026-02-20)

Command run:

```bash
PYTHONPATH=oaktree ./venv/bin/python scripts/benchmark.py --expanded --n-gene-trees 60 --seed 6 --replicates 2 --seed-step 1 --ci-bootstrap-samples 200 --guardrail-mode both --low-signal-mode adaptive --low-signal-threshold 0.2 --astral-jar /mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/Astral/astral.5.7.8.jar --output tables/validation_astral_core_vs_guardrail_confshrink2.json
PYTHONPATH=oaktree ./venv/bin/python scripts/report_validation.py --input tables/validation_astral_core_vs_guardrail_confshrink2.json --output-md tables/validation_astral_core_vs_guardrail_confshrink2_summary.md --output-fig tables/validation_astral_core_vs_guardrail_confshrink2_rf.png
```

Mean RF across expanded 8-taxon datasets:
- Guardrailed OAKTREE Phase 2: `0.000`
- Guardrailed OAKTREE Phase 4: `0.000`
- Core OAKTREE Phase 2: `0.500`
- Core OAKTREE Phase 4: `0.500`
- ASTRAL: `0.750`

Artifacts:
- `tables/validation_astral_core_vs_guardrail_confshrink2.json`
- `tables/validation_astral_core_vs_guardrail_confshrink2_summary.md`
- `tables/validation_astral_core_vs_guardrail_confshrink2_rf.png`
