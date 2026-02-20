# OAKTREE Milestone Status

Update this file when a milestone’s tests are green. Keep dates in YYYY-MM-DD.

## M0 Foundation
- [x] Complete
- Date achieved: 2026-02-19
- Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_trees.py` (10 passed)
  - Notes: Tree I/O, shared taxa, quintet topology enumeration/canonicalization, induced subtree extraction, and reproducible quintet sampling are implemented in `oaktree/oaktree/trees.py`.

## M1 MSC Core
- [x] Complete
- Date achieved: 2026-02-19
- Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_msc.py` (18 passed; includes published-value checks, closed-form+exact+simulation anomaly-zone quartet checks, msprime reference validation, lookup/interpolation error-bound tests, and lookup precompute performance-guard tests)
  - Notes: Implemented exact `coalescent_probability`, exact quintet likelihood via finite-state coalescent DP, exact rooted gene-tree distribution utility, lookup-table precompute + bilinear interpolation, and larger-grid profiling support (`scripts/profile_lookup_tables.py`).

## M2 Graph Partitioning
- [ ] Complete
- Date achieved: 
- Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_graphs.py` (8 passed), `./venv/bin/pytest -q tests/test_inference.py` (5 passed)
  - Notes: Initial Phase 2 core implemented in `oaktree/oaktree/graphs.py`:
    - quintet bipartition weighting
    - taxon graph construction from quintet observations
    - spectral signed max-cut with local refinement
    - recursive partitioning with artificial taxon bookkeeping and n2 projection support
    - recursive partition output conversion to concrete Newick via `partition_tree_to_newick`
    - end-to-end 8-taxon species-tree topology recovery from msprime gene trees via `infer_species_tree_newick_phase2`
    - additional asymmetric and missing-taxa integration validation coverage
    - low-signal unresolved/ambiguous split handling via partition-strength thresholding
    - adaptive low-signal mode added (`low_signal_mode='adaptive'`) using pairwise edge-coherence to set per-node thresholds
    - production default policy set to adaptive mode with cap `low_signal_threshold=0.5`
    - threshold calibration workflow added (`scripts/calibrate_threshold.py`, `oaktree.validation.calibrate_low_signal_threshold`)
    - initial sweep artifact: `tables/threshold_calibration.json` and `tables/threshold_calibration.md`
    - harder-regime sweep artifact: `tables/threshold_calibration_hard.json` and `tables/threshold_calibration_hard.md`
    - widened-threshold sweep artifact: `tables/threshold_calibration_hard_wide.json` and `tables/threshold_calibration_hard_wide.md`
    - adaptive-mode sweep artifact: `tables/threshold_calibration_adaptive.json` and `tables/threshold_calibration_adaptive.md`
    - confidence-shrunken quintet aggregation added before recursive partitioning (retain full counts for high-confidence quintets; margin-shrink ambiguous quintets)

## M3 Branch Lengths
- [x] Complete
- Date achieved: 2026-02-19
- Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_branch_lengths.py` (5 passed)
  - Notes: Phase 3 baseline implemented in `oaktree/oaktree/branch_lengths.py`:
    - edge-informative quintet frequency counting
    - method-of-moments branch-length estimator
    - baseline internal-edge length assignment across a species tree
    - ML-style branch-length refinement via L-BFGS-B (`optimize_branch_lengths_ml`)

## M4 EM Loop
- [x] Complete
- Date achieved: 2026-02-19
- Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_weights.py` (5 passed), `./venv/bin/pytest -q tests/test_inference.py::test_phase4_em_wrapper_runs_from_phase2_init` (passed)
  - Notes: Phase 4 entry path wired:
    - `compute_gene_tree_weights` and `em_refine_species_tree_newick` in `oaktree/oaktree/weights.py`
    - `infer_species_tree_newick_phase4_em` wrapper in `oaktree/oaktree/inference.py`
    - Branch-length optimization integrated in EM initialization and each iteration update
    - Explicit convergence criteria implemented (RF/topology + branch-length delta tolerances)
    - Added noisy/short-branch EM validation coverage
    - Added controlled clean-vs-outlier weight-separation test coverage
    - EM update now uses weighted quintet observations directly (no bootstrap resampling of gene trees), reducing iteration variance
    - EM gene-tree scoring objective upgraded to MSC likelihood when branch lengths are present (with topology-consistency fallback for unlengthed trees)

## M5 CLI
- [x] Complete
- Date achieved: 2026-02-19
- Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_cli.py` (2 passed)
  - Notes:
    - End-to-end CLI path implemented (`oaktree/oaktree/cli.py`) for Phase 2 / Phase 4 modes
    - Supports input gene trees, optional output file, taxa selection, seed, quintet sampling cap, and EM iteration count
    - Supports Phase 2 low-signal controls via `--low-signal-threshold` and `--low-signal-mode {fixed,adaptive}`
    - Default low-signal settings now match recommended policy: `--low-signal-mode adaptive --low-signal-threshold 0.5`
    - Console entrypoint added in `setup.py` (`oaktree=oaktree.cli:main`)

## M6 Validation
- [ ] Complete
- Date achieved: 
  - Evidence:
  - Tests: `./venv/bin/pytest -q tests/test_validation.py` (8 passed)
  - Notes:
    - Validation helpers implemented in `oaktree/oaktree/validation.py`
    - Simulation and benchmark drivers added (`scripts/simulate.py`, `scripts/benchmark.py`)
    - Baseline benchmark results recorded in `docs/VALIDATION_BASELINE.md` and `tables/validation_baseline.json`
    - Expanded benchmark run recorded in `tables/validation_expanded.json` (`--expanded` datasets)
    - External baseline set now includes majority consensus, strict consensus, NJ, and UPGMA RF comparisons
    - Reporting artifacts generated via `scripts/report_validation.py`:
      - `tables/validation_expanded_summary.md`
      - `tables/validation_expanded_rf.png`
    - Multi-seed replicate aggregation with mean/std/95% CI emitted to:
      - `tables/validation_expanded_replicates.json`
      - `tables/validation_expanded_replicates_summary.md`
      - `tables/validation_expanded_replicates_rf.png`
    - Replicate CI method upgraded to bootstrap-percentile (configurable samples via `--ci-bootstrap-samples`)
    - Benchmark runner now supports core-vs-guardrailed evaluation modes (`--guardrail-mode guardrailed|core|both`) for fair method attribution
    - Recommended-run artifact set generated:
      - `tables/validation_recommended.json`
      - `tables/validation_recommended_summary.md`
      - `tables/validation_recommended_rf.png`
    - Core-vs-guardrailed ASTRAL snapshot generated:
      - `tables/validation_astral_core_vs_guardrail.json`
      - `tables/validation_astral_core_vs_guardrail_summary.md`
      - `tables/validation_astral_core_vs_guardrail_rf.png`
    - Updated ASTRAL head-to-head (confidence-shrink + adaptive cap 0.2) generated:
      - `tables/validation_astral_core_vs_guardrail_confshrink2.json`
      - `tables/validation_astral_core_vs_guardrail_confshrink2_summary.md`
      - `tables/validation_astral_core_vs_guardrail_confshrink2_rf.png`
    - Runtime profiling harness added:
      - `scripts/profile_pipeline.py` (cProfile hotspot extraction for Phase 2/4 end-to-end runs)

## Current Suite
- Latest targeted run: `./venv/bin/pytest -q tests/test_inference.py tests/test_weights.py tests/test_validation.py tests/test_cli.py` (26 passed)
- Date: 2026-02-20
