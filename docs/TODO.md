# OAKTREE Task Backlog

This is the operational checklist. Update as work progresses.

## Next up
- Fix current regression in default-policy equivalence:
  - `tests/test_inference.py::test_phase2_default_policy_matches_explicit_adaptive_noisy`
- Strengthen large-taxa core accuracy:
  - improve Phase 2 core partition recovery on 32/64 taxa regimes where core RF remains high.
- Close runtime gap vs ASTRAL on large taxa:
  - prioritize EM scoring/quintet extraction hotspots in Phase 4 guardrailed path.
- Run larger-scale statistical validation package:
  - `replicates=10` for 32-taxon and 64-taxon regimes, with aggregate RF/runtime summaries.
- Finalize publication-quality benchmark bundle:
  - consolidated tables/figures comparing core, guardrailed, NJ, UPGMA, and ASTRAL by regime.

## Phase 0
- Verify remaining Phase 0 acceptance criteria and record evidence in `docs/STATUS.md`.

## Phase 1
- Monitor runtime stability of lookup precompute guard thresholds as environments evolve.

## Phase 2
- Monitor adaptive-threshold behavior on larger/noisier regimes and tune coherence-to-threshold mapping if needed.
- Investigate and reduce topology errors in core mode on 32/64-taxon balanced simulations.
- Evaluate whether guardrail-derived priors can safely improve core initialization without masking core behavior.

## Phase 3
- Add additional calibration tests against known branch lengths under controlled simulations.

## Phase 4
- Monitor robustness of weighted-observation EM updates under larger outlier fractions and alternate noise models.
- Continue tuning topology-update acceptance and freezing logic for >24 taxa regimes.
- Add dedicated convergence-quality tests for large-taxa guardrailed EM runs.

## Phase 5
- Expand CLI input validation/warnings and add richer user-facing diagnostics.

## Phase 6
- Implement simulation scripts and benchmarking.
- Run expanded validation study and produce figures.
- Add external baseline comparisons and aggregate reporting.
- Increase replicate count and improve CI methodology (bootstrap / bounded intervals).
- Promote recommended-run artifacts to final validation package after larger-scale rerun.
- Add explicit large-scale (32/64 taxa, 220 genes) benchmark sections to final report artifacts.
