# OAKTREE Task Backlog

This is the operational checklist. Update as work progresses.

## Next up
- Beat ASTRAL in standalone core mode on 64-taxon hard regimes:
  - target `shortbranch64` and `shortbranch64_missing_noisy` in `tables/validation_scaled64_complex_g220_r1_core_vs_astral_robustw_v3_fastci.json`.
- Implement hard-regime topology improvement pass (core only):
  - local NNI/SPR neighborhood search around top Phase2/NJ/UPGMA candidates, scored with weighted MSC objective.
- Add explicit selector diagnostics for standalone core:
  - persist per-candidate quick/MSC scores and winner rationale for postmortem on hard regimes.
- Add regime-focused acceptance tests:
  - assert non-regression on `balanced64` / `balanced64_missing`
  - require improved RF on at least one hard regime before promoting changes.
- Run larger-scale statistical validation package:
  - `replicates=10` for 32/64 taxa core + guardrailed + ASTRAL (+ TREE-QMC where available).
- Finalize publication-quality benchmark bundle:
  - consolidated tables/figures comparing core, guardrailed, NJ, UPGMA, ASTRAL, TREE-QMC by regime.

## Phase 0
- Verify remaining Phase 0 acceptance criteria and record evidence in `docs/STATUS.md`.

## Phase 1
- Monitor runtime stability of lookup precompute guard thresholds as environments evolve.

## Phase 2
- Monitor adaptive-threshold behavior on larger/noisier regimes and tune coherence-to-threshold mapping if needed.
- Investigate and reduce topology errors in core mode on 32/64-taxon balanced simulations.
- Investigate targeted hard-regime search (NNI/SPR) with weighted MSC scoring for short-branch+missing+noisy conditions.
- Keep external-candidate injection as optional experimental guardrail mode; exclude from standalone-core claims.

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
