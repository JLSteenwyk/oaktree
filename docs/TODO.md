# OAKTREE Task Backlog

This is the operational checklist. Update as work progresses.

## Next up
- Re-run the improved ASTRAL head-to-head profile at larger scale (`replicates=10`) to verify statistical stability of the observed win.
- Validate the same profile on 16-taxon quick datasets and confirm whether the ASTRAL gap remains closed.
- Run the recommended benchmark profile at larger scale offline (`replicates=10`, optionally `n_gene_trees=120`) and publish those artifacts as final Phase 6 reference.

## Phase 0
- Verify remaining Phase 0 acceptance criteria and record evidence in `docs/STATUS.md`.

## Phase 1
- Monitor runtime stability of lookup precompute guard thresholds as environments evolve.

## Phase 2
- Monitor adaptive-threshold behavior on larger/noisier regimes and tune coherence-to-threshold mapping if needed.

## Phase 3
- Add additional calibration tests against known branch lengths under controlled simulations.

## Phase 4
- Monitor robustness of weighted-observation EM updates under larger outlier fractions and alternate noise models.

## Phase 5
- Expand CLI input validation/warnings and add richer user-facing diagnostics.

## Phase 6
- Implement simulation scripts and benchmarking.
- Run expanded validation study and produce figures.
- Add external baseline comparisons and aggregate reporting.
- Increase replicate count and improve CI methodology (bootstrap / bounded intervals).
- Promote recommended-run artifacts to final validation package after larger-scale rerun.
