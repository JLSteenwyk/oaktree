# OAKTREE Milestones

Each milestone ends with tests green. Do not advance until previous milestone passes.

## M0 Foundation
- Implement Phase 0 tree IO, quintet canonicalization, and subtree extraction.
- Tests: `test_trees.py` suite passes.

## M1 MSC Core
- Implement Phase 1 coalescent history enumeration and probability functions.
- Build lookup table and interpolation.
- Tests: `test_msc.py` suite passes.

## M2 Graph Partitioning
- Implement taxon graph construction, spectral max-cut, recursive partitioning.
- Tests: `test_graphs.py` suite passes.

## M3 Branch Lengths
- Implement frequency counting, method-of-moments, and ML optimization.
- Tests: `test_branch_lengths.py` suite passes.

## M4 EM Loop
- Implement weighting, iteration, and convergence logic.
- Tests: `test_weights.py` and `test_inference.py` suites pass.

## M5 CLI
- Implement CLI with validation and outputs.
- Tests: `test_cli_*` pass.

## M6 Validation
- Simulation study and comparison against baselines.
- Results reported and analyzed.

## Current status snapshot (2026-02-19)
- Completed: M0, M1, M3, M4, M5.
- In progress: M2 (threshold calibration/closing criteria), M6 (external-baseline expansion and figure/report generation).
- Full test suite: `./venv/bin/pytest -q` -> 64 passed.
