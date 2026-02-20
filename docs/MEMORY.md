# OAKTREE Memory

This file is the long-term memory for the project. It captures constraints, design decisions, and cross-phase dependencies that must stay stable across future work.

## Non-negotiable constraints
- Correctness over speed. Tests first, then implementation.
- No ML or neural nets anywhere.
- MSC likelihood must use exact analytic formulas, no approximations.
- Always use a tree library for Newick parsing and manipulation. No string parsing.
- Do not assume trees are rooted unless explicitly stated.
- Do not proceed to the next phase until all tests in the current phase pass.

## Core architecture
- Language: Python.
- Primary libs: numpy, scipy, networkx, treeswift.
- Validation libs: msprime, dendropy, matplotlib.
- Package layout is defined in `PLAN.md` and must be preserved.

## Quintet representation invariants
- Taxa for a quintet are sorted alphabetically.
- Canonical unrooted topology is stored as two internal edges.
- Each internal edge stored as the smaller side of a bipartition.
- Topology representation is a tuple of two sorted tuples.

## MSC probability invariants
- Probabilities for 15 topologies sum to 1.0 (tolerance 1e-10).
- Star tree (all internal lengths 0) gives uniform 1/15.
- Long branches (tau > 5) must make matching topology probability approach 1.
- The 2->1 single-branch formula is P = 1 - exp(-tau).
- Quartet anomaly-zone threshold for rooted caterpillar uses closed-form `a(x)`; anomaly when younger branch `y < a(x)`.

## Graph partitioning invariants
- Max-cut uses spectral method + local refinement.
- Artificial taxon represents the complement set.
- n2 normalization required for artificial taxa.

## EM loop invariants
- Initialization uses a distance method, not random.
- Gene tree weights derived from mean log likelihood across quintets.
- Temperature schedule default: 1.0 + 0.5 * iteration.
- Convergence by RF distance between successive species trees.

## Testing invariants
- Write tests before implementation in each phase.
- Cross-validate subtree extraction against dendropy.
- Cross-validate coalescent histories with Degnan & Salter tables.

## Reproducibility
- Any randomized step must accept a seed.
- Sampling must be deterministic under fixed seed.

## Performance guardrails
- No optimization until correctness proven.
- Prefer clarity, determinism, and measurable outputs.

## Milestone progress
- 2026-02-19: M1 (MSC Core) completed with exact coalescent transition probabilities, finite-state DP quintet likelihoods, msprime reference validation, and lookup-table interpolation tests.
- 2026-02-19: M3 (Branch Lengths), M4 (EM Loop), and M5 (CLI) completed with passing tests.
- 2026-02-19: Phase 2 includes low-signal unresolved split handling and Newick polytomy emission for unresolved leaves.
- 2026-02-19: Phase 6 expanded benchmark artifact generated at `tables/validation_expanded.json`.

## Profiling baselines
- 2026-02-19 lookup precompute baseline (`scripts/profile_lookup_tables.py`, max_tau=2.0, all 15 species topologies):
  - grid 9x9: ~1.04s
  - grid 13x13: ~1.97s
  - grid 17x17: ~3.62s
- Guard-test budgets currently set to:
  - grid 9x9: < 4.0s
  - grid 13x13: < 7.0s
  - grid 17x17: < 12.0s

## Current validation snapshot
- 2026-02-19 full suite: `./venv/bin/pytest -q` -> 64 passed.
- 2026-02-19 validation artifacts:
  - `tables/validation_expanded.json`
  - `tables/validation_expanded_summary.md`
  - `tables/validation_expanded_rf.png`
  - `tables/validation_expanded_replicates.json`
  - `tables/validation_expanded_replicates_summary.md`
  - `tables/validation_expanded_replicates_rf.png`
  - `tables/threshold_calibration.json`
  - `tables/threshold_calibration.md`
  - `tables/threshold_calibration_hard.json`
  - `tables/threshold_calibration_hard.md`
  - `tables/threshold_calibration_hard_wide.json`
  - `tables/threshold_calibration_hard_wide.md`
  - `tables/threshold_calibration_adaptive.json`
  - `tables/threshold_calibration_adaptive.md`
  - `tables/validation_recommended.json`
  - `tables/validation_recommended_summary.md`
  - `tables/validation_recommended_rf.png`
