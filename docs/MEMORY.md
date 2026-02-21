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
- 2026-02-20: Added large-taxa optimization and stabilization changes in Phase 2/4 paths:
  - exact uniform quintet subset sampling via combinatorial unranking
  - guardrailed large-taxa budget floors in candidate construction/scoring
  - Phase 4 large-taxa topology-freeze behavior (branch-length optimization without topology moves)
  - topology-change acceptance gating for large-taxa guardrailed EM proposals

## Profiling baselines
- 2026-02-19 lookup precompute baseline (`scripts/profile_lookup_tables.py`, max_tau=2.0, all 15 species topologies):
  - grid 9x9: ~1.04s
  - grid 13x13: ~1.97s
  - grid 17x17: ~3.62s
- 2026-02-20 end-to-end inference profiling harness added (`scripts/profile_pipeline.py`).
  - Representative run: `shortbranch8`, Phase 4, 40 gene trees, 4 EM iterations, core mode.
  - Timing: simulate ~0.021s, inference ~9.463s, total ~9.484s.
  - Dominant cumulative hotspots:
    - `weights.py:score_gene_tree_against_species_tree`
    - `trees.py:extract_induced_subtree`
    - `branch_lengths.py:optimize_branch_lengths_ml`
    - `branch_lengths.py:count_quintet_frequencies`
    - `msc.py:quintet_probability`
- Guard-test budgets currently set to:
  - grid 9x9: < 4.0s
  - grid 13x13: < 7.0s
  - grid 17x17: < 12.0s
- 2026-02-20 larger-scale runtime checkpoints:
  - 32 taxa / 220 genes fullset (`tables/validation_balanced32_g220_fullset_r1.json`):
    - phase2_guardrailed ~84.88s; phase4_guardrailed_i4 ~194.38s
  - 32 taxa / 220 genes post-opt exact sampling (`tables/validation_balanced32_g220_guardrail_speed_postopt_sampling_exact_adaptive2.json`):
    - phase2_guardrailed ~26.28s; phase4_guardrailed_i4 ~55.63s
  - 64 taxa / 220 genes fullset (`tables/validation_balanced64_g220_fullset_r1.json`):
    - phase2_guardrailed ~51.45s; phase4_guardrailed_i4 ~70.75s; astral ~2.23s

## Current validation snapshot
- 2026-02-20 targeted suite: `./venv/bin/pytest -q tests/test_inference.py tests/test_validation.py` -> 18 passed, 1 failed.
- Current failing test:
  - `tests/test_inference.py::test_phase2_default_policy_matches_explicit_adaptive_noisy`
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
  - `tables/validation_balanced32_g220_fullset_r1.json`
  - `tables/validation_balanced32_g220_guardrail_speed_postopt.json`
  - `tables/validation_balanced32_g220_guardrail_speed_postopt_sampling_exact_adaptive2.json`
  - `tables/validation_balanced32_g220_guardrail_seed_sweep_r3.json`
  - `tables/validation_balanced32_g220_guardrail_seed_sweep_r3_topology_frozen_phase4.json`
  - `tables/validation_balanced32_g220_guardrail_vs_astral_r3.json`
  - `tables/validation_balanced64_g220_fullset_r1.json`
  - `tables/runtime_oaktree_vs_astral_postopt.json`
  - `tables/speed_accuracy_oaktree_vs_astral.json`
