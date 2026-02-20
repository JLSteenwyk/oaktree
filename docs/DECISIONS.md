# OAKTREE Decisions Log

This file records major decisions and the rationale. Add new entries as choices are made.

## 2026-02-19
- Adopt `treeswift` as the default tree library.
  Rationale: fastest and stable for large tree IO and manipulation.
- Quintet topology canonicalization uses bipartition smaller-side encoding.
  Rationale: avoids unrooted ambiguity and supports deterministic hashing.
- Max-cut uses spectral partitioning plus local refinement.
  Rationale: robust and fast for repeated recursive splits.
- Precompute MSC probabilities on a branch length grid.
  Rationale: supports fast inference during EM iterations.
- Start with ASTRID-like distance initialization and NJ tree.
  Rationale: strong baseline and fast bootstrap into EM.
- Implement MSC quintet likelihood with a finite-state coalescent DP over lineage forests.
  Rationale: exact probability computation with deterministic state transitions and direct validation against msprime.
- Use 2D quintet lookup tables over internal branch lengths with bilinear interpolation.
  Rationale: practical acceleration path for inference while preserving bounded interpolation error in tests.
- Add low-signal unresolved split handling in recursive partitioning.
  Rationale: avoid overconfident binary splits when partition evidence is weak; preserve ambiguity as polytomies.
- Keep consensus-tree baseline in validation benchmarks alongside Phase 2 and Phase 4 outputs.
  Rationale: provides a simple external reference point for RF-distance comparisons on simulated datasets.
- Expand external validation baselines to strict-consensus and distance-based trees (NJ, UPGMA).
  Rationale: gives broader, method-diverse references for topology recovery quality across data regimes.
- Add multi-seed replicate benchmarking with aggregate RF mean/std/95% CI outputs.
  Rationale: reduces single-seed sensitivity and provides uncertainty-aware baseline comparisons.
- Add a dedicated Phase 2 threshold calibration workflow and artifact outputs.
  Rationale: make low-signal threshold tuning reproducible and data-backed across regimes.
- Include harder calibration regimes (missing and label-noise stress) and wide threshold sweeps.
  Rationale: small threshold ranges were non-informative; stress scenarios exposed threshold-response behavior.
- Add `adaptive` low-signal policy based on edge-coherence with configurable cap.
  Rationale: avoid brittle single global threshold while preserving conservative behavior in low-coherence subproblems.
- Set production default low-signal policy to `adaptive` with cap `0.5`.
  Rationale: calibration showed stronger robustness in mixed/noisy regimes without harming clean regimes.
- Switch replicate CI reporting to bootstrap-percentile intervals.
  Rationale: nonparametric uncertainty estimates are more defensible at moderate replicate counts than normal-approximation CIs.
