# OAKTREE Risks

## High risk
- Quintet canonicalization errors causing silent topology misclassification.
- Coalescent history enumeration off-by-one or incorrect branch mapping.
- Artificial taxon handling errors producing biased partitions.
- Likelihood underflow or numerical instability in MSC probability computations.
- Optimization failures for short internal branches.

## Mitigations
- Exhaustive canonicalization tests across label permutations.
- Cross-validation with published tables and secondary implementations.
- Explicit tests for artificial taxon normalization.
- Log-space computations and stability checks.
- Start optimization from moment estimates with bounds.

## Medium risk
- Performance bottlenecks during quintet sampling or probability lookup.
- Memory use of large lookup tables.
- Over/under-sensitive low-signal split threshold leading to unnecessary polytomies or false resolution.
- Validation overfitting to in-repo simulators without enough external baseline comparisons.

## Mitigations
- Profiling only after correctness.
- Discrete grid resolution tuned by tests.
- Use on-disk tables and lazy loading.
- Calibrate low-signal threshold across balanced/asymmetric/short-branch regimes.
- Add external baseline methods in expanded Phase 6 benchmark reporting.
