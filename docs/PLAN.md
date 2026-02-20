# OAKTREE: Optimized Analytic K-taxon Tree Reconstruction with EM Estimation — Agent Coding Plan

## Project overview

OAKTREE is a novel coalescent-based species tree inference method that improves on ASTRAL and TreeQMC by:
1. Using quintet (5-taxon) subtrees instead of quartets as the atomic comparison unit
2. Scoring subtrees with analytic MSC likelihoods instead of count-based quartet support
3. Iterating between species tree estimation and gene tree reweighting in an EM-like loop

The method takes a set of gene trees (Newick format) as input and outputs a species tree (Newick format) with branch lengths in coalescent units.

## Progress snapshot (2026-02-19)
- Implemented through CLI and validation scaffolding (Phases 0-6 code paths present).
- Current full-suite health: `./venv/bin/pytest -q` -> 64 passed.
- Active focus areas: Phase 6 external baseline comparisons and reporting, plus remaining Phase 2 threshold calibration work.

**Language:** Python (NumPy/SciPy for numerics, NetworkX for graphs)
**Target:** Working prototype that is correct on small-to-medium datasets (10–200 taxa, 100–5,000 gene trees)

---

## Critical constraints for the agent

### Correctness over speed
This is a scientific method. A subtle numerical bug produces wrong species trees that look plausible. At every phase:
- Write unit tests BEFORE implementation
- Validate against known analytical results
- Never optimize before correctness is proven

### Do not
- Use machine learning or neural networks anywhere — this is a statistical/combinatorial method
- Approximate the MSC likelihood computation — use exact analytic formulas
- Skip the iteration convergence check
- Generate synthetic test data without validating it against an independent simulator (ms, msprime)
- Assume any tree is rooted unless explicitly stated
- Use string manipulation to parse or modify Newick trees — always use a tree library (ete3, dendropy, or treeswift)

### File structure
```
oaktree/
├── oaktree/
│   ├── __init__.py
│   ├── trees.py           # Tree I/O, Newick parsing, subtree extraction
│   ├── msc.py             # MSC likelihood computation and lookup tables
│   ├── graphs.py          # Taxon graph construction and max-cut partitioning
│   ├── inference.py        # Main species tree inference loop
│   ├── weights.py          # Gene tree weighting and EM iteration
│   ├── branch_lengths.py   # Branch length estimation in coalescent units
│   └── utils.py            # Shared utilities
├── tests/
│   ├── test_trees.py
│   ├── test_msc.py
│   ├── test_graphs.py
│   ├── test_inference.py
│   ├── test_weights.py
│   ├── test_branch_lengths.py
│   └── fixtures/           # Small test tree files
├── tables/
│   └── quintet_probabilities.json  # Precomputed MSC probability tables
├── scripts/
│   ├── generate_tables.py  # Script to generate MSC lookup tables
│   ├── simulate.py         # Simulation wrapper for validation
│   ├── benchmark.py        # Benchmark runner across datasets/methods
│   ├── calibrate_threshold.py # Phase 2 low-signal threshold sweep
│   └── report_validation.py # Validation table/figure generation
├── cli.py                  # Command-line interface
├── setup.py
└── README.md
```

---

## Phase 0: Foundation — Tree I/O and subtree extraction

### Goal
Read gene trees, extract induced 5-taxon subtrees, and represent them canonically.

### Steps

#### 0.1 — Tree data structures
- Install and use `treeswift` (fastest Python tree library) for tree I/O
- Implement a function `read_gene_trees(path: str) -> list[Tree]` that reads a file of Newick trees (one per line)
- Implement a function `get_leaf_set(tree: Tree) -> set[str]` returning all tip labels
- Implement a function `get_shared_taxa(trees: list[Tree]) -> set[str]` returning the intersection of all leaf sets

#### 0.2 — Canonical quintet representation
A quintet (unrooted 5-taxon tree) has 15 possible topologies. Define a canonical labeling:
- Sort the 5 taxa alphabetically → (a, b, c, d, e)
- Represent each topology as a frozenset of splits. An unrooted binary tree on 5 taxa has 2 internal edges, each defining a bipartition. Store the topology as a tuple of the two smaller sides of each bipartition, sorted.
- Example: `((a, b), (a, b, c))` means one internal edge separates {a,b} from {c,d,e} and the other separates {a,b,c} from {d,e}
- Implement `canonicalize_quintet(tree: Tree, taxa: tuple[str, str, str, str, str]) -> QuintetTopology`
- Implement `enumerate_quintet_topologies() -> list[QuintetTopology]` returning all 15 unrooted topologies for a labeled 5-taxon set

#### 0.3 — Subtree extraction
- Implement `extract_induced_subtree(tree: Tree, taxa: tuple[str, ...]) -> Tree` that prunes a gene tree to a subset of taxa and suppresses degree-2 nodes
- Implement `sample_quintets(tree: Tree, n_samples: int, rng: np.random.Generator) -> list[tuple[tuple[str,...], QuintetTopology]]` that:
  - Randomly samples `n_samples` 5-taxon subsets from the tree's leaf set
  - Extracts the induced subtree for each
  - Returns a list of (taxa_tuple, topology) pairs
- If the gene tree has missing taxa relative to the full taxon set, skip quintets that include missing taxa

### Tests
- `test_canonicalize`: Verify that all labelings of the same topology produce the same canonical form
- `test_enumerate`: Verify exactly 15 topologies are produced for 5 labeled taxa
- `test_extract_induced`: On a known 10-taxon tree, verify that extracted 5-taxon subtrees match hand-computed topologies (use at least 5 specific examples)
- `test_sample_reproducibility`: With a fixed random seed, verify sampling produces identical results

### Acceptance criteria
- All 15 quintet topologies correctly enumerated and distinguishable
- Induced subtree extraction agrees with dendropy's `extract_tree()` on 20 random examples from a 50-taxon tree
- Subtree extraction handles multifurcating gene trees by treating them as soft polytomies (randomly resolving or enumerating resolutions)

---

## Phase 1: MSC likelihood computation

### Goal
Compute the probability of each quintet topology given a species tree with branch lengths in coalescent units.

### Background
Under the multispecies coalescent, the probability of an unrooted gene tree topology for n species can be computed by:
1. Enumerating all possible ranked histories (orderings of coalescent events)
2. For each ranked history, computing a product of exponential waiting-time probabilities
3. Summing over all ranked histories

For 5 taxa, the number of ranked histories per topology is small (at most ~100 depending on the topology), making exact computation feasible.

### Steps

#### 1.1 — Coalescent history enumeration for quintets
- Implement the Degnan & Salter (2005) algorithm for enumerating coalescent histories
- For a given species tree topology and gene tree topology (both on the same 5 taxa), enumerate all valid coalescent histories where each gene tree coalescence occurs in a branch of the species tree that is ancestral to both descendant lineages
- Store histories as sequences of (coalescence_event, species_tree_branch) pairs

#### 1.2 — Probability computation for a single history
- For a given coalescent history and species tree branch lengths (in coalescent units, τ = t/2N):
  - In each branch of the species tree, compute the probability that exactly the specified coalescences occur given k lineages entering and j lineages leaving
  - This uses the formula: P(k→j in time τ) involves a sum over intermediate states with exponential terms exp(-i(i-1)τ/2) for i lineages
  - Use the exact formulas from Degnan & Salter (2005) or Rosenberg (2002)
- Implement `coalescent_probability(k_in: int, k_out: int, tau: float) -> float`

#### 1.3 — MSC quintet probability function
- Implement `quintet_probability(species_tree: Tree, gene_tree_topology: QuintetTopology, taxa: tuple[str,...]) -> float`
  - Extract the induced 5-taxon species tree
  - Enumerate all valid coalescent histories
  - For each history, compute the probability given current species tree branch lengths
  - Sum probabilities over all histories
- This function is the computational core of the method

#### 1.4 — Precompute lookup table
For computational efficiency during inference:
- For each of the 15 possible quintet species tree topologies:
  - For each of the 15 possible quintet gene tree topologies:
    - Precompute the probability as a function of branch lengths
    - For efficiency, discretize branch lengths into a grid (e.g., 0.01 to 10.0 in 0.01 increments) and store a lookup table
    - Use interpolation for intermediate values
- Save tables to `tables/quintet_probabilities.json` or `.npz`
- Implement `precompute_quintet_tables(branch_length_grid: np.ndarray) -> dict`
- Implement `lookup_quintet_probability(species_topology_id: int, gene_topology_id: int, branch_lengths: tuple[float,...]) -> float` using the precomputed table with linear interpolation

### Tests
- `test_coalescent_probability_boundary`: P(k→k, τ=0) = 1.0; P(k→1, τ=∞) = 1.0
- `test_single_branch_two_lineages`: P(2→1, τ) = 1 - exp(-τ). This is the simplest case and must be exact.
- `test_symmetric_species_tree`: For a symmetric 5-taxon species tree (balanced), the matching gene tree topology should have the highest probability; the 14 non-matching topologies should each have strictly lower probability
- `test_anomaly_zone_quartet`: Reproduce the known anomaly zone result for 4 taxa — when internal branch < ln(2/3) coalescent units, the asymmetric quartet topology is most probable. Use this as a sanity check (compute for 4-taxon subcase).
- `test_probabilities_sum_to_one`: For any species tree, the 15 quintet topology probabilities must sum to 1.0 (within numerical tolerance 1e-10)
- `test_long_branches`: With very long species tree branches (τ > 5), the matching topology should have probability approaching 1.0
- `test_star_tree`: With all internal branches = 0 (star tree), all 15 topologies should have equal probability 1/15

### Acceptance criteria
- Probabilities sum to 1.0 for 100 random species trees with random branch lengths
- Known anomaly zone behavior reproduced exactly
- Star tree produces uniform distribution
- Lookup table interpolation error < 1e-6 compared to exact computation on 1000 random test cases

---

## Phase 2: Graph construction and partitioning

### Goal
Build a taxon graph from quintet likelihoods and find the maximum cut to bipartition the taxon set, following the TreeQMC recursive framework but using likelihood-weighted quintets instead of quartets.

### Steps

#### 2.1 — Quintet-to-bipartition contribution
For a quintet on taxa (a, b, c, d, e) with an observed gene tree topology and a candidate bipartition of the full taxon set S into S₁ and S₂:
- Determine which of the 5 taxa fall on each side
- The quintet provides evidence for or against this bipartition based on whether its topology is consistent with splitting these taxa this way
- Implement `quintet_bipartition_weight(quintet_taxa: tuple, gene_topology: QuintetTopology, species_topology: QuintetTopology, msc_likelihood: float, bipartition_side1: set, bipartition_side2: set) -> float`
  - If the quintet's gene tree topology is more probable under a species tree that contains this bipartition → positive weight (good graph)
  - If the quintet's gene tree topology is more probable under a species tree that does NOT contain this bipartition → negative weight (bad graph)

#### 2.2 — Pairwise taxon graph construction
For the current taxon set at any recursion level:
- For each pair of taxa (X, Y), compute a net weight:
  - Positive weight → X and Y should be on opposite sides of the next bipartition
  - Negative weight → X and Y should be on the same side
- Aggregate weights across all sampled quintets that include both X and Y
- Implement `build_taxon_graph(taxa: list[str], quintet_observations: list[QuintetObservation], species_tree_estimate: Tree) -> nx.Graph`
  - Nodes = taxa
  - Edge weights = net good-minus-bad weight for each pair

#### 2.3 — Maximum cut
- Implement max-cut on the taxon graph using a spectral method:
  - Compute the Laplacian matrix L of the graph
  - Find the eigenvector corresponding to the second-smallest eigenvalue (Fiedler vector)
  - Partition taxa by the sign of the Fiedler vector
  - This is a well-known O(n²) approximation to max-cut
- Implement `spectral_max_cut(graph: nx.Graph) -> tuple[set[str], set[str]]`
- Also implement a refinement step: after the initial spectral cut, try moving each taxon to the other side and keep the move if it increases the cut value (Kernighan-Lin style local search)

#### 2.4 — Recursive bipartitioning
- Implement the top-level recursive algorithm:
  ```
  def recursive_partition(taxa, quintet_observations, species_tree_estimate):
      if len(taxa) <= 3:
          return base_case_tree(taxa, quintet_observations)
      graph = build_taxon_graph(taxa, quintet_observations, species_tree_estimate)
      left, right = spectral_max_cut(graph)
      # Create artificial taxon for complement (as in TreeQMC)
      left_tree = recursive_partition(left + [artificial_right], ...)
      right_tree = recursive_partition(right + [artificial_left], ...)
      return merge(left_tree, right_tree)
  ```
- Handle the artificial taxon bookkeeping carefully — this is where TreeQMC's n2 normalization matters

#### 2.5 — Normalization
- Implement TreeQMC-style n2 normalization: quintets involving artificial taxa (representing complement sets) are downweighted proportionally to the number of real taxa the artificial taxon represents
- This prevents large complement sets from dominating the partition decision

### Tests
- `test_spectral_cut_planted`: Generate a graph with a planted partition (two clusters connected by weak edges) and verify the spectral cut recovers it
- `test_symmetric_easy_case`: 8-taxon balanced species tree, 1000 gene trees from msprime with long branches → recursive partitioning should recover the exact species tree topology
- `test_artificial_taxon`: Verify that after splitting {A,B,C,D} into {A,B} and {C,D}, the artificial taxon in the {A,B,artificial_CD} subproblem correctly represents {C,D}
- `test_normalization_effect`: On an asymmetric tree where one side has 3 taxa and the other has 20, verify n2 normalization produces a more accurate tree than unnormalized (n0) version

### Acceptance criteria
- Spectral max-cut recovers planted partitions with >95% accuracy on 100 random graphs
- On a known 8-taxon species tree with 500 gene trees (no estimation error, simulated under MSC with msprime), the full recursive algorithm recovers the correct topology
- With estimation error (gene trees inferred by RAxML from short simulated sequences), accuracy degrades gracefully rather than catastrophically

---

## Phase 3: Branch length estimation

### Goal
Estimate species tree branch lengths in coalescent units from the observed quintet frequency distribution.

### Steps

#### 3.1 — Quintet frequency counting
- For each internal edge e in the current species tree estimate, identify the set of quintets that are informative about e (i.e., where the 5 taxa span both sides of e)
- Count the frequency of each of the 15 quintet topologies among these informative quintets
- Implement `count_quintet_frequencies(species_tree: Tree, edge: Edge, quintet_observations: list) -> np.ndarray` (length-15 frequency vector)

#### 3.2 — Method-of-moments estimator
- The MSC predicts the quintet frequency distribution as a function of branch lengths
- For each internal edge, use the observed frequencies and the predicted frequencies to estimate the branch length
- Start with a simple approach: for internal edge e with length τ, the probability of the matching quintet topology is a monotonically decreasing function of ILS. Invert this relationship numerically.
- Implement `estimate_branch_length(observed_frequencies: np.ndarray, species_topology: QuintetTopology) -> float`

#### 3.3 — Maximum likelihood refinement
- After moment estimates, refine all branch lengths jointly by maximizing the total log-likelihood:
  - L(τ₁, ..., τₘ) = Σ_quintets log P(observed_topology | species_tree, τ₁, ..., τₘ)
- Use `scipy.optimize.minimize` with L-BFGS-B (branch lengths must be > 0)
- Implement `optimize_branch_lengths(species_tree: Tree, quintet_observations: list) -> Tree`

### Tests
- `test_known_branch_lengths`: Simulate gene trees under a known species tree with specific branch lengths. Estimate branch lengths from the gene trees. Verify estimates are within 20% of true values with 1000 gene trees.
- `test_long_branch_estimation`: Very long branches (τ > 3) should be estimated as large (exact value less important, but should be > 2)
- `test_short_branch_estimation`: Very short branches (τ < 0.1) should be estimated as small with higher variance (wider confidence interval is acceptable)
- `test_optimization_improves_moments`: The ML-refined branch lengths should have higher total log-likelihood than the moment estimates

### Acceptance criteria
- On 50 simulated datasets with known branch lengths, mean absolute error < 0.3 coalescent units for branches between 0.1 and 5.0
- ML optimization converges in < 100 iterations on all test cases
- Branch length estimates are positive (enforced by bounds in optimizer)

---

## Phase 4: Iterative EM loop

### Goal
Implement the iterative reweighting scheme where the species tree estimate informs gene tree reliability, which in turn improves the species tree estimate.

### Steps

#### 4.1 — Gene tree likelihood scoring
- Given a current species tree estimate with branch lengths, compute a reliability score for each gene tree:
  - For each gene tree g, compute L(g) = mean of log P(quintet_topology | species_tree) across all sampled quintets from g
  - High L(g) → gene tree is consistent with species tree → trustworthy
  - Low L(g) → gene tree is inconsistent → likely contains estimation error
- Implement `score_gene_trees(gene_trees: list[Tree], species_tree: Tree, quintet_cache: dict) -> np.ndarray`

#### 4.2 — Weight computation
- Convert gene tree scores to weights:
  - w(g) = exp(L(g) - max_L) to prevent numerical underflow
  - Normalize: w(g) = w(g) / Σ w(g) * n_genes (so weights sum to n_genes, preserving scale)
  - Alternatively, use a softmax with temperature parameter β: w(g) = exp(β * L(g)) / Σ exp(β * L(j))
  - β controls how aggressively low-quality gene trees are downweighted. Start with β=1 and increase across iterations.
- Implement `compute_gene_weights(scores: np.ndarray, temperature: float) -> np.ndarray`

#### 4.3 — Weighted quintet aggregation
- Modify the graph construction (Phase 2) to weight each quintet by the gene tree weight from which it was extracted
- A quintet from a high-weight gene tree contributes more to the taxon graph than one from a low-weight gene tree
- This requires threading weights through all of Phase 2

#### 4.4 — Main iteration loop
```
def oaktree(gene_trees, n_quintets_per_gene, max_iterations=10, convergence_threshold=0):
    # Initialize with ASTRID-like distance method
    species_tree = initialize_species_tree(gene_trees)
    species_tree = estimate_branch_lengths(species_tree, ...)
    
    gene_weights = np.ones(len(gene_trees))  # uniform initial weights
    
    for iteration in range(max_iterations):
        # Sample quintets with current weights
        quintets = sample_all_quintets(gene_trees, n_quintets_per_gene, gene_weights)
        
        # Infer species tree topology
        new_species_tree = recursive_partition(all_taxa, quintets, species_tree)
        
        # Estimate branch lengths
        new_species_tree = optimize_branch_lengths(new_species_tree, quintets)
        
        # Check convergence (RF distance between successive trees)
        rf = robinson_foulds(species_tree, new_species_tree)
        if rf <= convergence_threshold:
            break
        
        species_tree = new_species_tree
        
        # Update gene tree weights
        scores = score_gene_trees(gene_trees, species_tree, ...)
        gene_weights = compute_gene_weights(scores, temperature=1.0 + iteration * 0.5)
    
    return species_tree
```

#### 4.5 — Initialization
- Use a fast initial species tree estimate to bootstrap the iteration:
  - Compute an average internode distance matrix from gene trees (as ASTRID does)
  - Build a neighbor-joining tree from this matrix
  - Estimate initial branch lengths using quartet-based coalescent unit estimation
- Implement `initialize_species_tree(gene_trees: list[Tree]) -> Tree`

### Tests
- `test_convergence_easy`: 20-taxon balanced tree, long branches, 500 gene trees → should converge in 1–2 iterations
- `test_convergence_hard`: 20-taxon tree with one very short internal branch (τ=0.1), 500 gene trees with estimation error → should converge in 3–6 iterations
- `test_iteration_improves_accuracy`: RF distance to true species tree should decrease (or not increase) with each iteration on 20 simulated datasets
- `test_weight_separation`: After iteration, gene trees simulated with high error should have lower weights than gene trees simulated with no error (create a mixed dataset where half the gene trees are correct and half have random NNI perturbations)
- `test_initialization_independence`: Starting from two different initial trees (NJ and random), the method should converge to the same (or very similar) species tree

### Acceptance criteria
- On 50 simulated 20-taxon datasets: (a) converges within 10 iterations on all, (b) final tree is at least as accurate as ASTRAL's tree on >80% of datasets
- Iteration never increases RF distance to the true tree by more than 2 on any single step (monotonic improvement is not guaranteed but catastrophic regression should not occur)
- Temperature scheduling produces meaningful weight differentiation — KL divergence between final weights and uniform weights > 0.1 on datasets with gene tree error

---

## Phase 5: CLI and integration

### Goal
Package as a usable command-line tool.

### Steps

#### 5.1 — Command-line interface
```bash
# Basic usage
oaktree -i gene_trees.nwk -o species_tree.nwk

# Options
oaktree -i gene_trees.nwk -o species_tree.nwk \
    --quintets-per-gene 200 \      # number of quintet samples per gene tree
    --max-iterations 10 \           # maximum EM iterations
    --convergence-threshold 0 \     # RF distance for convergence (0 = exact match)
    --seed 42 \                     # random seed for reproducibility
    --threads 4 \                   # parallel quintet sampling
    --verbose                       # print iteration progress
```

- Use `argparse` for CLI
- Implement `--threads` using `multiprocessing.Pool` for quintet sampling (embarrassingly parallel)

#### 5.2 — Output format
- Primary output: species tree in Newick format with branch lengths in coalescent units
- Optional outputs:
  - `--gene-weights weights.tsv`: per-gene-tree weights from final iteration
  - `--quintet-support support.tsv`: per-branch quintet support values (analogous to local posterior probability in ASTRAL)
  - `--log run.log`: iteration-by-iteration log with RF distance, total log-likelihood, and runtime

#### 5.3 — Input validation
- Verify all gene trees parse correctly
- Warn about taxa present in <50% of gene trees
- Error if fewer than 4 shared taxa
- Warn if fewer than 100 gene trees (insufficient signal for quintet-based inference)

### Tests
- `test_cli_basic`: Run the full CLI on a small test dataset and verify output tree is valid Newick
- `test_cli_reproducibility`: Same input + same seed → identical output
- `test_cli_missing_taxa`: Gene trees with missing taxa produce valid output with appropriate warnings

---

## Phase 6: Validation against ASTRAL and TreeQMC

### Goal
Demonstrate correctness and (potential) accuracy improvement through systematic simulation.

### Steps

#### 6.1 — Simulation framework
Use `msprime` to simulate gene trees under the MSC:
- Species tree topologies: balanced, caterpillar, random (birth-death)
- Number of taxa: 10, 20, 50, 100, 200
- Number of gene trees: 100, 200, 500, 1000, 2000
- ILS levels: low (all branches > 2 CU), moderate (shortest branch 0.5 CU), high (shortest branch 0.1 CU)
- Gene tree estimation error: none (true gene trees), low (1000 bp sequences → RAxML), high (200 bp sequences → RAxML)

#### 6.2 — Accuracy metrics
- Normalized Robinson-Foulds distance (nRF) to true species tree
- Quartet score relative to true species tree
- Branch-specific accuracy: for each true internal branch, is it recovered?

#### 6.3 — Comparison protocol
For each simulated dataset:
1. Run ASTRAL-III (or ASTER ASTRAL-IV)
2. Run wASTRAL (hybrid weighting)
3. Run TreeQMC-n2
4. Run wTreeQMC
5. Run OAKTREE with default parameters
6. Record nRF, quartet score, and runtime for each

#### 6.4 — Statistical analysis
- Paired Wilcoxon signed-rank tests with Bonferroni correction (as in the TreeQMC paper)
- Identify conditions where OAKTREE outperforms (expected: high ILS + high gene tree error)
- Identify conditions where OAKTREE underperforms (expected: very few gene trees where quintet signal is too sparse)

### Acceptance criteria
- OAKTREE matches ASTRAL accuracy (nRF within 0.02) on easy datasets (low ILS, true gene trees)
- OAKTREE shows statistically significant improvement over ASTRAL on at least one difficult condition (high ILS + gene tree error) with p < 0.01
- OAKTREE runtime is within 5× of ASTRAL on 100-taxon datasets (if much slower, profile and identify bottleneck)

---

## Dependency list

```
# requirements.txt
numpy>=1.24
scipy>=1.10
networkx>=3.0
treeswift>=1.1
msprime>=1.2       # for simulation/validation only
dendropy>=4.6      # for RF distance computation and cross-validation
matplotlib>=3.7    # for plotting results
```

---

## Agent execution order

1. **Phase 0** → Phase 1 → run all Phase 0+1 tests → fix until green
2. **Phase 2** → run Phase 2 tests using hardcoded likelihood values → fix until green
3. **Phase 3** → run Phase 3 tests → fix until green
4. **Phase 2+3 integration** → end-to-end test on 8-taxon simulated data → fix until green
5. **Phase 4** → run iteration tests on 20-taxon simulated data → fix until green
6. **Phase 5** → CLI tests → fix until green
7. **Phase 6** → systematic simulation study → analyze results → report

**Do not proceed to the next phase until all tests in the current phase pass.**

---

## Known difficult spots where extra care is needed

1. **Quintet topology canonicalization (Phase 0.2)**: The mapping from a 5-leaf tree to one of 15 topologies is error-prone. The standard approach uses sorted bipartition sets, but be very careful with the unrooted representation — a split {a,b}|{c,d,e} is the same as {c,d,e}|{a,b}. Always normalize to the smaller side.

2. **Coalescent history enumeration (Phase 1.1)**: This is the trickiest piece of math in the project. For a given species tree and gene tree, a valid coalescent history must place each gene tree coalescence in a species tree branch that is ancestral to all taxa descended from that coalescence. Off-by-one errors in branch assignment are common. Cross-validate against published tables in Degnan & Salter (2005) and Rosenberg (2002).

3. **Artificial taxon handling (Phase 2.4)**: When recursing, the artificial taxon representing the complement set must correctly aggregate information. The complement set's representative must participate in quintet sampling, but its contribution must be normalized. This is where the n2 normalization is essential. Getting this wrong produces trees that are systematically wrong on asymmetric species trees.

4. **Branch length estimation convergence (Phase 3.3)**: The MSC likelihood surface can be multimodal for very short branches. Always start the optimizer from the moment estimate, not from a random point. If optimization fails, fall back to the moment estimate.

5. **EM iteration stability (Phase 4.4)**: The temperature parameter β must be tuned carefully. Too aggressive (β >> 1 early) will zero out gene trees that are merely unlucky, not erroneous. Too conservative (β ≈ 0) provides no benefit over unweighted analysis. The schedule `β = 1.0 + iteration * 0.5` is a starting point; expose it as a tunable parameter.
