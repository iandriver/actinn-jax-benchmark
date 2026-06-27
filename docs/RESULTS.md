# Benchmark results (final)

Mac M5 Pro, CPU. Each method runs as an isolated subprocess via `benchmark.runner`;
timings are the **mean of 3 repeats**. Tier-1 classical methods: actinn-jax,
CellTypist (HVG + SGD), SVM (SGD-hinge), kNN-on-PCA. See
[RESULTS_actinn_orig.md](RESULTS_actinn_orig.md) for the original-TF-ACTINN baseline.
Raw tables: [results_lung_final.csv](results_lung_final.csv),
[results_pbmc.csv](results_pbmc.csv).

## Headline

- **Accuracy is a near-tie** across methods on every dataset; **macro-F1** (rare-type
  sensitivity) and **runtime** are what separate them.
- **CellTypist** has the edge on rare-type macro-F1 (it was purpose-built for immune
  annotation); **kNN** is fastest but weakest on rare types; **SVM (SGD)** is the best
  accuracy/runtime balance; **actinn-jax** is mid-pack on accuracy and, after the
  sparse/per-minibatch rewrite, no longer a memory outlier.
- Cross-atlas exact accuracy is low (label-vocabulary mismatch) but **ontology
  concordance ~0.83** — the fair number.

## Lung — krasnow within-dataset CV (24,558 train / 8,169 test, 46 types)

| method | accuracy | macro-F1 | ontology | fit (s) | predict (s) | mem (MB) |
|---|---|---|---|---|---|---|
| actinn-jax | 0.910 | 0.888 | 0.931 | 41.4 | 0.48 | 4,103 |
| celltypist | 0.911 | **0.896** | 0.933 | 83.3 | 0.53 | 4,058 |
| svm        | 0.911 | 0.886 | 0.932 | 46.7 | 0.15 | 3,709 |
| knn        | **0.918** | 0.880 | 0.941 | **2.4** | 0.43 | 3,970 |

kNN has the highest *overall* accuracy but the lowest macro-F1 — it nails the common
types and misses rare ones (the accuracy-vs-macro-F1 tradeoff).

### R methods on the same split (SingleR, scmap — single run)

| method | accuracy | macro-F1 | fit (s) | predict (s) | mem (MB) |
|---|---|---|---|---|---|
| singler       | 0.876 | 0.852 | 1.1 | **165.4** | 3,615 |
| scmap-cluster | 0.851 | 0.801 | 0.9 | 26.0 | **13,159** |

On the fine-grained 46-type lung data the R methods drop **below** the Python
classical methods (~0.91) and are much costlier — SingleR's correlation scan is
165 s on 8k query cells, and scmap's index is memory-heavy (13 GB). This is the
*opposite* of PBMC (8 broad types), where SingleR led: correlation/projection
methods degrade on many closely-related fine types.

## Lung — HCLA → krasnow cross-atlas (22,859 train / 37,688 query)

| method | exact acc | macro-F1 | ontology | fit (s) | predict (s) | mem (MB) |
|---|---|---|---|---|---|---|
| actinn-jax | 0.387 | **0.185** | 0.825 | 43.8 | 2.01 | 9,709 |
| celltypist | 0.384 | 0.180 | 0.831 | 95.2 | 0.99 | 5,788 |
| svm        | 0.382 | 0.179 | **0.831** | 59.6 | 0.60 | 4,863 |
| knn        | 0.374 | 0.175 | 0.822 | **2.4** | 1.64 | 5,597 |

All methods agree ~0.83 by ontology. actinn-jax's higher memory here is the cold
predict densifying selected genes for 37.7k query cells in one chunk — tunable via
`chunk_size`.

## PBMC / immune — pbmc3k within-dataset CV (≈1,979 train / 659 test, 8 types)

The imbalanced immune subtypes (Megakaryocytes: 15 cells) **spread the methods apart
on macro-F1**, where lung had them tied:

| method | accuracy | macro-F1 | fit (s) | predict (s) | mem (MB) |
|---|---|---|---|---|---|
| celltypist | 0.919 | **0.880** | 2.2 | 0.51 | 609 |
| svm        | 0.907 | 0.844 | 1.9 | 0.02 | 557 |
| actinn-jax | 0.913 | 0.795 | 5.3 | 0.09 | 893 |
| knn        | 0.907 | **0.730** | 1.6 | 0.04 | 557 |

CellTypist clearly leads on rare-type macro-F1; kNN collapses on the rare types.
This matches the survey's expectation that immune subtypes stress methods that
lung's broad types do not.

## All tiers — PBMC (classical + R + deep), pbmc3k within-CV (mean of 2 reps)

The full method roster across three tiers (deep methods on Apple **MPS**), sorted by
macro-F1. Raw table: [results_pbmc_all.csv](results_pbmc_all.csv).

| method | tier | accuracy | macro-F1 | fit (s) | predict (s) | mem (MB) |
|---|---|---|---|---|---|---|
| singler       | classical (R) | 0.925 | **0.884** | 0.1 | 4.4 | 1,353 |
| scarches      | deep (MPS) | **0.936** | 0.880 | 24.0 | 3.6 | 997 |
| celltypist    | classical | 0.916 | 0.874 | 2.3 | 0.5 | 612 |
| scanvi        | deep (MPS) | 0.919 | 0.872 | 0.0 | 24.2 | 1,053 |
| svm           | classical | 0.906 | 0.844 | 2.2 | 0.03 | 561 |
| actinn-jax    | classical | 0.913 | 0.795 | 5.8 | 0.10 | 885 |
| scmap-cluster | classical (R) | 0.871 | 0.778 | 0.1 | 4.6 | 2,826 |
| knn           | classical | 0.907 | 0.739 | 1.9 | 0.05 | 561 |

Reading:
- **Deep methods (scArches/scANVI) take the top accuracy (0.936/0.919)** but cost
  ~24 s (scArches fit / scANVI predict) vs sub-second for the classical classifiers —
  the central accuracy×runtime tradeoff. Runtime spans ~3 orders of magnitude
  (0.03 s → 24 s) for ≤ 0.16 accuracy difference.
- **SingleR is the standout classical method** on PBMC (0.925 accuracy, best macro-F1
  0.884, ~4 s) — correlation-to-reference works very well on clean immune types.
- scANVI vs scArches: scANVI is transductive (work in `predict`); scArches trains the
  reference once (`fit`) then maps queries by surgery in ~3.6 s — the right shape for
  repeated reference mapping.

### Method/tooling notes
- R methods (SingleR, scmap) run via a `.mtx` → `Rscript` bridge against a project R
  library; **scPred is excluded** — it is GitHub-only and calls the removed
  `harmony::HarmonyMatrix`, so it no longer installs (see `envs/README.md`).
- Deep methods use scvi-tools on Apple MPS in `.venv-scvi`; modest epochs
  (SCVI 40 / SCANVI 20 / query 40) for tractable laptop timing — not tuned for max
  accuracy. A CUDA run (see `docs/AWS_GPU.md`) would speed these up substantially.

## Memory: effect of the actinn-jax optimization

The sparse-preprocessing + per-minibatch-densification rewrite (see the actinn-jax
repo) cut training memory dramatically — on krasnow within-CV, actinn-jax peak memory
fell **12.3 GB → 4.1 GB**, bringing it in line with the sklearn methods (was the
clear outlier in the first pass).

## Caveats / next

- Single machine, CPU only; deep (scANVI/scArches/Symphony/Azimuth) and foundation
  (scGPT/scBERT) tiers not yet run — would need isolated envs (and likely a cloud GPU
  for fair foundation-model timing; see `docs/AWS_GPU.md`).
- pbmc3k labels are broad (louvain); a fine-grained immune set (e.g. experimentally
  labeled T-cell subtypes) would stress macro-F1 further.
- Original ACTINN (TF) head-to-head is in `RESULTS_actinn_orig.md`.
