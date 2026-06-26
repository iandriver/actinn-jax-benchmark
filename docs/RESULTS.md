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
