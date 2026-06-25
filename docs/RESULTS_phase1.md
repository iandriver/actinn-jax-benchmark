# Phase 1 results — lung (first real-data pass)

**Setup (first pass, deliberately small for fast iteration):** Mac M5 Pro, CPU,
full machine, math threads capped at 4 per method, `repeats: 1`. Reference
subsampled per label (krasnow 600/label; HCLA 300/label; cross query capped
1000/label). Single repeat → timings are indicative, not averaged.

## krasnow within-dataset CV (13,923 train / 4,628 test, 46 cell types)

| method | accuracy | macro-F1 | ontology | fit (s) | predict (s) | peak mem (MB) |
|---|---|---|---|---|---|---|
| actinn-jax | 0.904 | **0.900** | 0.931 | 63.6 | 5.1 | 12,294 |
| celltypist | 0.902 | 0.896 | 0.929 | 153.6 | 0.86 | 2,566 |
| svm (SGD)  | 0.901 | 0.894 | 0.929 | 43.2 | 0.20 | 2,504 |
| knn        | 0.901 | 0.882 | 0.932 | **4.4** | 0.39 | 2,509 |

## HCLA → krasnow cross-atlas (14,390 train / 26,077 query)

| method | exact acc | macro-F1 | ontology | fit (s) | predict (s) | peak mem (MB) |
|---|---|---|---|---|---|---|
| actinn-jax | 0.376 | 0.183 | 0.791 | 78.9 | 12.6 | 17,712 |
| celltypist | 0.372 | 0.180 | 0.795 | 86.5 | 1.24 | 4,223 |
| svm (SGD)  | 0.368 | 0.180 | 0.792 | 58.9 | 0.68 | 3,256 |
| knn        | 0.365 | 0.176 | 0.787 | 3.5 | 1.51 | 3,760 |

## Reading

- **Accuracy is a near-tie** (~0.90 within-atlas; cross-atlas exact ~0.37 but
  ontology-concordance ~0.79 for all). Consistent with the survey: simple
  task-specific methods are competitive, and no method dominates. Cross-atlas
  exact accuracy is depressed by label-vocabulary mismatch, not errors — ontology
  concordance is the fairer number.
- **Runtime separates them.** kNN is the fastest to fit (seconds) but has the
  weakest macro-F1 (worst on rare types). SVM (SGD) is the best accuracy/runtime
  balance here. CellTypist is the slowest to fit. actinn-jax matches the best on
  macro-F1.
- **actinn-jax memory is high (12–18 GB)** because `train_reference` densifies the
  full shared-gene matrix; predict is already chunked. → concrete optimization
  target for the library (chunk/stream training).
- **predict time** for actinn-jax (5–13 s) is inflated by per-process JAX JIT
  warmup (each method runs in a fresh subprocess). Warm/amortized predict is ~ms;
  the cached-ReferenceModel path is its real strength but isn't exercised by a
  single cold fit→predict.

## Caveats / next

- `repeats: 1` → bump to 3 for averaged timing; small subsamples → scale up
  reference sizes for final accuracy.
- Add PBMC/immune datasets (closely related subtypes stress macro-F1).
- Add original ACTINN (TF) baseline to quantify the rewrite's speedup; add R
  methods (SingleR, scmap, scPred) and Tier-2/3 in isolated envs.
