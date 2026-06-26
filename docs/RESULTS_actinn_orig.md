# actinn-jax vs original TensorFlow ACTINN

Head-to-head of the JAX reimplementation against the original TF1 (`tf.compat.v1`
graph-mode) ACTINN it replaced. Same machine (Mac M5 Pro, CPU), same data, each run
as an isolated subprocess via `benchmark.runner`. Config: `configs/lung_orig.yaml`.

## Single train + predict — krasnow within-CV (13,923 train / 4,628 test, 46 types)

| method | total time | fit | predict | peak mem | accuracy |
|---|---|---|---|---|---|
| **original ACTINN (TF1)** | 77.9 s | (coupled) | 77.8 s | 9,559 MB | 0.904 |
| **actinn-jax** | **23.8 s** | 23.5 s | 0.2 s | **2,713 MB** | 0.903 |
| **factor** | **3.3× faster** | — | — | **3.5× less mem** | tie |

Accuracy is identical; actinn-jax is ~3× faster and ~3.5× lighter on a single run.
(The original couples training and prediction in one call — joint normalization over
train+query, densified via `adata.to_df()` — so its time is reported under predict.)

## Reference mapping — the real gap (train once, map many)

The original **retrains and re-preprocesses on every query** (no model caching).
actinn-jax trains once and maps each additional query from the cached
`ReferenceModel` in ~0.2 s. Mapping *K* queries against one reference:

| K queries | original (77.9 s each) | actinn-jax (23.5 s once + 0.2 s each) | speedup |
|---|---|---|---|
| 1   | 78 s    | 24 s   | 3.3× |
| 10  | 779 s   | 25.5 s | **~30×** |
| 100 | 7,790 s | 43.5 s | **~175×** |

## Notes

- **TF version matters:** the original uses `tf.compat.v1` graph mode with multiple
  `Session`s, which **deadlocks (hangs at 0 % CPU) on TensorFlow 2.21**. Pin
  **TF 2.15** (see `envs/README.md`); that is what these numbers use.
- The original is vendored verbatim from the pre-rewrite package under
  `benchmark/vendor/actinn_original/` — not modified, so this is a faithful baseline.
- Single-run speedup (3.3×) understates the benefit; the cached reference-mapping
  path is where the rewrite pays off, plus the ~3.5× memory reduction.
