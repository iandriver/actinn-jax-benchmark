# Per-method environments

Heavy or conflicting methods run in their own isolated environments; the driver
invokes each via subprocess, so dependency sets never clash (R vs Python, jax vs
torch vs TF). Point a method at its interpreter in the config:

```yaml
methods:
  - name: scanvi
    python: /path/to/envs/scvi/bin/python
```

## Tier 1 — classical (core env)
Installed by `pip install -e .` at the repo root: actinn-jax, CellTypist, scikit-learn
(SVM, kNN). R-based classical methods (SingleR, scmap, scPred) use the R env below.

### `actinn-orig` — original TensorFlow ACTINN (the baseline actinn-jax replaced)
Vendored under `benchmark/vendor/actinn_original/`. Runs in its own TF env:
```
uv venv --python 3.11 .venv-tf
uv pip install --python .venv-tf/bin/python "tensorflow==2.15.*" scanpy anndata pandas scipy h5py pyarrow pyyaml psutil
```
**Pin TF 2.15** — the original uses `tf.compat.v1` graph mode + multiple `Session`s,
which **deadlocks (hangs at 0% CPU)** on TF 2.21. Point the method at it in the config:
```yaml
methods:
  - name: actinn-orig
    python: .venv-tf/bin/python
```

### R classical methods — SingleR, scmap (project R library `.Rlib`)
System R (4.6) + a project-local library so system R stays clean:
```
mkdir -p .Rlib
R_LIBS_USER=$PWD/.Rlib Rscript -e '.libPaths(Sys.getenv("R_LIBS_USER")); \
  if(!requireNamespace("BiocManager",quietly=TRUE)) install.packages("BiocManager"); \
  BiocManager::install(c("SingleR","SingleCellExperiment","scuttle","scmap"), \
    update=FALSE, ask=FALSE, lib=Sys.getenv("R_LIBS_USER"))'
```
The Python adapters (`benchmark/adapters/r_adapter.py`) export counts as `.mtx` and
shell out to `benchmark/r/run_r_method.R`. No config `python:` needed — they run from
the core env and call `Rscript` (set `R_LIBS` if `.Rlib` isn't at the repo root).

**scPred is currently NOT installable** here: it is GitHub-only and calls
`harmony::HarmonyMatrix`, which modern harmony (>=1.0) no longer exports, so it fails
to load. Pinning harmony 0.1.1 did not resolve it cleanly. Left as a known gap.

## Tier 2 — deep reference mapping
- `.venv-scvi`: `scvi-tools` (scANVI, scArches) — `uv pip install scvi-tools scanpy
  anndata pyarrow pyyaml psutil`. Uses Apple **MPS** when available. Point the methods
  at it via `python: .venv-scvi/bin/python` in the config.
- `r-bioc` (`.Rlib`): Symphony + Azimuth (Seurat) for the remaining deep methods —
  same R bridge pattern. (Not yet wired.)

## Tier 3 — foundation models
- `foundation`: PyTorch + scGPT / scBERT / scDeepSort / TOSICA, plus pretrained
  weights. Prefer MPS on Apple Silicon; expect to subsample queries.

Each env should be captured as a lockfile (e.g. `envs/scvi.lock.txt`) for repro.
Adapters for these tiers are added under `benchmark/adapters/` as they come online.
