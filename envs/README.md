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

## Tier 2 — deep reference mapping
- `scvi`: `scvi-tools` (scANVI, scArches) — `pip install scvi-tools` (MPS/CPU).
- `r-bioc`: R + Seurat + Azimuth + Symphony + SingleR + scmap + scPred
  (e.g. via conda `r-base`, `bioconductor-*`). Adapters shell out via the runner.

## Tier 3 — foundation models
- `foundation`: PyTorch + scGPT / scBERT / scDeepSort / TOSICA, plus pretrained
  weights. Prefer MPS on Apple Silicon; expect to subsample queries.

Each env should be captured as a lockfile (e.g. `envs/scvi.lock.txt`) for repro.
Adapters for these tiers are added under `benchmark/adapters/` as they come online.
