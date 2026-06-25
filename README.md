# actinn-jax-benchmark

A modern, **neutral** benchmark of reference-based single-cell **cell-type
annotation** methods, comparing **accuracy × runtime × memory** on commodity
hardware (Apple Silicon, CPU-first). Companion to
[actinn-jax](https://github.com/iandriver/actinn-jax) — which is included as one
method among many, not the centerpiece.

## Why

The most rigorous recent accuracy benchmark
([Huang et al. 2024, *Brief. Bioinform.*](https://academic.oup.com/bib/article/25/5/bbae392/7730135))
reports **no runtime or memory**; the classic one that did
([Abdelaal et al. 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1795-z))
predates foundation models. This repo measures both, across three method tiers, and
reports an **accuracy × runtime Pareto** view.

## Design

- **Subprocess isolation** — each method runs via `benchmark.runner` in its own
  (optionally separate) environment; I/O is standardized (`h5ad` in → parquet
  predictions + json metrics out). This sidesteps R/Python/jax/torch/TF conflicts.
- **Uniform adapters** — every method implements `fit(ref, label)` / `predict(query)`
  ([`benchmark/adapters/`](benchmark/adapters)).
- **Honest device reporting** — CPU/MPS only here; GPU-native methods are timed on
  Apple hardware (a no-CUDA laptop scenario). A cloud-GPU run is planned but deferred
  (see [docs/AWS_GPU.md](docs/AWS_GPU.md)).

## Method tiers

- **Tier 1 — classical (CPU):** actinn-jax, original ACTINN, CellTypist, SingleR,
  scmap, scPred, SVM, kNN. *(SVM, kNN, CellTypist, actinn-jax implemented; R methods
  and original ACTINN to follow.)*
- **Tier 2 — deep reference mapping:** scANVI, scArches, Symphony, Azimuth.
- **Tier 3 — foundation models:** scGPT, scBERT, scDeepSort, TOSICA.

## Quick start

```bash
pip install -e .                       # core + Tier-1 python deps
python -m benchmark.driver configs/smoke.yaml     # synthetic end-to-end smoke test
```

Lung atlas run (needs the two h5ad files + a Cell Ontology OBO for lineage scoring):

```bash
curl -L -o /tmp/cl-basic.obo http://purl.obolibrary.org/obo/cl/cl-basic.obo
python -m benchmark.driver configs/lung.yaml
```

Results are written to `results/<name>/results.csv` (one tidy row per
dataset × method × repeat).

## Adding a method

Subclass `AnnotationMethod`, implement `fit`/`predict`, and `@register` it
(see [`benchmark/adapters/svm_adapter.py`](benchmark/adapters/svm_adapter.py)).
Heavier methods get their own environment under [`envs/`](envs).

## Status

Phase 0 (harness + Tier-1 skeleton) — see the full
[benchmark plan](https://github.com/iandriver/actinn-jax/blob/main/BENCHMARK_PLAN.md).

## License

MIT.
