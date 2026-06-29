# actinn-jax-benchmark

## ⭐ Headline: foundation-model-shaped, CPU-fast cell typing — **[docs/MODEL_FLOW.md](docs/MODEL_FLOW.md)**

A two-stage flow that uses the **scPRINT** foundation model *once, offline, on a GPU*
to **discover cell-type structure**, then trains a small **[actinn-jax](https://github.com/iandriver/actinn-jax)**
classifier that annotates new data in **milliseconds on a CPU**.

- **scPRINT-discovered coarse→fine hierarchy beats a flat classifier on all 3 datasets**
  (lung 46 types, blood+gut 86, multi-organ Tabula Sapiens 83/8 organs), ties an expert
  biological hierarchy, and beats a random-grouping control.
- scPRINT's discovered groups **recover biological lineage** (ARI **0.54**), not organ
  (0.02) — it groups by cell-type identity, which is the point.
- **Inference is pure CPU, <1 s** for thousands of cells, in ~2 GB RAM — vs scPRINT as a
  predictor (~1 s/cell on CPU, and weak zero-shot). **Use its embeddings, not its labels.**
- The GPU step is cached: committed [`data/embeddings/`](data/embeddings) (7–16 MB each)
  let you reproduce the structure with **no GPU** — `python benchmark/explore/discover_hierarchy.py blood_gut --bio Lineage`.

- **Depth, not just breadth:** the same fast CPU model resolves *within*-cell-type
  structure — **hepatocyte zonation** (portal/mid/central) at ~0.99 within-1-zone,
  generalizing across donors and even across independent datasets (GSE158723↔GSE136103).
  See **[docs/ZONATION.md](docs/ZONATION.md)**.

Read the mini-paper: **[docs/MODEL_FLOW.md](docs/MODEL_FLOW.md)** · two-stage details in
[docs/TWO_STAGE.md](docs/TWO_STAGE.md) · zonation in [docs/ZONATION.md](docs/ZONATION.md).

---

The rest of this repo is a **neutral benchmark** of reference-based single-cell
cell-type annotation methods, comparing **accuracy × runtime × memory** on commodity
hardware (Apple Silicon, CPU-first).
[actinn-jax](https://github.com/iandriver/actinn-jax) is one method among many.

## Why

The most rigorous recent accuracy benchmark
([Huang et al. 2024, *Brief. Bioinform.*](https://academic.oup.com/bib/article/25/5/bbae392/7730135))
reports **no runtime or memory**; the classic one that did
([Abdelaal et al. 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1795-z))
predates foundation models. This repo measures both, across three method tiers, and
reports an **accuracy × runtime Pareto** view.

See [docs/METHODS_SURVEY.md](docs/METHODS_SURVEY.md) for a cited, adversarially-verified
survey of the method landscape that motivates the shortlist.

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

## Results

- **[docs/RESULTS.md](docs/RESULTS.md)** — final Tier-1 numbers on lung (krasnow CV +
  HCLA→krasnow cross-atlas) and PBMC/immune (pbmc3k), mean of 3 repeats.
- **[docs/RESULTS_actinn_orig.md](docs/RESULTS_actinn_orig.md)** — actinn-jax vs the
  original TensorFlow ACTINN (3.3× faster, 3.5× less memory single-run).
- **[docs/METHODS_SURVEY.md](docs/METHODS_SURVEY.md)** — cited methods landscape.
- **[docs/TWO_STAGE.md](docs/TWO_STAGE.md)** — scPRINT (broad) shaping a fast CPU classifier: a scPRINT-discovered coarse→fine hierarchy beats flat (macro-F1 0.71 vs 0.68), pure-CPU inference.

Tier-1 (classical) complete; deep + foundation tiers next. See the full
[benchmark plan](https://github.com/iandriver/actinn-jax/blob/main/BENCHMARK_PLAN.md).

## License

MIT.
