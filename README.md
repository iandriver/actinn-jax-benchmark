# actinn-jax-benchmark

A comparison of methods for labelling cells in single-cell data by mapping them onto an
already-labelled reference. It measures accuracy, runtime and memory together, on a
laptop, because those are the numbers that decide what you can actually run.

[actinn-jax](https://github.com/iandriver/actinn-jax) is one of the methods tested here.
This repository is where its claims get checked.

## What we found

Thirteen methods across eight splits of seven datasets, 8 to 151 cell types.

Accuracy barely separates the top methods. actinn-jax averages 0.831, against 0.839 for a
tuned linear pipeline, 0.833 for scArches and 0.832 for scANVI. That spread is smaller
than the run-to-run noise of the methods that use random initialisation, so treating it as
a ranking would be reading noise.

Cost separates them by two orders of magnitude. actinn-jax predicts a query in 0.54
seconds where scANVI takes 66.7, about 123 times faster. No method in the panel beats it
on both accuracy and speed.

It does not win everywhere. On the Allen brain split, which asks for 151 cluster-level
labels, actinn-jax places tenth of eleven. Fine-grained taxonomies are its weak point and
the report says so.

The full report is [docs/PAPER.md](docs/PAPER.md). Figures are in
[docs/figures/](docs/figures).

## Checked against an outside benchmark

Running our method through [Open Problems](https://openproblems.bio/benchmarks/label_projection),
using their datasets, their splits and their metrics, and scoring against their published
v2.0.0 leaderboard: actinn-jax places 3rd of 17 on accuracy, and 1st among the methods
that complete all six datasets. It beats the PCA-space `mlp` that is closest to it in
design, and posts the best accuracy on the hardest dataset, tabula_sapiens, with 160 cell
types.

Details in [docs/OPENPROBLEMS.md](docs/OPENPROBLEMS.md).

## The two-stage idea

Foundation models are good at understanding cell-type structure and slow at labelling
cells. So use them for the first job only.

Run [scPRINT](https://github.com/cantinilab/scPRINT) once, offline, on a GPU, to work out
how cell types group together. Train a small actinn-jax classifier on that grouping. From
then on, labelling new data takes milliseconds on a CPU.

What we measured:

- The scPRINT-derived grouping beats a flat classifier on all three datasets we tried
  (lung with 46 types, blood and gut with 86, Tabula Sapiens with 83 across 8 organs). It
  matches a hierarchy an expert wrote by hand, and beats a random grouping.
- The groups it finds track biological lineage (ARI 0.54), not which organ the cells came
  from (0.02). Grouping by cell identity is the point.
- Labelling is CPU-only and takes under a second for thousands of cells in about 2 GB of
  RAM. Using scPRINT itself as the labeller takes roughly a second per cell on CPU, and
  its zero-shot labels are weak. Use its embeddings, not its predictions.
- The GPU step is cached. The embeddings are committed in
  [data/embeddings/](data/embeddings) at 7 to 16 MB each, so you can reproduce the
  structure with no GPU:
  `python benchmark/explore/discover_hierarchy.py blood_gut --bio Lineage`

The same fast CPU model also resolves structure inside a cell type. It places hepatocytes
along the portal to central axis at about 0.99 within-one-zone accuracy, and that holds
across donors and across two independent datasets (GSE158723 and GSE136103). See
[docs/ZONATION.md](docs/ZONATION.md).

Write-up: [docs/MODEL_FLOW.md](docs/MODEL_FLOW.md), with details in
[docs/TWO_STAGE.md](docs/TWO_STAGE.md).

## Why this benchmark exists

The most careful recent accuracy comparison
([Huang et al. 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135)) reports
no runtime and no memory. The classic one that did
([Abdelaal et al. 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1795-z))
predates foundation models. This repository measures all three, and reports accuracy
against runtime as a trade-off rather than a ranking.

[docs/METHODS_SURVEY.md](docs/METHODS_SURVEY.md) is a cited survey of the field, which is
where the shortlist came from.

## Running it

```bash
pip install -e .
python -m benchmark.driver configs/smoke.yaml
```

The smoke config runs end to end on synthetic data and takes a minute. It is the fastest
way to check your install works.

A real run needs the data files and, for lineage-aware scoring, a Cell Ontology file:

```bash
curl -L -o /tmp/cl-basic.obo http://purl.obolibrary.org/obo/cl/cl-basic.obo
python -m benchmark.driver configs/lung.yaml
```

Results land in `results/<name>/results.csv`, one row per dataset, method and repeat.

To reproduce the report: `configs/paper.yaml`, then `configs/paper_baselines.yaml`,
`configs/paper_brain.yaml` and `configs/paper_brain_cluster.yaml`.

## How it is built

Each method runs in a separate process, and where necessary a separate environment, so
that R, PyTorch, TensorFlow and JAX never have to agree on dependency versions. A method
reads `h5ad` and writes parquet predictions plus a json of metrics.

Every method implements the same two calls, `fit(ref, label)` and `predict(query)`, in
[benchmark/adapters/](benchmark/adapters).

Timing is reported for the hardware it actually ran on, which is Apple Silicon CPU and
MPS. Methods designed for CUDA GPUs are timed on a laptop with no CUDA, which is a real
scenario for many labs but is not the hardware their authors assume. A cloud GPU run is
planned ([docs/AWS_GPU.md](docs/AWS_GPU.md)).

## Adding a method

Subclass `AnnotationMethod`, write `fit` and `predict`, and add `@register`. See
[benchmark/adapters/svm_adapter.py](benchmark/adapters/svm_adapter.py) for the shortest
example. Methods with awkward dependencies get their own environment under
[envs/](envs).

## Methods tested

Classical, on CPU: actinn-jax, the original TensorFlow ACTINN, CellTypist, SingleR,
scmap-cluster, SVM, kNN, scTOP, and a tuned linear pipeline.

Reference mapping with deep models: scANVI, scArches, ProtoCloud.

Foundation model: scPRINT.

That is the thirteen in the report. A fourteenth reference, distilled from Pan-human
Azimuth, is evaluated separately in
[docs/PANHUMAN_DISTILL.md](docs/PANHUMAN_DISTILL.md).

## Reports

- [docs/PAPER.md](docs/PAPER.md), the full comparison
- [docs/OPENPROBLEMS.md](docs/OPENPROBLEMS.md), the external benchmark
- [docs/MODEL_FLOW.md](docs/MODEL_FLOW.md), the two-stage method
- [docs/TWO_STAGE.md](docs/TWO_STAGE.md), a scPRINT-derived hierarchy beats a flat
  classifier, macro-F1 0.71 against 0.68, with CPU-only inference
- [docs/ZONATION.md](docs/ZONATION.md), hepatocyte zonation
- [docs/RESULTS.md](docs/RESULTS.md), lung and PBMC numbers, mean of 3 repeats
- [docs/RESULTS_actinn_orig.md](docs/RESULTS_actinn_orig.md), actinn-jax against the
  original TensorFlow ACTINN: 3.3 times faster, 3.5 times less memory
- [docs/METHODS_SURVEY.md](docs/METHODS_SURVEY.md), the field
- [docs/REFINE.md](docs/REFINE.md), what narrowing a broad reference does and does not fix
- [docs/HLICA_LIVER.md](docs/HLICA_LIVER.md), building a focused liver reference
- [docs/PANHUMAN_DISTILL.md](docs/PANHUMAN_DISTILL.md), distilling Pan-human Azimuth into
  an actinn-jax reference. On a withheld liver study the 324-class distilled model scores
  0.406 ontology concordance, against 0.338 for the census-built reference and 0.408 for
  the teacher it copied, at 6 to 9 times the teacher's throughput. Built from unlabelled
  data, no GPU.
- [docs/UPDATE_BROAD_REFERENCE.md](docs/UPDATE_BROAD_REFERENCE.md), rebuilding the
  census-wide reference, including the check that scores a rebuilt model before you keep
  it

## License

MIT.
