# scPRINT on CPU/MPS, and a fast CPU classifier from curated data

Two related explorations (scripts in [`benchmark/explore/`](../benchmark/explore)):
1. Can the **scPRINT** foundation model run locally (CPU / Apple MPS) for cell-type
   annotation, and how good/fast is it?
2. Can we skip the heavy model and instead use the **same curated, ontology-labeled
   data** (CELLxGENE / lamindb) to train a classifier that runs fast on a non-GPU
   machine?

## 1. scPRINT ([cantinilab/scPRINT](https://github.com/cantinilab/scPRINT))

A transformer foundation model (≈2M–100M params) pretrained on ~50M CELLxGENE cells.
Primary purpose is gene-network inference / embeddings; cell-type classification is a
secondary zero-shot head. Needs **lamindb** for its gene/cell ontologies.

**Getting it to run locally was non-trivial** (documented in `scprint_run.py`):
- `pip install scprint`; init lamindb anonymously (`lamindb_setup.init(modules="bionty")`).
- The bionty registries start **empty** — you must `populate_my_ontology()` for **both
  human and mouse** (the model uses both organisms in its Collator) or preprocessing
  fails with `Organism ... NoneType`.
- The published `small.ckpt` URL 404s; current checkpoints are `medium-v1.5.ckpt`
  (221 MB, smallest), `medium-v1.ckpt`, `large-v1.ckpt`, etc.
- **MPS doesn't work out of the box**: the Embedder calls
  `torch.autocast(device_type="mps")`, which PyTorch doesn't support. A one-line
  monkeypatch (no-op autocast on MPS, run fp32) fixes it.

**Results** — medium-v1.5, 400-cell krasnow lung subset:

| device | time (400 cells) | notes |
|---|---|---|
| CPU | **421.9 s** (~1 cell/s) | runs out of the box; impractically slow |
| Apple MPS | **13.9 s** (~30× faster) | needs the autocast monkeypatch |

Zero-shot accuracy vs krasnow's Cell-Ontology labels:

| metric | value |
|---|---|
| exact CL-id match | 0.06 |
| **ontology-lineage concordance** | **0.32** |

So scPRINT *runs* (and the Apple GPU is usable with a patch, ~30× faster than CPU),
but its **zero-shot cell-typing is weak here** (0.32 lineage) — far below the
supervised methods (0.79–0.93, see [RESULTS.md](RESULTS.md)). Expected: it's a
gene-network model, the classifier is untuned/zero-shot, on a small sample with the
medium checkpoint. Not competitive as a drop-in annotator on this hardware.

### As a formal benchmark method

scPRINT is wired into the harness (`benchmark/adapters/scprint_adapter.py`, runs in
`.venv-scprint` on MPS; `configs/scprint.yaml`). Because it predicts Cell-Ontology ids
rather than the dataset's label vocabulary, the harness scores it in CL-id space
(`Predictions.label_cl`) and ontology concordance is the comparable metric. On a
1,484-cell CL-labeled krasnow subset ([results_scprint.csv](results_scprint.csv)):

| method | tier | ontology | exact | predict (s) | device |
|---|---|---|---|---|---|
| actinn-jax | classical | **0.912** | 0.879 | **0.15** | CPU |
| scprint    | foundation | 0.218 | 0.026 | 32.7 | MPS |

scPRINT zero-shot is ~200× slower at predict and far less accurate than a small model
trained on the reference. (exact = exact-label for actinn-jax, exact-CL-id for scPRINT.)

## 2. Fast CPU classifier from curated atlas data

The better use of the *underlying* curated data: train a lightweight classifier on it.
Atlases like the **Human Lung Cell Atlas** (HCLA / Sikkema) are themselves integrations
of dozens of datasets with **harmonized Cell Ontology labels** — exactly the curated,
ontology-standardized data lamindb/CELLxGENE manage.

**actinn-jax trained on HCLA** (50 CL-labeled cell types; `atlas_classifier.py`):

| | value |
|---|---|
| train (CPU) | **34.9 s** (10,794 cells, 50 types) |
| predict | 0.49 s (3,596 cells) |
| peak memory | 4.1 GB |
| held-out accuracy | **0.913** |
| macro-F1 | 0.915 |
| ontology concordance | **0.933** |

**The contrast is the point:** a 35-second CPU training run on curated atlas data
yields a classifier with **0.93 ontology concordance** — versus the foundation model
needing a GPU (or ~1 cell/s on CPU) and scoring **0.32** zero-shot. For routine
cell-type annotation on a non-GPU machine, training a small model on curated,
ontology-labeled reference data is dramatically cheaper and more accurate.

### On pulling data directly from CELLxGENE census / lamindb

`census_fetch.py` + `census_train.py` show the direct path:
`cellxgene_census.get_anndata(...)` to pull a stratified, ontology-labeled blood
reference, then train actinn-jax on it. The census query found **13.8M** human blood
cells across 159 types — but **scattered-coordinate reads over the network were
prohibitively slow on a home connection** (a stratified subsample did not finish in
~1 h). For local work, using an already-downloaded curated atlas (as above) is far
faster; for production, build the reference once on a fast/cloud connection (or pull a
single contiguous dataset by `dataset_id` rather than a scattered sample) and cache it.

## Takeaway

scPRINT is interesting but GPU-bound and not a strong zero-shot annotator here. The
pragmatic recipe for fast, accurate, **non-GPU** cell typing is: take curated,
Cell-Ontology-labeled reference data (atlas or CELLxGENE/lamindb) and train a small
model (actinn-jax) on it — seconds of CPU, ~0.93 ontology concordance.
