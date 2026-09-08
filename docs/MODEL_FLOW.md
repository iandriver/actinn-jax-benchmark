# Foundation-model-shaped, CPU-fast cell-type annotation

*A two-stage flow that uses the scPRINT foundation model once, offline, to discover
cell-type structure, then trains a small JAX classifier that annotates new data in
milliseconds on a CPU, with no GPU at inference.*

## Abstract

Single-cell foundation models such as scPRINT carry broad, cross-tissue knowledge but are
slow and GPU-bound, and used zero-shot they are weak annotators, reaching ontology
concordance of 0.22 on lung here. Small supervised classifiers are fast and accurate but
blind to structure beyond their reference. We combine them: run scPRINT once, offline, to
embed the reference and discover a coarse-to-fine cell-type hierarchy, then train a fast
[`actinn-jax`](https://github.com/iandriver/actinn-jax) MLP on that hierarchy. Inference is
pure CPU and sub-second. Across three datasets (lung with 46 types, blood and gut with 86,
and multi-organ Tabula Sapiens with 83 across 8 organs) the scPRINT-shaped hierarchy beats
a flat classifier on every dataset, beats a random-grouping control, and matches an expert
biological hierarchy, while scPRINT's discovered groups recover biological lineage at ARI
0.54, far better than chance. The practical consequence is to use the foundation model's
embeddings, which carry structure, rather than its labels.

## The flow

```
                ┌─ OFFLINE, once, GPU/MPS ─┐   ┌──── OFFLINE, CPU ────┐   ┌─ INFERENCE, CPU ─┐
  reference ──► scPRINT embed (256-d) ──► discover coarse/fine  ──► train actinn-jax ──► annotate query
  (raw counts)   ~22 ms/cell, cached     hierarchy (cluster        coarse + per-group     <1 s / 1000s cells
                 → data/embeddings/*.npz  type centroids)           fine models
```

1. Embed, using scPRINT on GPU or MPS, one time. Embed the labeled reference and cache the
   256-d vectors. This is the only GPU step and it is paid once. The embeddings are
   committed (`data/embeddings/*.npz`, 7 to 16 MB each) so the step is reproducible without
   a GPU.
2. Discover, on CPU, in seconds. Cluster per-cell-type centroids in scPRINT space into
   coarse groups (`benchmark/explore/discover_hierarchy.py`).
3. Train, with actinn-jax on CPU, in about 30 s. A coarse classifier plus one fine model per
   group.
4. Annotate, on CPU, sub-second. Predict coarse, route, then predict fine. No scPRINT and no
   GPU.

## Results

### Accuracy against baselines, as the mean over the query

| dataset (types) | flat actinn-jax | **hierarchy-scprint** | hierarchy-random *(ctrl)* | biological hierarchy | scPRINT zero-shot |
|---|---|---|---|---|---|
| Lung (46), macro-F1 | 0.682 | **0.710** | 0.667 |, | 0.32 (ontology) |
| Blood+gut (86), macro-F1 | 0.862 | **0.869** | 0.810 | 0.869 (`Lineage`) |, |
| Tabula Sapiens (83/8 organs), macro-F1 | 0.402 | **0.413** | 0.373 | 0.397 (`organ`) |, |

The pattern is consistent: the scPRINT-shaped hierarchy beats flat, which beats random
grouping, and it ties the expert biological hierarchy. Accuracy moves the same way, with
full per-method tables in [TWO_STAGE.md](TWO_STAGE.md).

### scPRINT's structure is biology

Comparing scPRINT's discovered grouping of cell types to known biology, as Adjusted Rand
Index over types (`discover_hierarchy.py`):

| dataset | biological grouping | ARI |
|---|---|---|
| Blood+gut | `Lineage` (immune/epithelial/B/mesenchymal) | **0.543** |
| Tabula Sapiens | `organ` | 0.017 |

The high lineage ARI shows scPRINT recovers genuine cell-type structure. The near-zero
organ ARI is expected and correct: scPRINT groups by cell-type identity, putting
endothelial cells from all organs together, rather than by organ, which is exactly why it
beat an organ-based hierarchy.

### Speed, and the two roles of scPRINT

| operation | device | cost | when |
|---|---|---|---|
| scPRINT embed reference (**structure**) | MPS | ~22 ms/cell (~4 min / 10k cells) | **once, offline** (cached) |
| discover hierarchy | CPU | < 1 s | offline |
| train actinn-jax (flat / hierarchy) | CPU | 18-29 s / 35-58 s | offline |
| **annotate query (inference)** | **CPU** | **0.1-1.1 s for thousands of cells** | **per query** |
|, vs scPRINT as a **predictor** (labels) | MPS / CPU | ~22 ms/cell / **~1 s/cell** | per query, **not used** |

Using scPRINT as the runtime classifier is 100 to 1000 times slower than the small CPU
model and less accurate. This flow keeps scPRINT entirely offline.

### Memory footprint

| component | peak memory |
|---|---|
| actinn-jax training (sparse, per-minibatch) | **~1.9 GB** (was 12.3 GB before optimization; original TF ACTINN: 9.6 GB) |
| actinn-jax inference | tens of MB (model weights are a small `.npz`) |
| scPRINT embed (medium-v1.5, one-time) | ~4 GB; 211 MB checkpoint |

The deployed artifact is a few-MB JAX model running in about 2 GB of RAM on a CPU.

## When to use what

For fast, accurate, no-GPU annotation, train a flat actinn-jax model on curated reference
data. That takes 20 to 40 s of CPU and under a second at inference, and lands within 0.01
to 0.03 macro-F1 of the hierarchy. It is the simplest option and needs no scPRINT.

For the extra macro-F1 on rare and closely related types, add the scPRINT-discovered
hierarchy. That costs one offline GPU pass over the reference, cached here, and inference
stays pure CPU.

Do not use scPRINT, or any foundation model, as the runtime predictor for routine
annotation on commodity hardware. It is slow, and zero-shot it is inaccurate.

## Reproduce

```bash
pip install -e .                      # core env (actinn-jax, sklearn, scanpy)
# 1. Inspect the scPRINT-discovered hierarchy from the COMMITTED embeddings (no GPU):
python benchmark/explore/discover_hierarchy.py blood_gut --bio Lineage
# 2. Full two-stage comparison (needs raw counts; embeddings are cached):
#    embed (GPU, optional, cached) then compare (CPU)
#    see benchmark/explore/two_stage_{embed,compare}.py and configs in TWO_STAGE.md
```

`data/embeddings/{lung,blood_gut,tabula_sapiens}.npz` hold the scPRINT 256-d vectors and
labels for the reference and query of each dataset, which is the cached output of the one
GPU step, so structure discovery and analysis run anywhere on a CPU.

## Limitations

- Gains are modest, at macro-F1 plus 0.007 to 0.028, though consistent and stable in
  direction. The strongest signal is the biological-hierarchy match.
- scPRINT medium-v1.5, G=8 groups, a single split per dataset, and query sets that are small
  for some rare types. Sweeping G, using larger queries, and trying cheaper embedders such
  as scANVI are all worth doing.
- The final classifier is closed-set, with no novel-type rejection, so pair it with an
  uncertainty or rejection rule for open-world use.
