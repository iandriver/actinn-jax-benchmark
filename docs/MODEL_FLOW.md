# Foundation-model-shaped, CPU-fast cell-type annotation

*A two-stage flow that uses the scPRINT foundation model **once, offline, to discover
cell-type structure**, then trains a small JAX classifier that annotates new data in
**milliseconds on a CPU** — no GPU at inference, foundation-model-grade structure.*

## Abstract

Single-cell foundation models (e.g. scPRINT) carry broad, cross-tissue knowledge but
are slow and GPU-bound, and — used zero-shot — are weak annotators (ontology
concordance **0.22** on lung here). Small supervised classifiers are fast and accurate
but "blind" to structure beyond their reference. We combine them: run scPRINT **once,
offline**, to embed the reference and **discover a coarse→fine cell-type hierarchy**,
then train a fast [`actinn-jax`](https://github.com/iandriver/actinn-jax) MLP on that
hierarchy. Inference is **pure CPU, sub-second**. Across three datasets (lung, 46
types; blood+gut, 86; multi-organ Tabula Sapiens, 83 across 8 organs) the
scPRINT-shaped hierarchy **beats a flat classifier on every dataset**, beats a
random-grouping control, and **matches an expert biological hierarchy** — while
scPRINT's discovered groups recover biological lineage (ARI **0.54**) far better than
chance. The practical takeaway: **use the foundation model's embeddings (structure),
not its labels (predictions).**

## The flow

```
                ┌─ OFFLINE, once, GPU/MPS ─┐   ┌──── OFFLINE, CPU ────┐   ┌─ INFERENCE, CPU ─┐
  reference ──► scPRINT embed (256-d) ──► discover coarse→fine ──► train actinn-jax ──► annotate query
  (raw counts)   ~22 ms/cell, cached     hierarchy (cluster        coarse + per-group     <1 s / 1000s cells
                 → data/embeddings/*.npz  type centroids)           fine models
```

- **Stage 1 — embed (scPRINT, GPU/MPS, one-time).** Embed the labeled reference; cache
  the 256-d vectors. This is the only GPU step, paid once. We commit the embeddings
  (`data/embeddings/*.npz`, 7–16 MB each) so the step is **optional/reproducible
  without a GPU**.
- **Stage 2 — discover (CPU, seconds).** Cluster per-cell-type centroids in scPRINT
  space into coarse groups (`benchmark/explore/discover_hierarchy.py`).
- **Stage 3 — train (actinn-jax, CPU, ~30 s).** A coarse classifier + one fine model
  per group.
- **Stage 4 — annotate (CPU, sub-second).** Coarse → route → fine. No scPRINT, no GPU.

## Results

### Accuracy — scPRINT-shaped hierarchy vs baselines (mean over the query)

| dataset (types) | flat actinn-jax | **hierarchy-scprint** | hierarchy-random *(ctrl)* | biological hierarchy | scPRINT zero-shot |
|---|---|---|---|---|---|
| Lung (46) — macro-F1 | 0.682 | **0.710** | 0.667 | — | 0.32 (ontology) |
| Blood+gut (86) — macro-F1 | 0.862 | **0.869** | 0.810 | 0.869 (`Lineage`) | — |
| Tabula Sapiens (83/8 organs) — macro-F1 | 0.402 | **0.413** | 0.373 | 0.397 (`organ`) | — |

The pattern is consistent: **hierarchy-scprint > flat > random grouping**, and it
**ties the expert biological hierarchy**. (Accuracy moves the same way; full per-method
tables in [TWO_STAGE.md](TWO_STAGE.md).)

### scPRINT's structure *is* biology

Comparing scPRINT's discovered grouping of cell types to known biology (Adjusted Rand
Index over types; `discover_hierarchy.py`):

| dataset | biological grouping | ARI |
|---|---|---|
| Blood+gut | `Lineage` (immune/epithelial/B/mesenchymal) | **0.543** |
| Tabula Sapiens | `organ` | 0.017 |

The high lineage ARI shows scPRINT recovers genuine cell-type structure. The ~0 organ
ARI is *expected and correct*: scPRINT groups by **cell-type identity** (e.g.
endothelial cells from all organs together), not by organ — which is exactly why it
beat an organ-based hierarchy.

### Speed — the two roles of scPRINT, made explicit

| operation | device | cost | when |
|---|---|---|---|
| scPRINT embed reference (**structure**) | MPS | ~22 ms/cell (~4 min / 10k cells) | **once, offline** (cached) |
| discover hierarchy | CPU | < 1 s | offline |
| train actinn-jax (flat / hierarchy) | CPU | 18–29 s / 35–58 s | offline |
| **annotate query (inference)** | **CPU** | **0.1–1.1 s for thousands of cells** | **per query** |
| — vs scPRINT as a **predictor** (labels) | MPS / CPU | ~22 ms/cell / **~1 s/cell** | per query — **not used** |

Using scPRINT as the runtime classifier is **100–1000× slower** than the small CPU
model *and* less accurate. Our flow keeps scPRINT entirely offline.

### Memory footprint

| component | peak memory |
|---|---|
| actinn-jax training (sparse, per-minibatch) | **~1.9 GB** (was 12.3 GB before optimization; original TF ACTINN: 9.6 GB) |
| actinn-jax inference | tens of MB (model weights are a small `.npz`) |
| scPRINT embed (medium-v1.5, one-time) | ~4 GB; 211 MB checkpoint |

The deployed artifact is a few-MB JAX model running in ~2 GB RAM on a CPU.

## When to use what

- **Just want fast, accurate, no-GPU annotation?** Train **flat actinn-jax** on curated
  reference data — 20–40 s CPU, <1 s inference, within ~0.01–0.03 macro-F1 of the
  hierarchy. Simplest, no scPRINT needed.
- **Want the extra macro-F1 (rare/related types)?** Add the **scPRINT-discovered
  hierarchy** — one offline GPU pass on the reference (cached here), pure-CPU inference.
- **Don't** use scPRINT (or any foundation model) as the runtime predictor for routine
  annotation on commodity hardware: slow and, zero-shot, inaccurate.

## Reproduce

```bash
pip install -e .                      # core env (actinn-jax, sklearn, scanpy)
# 1. Inspect the scPRINT-discovered hierarchy from the COMMITTED embeddings (no GPU):
python benchmark/explore/discover_hierarchy.py blood_gut --bio Lineage
# 2. Full two-stage comparison (needs raw counts; embeddings are cached):
#    embed (GPU, optional — cached) -> compare (CPU)
#    see benchmark/explore/two_stage_{embed,compare}.py and configs in TWO_STAGE.md
```

`data/embeddings/{lung,blood_gut,tabula_sapiens}.npz` hold the scPRINT 256-d vectors +
labels for the reference and query of each dataset — the cached output of the one GPU
step, so the structure-discovery and analysis run anywhere on a CPU.

## Limitations

- Gains are modest (macro-F1 +0.007 to +0.028) though consistent and direction-stable;
  the strongest signal is the biological-hierarchy match.
- scPRINT medium-v1.5, G=8 groups, single split per dataset; query sets are small for
  some rare types. Worth: sweeping G, larger queries, and cheaper embedders (scANVI).
- The final classifier is closed-set (no novel-type rejection); pair with an
  uncertainty/rejection rule for open-world use.
