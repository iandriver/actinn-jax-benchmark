# Two-stage: scPRINT (broad) shaping a fast actinn-jax (fine) classifier

**Question:** can the scPRINT foundation model's broad, all-cell-types knowledge make
a *fast, non-GPU* classifier better — best of both worlds — rather than being used
directly (slow, weak zero-shot)?

**Setup:** krasnow lung within-dataset CV, 46 fine cell types. Reference 10,803 cells
/ query 2,747. scPRINT = medium-v1.5 on Apple MPS. All small models are actinn-jax
(CPU). scPRINT embeds the reference **once** (234 s offline); "online" methods also
embed the query (82 s). Scripts: `benchmark/explore/two_stage_embed.py`,
`two_stage_compare.py`.

## Results

| method | accuracy | macro-F1 | ontology | train (s) | infer (s) | n_ref | scPRINT |
|---|---|---|---|---|---|---|---|
| **hierarchy-scprint** (G8) | **0.912** | **0.710** | **0.932** | 34 | 0.58 | 10,803 | offline (ref embed) |
| flat-full (baseline) | 0.902 | 0.682 | 0.928 | 18 | 0.18 | 10,803 | none |
| hierarchy-random (G8) *(control)* | 0.900 | 0.667 | 0.927 | 33 | 0.63 | 10,803 | none |
| random-50/type | 0.828 | 0.601 | 0.873 | 3.3 | 0.14 | 2,161 | none |
| coreset-50/type | 0.823 | 0.598 | 0.878 | 3.2 | 0.13 | 2,161 | offline (ref embed) |
| routing (G8) | 0.610 | 0.446 | 0.824 | 34 | 0.46 | 10,803 | **online** (+82 s) |
| scoping (26/46 types) | 0.600 | 0.420 | 0.715 | 9 | 0.15 | 6,401 | **online** (+82 s) |
| scprint-alone (zero-shot) | 0.035 | 0.008 | 0.197 | — | — | 0 | **online** (full) |

## What it shows

1. **The win: a scPRINT-discovered coarse→fine hierarchy.** Clustering the 46 fine
   types into 8 coarse groups *by their scPRINT embeddings*, then training a coarse
   actinn-jax + one fine model per group, **beats flat on every metric — macro-F1
   0.710 vs 0.682** (rare types benefit most). Inference is **pure CPU, 0.58 s** (no
   scPRINT at query time). This is the best-of-both-worlds result.

2. **It's scPRINT's structure, not just "a hierarchy."** The control with the *same*
   two-stage structure but **random** type groupings scores **0.667 macro-F1 —
   *worse* than flat** (0.682). So an arbitrary hierarchy hurts; scPRINT's
   *meaningful* grouping is what helps (+0.043 over random, +0.028 over flat).

3. **scPRINT-guided coreset ≈ random subsampling.** Picking medoid cells in scPRINT
   embedding space (coreset-50) is no better than a random 50/type subsample
   (0.823 vs 0.828 acc). For this data, scPRINT embeddings don't select better
   training cells than chance.

4. **Using scPRINT's *predictions* online HURTS.** Scoping (restrict reference
   classes to scPRINT's query calls) and routing (let scPRINT's coarse call pick the
   fine model) both collapse to ~0.60 accuracy — because scPRINT's zero-shot
   predictions are noisy (0.035 exact / 0.197 ontology). They inject the foundation
   model's errors, and they're *slower* (need scPRINT on every query).

## Takeaway

**Use scPRINT's embeddings (geometry/structure), not its labels (predictions).** The
durable recipe: run scPRINT once, *offline*, on the reference to discover a
coarse→fine cell-type hierarchy; train small actinn-jax models on it; serve entirely
on CPU. That lifts fine-grained accuracy (macro-F1) above a flat classifier while
keeping millisecond CPU inference — and decisively beats both the foundation model
alone and any use of its query-time predictions.

### Caveats / next
- One dataset (lung, 46 types), one checkpoint (medium-v1.5), G=8 groups, single
  split. The hierarchy gain is modest (+0.028 macro-F1) — worth confirming on PBMC /
  Tabula Sapiens, sweeping G, and trying scANVI embeddings as an alternative
  structure source.
- The hierarchy trains more models (34 s vs 18 s) for the macro-F1 gain; inference
  stays sub-second.
