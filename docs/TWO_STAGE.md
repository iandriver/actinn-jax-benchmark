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

## Generalization: 86-type multi-tissue atlas (blood + gut)

Repeated on a more diverse atlas (Sanger blood + gut immune/epithelial; 86 fine
`Final_labels`, ref 12,769 / query 4,100). This atlas ships a real **biological
hierarchy** (`Lineage`), so we add a third control: the *expert* grouping.
Raw table: [results_two_stage_atlas.csv](results_two_stage_atlas.csv).

| method | accuracy | macro-F1 | train (s) | infer (s) | n_ref |
|---|---|---|---|---|---|
| **hierarchy-scprint** (G8) | **0.880** | **0.869** | 58 | 1.04 | 12,769 |
| **hierarchy-biological** (`Lineage`) | **0.880** | **0.869** | 56 | 0.77 | 12,769 |
| flat-full | 0.873 | 0.862 | 28 | 0.36 | 12,769 |
| hierarchy-random (G8) *(control)* | 0.840 | 0.810 | 58 | 0.97 | 12,769 |
| random-50/type | 0.817 | 0.796 | 9.5 | 0.31 | 4,300 |
| coreset-50/type | 0.810 | 0.790 | 9.3 | 0.27 | 4,300 |

The same pattern holds on 2× the label diversity — and the new control sharpens it:

- **scPRINT's discovered hierarchy ties the *real biological* hierarchy** (both
  0.880 / 0.869), and both beat flat (0.862). scPRINT's embedding geometry recovers
  the expert `Lineage` grouping well enough to match it — strong evidence it's
  capturing genuine biological structure, not noise.
- **Random grouping is again worse than flat** (0.810 vs 0.862): an arbitrary
  hierarchy hurts; a *meaningful* one (scPRINT or biological) helps.
- **Coreset ≈ random** once more — no benefit from scPRINT-guided cell selection.

## Bottom line (both datasets)

Across lung (46 types) and a blood+gut atlas (86 types): **a coarse→fine hierarchy
whose groups come from scPRINT embeddings beats a flat classifier and matches the
expert biological hierarchy, with pure-CPU inference.** scPRINT-guided coreset
subsampling does not beat random, and using scPRINT's query-time *predictions*
(scoping/routing) hurts. **Use scPRINT's embeddings (structure), not its labels.**

### Caveats / next
- Two datasets, one checkpoint (medium-v1.5), G=8, single split each; gains are
  modest (macro-F1 +0.028 lung, +0.007 atlas) but consistent and direction-stable,
  with the biological-hierarchy match as the strongest signal. Worth: sweeping G,
  a true multi-organ atlas (Tabula Sapiens, ~25 organs), and scANVI embeddings as a
  cheaper structure source.
- Hierarchy trains more models (≈2× train time) for the gain; inference stays
  sub-second and pure-CPU.
