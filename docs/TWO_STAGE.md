# Two-stage: scPRINT shaping a fast actinn-jax classifier

> **Where the runnable workflow lives.** This repo holds the reasoning and comparisons,
> meaning the ablations, controls and datasets. The productized, importable workflow ships
> in [**actinn-jax**](https://github.com/iandriver/actinn-jax) (`actinn_jax.hierarchy`,
> `examples/`): `discover_hierarchy`, `build_hierarchical_reference`, `annotate`,
> `HierarchicalReferenceModel`, the optional `scprint_embed`, and a pre-trained broad-human
> reference sampled across the whole CELLxGENE census (about 800 cell types, 314 tissues and
> 440 datasets, with an abstain threshold for out-of-distribution cells) that annotates
> unknown data on CPU out of the box. To rebuild that reference see
> [UPDATE_BROAD_REFERENCE.md](UPDATE_BROAD_REFERENCE.md): one driver over
> `fetch_census_wide.py`, `embed_broad.py` and `build_census_model.py`, plus a verification
> gate (Tabula-Sapiens-only variant: `fetch_broad_reference.py` and `build_from_emb.py`).

Can the scPRINT foundation model's broad, all-cell-types knowledge make a fast, non-GPU
classifier better, rather than being used directly, where it is slow and weak zero-shot?

The setup is krasnow lung within-dataset CV over 46 fine cell types, with a reference of
10,803 cells and a query of 2,747. scPRINT is medium-v1.5 on Apple MPS, and all small
models are actinn-jax on CPU. scPRINT embeds the reference once, taking 234 s offline;
the online methods also embed the query, taking 82 s. Scripts:
`benchmark/explore/two_stage_embed.py`, `two_stage_compare.py`.

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
| scprint-alone (zero-shot) | 0.035 | 0.008 | 0.197 |, |, | 0 | **online** (full) |

## What it shows

1. A scPRINT-discovered coarse-to-fine hierarchy is the win. Clustering the 46 fine types
   into 8 coarse groups by their scPRINT embeddings, then training a coarse actinn-jax model
   plus one fine model per group, beats flat on every metric, with macro-F1 at 0.710 against
   0.682, and rare types benefit most. Inference is pure CPU at 0.58 s, with no scPRINT at
   query time.

2. It is scPRINT's structure rather than the mere presence of a hierarchy. The control with
   the same two-stage structure but random type groupings scores 0.667 macro-F1, which is
   worse than flat at 0.682. An arbitrary hierarchy hurts, and scPRINT's meaningful grouping
   is what helps, by 0.043 over random and 0.028 over flat.

3. A scPRINT-guided coreset is no better than random subsampling. Picking medoid cells in
   scPRINT embedding space (coreset-50) matches a random 50-per-type subsample, at 0.823
   against 0.828 accuracy. For this data, scPRINT embeddings do not select better training
   cells than chance.

4. Using scPRINT's predictions online hurts. Scoping, which restricts reference classes to
   scPRINT's query calls, and routing, which lets scPRINT's coarse call pick the fine model,
   both collapse to about 0.60 accuracy, because scPRINT's zero-shot predictions are noisy
   at 0.035 exact and 0.197 ontology. They inject the foundation model's errors, and they
   are slower, since they need scPRINT on every query.

The durable recipe is to use scPRINT's embeddings, meaning its geometry and structure,
rather than its labels. Run scPRINT once, offline, on the reference to discover a
coarse-to-fine cell-type hierarchy, train small actinn-jax models on it, and serve entirely
on CPU. That lifts fine-grained accuracy above a flat classifier while keeping millisecond
CPU inference, and it beats both the foundation model alone and any use of its query-time
predictions.

## Generalization: 86-type multi-tissue atlas (blood and gut)

Repeated on a more diverse atlas, the Sanger blood and gut immune and epithelial set, with
86 fine `Final_labels`, a reference of 12,769 and a query of 4,100. This atlas ships a real
biological hierarchy in `Lineage`, so we add a third control: the expert grouping. Raw
table: [results_two_stage_atlas.csv](results_two_stage_atlas.csv).

| method | accuracy | macro-F1 | train (s) | infer (s) | n_ref |
|---|---|---|---|---|---|
| **hierarchy-scprint** (G8) | **0.880** | **0.869** | 58 | 1.04 | 12,769 |
| **hierarchy-biological** (`Lineage`) | **0.880** | **0.869** | 56 | 0.77 | 12,769 |
| flat-full | 0.873 | 0.862 | 28 | 0.36 | 12,769 |
| hierarchy-random (G8) *(control)* | 0.840 | 0.810 | 58 | 0.97 | 12,769 |
| random-50/type | 0.817 | 0.796 | 9.5 | 0.31 | 4,300 |
| coreset-50/type | 0.810 | 0.790 | 9.3 | 0.27 | 4,300 |

The same pattern holds at twice the label diversity, and the new control sharpens it.

- scPRINT's discovered hierarchy ties the real biological hierarchy, both at 0.880 accuracy
  and 0.869 macro-F1, and both beat flat at 0.862. scPRINT's embedding geometry recovers the
  expert `Lineage` grouping well enough to match it, which is strong evidence it captures
  genuine biological structure rather than noise.
- Random grouping is again worse than flat, at 0.810 against 0.862. An arbitrary hierarchy
  hurts while a meaningful one, whether scPRINT's or biological, helps.
- Coreset matches random once more, with no benefit from scPRINT-guided cell selection.

## True multi-organ: Tabula Sapiens (8 organs, 83 types)

The strongest test of diverse, mostly non-overlapping cell types is a CZ Biohub Tabula
Sapiens slice spanning 8 organs (pancreas, skin, liver, trachea, heart, bone marrow,
stomach, eye), pulled contiguously by `dataset_id` from the CELLxGENE census
(`fetch_tabula_sapiens.py`, which avoids the scattered-read slowness, taking about 3.5 min
for roughly 16k cells). Reference 6,012, query 840. Raw table:
[results_two_stage_tabula_sapiens.csv](results_two_stage_tabula_sapiens.csv).

| method | accuracy | macro-F1 | train (s) | infer (s) | n_ref |
|---|---|---|---|---|---|
| **hierarchy-scprint** (G8) | **0.739** | **0.413** | 38 | 0.46 | 6,012 |
| hierarchy-biological (`organ`) | 0.738 | 0.397 | 37 | 0.45 | 6,012 |
| flat-full | 0.732 | 0.402 | 19 | 0.11 | 6,012 |
| hierarchy-random (G8) *(control)* | 0.694 | 0.373 | 38 | 0.57 | 6,012 |
| coreset-30/type | 0.640 | 0.329 | 6.8 | 0.06 | 2,210 |
| random-30/type | 0.633 | 0.333 | 7.0 | 0.08 | 2,210 |

The same pattern holds across organs, with hierarchy-scprint ahead of flat, which is ahead
of hierarchy-random, and coreset matching random. Here scPRINT's grouping beats the
organ-based grouping on macro-F1, 0.413 against 0.397, because shared types such as
endothelial cells, macrophages and T cells span organs, which makes "organ" a noisy
hierarchy while scPRINT's embedding captures the actual cell-type structure. Overall
accuracy is lower, near 0.73, because multi-organ annotation with many cross-organ-shared
types is harder.

CL ids were not carried into this cache, so ontology concordance is not available here.
Accuracy and macro-F1 on exact `cell_type` are the scoring, and the ranking is unambiguous.

## Breadth check: 8 distinct organs

A diverse organ set pulled contiguously from census (`fetch_multitissue.py`): heart, kidney
(KPMP, which is not in Tabula Sapiens), liver, fat, pancreas, skin, stomach and bone marrow,
giving 81 types with a reference of 5,766 and a query of 814. Raw table:
[results_two_stage_8organ.csv](results_two_stage_8organ.csv).

| method | accuracy | macro-F1 | ontology |
|---|---|---|---|
| flat-full | 0.725 | 0.358 | **0.808** |
| hierarchy-scprint (G8) | 0.711 | **0.364** | 0.806 |
| hierarchy-random (G8) | 0.682 | 0.318 | 0.767 |
| scoping / routing / scprint-alone | ≤ 0.60 | ≤ 0.32 | ≤ 0.68 |

Here hierarchy-scprint matches flat, with macro-F1 marginally up and accuracy marginally
down, so the hierarchy gain washes out on this set. The likely reasons are the small
814-cell query and the many cross-organ shared immune and stromal types. The controls still
hold: random grouping is worse, and scoping, routing and scprint-alone all hurt. This is the
fourth data point, and it says the gain is real but modest and not universal.

## Bottom line, over four datasets

Across lung (46 types), a blood and gut atlas (86), multi-organ Tabula Sapiens (83 types
over 8 organs) and a separate 8-organ census set (81), a coarse-to-fine hierarchy whose
groups come from scPRINT embeddings beats a flat classifier on 3 of 4 datasets and washes
out on the fourth, always beats a random-grouping control, and matches the expert biological
hierarchy, all with pure-CPU inference. The gain is modest and not universal.

What is universal: a meaningful grouping, whether scPRINT's or biological, helps and a
random one hurts; scPRINT-guided coreset subsampling does not beat random; and using
scPRINT's query-time predictions, through scoping, routing or zero-shot, consistently hurts.
Use scPRINT's embeddings rather than its labels.

### Caveats and next steps

- Four datasets, one checkpoint (medium-v1.5), G=8 and a single split each. Gains are modest,
  at macro-F1 plus 0.028 on lung and plus 0.007 on the blood and gut atlas, but consistent
  and stable in direction, with the biological-hierarchy match as the strongest signal.
  Worth doing: sweeping G, a wider multi-organ atlas than the 8-organ slices used here, and
  scANVI embeddings as a cheaper structure source.
- The hierarchy trains more models, at roughly twice the train time, for that gain.
  Inference stays sub-second and pure CPU.
