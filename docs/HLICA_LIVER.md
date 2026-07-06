# A focused liver reference from HLiCA — retraining, not masking, closes the gap

Direct follow-up to [REFINE.md](REFINE.md)'s conclusion: masking the broad reference is
safe but doesn't fix accuracy; retraining on a narrower, better reference does. This is
that experiment, on real data, at scale.

## The data: HLiCA

[Edgar, Portman, Hu et al., "HLiCA: An integrated cell atlas of the healthy human liver"](https://doi.org/10.64898/2026.06.30.735539)
(bioRxiv, posted 2026-07-04) integrates **522,730 cells from 110 donors across 7 published
studies**, expert-curated into 6 lineages (hepatocyte, cholangiocyte, endothelial, myeloid,
mesenchyme, lymphocyte) and 38 fine cell types. Processed data: CELLxGENE collection
[`059202e1-...`](https://cellxgene.cziscience.com/collections/059202e1-1f1b-483f-9151-f3a25a380c39),
CC-BY 4.0. Downloaded to local SSD storage (`/Volumes/IanSSD/hlica/`, 11 GB, 7 files).

This replaces the ~26k-cell, 1-2-dataset liver/zonation reference used previously
([ZONATION.md](ZONATION.md)) — a **20x increase in reference cells**, from 110 donors
instead of 1-2, expert-curated rather than self-assembled.

**Bonus finding:** the hepatocyte-specific file's `author_cell_type` column carries 7
substates (Periportal, Pericentral, Ribosomal+, Mito+, SERPINE1+, UGT+, Cycling) — richer
than the standardized `cell_type` column, which collapses everything except the two
zonation poles to generic "hepatocyte". We used `author_cell_type` for hepatocytes (after
dropping "Cycling" — verified via cross-tab to be 1,179 mislabeled lymphocytes, not a real
hepatocyte substate) and the standard `cell_type` for the other 5 lineages.

## Building the reference — no scPRINT needed this time

HLiCA already ships expert-curated into 6 lineage files. That split **is** the coarse
grouping — no foundation-model embedding step required. Each file's own cell-type column
becomes the fine label within its lineage. `actinn_jax.build_hierarchical_reference`'s
`hierarchy=` parameter (a precomputed `{type: group}` dict) takes this directly.

## Validation: cross-study, not random split

**Important byproduct of the earlier work:** our previous liver query
(`benchmark/explore/fetch_liver_query.py`, CELLxGENE dataset `ddb22b3d-...`) turned out to
literally **be** HLiCA's `Andrews_2022` component study (confirmed via the `STUDY` obs
column). Reusing it as an "external" test would leak training data. So validation instead
holds out `Andrews_2022` **entirely** (56,545 cells) — trains on the other 6 studies,
tests only on the one held out. This is a stricter test than a random split: different
donors *and* a different research center's protocol.

## Results

| model | exact-CL | ontology-concordant |
|---|---|---|
| **broad_human_v1** (798-type census reference) on this held-out set | 0.231 | 0.580 |
| **HLiCA-focused reference** (38 types, cross-study held-out) | **0.728** | **0.858** |

This is a **3x jump in exact accuracy** — directly confirming REFINE.md's central claim
with a real, large-scale retrain: a focused reference built on much more per-type data
(thousands of cells/type vs. the census reference's 15-40) closes the gap that masking
alone could not.

### Zonation specifically (cross-study held-out, `Andrews_2022`)

| lineage | cells | exact-zone | portal↔central flip rate |
|---|---|---|---|
| hepatocyte (periportal / pericentral) | 27,740 | 0.787 | 0.182 |
| endothelial (periportal / pericentral sinusoid) | 5,382 | 0.711 | 0.264 |

Note this reference models zonation as **2 poles** (periportal/pericentral) plus 4
orthogonal metabolic/stress hepatocyte substates, rather than the 3-tier
periportal/midzonal/centrilobular axis used in the original GSE158723-based build
(ZONATION.md) — HLiCA's own curation doesn't define a midzonal category. Direct
comparison to the old within-1-zone numbers (~0.99) isn't apples-to-apples: that was
same-study internal validation on a 3-tier scheme; this is cross-study on a 2-tier one, a
harder and more honest test. Both the exact-zone accuracy and the low flip rate (errors
land on the metabolic-substate labels, not the opposite zone) support the same
conclusion: zonation is a real, learnable signal that generalizes across studies.

## Shipped

`actinn_jax/references/liver_hlica_v1/` — 38 types, 6 lineages, 11.8 MB, trained on all
522,730 cells (the held-out split above uses a separate validation-only model that never
saw `Andrews_2022`; the shipped artifact is retrained on everything for maximum coverage,
standard practice — same pattern as `broad_human_v1`).

```python
model = aj.bundled_reference("liver_hlica_v1")
adata = aj.annotate(adata, model)
```

## Attribution

Built from data made available by Edgar, R.D., Portman, J.R., Hu, H. et al. **HLiCA: An
integrated cell atlas of the healthy human liver.** bioRxiv (2026).
https://doi.org/10.64898/2026.06.30.735539. CC-BY 4.0. Please cite the original paper if
you use this derived reference.

Build script: `benchmark/explore/build_hlica_liver.py`. Full log: this file's git history.
