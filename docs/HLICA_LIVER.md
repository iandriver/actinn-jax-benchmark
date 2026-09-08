# A focused liver reference from HLiCA: retraining, not masking, closes the gap

A direct follow-up to [REFINE.md](REFINE.md)'s conclusion that masking the broad reference
is safe but does not fix accuracy, while retraining on a narrower, better reference does.
This is that experiment, on real data, at scale.

## The data: HLiCA

[Edgar, Portman, Hu et al., "HLiCA: An integrated cell atlas of the healthy human liver"](https://doi.org/10.64898/2026.06.30.735539)
(bioRxiv, posted 2026-07-04) integrates 522,730 cells from 110 donors across 7 published
studies, expert-curated into 6 lineages (hepatocyte, cholangiocyte, endothelial, myeloid,
mesenchyme, lymphocyte) and 38 fine cell types. Processed data: CELLxGENE collection
[`059202e1-...`](https://cellxgene.cziscience.com/collections/059202e1-1f1b-483f-9151-f3a25a380c39),
CC-BY 4.0. Downloaded to local SSD storage (`/Volumes/IanSSD/hlica/`, 11 GB, 7 files).

This replaces the roughly 26k-cell liver and zonation reference built from one or two
datasets ([ZONATION.md](ZONATION.md)), a twentyfold increase in reference cells, from 110
donors rather than one or two, and expert-curated rather than self-assembled.

The hepatocyte-specific file's `author_cell_type` column carries 7 substates (Periportal,
Pericentral, Ribosomal+, Mito+, SERPINE1+, UGT+, Cycling), which is richer than the
standardized `cell_type` column that collapses everything except the two zonation poles to
a generic "hepatocyte". We use `author_cell_type` for hepatocytes, after dropping "Cycling",
which a cross-tab verified to be 1,179 mislabeled lymphocytes rather than a real hepatocyte
substate, and the standard `cell_type` for the other 5 lineages.

## Building the reference, with no scPRINT needed

HLiCA already ships expert-curated into 6 lineage files, and that split is the coarse
grouping, so no foundation-model embedding step is required. Each file's own cell-type
column becomes the fine label within its lineage.
`actinn_jax.build_hierarchical_reference`'s `hierarchy=` parameter, a precomputed
`{type: group}` dict, takes this directly.

## Validation: cross-study, not random split

Our previous liver query (`benchmark/explore/fetch_liver_query.py`, CELLxGENE dataset
`ddb22b3d-...`) is HLiCA's `Andrews_2022` component study, confirmed via the `STUDY` obs
column, so reusing it as an external test would leak training data. Validation instead
holds out `Andrews_2022` entirely (56,545 cells), training on the other 6 studies and
testing only on the one held out. That is a stricter test than a random split, since it
crosses both donors and a different research center's protocol.

## Results

| model | exact-CL | ontology-concordant |
|---|---|---|
| **broad_human_v1** (798-type census reference) on this held-out set | 0.231 | 0.580 |
| **HLiCA-focused reference** (38 types, cross-study held-out) | **0.728** | **0.858** |

Exact accuracy triples, which confirms REFINE.md's central claim with a real, large-scale
retrain: a focused reference built on much more per-type data, thousands of cells per type
against the census reference's 15 to 40, closes the gap that masking alone could not.

### Zonation specifically, cross-study held-out on `Andrews_2022`

| lineage | cells | exact-zone | portal/central flip rate |
|---|---|---|---|
| hepatocyte (periportal / pericentral) | 27,740 | 0.787 | 0.182 |
| endothelial (periportal / pericentral sinusoid) | 5,382 | 0.711 | 0.264 |

This reference models zonation as 2 poles, periportal and pericentral, plus 4 orthogonal
metabolic and stress hepatocyte substates, rather than the 3-tier periportal, midzonal and
centrilobular axis used in the original GSE158723-based build (ZONATION.md), because
HLiCA's own curation does not define a midzonal category. Direct comparison to the older
within-1-zone numbers near 0.99 is not like for like: that was same-study internal
validation on a 3-tier scheme, and this is cross-study on a 2-tier one, which is the harder
test. Both the exact-zone accuracy and the low flip rate, where errors land on the
metabolic-substate labels rather than the opposite zone, support the same conclusion, that
zonation is a real and learnable signal that generalizes across studies.

## Shipped

`liver_hlica_v2` is the current bundled reference: 48 types, 6 lineages, 11.8 MB, built by
`build_hlica_liver_v2.py` after [HLICA_EDGE_CASES.md](HLICA_EDGE_CASES.md) checked this
build against the edge cases the HLiCA paper states and found two coverage gaps we had
introduced. Closing them cost nothing on the core metrics (exact-CL 0.724 against v1's
0.728, ontology 0.859 against 0.858, hepatocyte zonation 0.789 against 0.787).

`liver_hlica_v1`, the 38-type reference the results above were measured on, still ships for
anyone who wants the smaller taxonomy.

```python
model = aj.bundled_reference("liver_hlica_v2")
adata = aj.annotate(adata, model)
```

Both are trained on all 522,730 cells. The held-out split above uses a separate
validation-only model that never saw `Andrews_2022`; the shipped artifacts are retrained on
everything for maximum coverage, which is the same pattern as `broad_human_v1`.

## Attribution

Built from data made available by Edgar, R.D., Portman, J.R., Hu, H. et al. **HLiCA: An
integrated cell atlas of the healthy human liver.** bioRxiv (2026).
https://doi.org/10.64898/2026.06.30.735539. CC-BY 4.0. Please cite the original paper if
you use this derived reference.

Build script: `benchmark/explore/build_hlica_liver.py`.
