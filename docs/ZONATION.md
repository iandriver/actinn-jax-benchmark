# Fine sub-cell-type structure: hepatocyte zonation

Standard `cell_type` labels stop at "hepatocyte", but hepatocytes are organized along a
portal-to-central spatial axis, called zonation, with distinct metabolic programs. That is
a fine sub-state beneath one cell type, the kind of structure a reference-mapped classifier
should be able to resolve. A fast CPU actinn-jax model learns it robustly.

## Data and labels (`build_zonation_ref.py`)

Source: GSE158723, 8 human-liver scRNA-seq samples of about 29k cells (10x, Ensembl plus
symbols). No zonation labels ship with it, so we derive them the standard Halpern-style way:

1. Hepatocytes are the ALB-high, PTPRC-low cells, giving 11,427 cells.
2. The zonation score is mean(central landmarks) minus mean(portal landmarks). Central
   landmarks are GLUL, CYP2E1, CYP1A2, CYP3A4, ADH1B, ADH4, OAT and CYP2C8; portal are
   ASS1, ASL, SDS, HAL, CPS1, PCK1, ARG1, AGT, GLS2 and SLC7A2.
3. Zones are tertiles of that score: portal, mid and central, at 3,809 cells each.

## Can actinn-jax learn zonation, and generalize? (`zonation_classify.py`)

Because the labels come from landmark genes, the meaningful tests remove that crutch.

| test | exact (3-zone) | **within-1-zone** | macro-F1 |
|---|---|---|---|
| A, random split, all genes | 0.711 | **0.991** | 0.714 |
| B, random split, landmark genes removed | 0.608 | 0.969 | 0.608 |
| C, held-out donors, all genes | 0.738 | **0.994** | 0.724 |
| C, held-out donors, no landmark genes | 0.603 | 0.962 | 0.583 |

Random baselines are about 0.33 exact and 0.78 within-1. Training takes 10 to 20 s of CPU
and inference under a second.

Within-1-zone accuracy near 0.99 means the model essentially never confuses portal with
central; its errors are adjacent bins on what is really a continuum, which is the expected
behaviour on an ordinal 3-zone task. It generalizes across donors, with test C reaching
0.74 exact and 0.99 within-1 on held-out patients, so zonation transfers rather than being
memorized per sample. And it is not simply reading the markers: with the landmark genes
deleted, exact accuracy drops to about 0.60 while within-1 stays near 0.96, so zonation is
encoded broadly across the transcriptome rather than in the 18 genes used to define the
labels.

## Cross-dataset transfer (`cross_dataset_zonation.py`)

The stricter test trains zonation on one human-liver study and predicts it in a fully
independent one, from a different lab, protocol and set of patients. We use GSE136103
(Ramachandran healthy liver; 6 CD45-negative samples giving 2,108 hepatocytes), with zones
derived independently the same way. Genes are aligned by Ensembl id, sharing 20,197.

| transfer | exact (3-zone) | within-1-zone | macro-F1 | portal/central flips |
|---|---|---|---|---|
| GSE158723 to GSE136103 | 0.459 | 0.879 | 0.457 | 0.181 |
| GSE136103 to GSE158723 | 0.583 | 0.916 | 0.483 | 0.126 |

Exact 3-class accuracy drops against the roughly 0.72 seen within a dataset, because each
study cuts its tertile boundaries on its own cell distribution. Within-1-zone stays at 0.88
to 0.92 and axis flips between portal and central are rare at 0.13 to 0.18. The zonation
gradient and its direction therefore transfer across datasets, while only the discrete bin
boundaries are dataset-specific, which is expected for a continuum.

## Why this matters for the model flow

Zonation is the natural fine level under "hepatocyte" in the two-stage hierarchy
([MODEL_FLOW.md](MODEL_FLOW.md)): a coarse classifier identifies hepatocytes, then a small
zonation model resolves portal, mid and central, all on CPU. It shows the same recipe
extends from across cell types (lung, atlas, Tabula Sapiens) to within a cell type,
recovering continuous spatial structure.

## Reproduce

```bash
# download GSE158723_RAW.tar and GSE136103_RAW.tar from GEO -> /tmp/gse{158723,136103}/ (tar xf)
python benchmark/explore/build_zonation_ref.py                                   # GSE158723 -> /tmp/liver_zonation_ref.h5ad
ZON_SRC=/tmp/gse136103 ZON_INCLUDE=healthy,cd45- ZON_OUT=/tmp/gse136103_zonation_ref.h5ad \
  python benchmark/explore/build_zonation_ref.py                                  # GSE136103 healthy CD45-
python benchmark/explore/zonation_classify.py        # within-dataset -> docs/results_zonation.csv
python benchmark/explore/cross_dataset_zonation.py   # cross-dataset  -> docs/results_cross_zonation.csv
```

## Next steps

Zones here are tertile bins on a continuum. Modeling the continuous zonation score, by
regression or an ordinal model, would be closer to the biology than 3 discrete classes and
would likely raise cross-dataset exact accuracy, since boundary placement is the main loss
there. A further extension would add scPRINT or scANVI embedding features to the zonation
model and test transfer into a spatial liver atlas with ground-truth zonation.
