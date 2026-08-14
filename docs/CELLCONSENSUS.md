# CellConsensus: an exploratory comparison

**Status: not in the paper.** This is a scoping run against a method that appeared after the
matrix was frozen, kept here so the numbers and the failure analysis are recoverable if we
decide to include it.

CellConsensus (de Mathelin, Quinn, Tosh & Tansey, bioRxiv 2026,
[doi:10.64898/2026.08.07.743503](https://doi.org/10.64898/2026.08.07.743503),
[tansey-lab/cellconsensus](https://github.com/tansey-lab/cellconsensus), MIT) assigns cell
types from a consensus corpus of marker genes — 2,607 curated sources plus de novo mining of
1,174 papers — rather than from a labeled reference. It needs no training data, no GPU, and
makes no network calls at inference.

## How it has to be scored

It answers in its own fixed vocabulary at three levels: 13 broad classes, 45 subtypes, 77
fine types. Exact match against a dataset's own label strings would therefore measure
vocabulary overlap rather than accuracy — the same situation as the pretrained annotators in
§2.2 of the paper. It exposes `predict(output="cl_id")`, so it is scored the way those are:
ontology-aware concordance against the query's Cell Ontology ids, on the identical splits.

## Results

Ontology concordance, same splits and same pinned ontology release as the paper:

| | lung_intra | liver_intra |
|---|---:|---:|
| actinn-jax (reference-trained, Table 4) | **0.917** | **0.846** |
| CellConsensus, level 1 (13 classes) | 0.762 | 0.691 |
| CellConsensus, level 2 (45 classes) | 0.813 | 0.533 |
| CellConsensus, level 3 (77 classes) | 0.642 | 0.491 |
| Pan-human Azimuth (pretrained, Table 4) | 0.700 | 0.521 |
| scPRINT (zero-shot, Table 4) | 0.201 | — |

Fit is 11–12 s on 1.3–2.7k cells, CPU only.

It beats the pretrained annotator on both tissues, which is notable for a method that needs
no reference at all, and trails the reference-trained model. **Read the levels with care:**
ontology concordance credits an ancestor of the truth, so a coarse prediction collects credit
a fine one does not, and the coarsest level scores best on liver for that reason. Comparing
its 77-class level against a 36-type reference model is the fairer fine-grained reading.

## Where liver goes wrong

98 of 111 true hepatocytes (88%) are labeled **adipocyte** (CL:0000136). Concordance on those
cells is 0.000 — adipocyte is neither an ancestor nor a descendant of hepatocyte, so the
metric gives no partial credit, correctly. Cholangiocytes (20) and hepatic stellate cells (8)
land there too, so the whole parenchymal compartment is affected.

The cause is the hierarchy, not the markers:

- `hepatocyte` exists **only at level 3** (931 markers). There is no hepatocyte class at
  level 1 or level 2. The 13 level-1 classes are adipocyte, B/plasma, endothelial,
  epithelial, erythroid/megakaryocyte, fibroblast, germ cell, mast, myeloid, neural,
  smooth muscle/pericyte, stem/progenitor, T cell.
- Assignment is level 1 first, then refinement **within that branch only**. A hepatocyte can
  reach the leaf only if level 1 routes it to `epithelial`. Twelve did; the rest did not.
- On true hepatocytes every level-1 class scores ~0.0001 — an effective tie. Adipocyte's 344
  markers are lipid-metabolism genes (ACACA, ACACB, ACSL1, ACSS2), which hepatocytes express,
  so it wins a coin-flip and refinement never revisits the branch.
- Scored directly, the hepatocyte markers are fine: 6.6× separation between true hepatocytes
  and the rest, and they beat adipocyte on 99% of those cells.

There is no tissue-context parameter to work around this — `fit()` takes only
`include_cancer` / `cancer_types`. A tissue prior could only help by changing level-1
routing, and no prior can route to a class that does not exist at level 1.

## What removing the hierarchy does

Scoring all 77 level-3 types directly and taking a flat argmax (no routing) recovers
110 of 111 hepatocytes and lifts whole-query concordance 0.491 → 0.562. But it is not a free
win: the hierarchy is doing real work elsewhere, and dropping it retreats to catch-all
classes in the immune compartment.

| truth | hierarchy | flat argmax |
|---|---|---|
| centrilobular hepatocyte | adipocyte (37/37) | **hepatocyte (37)** |
| periportal hepatocyte | adipocyte (35/37) | **hepatocyte (36)** |
| dendritic cell | macrophage (30) | **cDC2 (28)** |
| B cell | naive/immature B cell (23) | other B/plasma lineage (34) |
| CD8 cytotoxic T | effector/cytotoxic CD8 (26) | other CD8+ T cell (34) |
| helper T cell | naive CD4 (23) | other CD4+ T cell (30) |
| hepatic stellate cell | pericyte (13) | other mural cell (20) |

So: the hierarchy buys immune specificity and loses a whole parenchymal compartment. That is
a corpus-coverage limitation on parenchymal tissue, not a flaw in the consensus idea, and it
is worth reporting upstream — the near-zero tied level-1 scores are a signal the method could
expose as low confidence instead of committing to a confident wrong answer.

## If this goes in the paper

It would belong in the Table 4 block (fixed-vocabulary methods scored ontology-only), not the
main matrix, and would need: the coarseness caveat stated explicitly, at least one more
tissue, and the liver failure described rather than averaged away.

## Reproducing

Environment is separate, as the harness requires; `.venv-cellconsensus` with
`pip install cellconsensus`.

    .venv-cellconsensus/bin/python benchmark/explore/compare_cellconsensus.py \
        --dataset liver_intra --diagnose
    .venv-cellconsensus/bin/python benchmark/explore/compare_cellconsensus.py \
        --dataset lung_intra

Two guards in that script matter, because either failure is silent. Marker sets are keyed by
gene symbol and these matrices by Ensembl id, so `var_names` are remapped and the overlap is
asserted (92% liver, 99% lung) rather than assumed — handing over Ensembl ids yields one
constant label with no error. And `fit()` documents raw counts and normalizes in place, so
the raw layer is used where present; the matrix passed is integral with a median of 3,239
counts per cell.

The ontology file was re-fetched for this run and hashes to `73996c63…`, the release the
paper pins, so these numbers are directly comparable to Table 4 rather than approximately so.
