# Fine sub-cell-type structure: hepatocyte zonation

Standard `cell_type` labels stop at "hepatocyte". But hepatocytes are organized along a
**portal→central** spatial axis (zonation) with distinct metabolic programs. This is a
*fine sub-state* beneath one cell type — the kind of structure a good reference-mapped
classifier should resolve. We show a fast CPU **actinn-jax** model learns it robustly.

## Data & labels (`build_zonation_ref.py`)

- **Source:** GSE158723 — 8 human-liver scRNA-seq samples (~29k cells; 10x, Ensembl +
  symbols). No zonation labels ship with it, so we derive them the standard
  (Halpern-style) way:
  1. **Hepatocytes** = ALB-high, PTPRC-low cells → 11,427 cells.
  2. **Zonation score** = mean(central landmarks) − mean(portal landmarks)
     (central: GLUL, CYP2E1, CYP1A2, CYP3A4, ADH1B, ADH4, OAT, CYP2C8; portal: ASS1,
     ASL, SDS, HAL, CPS1, PCK1, ARG1, AGT, GLS2, SLC7A2).
  3. **Zones** = tertiles of the score → portal / mid / central (3,809 each).

## Can actinn-jax learn zonation — and generalize? (`zonation_classify.py`)

Because the labels come from landmark genes, the meaningful tests *remove* that crutch:

| test | exact (3-zone) | **within-1-zone** | macro-F1 |
|---|---|---|---|
| A — random split, all genes | 0.711 | **0.991** | 0.714 |
| B — random split, **landmark genes removed** | 0.608 | 0.969 | 0.608 |
| C — **held-out donors**, all genes | 0.738 | **0.994** | 0.724 |
| C — held-out donors, no landmark genes | 0.603 | 0.962 | 0.583 |

(Random baselines: exact ≈ 0.33, within-1 ≈ 0.78. Train ~10–20 s CPU; inference <1 s.)

**Reading:**
- **Within-1-zone ≈ 0.99** — the model essentially never confuses portal with central;
  its only "errors" are adjacent bins on what is really a continuum. For an ordinal
  3-zone task that is excellent.
- **Generalizes across donors** (test C: 0.74 / 0.99 on held-out patients) — zonation
  transfers, not memorized per-sample.
- **Not just the markers** (tests B/C, landmark genes deleted): exact drops to ~0.60
  but **within-1 stays ~0.96** — zonation is encoded *broadly* across the transcriptome,
  so the classifier learns real spatial biology rather than echoing the 18 genes used
  to define the labels.

## Why this matters for the model flow

Zonation is the natural **fine level under "hepatocyte"** in the two-stage hierarchy
([MODEL_FLOW.md](MODEL_FLOW.md)): a coarse classifier identifies hepatocytes, then a
small zonation model resolves portal/mid/central — all on CPU. It demonstrates the same
recipe extends from *across* cell types (lung/atlas/Tabula Sapiens) to *within* a cell
type, recovering continuous spatial structure.

## Reproduce

```bash
# 1. download GSE158723_RAW.tar from GEO -> /tmp/gse158723/ (tar xf)
python benchmark/explore/build_zonation_ref.py     # -> /tmp/liver_zonation_ref.h5ad
python benchmark/explore/zonation_classify.py       # -> docs/results_zonation.csv
```

## Notes / next

- Zones are derived (tertile bins on a continuum); modeling the **continuous zonation
  score** (regression / ordinal) would be closer to the biology than 3 discrete classes.
- A natural extension: add scPRINT/scANVI embedding features to the zonation model, and
  test cross-*dataset* transfer (train GSE158723 → predict zonation in Tabula Sapiens or
  a spatial liver atlas).
