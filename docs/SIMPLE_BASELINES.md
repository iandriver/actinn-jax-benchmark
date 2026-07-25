# Simple baselines: scTOP and a tuned linear pipeline

> **Superseded by the unified matrix.** The head-to-head numbers below are from the initial
> **4-dataset, per-label-subsampled** probe. Both methods have since been run across the
> **full 6-dataset paper matrix on the identical `paper.yaml` splits** as every other method;
> those are the authoritative figures ([`PAPER.md`](PAPER.md) §3.1–§3.2, raw in
> [`results_paper_matrix_unified.csv`](results_paper_matrix_unified.csv)). On the full matrix
> the linear pipeline leads accuracy at **0.839 vs actinn-jax 0.831** (5-dataset mean) and
> fits **4× faster**; scTOP is **0.739** (mid-pack). This doc is kept as the original probe;
> the conclusions are unchanged, only the exact numbers differ with the larger splits.

Souza & Mehta ([bioRxiv 2026](https://doi.org/10.64898/2026.02.11.705358)) show that
parameter-free and carefully-tuned linear methods rival single-cell foundation models.
That is the same class actinn-jax lives in, so their absence from our benchmark was its
weakest point. This adds both:

- **`sctop`** — parameter-free projection onto per-cell-type bases of rank z-scored
  expression (Yampolskaya, Souza & Mehta). No training.
- **`linear-anova-pca`** — their Tabula Sapiens recipe: per-cell normalization → ANOVA
  gene selection → standardization → PCA(220) → multinomial logistic regression. Fit on
  the reference only, replayed on the query.

Configs: [`simple_baselines.yaml`](../configs/simple_baselines.yaml) +
[`simple_baselines_hlica.yaml`](../configs/simple_baselines_hlica.yaml). All four methods
share the same subsampled splits as the ProtoCloud comparison. CPU throughout.

## Results — 4 datasets

| dataset (types) | method | accuracy | macro-F1 | fit (s) | peak mem (MB) |
|---|---|---|---|---|---|
| **pbmc3k** (8) | actinn-jax | **0.913** | 0.795 | 5.8 | 892 |
| | scTOP | 0.910 | **0.837** | **1.0** | **507** |
| | linear-anova-pca | **0.913** | 0.829 | 1.6 | 1521 |
| | ProtoCloud | 0.880 | 0.770 | 44.9 | 1123 |
| **lung** (46) | actinn-jax | 0.894 | 0.901 | 15.7 | **1957** |
| | scTOP | 0.828 | 0.830 | **1.1** | 1599 |
| | linear-anova-pca | 0.898 | 0.904 | 4.6 | 4837 |
| | ProtoCloud | **0.932** | **0.932** | 167.2 | 1967 |
| **liver** (36) | actinn-jax | 0.802 | 0.798 | 10.3 | 1541 |
| | scTOP | 0.649 | 0.644 | **1.2** | **1349** |
| | linear-anova-pca | **0.804** | **0.804** | 2.8 | 2681 |
| | ProtoCloud | 0.695 | 0.689 | 100.5 | 1618 |
| **blood+gut** (86) | actinn-jax | 0.860 | 0.860 | 19.1 | 2591 |
| | scTOP | 0.795 | 0.789 | **1.3** | **2076** |
| | linear-anova-pca | **0.902** | **0.903** | 4.8 | 5543 |
| | ProtoCloud | 0.841 | 0.839 | 204.8 | 2065 |

**Means across the four datasets**

| method | accuracy | macro-F1 | fit (s) | peak mem (MB) |
|---|---|---|---|---|
| **linear-anova-pca** | **0.880** | **0.860** | 3.5 | 3645 |
| actinn-jax | 0.867 | 0.839 | 12.7 | **1745** |
| ProtoCloud | 0.837 | 0.807 | 129.3 | 1693 |
| scTOP | 0.796 | 0.775 | **1.2** | 1383 |

## Reading — including what this costs us

**The tuned linear pipeline is the most accurate method here, and it is not close on
cost.** `linear-anova-pca` has the best mean accuracy (0.880) and macro-F1 (0.860),
beating actinn-jax on every dataset — a tie on pbmc3k, slight edges on lung and liver, and
a decisive **+4.2 points on the 86-type blood+gut set** (0.902 vs 0.860) — while fitting
**3.7× faster**. It also beats ProtoCloud, a GPU-class deep generative model, on 3 of 4.

This is a genuine correction to §3.1. Our reported margins over the "classical tier" were
margins over *untuned* baselines (SVM, kNN, CellTypist). Against a linear pipeline built
with the care Souza & Mehta describe, **actinn-jax does not win on accuracy.**

**Its one clear advantage is memory.** actinn-jax averages **1745 MB vs 3645 MB** — 2.1×
lighter — and the gap widens with the data: 5.5 GB for the linear pipeline on blood+gut vs
2.6 GB. The reason is structural: ANOVA + PCA densify a (cells × genes) matrix, so the
pipeline's footprint grows with both dimensions, while actinn-jax stays sparse and
minibatched. At these subsampled sizes that is a 2× inconvenience; at atlas scale it is
the difference between running and not. (We have not measured that regime here, so this is
a projection from the trend, not a result.)

**scTOP is the cheapest annotation method we have benchmarked, and it degrades with
cardinality.** 1.2 s fit and 1.4 GB — and on 8 immune types it has the *best* macro-F1 of
any method (0.837). But accuracy falls off as classes multiply: 0.910 (8 types) → 0.828
(46) → 0.795 (86) → 0.649 (liver, 36 closely-related types). A parameter-free projection
onto class averages cannot sharpen boundaries between similar types; that is where a
discriminative model still earns its keep.

**ProtoCloud** wins only the finest-grained set (lung, 0.932) at 10–170× the fit cost.

## Consequence for actinn-jax's positioning

Raw accuracy no longer distinguishes actinn-jax from a well-built linear pipeline — it is
*behind* on these four datasets. What still holds:

1. **Bounded, sparse memory** — 2.1× lighter on average, widening with dataset size.
2. **The cached reference** — `train_reference` → `save`/`load` amortizes across repeated
   queries; the linear pipeline refits scaler/PCA/classifier each time.
3. **The workflow** — broad→refined, abstain, tissue-aware refinement, novel-type
   detection (§3.6, §3.7). Neither scTOP nor the linear pipeline provides any of these;
   ProtoCloud does provide uncertainty, attribution and a fine-tuning refinement path, so
   the genuinely distinct pieces are the **shipped ready-to-run broad reference** and
   **sub-second frozen-model refinement** (see [`PROTOCLOUD.md`](PROTOCLOUD.md)).
4. **Cardinality robustness** relative to scTOP.

The honest summary: actinn-jax's case rests on **cost profile and workflow, not accuracy**.
A reader who only needs one-shot labels on a laptop-sized dataset should probably reach for
the linear pipeline first — and our paper should say so.
