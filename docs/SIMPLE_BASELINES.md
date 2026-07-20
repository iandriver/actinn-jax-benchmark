# Simple baselines: scTOP and a tuned linear pipeline

Souza & Mehta ([bioRxiv 2026](https://doi.org/10.64898/2026.02.11.705358)) show that
parameter-free and carefully-tuned linear methods rival single-cell foundation models.
That is the same class actinn-jax lives in, so their absence from our benchmark was its
weakest point. This adds both:

- **`sctop`** — parameter-free projection onto per-cell-type bases of rank z-scored
  expression (Yampolskaya, Souza & Mehta). No training.
- **`linear-anova-pca`** — their Tabula Sapiens recipe: per-cell normalization → ANOVA
  gene selection → standardization → PCA(220) → multinomial logistic regression. Fit on
  the reference only, replayed on the query.

Config: [`configs/simple_baselines.yaml`](../configs/simple_baselines.yaml); same
subsampled splits as the ProtoCloud comparison, so all four methods are directly
comparable. CPU throughout.

## Results

**pbmc3k — 8 immune types (1,981 ref / 657 query)**

| method | accuracy | macro-F1 | fit (s) | predict (s) | peak mem (MB) |
|---|---|---|---|---|---|
| actinn-jax | **0.913** | 0.795 | 5.8 | 0.24 | 892 |
| **scTOP** | 0.910 | **0.837** | **1.0** | 0.64 | **507** |
| **linear-anova-pca** | **0.913** | 0.829 | 1.6 | 0.08 | 1521 |
| ProtoCloud | 0.880 | 0.770 | 44.9 | 0.41 | 1123 |

**Krasnow lung — 46 types (8,109 ref / 2,694 query)**

| method | accuracy | macro-F1 | fit (s) | predict (s) | peak mem (MB) |
|---|---|---|---|---|---|
| actinn-jax | 0.894 | 0.901 | 15.7 | 0.36 | 1957 |
| scTOP | 0.828 | 0.830 | **1.1** | 0.92 | 1599 |
| **linear-anova-pca** | 0.898 | 0.904 | 4.6 | 0.30 | 4837 |
| **ProtoCloud** | **0.932** | **0.932** | 167.2 | 1.15 | 1967 |

(HLiCA liver and blood+gut are in the config but did not run — the external volume
holding them was unmounted. They remain to be filled in.)

## Reading — including what this costs us

**A tuned linear pipeline matches actinn-jax.** `linear-anova-pca` ties actinn-jax on
pbmc3k accuracy (0.913) with a *better* macro-F1 (0.829 vs 0.795), and edges it on lung
(0.898/0.904 vs 0.894/0.901) — while fitting 3–4× faster. This is a real result and it
tempers the accuracy claims in §3.1: **against a properly-tuned linear baseline,
actinn-jax's accuracy advantage over the classical tier largely disappears.** Our SVM/kNN
numbers flattered the MLP because those baselines were untuned. This is exactly the
comparison Souza & Mehta's paper implies, and it lands the way they predict.

**scTOP is astonishingly cheap and strong at low cardinality, and degrades with it.**
1.0 s fit and 507 MB — the lightest method here — and the *best* macro-F1 on pbmc3k
(0.837). But on 46 lung types it falls to 0.828 accuracy vs 0.894 for actinn-jax. A
parameter-free class-average projection has no way to sharpen boundaries between many
closely related types, which is the regime where a discriminative model still pays.

**Where the linear pipeline is not free: memory.** On lung it peaks at **4.8 GB vs
2.0 GB** for actinn-jax — 2.5×, because it densifies the full gene matrix for ANOVA/PCA.
actinn-jax stays sparse and bounded. On a laptop, or at atlas scale, that gap matters more
than the few seconds of fit time.

**ProtoCloud** remains the accuracy leader on the fine-grained set (0.932) at 10–35× the
fit cost.

## Consequence for actinn-jax's positioning

Accuracy alone no longer distinguishes actinn-jax from a well-built linear pipeline on
these datasets. What still does:

1. **Bounded, sparse memory** — 2.5× lighter than the linear pipeline on the larger set,
   and the gap grows with genes × cells.
2. **The cached reference** — `train_reference` → `save`/`load`, so repeated annotation
   against one reference is amortized; the linear pipeline refits its scaler/PCA/classifier.
3. **The workflow** — broad→refined, abstain, tissue-aware refinement, novel-type
   detection (§3.6, §3.7) — none of which the baselines provide.
4. **Cardinality robustness** relative to scTOP.

The honest summary is that actinn-jax's case rests on **cost profile and workflow, not on
beating a tuned linear model at raw accuracy**. Reporting it otherwise would not survive
contact with this baseline.
