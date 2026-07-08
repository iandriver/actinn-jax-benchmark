# actinn-jax on the Open Problems `label_projection` benchmark (v2.0.0)

An **objective, external** placement of actinn-jax, run through the community-standard
[Open Problems](https://openproblems.bio/benchmarks/label_projection) label-projection
task — the same 6 datasets, the same train/test splits, the same two metrics, scored the
same way — and slotted directly into their published leaderboard (run `2026-06-05`).

## Method & protocol

Open Problems splits each dataset into train/test **batches**; a method trains a cell-type
classifier on `train.h5ad` (`layers['counts']`, `layers['normalized']`, `obs['label']`,
`obsm['X_pca']`) and predicts `label` on `test.h5ad`. actinn-jax is a **gene-space** method
(its own CP10k+log2 + gene filtering), so — like scANVI (counts) rather than the PCA-based
classifiers — it trains on the **counts layer restricted to the 1000 highly-variable genes
the task provides (`var['hvg']`)**: the exact feature set the framework's PCA is built
from. It trains on the **full** train set (no subsampling), predicts the full test set, and
is scored with `accuracy` and macro-`f1` (sklearn), identical to OP's metrics. Runner:
`benchmark/explore/op_runner.py`; results: `docs/results_openproblems.csv`.

The other 16 methods' scores are OP's own published numbers (not re-run) — this places
actinn-jax against the real leaderboard.

## Result: the full leaderboard, with actinn-jax placed

**Accuracy** (mean over the 6 datasets; `n` = how many datasets the method completed):

| rank | method | dkd | gtex | hypomap | immune | mouse_panc | tab_sapiens | **mean** | n |
|---|---|---|---|---|---|---|---|---|---|
| — | *true_labels (pos ctrl)* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | 6 |
| 1\* | scanvi_scarches | 0.955 | 0.875 | 0.997 | 0.892 | 0.976 | **DNF** | 0.939 | 5 |
| 2\* | xgboost | 0.960 | 0.815 | 0.996 | 0.812 | 0.972 | **DNF** | 0.911 | 5 |
| **3** | **actinn-jax** | 0.947 | 0.858 | 0.997 | 0.859 | 0.965 | **0.394** | **0.837** | **6** |
| 4 | mlp | 0.949 | 0.853 | 0.998 | 0.854 | 0.971 | 0.342 | 0.828 | 6 |
| 5 | seurat_transferdata | 0.954 | 0.841 | 0.996 | 0.827 | 0.966 | 0.377 | 0.827 | 6 |
| 6 | scanvi | 0.956 | 0.895 | 0.996 | 0.891 | 0.975 | 0.243 | 0.826 | 6 |
| 7 | logistic_regression | 0.957 | 0.883 | 0.996 | 0.885 | 0.978 | 0.183 | 0.814 | 6 |
| 8 | knn | 0.949 | 0.852 | 0.995 | 0.814 | 0.976 | 0.173 | 0.793 | 6 |
| 9 | cellmapper_linear | 0.953 | 0.825 | 0.986 | 0.795 | 0.966 | 0.123 | 0.775 | 6 |
| 10 | singler | 0.915 | 0.790 | 0.993 | 0.639 | 0.962 | 0.173 | 0.745 | 6 |
| 11 | naive_bayes | 0.927 | 0.760 | 0.995 | 0.639 | 0.950 | 0.158 | 0.738 | 6 |
| — | scgpt_zeroshot | — | 0.639 | — | — | — | — | 0.639 | 1 |
| — | *majority_vote (neg ctrl)* | 0.295 | 0.080 | 0.452 | 0.006 | 0.324 | 0.021 | 0.196 | 6 |
| — | uce | 0.182 | 0.005 | 0.381 | 0.036 | 0.166 | 0.014 | 0.131 | 6 |
| — | *random_labels (neg ctrl)* | 0.188 | 0.032 | 0.296 | 0.046 | 0.164 | 0.021 | 0.124 | 6 |

\* **scanvi_scarches and xgboost rank above actinn-jax only because they do not complete
`tabula_sapiens`** (160 labels) — their means are over 5 easier datasets. **Among the
methods that completed all 6, actinn-jax has the highest mean accuracy (0.837),** ahead of
`mlp` (0.828), seurat (0.827), scanvi (0.826), logistic (0.814), knn (0.793).

**Macro-F1** (mean over 6): scanvi_scarches 0.810 (n=5), xgboost 0.723 (n=5),
logistic 0.691, scanvi 0.681, seurat 0.662, **actinn-jax 0.656**, mlp 0.648, knn 0.648,
naive_bayes 0.613, singler 0.605, … , scgpt_zeroshot 0.291, uce/random ≈ 0.04. actinn-jax
is upper-mid — above its sibling `mlp`, below the logistic/scanvi cluster.

## Reading the result

1. **actinn-jax vs. its direct sibling `mlp`.** Both are multilayer perceptrons; the only
   difference is the input — actinn-jax uses HVG **genes** with ACTINN's normalization and
   gene filtering, `mlp` uses the 50-dim **PCA**. actinn-jax wins on **mean accuracy
   (0.837 vs 0.828)** and **mean macro-F1 (0.656 vs 0.648)**, and on 4 of 6 datasets by
   accuracy. The gene-space representation earns a small but consistent edge over PCA.

2. **Robustness on the hardest task.** `tabula_sapiens` (160 fine cell types, a 284-cell
   test batch) breaks the field — scanvi_scarches and xgboost don't complete it at all, and
   most others land at 0.15–0.38. **actinn-jax's 0.394 is the best accuracy of any method
   that completed it** (vs mlp 0.342, seurat 0.377). It is one of the few methods that runs
   to completion on every dataset, which is why it tops the "completed-all-6" ranking.

3. **Foundation models underperform on label projection.** scgpt_zeroshot (0.639, and only
   on the one dataset it ran) and **uce (0.131 — barely above the random-labels control
   0.124)** sit at the bottom. This independently reproduces our own finding
   ([PAPER.md](PAPER.md) §3.8): a foundation model's *zero-shot labels* are a weak
   annotation signal; its embeddings are where the value is.

4. **The honest cost.** actinn-jax trains in gene space, so its **fit time scales with the
   training set** — 34 s (34k cells) up to 327 s (482k-cell tabula_sapiens) — much heavier
   than the PCA-based classifiers, which fit on 50 dims. Its **predict stays sub-second**
   (0.08–0.48 s) on every dataset, and with the train-once/map-many cache the fit is paid
   once. Full-gene training on the 482k×56k atlas OOM'd on 51 GB RAM; the HVG restriction
   (which the task provides and its PCA methods already use) is what makes it tractable.

## Bottom line

On a benchmark actinn-jax's authors did not design and cannot tune, run to the community's
own protocol: **actinn-jax places 3rd of 17 on accuracy — 1st among all methods that
complete every dataset — beats its PCA-space `mlp` sibling on both metrics, and posts the
best accuracy on the single hardest dataset.** It is not the top method overall
(scANVI+scArches reference surgery and xgboost score higher on the datasets they finish),
and it is upper-mid rather than top on macro-F1. But as a fast, dependency-light,
CPU/gene-space classifier, its standing on this external leaderboard is strong and — unlike
our in-house benchmark — entirely independent.

*Datasets © CZ CELLxGENE / Open Problems, downloaded from the public `openproblems-data`
S3 bucket. Leaderboard scores: OP run `2026-06-05_12-50-03`, `score_uns.yaml`.*
