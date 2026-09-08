# actinn-jax on the Open Problems `label_projection` benchmark (v2.0.0)

An external placement of actinn-jax, run through the community-standard
[Open Problems](https://openproblems.bio/benchmarks/label_projection) label-projection
task, using the same 6 datasets, the same train and test splits, the same two metrics,
scored the same way, and slotted directly into their published leaderboard (run
`2026-06-05`).

## Method and protocol

Open Problems splits each dataset into train and test batches. A method trains a cell-type
classifier on `train.h5ad` (`layers['counts']`, `layers['normalized']`, `obs['label']`,
`obsm['X_pca']`) and predicts `label` on `test.h5ad`. actinn-jax is a gene-space method,
applying its own CP10k and log2 normalization and gene filtering, so like scANVI (counts)
rather than the PCA-based classifiers it trains on the counts layer restricted to the 1000
highly-variable genes the task provides (`var['hvg']`), which is the exact feature set the
framework's PCA is built from. It trains on the full train set without subsampling,
predicts the full test set, and is scored with `accuracy` and macro-`f1` (sklearn),
identical to OP's metrics. Runner: `benchmark/explore/op_runner.py`; results:
`docs/results_openproblems.csv`.

The other 16 methods' scores are OP's own published numbers, not re-run, so this places
actinn-jax against the real leaderboard.

## Result: the full leaderboard, with actinn-jax placed

Accuracy, as the mean over the 6 datasets, with `n` counting how many datasets the method
completed:

| rank | method | dkd | gtex | hypomap | immune | mouse_panc | tab_sapiens | **mean** | n |
|---|---|---|---|---|---|---|---|---|---|
|  | *true_labels (pos ctrl)* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | 6 |
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
|  | scgpt_zeroshot | n/a | 0.639 | n/a | n/a | n/a | n/a | 0.639 | 1 |
|  | *majority_vote (neg ctrl)* | 0.295 | 0.080 | 0.452 | 0.006 | 0.324 | 0.021 | 0.196 | 6 |
|  | uce | 0.182 | 0.005 | 0.381 | 0.036 | 0.166 | 0.014 | 0.131 | 6 |
|  | *random_labels (neg ctrl)* | 0.188 | 0.032 | 0.296 | 0.046 | 0.164 | 0.021 | 0.124 | 6 |

scanvi_scarches and xgboost rank above actinn-jax only because they do not complete
`tabula_sapiens` (160 labels), so their means are over 5 easier datasets. Among the methods
that completed all 6, actinn-jax has the highest mean accuracy at 0.837, ahead of `mlp`
(0.828), seurat (0.827), scanvi (0.826), logistic (0.814) and knn (0.793).

Macro-F1, mean over 6: scanvi_scarches 0.810 (n=5), xgboost 0.723 (n=5), logistic 0.691,
scanvi 0.681, seurat 0.662, actinn-jax 0.656, mlp 0.648, knn 0.648, naive_bayes 0.613,
singler 0.605, down to scgpt_zeroshot 0.291 and uce and random at about 0.04. actinn-jax is
upper-mid: above its sibling `mlp`, below the logistic and scanvi cluster.

### Accuracy, macro-F1, runtime and memory together

Sorted by mean accuracy, with `n` the number of datasets completed out of 6. Runtime and
memory here are cross-hardware and indicative: actinn-jax (†) is the same-hardware AWS
mean per dataset (`r7i.8xlarge`), while every other method is OP's cloud-CI Nextflow trace.
For the CPU tier measured like for like, see
[the controlled run](#controlled-same-hardware-run).

| method | mean acc | macro-F1 | n | runtime (s) | peak (GB) |
|---|---|---|---|---|---|
| scanvi_scarches | **0.939** | 0.810 | 5 | 1542 | 39.3 |
| xgboost | 0.911 | 0.723 | 5 | 999 | 41.4 |
| **actinn-jax + std** | 0.839 | 0.668 | 6 | 205† | 21.0† |
| **actinn-jax** | 0.837 | 0.656 | 6 | 165† | 21.0† |
| mlp | 0.828 | 0.648 | 6 | 809 | 10.1 |
| seurat_transferdata | 0.827 | 0.662 | 6 | 934 | 48.8 |
| scanvi | 0.826 | 0.681 | 6 | 647 | 49.2 |
| logistic_regression | 0.814 | 0.691 | 6 | 74 | 10.4 |
| knn | 0.793 | 0.648 | 6 | 21 | 10.0 |
| cellmapper_linear | 0.775 | 0.561 | 6 | 160 | 17.3 |
| cellmapper_scvi | 0.753 | 0.550 | 5 | 1312 | 25.7 |
| singler | 0.745 | 0.605 | 6 | 3913 | 32.5 |
| naive_bayes | 0.738 | 0.613 | 6 | 17 | 8.5 |
| scimilarity_knn | 0.711 | 0.566 | 4 | 814 | 38.9 |
| scgpt_zeroshot | 0.639 | 0.291 | 1 | 3259 | ~0 |
| uce | 0.131 | 0.043 | 6 | 11825 | 129.0 |

The two methods above actinn-jax on mean accuracy, scanvi_scarches and xgboost, do not
complete tabula_sapiens (n=5, means over easier datasets) and cost 6 to 9 times the runtime
at twice the memory. Among all-6 completers, actinn-jax leads. Its opt-in
`standardize=True` moves it to 0.839 accuracy and 0.668 macro-F1, above `mlp` on both, for
about 24% more fit time.

## Runtime and peak memory

### Controlled same-hardware run

actinn-jax and the CPU and Python tier were re-run through OP's own Nextflow pipeline on a
single AWS `r7i.8xlarge` (32 vCPU, 256 GB), so hardware, harness, storage and `trace`
instrumentation are identical for every method. Mean accuracy and macro-F1 are over the 6
datasets, runtime is mean per-dataset `realtime`, and peak is max per-dataset `peak_rss`.
`singler` and `seurat_transferdata` are omitted, because SingleR did not finish a single
dataset in over 2 hours; their OP-CI figures are in the indicative table below.

| method | mean acc | macro-F1 | runtime/dataset (s) | peak RSS (GB) |
|---|---|---|---|---|
| mlp | 0.838 | 0.664 | 327 | 19.9 |
| **actinn-jax** | **0.837** | 0.655 | **165** | 21.0 |
| logistic_regression | 0.813 | 0.689 | 29 | 20.0 |
| xgboost | 0.795 | 0.613 | 905 | 80.7 |
| knn | 0.793 | 0.648 | 11 | 19.5 |
| cellmapper_linear | 0.776 | 0.553 | 58 | 31.5 |
| naive_bayes | 0.738 | 0.613 | 31 | 19.5 |

What this run shows:

- Against its sibling `mlp`, a gene-space MLP against a PCA-space one, accuracy is a dead
  heat (0.837 against 0.838) and so is macro-F1 (0.655 against 0.664), while actinn-jax is
  about twice as fast on the same box (165 against 327 s per dataset) at essentially equal
  memory.
- Against xgboost, the only heavyweight in this tier, actinn-jax is more accurate (0.837
  against 0.795), about 5.5 times faster (165 against 905 s) and about 4 times lighter (21
  against 81 GB).
- The methods that beat it on speed, knn at 11 s, logistic at 29 s and naive_bayes at 31 s,
  are all less accurate (0.79, 0.81, 0.74). actinn-jax buys the best accuracy in this tier
  for a mid-pack runtime, and remains Pareto-efficient.
- Peak memory is I/O-bound rather than algorithmic. The whole light tier sits near 20 GB
  because that is the cost of loading the 1 to 20 GB train, test and solution h5ads, with
  the tabula_sapiens trio dominating. actinn-jax's model is a few hundred MB and its
  predict is sub-second. Only xgboost (81 GB) and cellmapper (31 GB) exceed the load floor.

Per-method same-hardware numbers: `docs/results_openproblems_samehw.csv`.

### Full field, from OP cloud CI

For the GPU, foundation and R methods not re-run above, these are OP's own Nextflow-trace
figures from their cloud CI, including container startup and h5ad I/O, on different
hardware. Read them as tiers rather than exact factors. actinn-jax's row here is its
Mac-standalone measure, shown only for rough placement; use the controlled table above for
any comparison.

| method | accuracy | macro-F1 | runtime (s) | peak mem (GB) |
|---|---|---|---|---|
| scanvi_scarches | 0.939 | 0.810 | 1542 | 39.3 |
| xgboost | 0.911 | 0.723 | 999 | 41.4 |
| seurat_transferdata | 0.827 | 0.662 | 934 | 48.8 |
| scanvi | 0.826 | 0.681 | 647 | 49.2 |
| singler | 0.745 | 0.605 | 3913 | 32.5 |
| scimilarity_knn | 0.711 | 0.566 | 814 | 38.8 |
| scgpt_zeroshot | 0.639 | 0.291 | 3259 | ~0* |
| uce | 0.131 | 0.043 | 11825 | 129.0 |

*scgpt_zeroshot's trace peak_rss is anomalous, being GPU-resident. The tier structure is
unambiguous: every method more accurate than actinn-jax here is far heavier, with
scanvi_scarches (0.939) at 39 GB, xgboost (0.911) at 41 GB, and seurat and scanvi, at about
actinn-jax's accuracy, near 49 GB. The foundation models are the extreme: uce needs 129 GB
and about 3.3 hours to score 0.131, roughly random, and singler takes 3913 s. This
independently reproduces our own finding ([PAPER.md](PAPER.md) §3.2) that a foundation
model's zero-shot labels are a weak annotation signal.

## actinn-jax against the top method, scanvi_scarches

scanvi_scarches (scVI latent plus scArches reference surgery) is 1st on mean accuracy. The
mean hides two things: it does not complete tabula_sapiens, and where both run the gap is
modest. Per dataset, showing actinn-jax baseline, with `standardize=True`, and scArches,
where gap is scArches minus standardized:

**Accuracy**

| dataset | actinn-jax | +std | scanvi_scarches | gap |
|---|---|---|---|---|
| dkd | 0.9471 | 0.9451 | 0.9551 | +1.00 |
| gtex_v9 | 0.8576 | 0.8613 | 0.8745 | +1.32 |
| immune_cell_atlas | 0.8590 | 0.8574 | 0.8917 | **+3.43** |
| mouse_pancreas | 0.9653 | 0.9659 | 0.9755 | +0.96 |
| hypomap | 0.9973 | 0.9982 | 0.9973 | −0.09 |
| tabula_sapiens | 0.3944 | 0.4049 | **DNF** | n/a |
| **mean (5 common)** | 0.9253 | 0.9256 | **0.9388** | **+1.32** |

**Macro-F1**

| dataset | +std | scanvi_scarches | gap |
|---|---|---|---|
| dkd | 0.9300 | 0.9404 | +1.04 |
| gtex_v9 | 0.4095 | 0.4822 | +7.27 |
| immune_cell_atlas | 0.7593 | 0.7517 | **−0.76** |
| mouse_pancreas | 0.7695 | 0.8843 | **+11.48** |
| hypomap | 0.9950 | 0.9915 | −0.35 |
| **mean (5 common)** | 0.7727 | **0.8100** | **+3.73** |

- The edge is modest on accuracy, about 1.3 points, and larger on macro-F1, about 3.7
  points, across the five datasets both complete. The F1-heavy gap is the tell: scArches's
  batch-corrected latent helps rare and fine cell types most, with mouse_pancreas F1 up
  11.5 on rare islet subtypes and gtex F1 up 7.3, which is where a gene-space MLP lags.
- It is not uniformly ahead. actinn-jax with standardization beats scArches on macro-F1 for
  immune (0.76 points) and hypomap (0.35), and ties it on hypomap accuracy.
- The structural trade is that scArches is 1 to 4 points better where it runs but fails on
  tabula_sapiens, which actinn-jax completes at 0.405. That is why actinn-jax tops the
  all-6-completers ranking.
- Standardization does not close the gap. On the 5 common datasets it barely moves
  actinn-jax, from 0.9253 to 0.9256; its gains land on tabula_sapiens, where scArches does
  not finish, and on gtex F1. The residual 1.3 accuracy and 3.7 F1 is a genuine VAE-latent
  benefit on fine and rare types, not something a CPU normalization recovers.

### Annotated UMAPs

![immune_cell_atlas annotated UMAP](figures/umap_immune_cell_atlas.png)

![gtex_v9 annotated UMAP](figures/umap_gtex_v9.png)

Test cells embedded from the task-provided PCA, coloured by true label and by each model's
prediction on a shared palette. Coarse compartments are annotated correctly: in immune, NK
cells, B cells and monocytes are clean. Errors concentrate in the dense central T-cell
subtypes (CD4, CD8, memory, naive), which is where scArches's 3.4-point immune lead comes
from. The baseline and `+std` panels are near-identical by eye, since their difference is
rare-class recall rather than gross reassignment. Regenerate or add datasets with
`benchmark/explore/plot_umaps.py <dataset_dir> <out.png>`.

### On equal features, the two converge

![immune 4-panel: truth / actinn / actinn+std / scArches](figures/umap4_immune_cell_atlas.png)

scANVI with scArches was run locally on CPU (`benchmark/explore/run_scarches.py`, OP's exact
SCVI to SCANVI to `load_query_data` recipe) restricted to the same 1000 HVGs actinn-jax uses,
with a 30k-cell reference, both for CPU feasibility. Under that equal feature budget it
scores accuracy 0.857 and macro-F1 0.702 on immune, a tie with actinn-jax's 0.859 and 0.734,
which wins on macro-F1. All three prediction panels are visually near-identical.

The gap to scArches's all-gene leaderboard number of 0.892 therefore comes substantially
from feature and data budget, meaning 29k genes and the full 325k-cell reference, rather
than from the VAE latent itself. This is a single dataset in a resource-constrained CPU run,
so it is illustrative rather than a full re-benchmark; CPU cost was about 23 min at 1k HVG
and 30k cells, against days for the full all-gene config.

### Does actinn-jax improve with more genes?

OP feeds every method the same 1000 HVGs, and a gene-space MLP is the one most starved by
that. Sweeping actinn-jax's input gene budget on all 6 datasets, same hardware (seurat_v3
HVG, `standardize=True`, full reference; `benchmark/explore/gene_budget.py`):

![actinn-jax accuracy and macro-F1 against gene budget, all 6 datasets](figures/gene_budget_curve.png)

| dataset | acc 1k | acc 5k | Δacc | F1 1k | F1 5k | ΔF1 | fit 1k to 5k | peak GB 1k to 5k |
|---|---|---|---|---|---|---|---|---|
| dkd | 0.945 | 0.949 | +0.4 | 0.930 | 0.938 | +0.8 | 7 to 16 s | 3.5 to 3.7 |
| gtex_v9 | 0.861 | 0.891 | **+3.0** | 0.409 | 0.455 | +4.5 | 94 to 119 s | 6.1 to 5.8 |
| immune_cell_atlas | 0.857 | 0.891 | **+3.4** | 0.759 | 0.816 | +5.7 | 65 to 158 s | 11.9 to 7.4 |
| mouse_pancreas | 0.966 | 0.974 | +0.9 | 0.769 | 0.845 | +7.6 | 56 to 147 s | 13.3 to 9.4 |
| hypomap | 0.998 | 0.996 | −0.2 | 0.995 | 0.978 | −1.7 | 70 to 188 s | 17.9 to 13.7 |
| tabula_sapiens | 0.405 | **0.303** | **−10.2** | 0.147 | 0.135 | −1.1 | 101 to 223 s | 23.5 to 21.7 |
| **mean (6)** | **0.839** | **0.834** | **−0.5** | | | | | |
| mean (excl. TS) | 0.926 | 0.940 | **+1.5** | | | | | |

Four conclusions, of which the exception is the important one.

1. More genes help the ordinary datasets. Four of 6 gain, led by gtex at 3.0 and immune at
   3.4 points, and on those five the mean rises 1.5 points of accuracy. On immune, 5000
   genes reaches 0.891, matching scArches's full-config 0.892 (see the four-way above), so
   that gap was gene budget rather than the model.
2. tabula_sapiens regresses hard, by 10.2 points, from 0.405 to 0.303, monotonically worse
   with more genes. Its 284-cell test batch spans 160 fine types across many organs, so
   extra genes let the MLP overfit reference and organ-specific expression that the tiny
   shifted test batch cannot exploit. This is the classic regime where more features hurt
   under domain shift. hypomap is already saturated at 0.998 and gains nothing.
3. Net over all 6, 5000 genes is slightly worse, by 0.5 points, because tabula_sapiens's
   collapse outweighs the others' gains. A blanket `n_hvg=5000` would trade a modest lift on
   easy references for a large loss on the hardest one, which is the dataset that gives
   actinn-jax its all-6-completion standing. `n_hvg=1000` is the more robust default.
4. Cost is modest and bounded. 5000 genes is about 2 to 2.5 times the fit time, still under
   4 minutes, at comparable and data-load-dominated peak memory of 3 to 24 GB, sometimes
   lower than at 1000.

The gene budget is a tunable knob rather than a free lunch. Keep `n_hvg=1000` as the robust
default, raise it toward 5000 for well-behaved same-domain references, and for hard
cross-domain or fine-grained targets, fewer genes are safer. An adaptive rule, widening the
gene set only when train and test distributions look similar, is the real fix. Full sweep:
`docs/results_gene_budget.csv`.

#### Choosing the gene budget without test labels

If more genes help some datasets and wreck others, the practical question is whether the
budget can be chosen at inference, from the reference and the unlabeled query, without
peeking at test labels (which would be leakage) or looping over held-out test accuracy. Two
signals, computed with no test labels, checked against the actual 1k to 5k transfer outcome
on all 6 (`benchmark/explore/probe_signals.py`):

![deterministic label-free signals against transfer outcome](figures/gene_budget_signals.png)

| dataset | query cells/class | Δ ref held-out CV (1k to 5k) | Δ transfer acc | outcome |
|---|---|---|---|---|
| dkd | 406 | +1.2 | +0.4 | help |
| gtex_v9 | 217 | +4.8 | +3.0 | help |
| immune_cell_atlas | 123 | +5.2 | +3.4 | help |
| mouse_pancreas | 726 | +0.6 | +0.9 | help |
| hypomap | 84 | +3.4 | −0.2 | saturated |
| **tabula_sapiens** | **1.8** | **−9.9** | **−10.2** | **HURT** |

- Reference held-out CV is predictive. Sweep the gene budget and cross-validate on the
  reference, using no test labels. It rises for every safe dataset and is the one signal
  that drops for tabula_sapiens, by 9.9 points, tracking the 10.2-point transfer collapse
  almost exactly. So "expand genes only while reference CV improves" would correctly stop
  tabula_sapiens at 1000 and grow the others.
- Query cells per class is a free pre-flag. `n_query / n_ref_classes` needs no training,
  since the label space is the reference's types and query size is known. tabula_sapiens is
  1.8 and everyone else is at least 84. Below about 10 means a thin query over a fine label
  space, so do not expand.
- One signal did not work: a reference-against-query domain-classifier AUC saturates near
  1.0 for all datasets, because OP label-projection is cross-batch by construction, so it
  cannot discriminate.

The resulting rule needs no test-label loop: default `n_hvg=1000`, and expand toward 5000
only while reference held-out CV keeps improving and query cells per class stays comfortably
above 10. On these 6 that rule expands gtex, immune, mouse and dkd, leaves tabula_sapiens at
1000 and so dodges the 10-point loss, and merely over-recommends on saturated hypomap, which
wastes compute at no accuracy cost. With n=6 and a single clean failure case, this is
directional evidence for the rule rather than a tuned threshold. Signals:
`docs/results_probe_signals.csv`.

#### Fine sweep on the saturated case: hypomap, 1000 to 10000

![hypomap fine gene-budget sweep](figures/gene_budget_hypomap.png)

hypomap starts near its ceiling at 0.998 accuracy and 0.995 F1, so the 1.7-point macro-F1
drop seen at 5000 is worth resolving into a slide or a step. It is a step onto a hard
plateau. F1 falls only over 1000 to 3000, as a handful of rare-class cells flip (0.9950,
0.9915, 0.9783), and then accuracy and F1 are identical to four decimals from 3000 to
10000: the models retrain with different fit time and memory but emit the same test labels,
so the extra 7000 genes are inert. Meanwhile fit time climbs linearly, from 75 to 273 s, a
factor of 2.6. On an already-saturated, cleanly-separated reference, gene budget past about
1000 to 3000 is wasted compute, and the small F1 dip is a fixed rare-class reassignment
rather than a worsening trend.
(`benchmark/explore/gene_budget.py hypomap out.csv 1000_op,2000,3000,5000,7500,10000`)

## What the smoke-test resource revealed about the defaults

Every component in `openproblems_component/` is developed against OP's `resources_test`
copy of `cxg_immune_cell_atlas`: 437 training cells, 163 test cells, 31 classes, and the
task's own top-1000 HVG mask. It exists as a pre-flight check that a component runs at all,
not as a measurement. It measured something anyway.

Scored against ground truth on that split, actinn-jax at its defaults sat near the
majority-vote control while every convex baseline was fine:

| method | accuracy |
|---|---|
| celltypist | 0.503 |
| svm_sgd | 0.491 |
| linear_anova_pca | 0.485 |
| sctop | 0.393 |
| **actinn-jax (defaults)** | **0.086** |

The model was not the problem; the schedule was. Holding everything else fixed and training
longer walks it back up:

| actinn-jax setting | accuracy | macro-F1 |
|---|---|---|
| `num_epochs=50` (the default) | 0.086 | 0.035 |
| `num_epochs=200` | 0.264 | 0.108 |
| `num_epochs=500` | 0.307 | 0.138 |
| `num_epochs=500, batch_size=32` | 0.405 | 0.239 |
| `num_epochs=500, batch_size=32, lr=1e-3` | 0.399 | 0.248 |

The cause is that `auto_batch_size` clamped the minibatch to a floor of 128 while
`num_epochs` was a fixed 50, so the number of gradient steps a reference receives was
proportional to its cell count. 437 cells at batch 128 is three steps per epoch, about 170
updates for the entire fit, and the network barely leaves its initialisation. Accuracy on
this split tracks total gradient steps almost monotonically, which is the signature. Nothing
warned: no error, no convergence check, just bad labels.

The fix ([iandriver/actinn-jax#1](https://github.com/iandriver/actinn-jax/pull/1)) makes
`num_epochs` default to `'auto'` alongside `batch_size`, both pivoting on a 5,000-cell
threshold. Below it the batch shrinks toward 32 and the epoch count rises as 1/n; at 437
cells `'auto'` resolves to batch 32 and 573 epochs, the configuration of the best row above.
Below about 1,500 total steps it now warns instead of failing silently.

None of this moves anything on this page. At or above 5,000 cells the auto rules return
exactly the historical schedule (50 epochs; batch 128 up to 12.8k cells, then n/100 capped
at 1024), verified bit-identical in both weights and stdout. The smallest training set among
the six leaderboard datasets is dkd at 33,898 cells and the largest is tabula_sapiens at
482,868, so the entire leaderboard sits far above the pivot and every number above stands.
The defect only ever affected references of a few hundred cells, a regime this benchmark
never enters and the smoke test always does.

## Reading the result

1. **actinn-jax against its direct sibling `mlp`.** Both are multilayer perceptrons, and the
   only difference is the input: actinn-jax uses HVG genes with ACTINN's normalization and
   gene filtering, while `mlp` uses the 50-dimensional PCA. On mean accuracy the two are a
   statistical dead heat, with OP's leaderboard putting mlp at 0.828 and our same-hardware
   rerun at 0.838, bracketing actinn-jax's 0.837. The gene-space representation matches PCA
   on accuracy while staying directly interpretable in marker space, at about half the
   runtime on identical hardware (165 against 327 s per dataset), with a clear win on the
   hardest dataset.

2. **Robustness on the hardest task.** `tabula_sapiens`, with 160 fine cell types and a
   284-cell test batch, breaks the field: scanvi_scarches and xgboost do not complete it at
   all, and most others land at 0.15 to 0.38. actinn-jax's 0.394 is the best accuracy of any
   method that completed it, against mlp's 0.342 and seurat's 0.377. It is one of the few
   methods that runs to completion on every dataset, which is why it tops the
   completed-all-6 ranking.

3. **Foundation models underperform on label projection.** scgpt_zeroshot (0.639, and only
   on the one dataset it ran) and uce (0.131, barely above the random-labels control at
   0.124) sit at the bottom. This independently reproduces our own finding
   ([PAPER.md](PAPER.md) §3.2): a foundation model's zero-shot labels are a weak annotation
   signal, while its embeddings are where the value is.

4. **The cost.** actinn-jax trains in gene space, so its fit time scales with the training
   set. On the controlled AWS run it averages 165 s per dataset for load, fit and predict
   through the OP harness, which is mid-pack in the CPU tier and about twice as fast as mlp.
   A Mac-standalone breakdown puts fit at 8.6 s on 34k cells rising to 112.6 s on the
   482k-cell tabula_sapiens, with predict under a tenth of a second (0.07 to 0.11 s), so
   with the train-once and map-many cache the fit is paid once. Peak memory, about 21 GB on
   the same hardware, is dominated by loading the h5ads rather than by the model. Full-gene
   training on the 482k by 56k atlas ran out of memory on 51 GB of RAM; the HVG restriction,
   which the task provides and its PCA methods already use, is what makes it tractable.

## Bottom line

On a benchmark actinn-jax's authors did not design and cannot tune, run to the community's
own protocol, actinn-jax places 3rd of 17 on accuracy and 1st among all methods that
complete every dataset. It beats its PCA-space `mlp` sibling on both metrics and posts the
best accuracy on the single hardest dataset. It is not the top method overall, since scANVI
with scArches reference surgery and xgboost score higher on the datasets they finish, and it
is upper-mid rather than top on macro-F1. As a fast, dependency-light, CPU gene-space
classifier, its standing on this external leaderboard is strong, and unlike our in-house
benchmark it is entirely independent.

*Datasets © CZ CELLxGENE / Open Problems, downloaded from the public `openproblems-data`
S3 bucket. Leaderboard scores: OP run `2026-06-05_12-50-03`, `score_uns.yaml`.*
