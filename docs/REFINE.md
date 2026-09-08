# Narrowing a large reference at runtime: how much it helps, and what it cannot fix

The shipped `broad_human_v1` reference covers about 800 cell types across the whole body,
while any one dataset realistically contains a few dozen. Two questions follow. Does the
crowd of irrelevant types hurt fine-grained accuracy, and can the reference be narrowed to
the types present in a specific query automatically, without ground truth, and without ever
discarding a type that is genuinely there?

The setup is the shipped `broad_human_v1` model (798 types, 28 coarse groups) against two
real ground-truth queries: the Krasnow lung atlas (65,662 cells, 46 true types) and a
CELLxGENE liver pull with zonation labels (5,566 cells, 12 true types). Scripts:
`benchmark/explore/refine_experiment.py`, `tune_refine_threshold.py`. Full numbers:
`docs/results_refine.csv`, `docs/results_refine_threshold_sweep.csv`.

## How bad is it?

| query | baseline (798 types) exact-CL | ontology-concordant |
|---|---|---|
| lung (46 types) | **0.13** | 0.52 |
| liver (12 types) | **0.29** | 0.46 |

Exact name and CL accuracy is low, while ontology-concordant accuracy, which credits a
macrophage subtype called for a macrophage, is much better. Most of the error is
fine-grained sibling confusion rather than wild misclassification. The
[timing notebook](../../actinn-jax/examples/annotate_with_timing.ipynb) and
[liver zonation notebook](../../actinn-jax/examples/liver_zonation.ipynb) show that
distinction in context.

## Two separate problems, not one

Breaking the baseline error down by pipeline stage, over only those cells whose true type
is in the 798-type vocabulary, which is 100% for both queries:

| query | coarse-routing accuracy | fine accuracy *given correct coarse group* |
|---|---|---|
| lung | 0.63 | 0.21 |
| liver | 0.62 | 0.46 |

About 37 to 38% of cells never reach the right coarse group at all. That is a 28-way
routing decision which is itself imperfect, because each coarse classifier is trained on
the same sparse per-type data, roughly 15 to 40 cells per type at this reference's scale.
No amount of fine-level narrowing can recover a cell routed to the wrong bucket in the
first place; that is a training-data and model-capacity problem rather than a
candidate-set problem.

Even within the correct group, fine accuracy is only 0.21 to 0.46. This is the part
narrowing can address, where too many biologically similar sibling types compete in one
softmax, each with only a handful of training cells.

## The ceiling, measured with ground truth

| query | method | exact-CL | ontology | classes used |
|---|---|---|---|---|
| lung | baseline | 0.134 | 0.524 | 798 |
| lung | **oracle-mask** (mask to the true 46, no retrain) | 0.299 | 0.447 | 46 |
| lung | **oracle-retrain** (retrain on the true 46) | **0.522** | **0.668** | 46 |
| liver | baseline | 0.286 | 0.461 | 798 |
| liver | **oracle-mask** | 0.398 | 0.470 | 12 |
| liver | **oracle-retrain** | **0.453** | **0.505** | 12 |

Three things stand out.

1. Masking alone recovers real accuracy, from 0.13 to 0.30 on lung and 0.29 to 0.40 on
   liver, just by removing implausible competitors from the softmax, with no retraining and
   no extra data.
2. Retraining does meaningfully better than masking, from 0.30 to 0.52 on lung and 0.40 to
   0.45 on liver. Masking restricts a frozen classifier's candidate set, while retraining
   reshapes the decision boundary itself using only the relevant classes, which is a
   materially different and better function rather than a restricted view of the old one.
   The coarse classifier also becomes an easier few-way problem instead of a 28-way one
   when retrained.
3. Oracle-masking's ontology concordance on lung is slightly worse than baseline, 0.447
   against 0.524, even though exact match improves. Forcing the model to choose among only
   the 46 true types removes the option to hedge onto a generic ancestor label such as
   plain "macrophage", which would have counted as ontology-correct. Masking therefore
   trades some hedged-but-lineage-correct calls for specific-but-wrong ones, which is worth
   knowing if ontology-level correctness matters more than exact labels.

## Can this be done without ground truth?

`actinn_jax.refine_to_query(model, adata)` reads the model's own predictions on the query,
meaning probability mass per class, argmax win counts and confidence, and masks out classes
with no supporting evidence. No ground truth, no retraining, no extra data, using the same
mask-and-renormalize mechanism validated above.

Recall, meaning not dropping real types, works well. On lung it recovered 44 of 46 true
types and on liver 9 of 12, missing `B cell`, `natural killer cell` and `erythroid lineage
cell`. We checked those three misses directly: the model's own fine classifier assigns them
zero argmax wins across 500, 500 and 66 real cells of those exact types, so the evidence a
detector could act on is not there. No threshold can recover a class the underlying
classifier never once favors, which makes it a retraining problem rather than a
detection-threshold problem.

Precision, meaning not admitting absent types, does not work well, and that is the
important finding. We swept six detection rules: absolute mass and count thresholds,
per-group relative thresholds, confidence floors, and a per-group cumulative-coverage
"elbow" rule.

| query | rule | classes kept (of 798) | precision vs. oracle | recall vs. oracle | resulting exact-CL |
|---|---|---|---|---|---|
| lung | current default | 460 | 0.10 | 0.96 | 0.134 (≈ baseline) |
| lung | tightest tested (top1_frac≥1%, conf≥0.5) | 130 | 0.22 | 0.61 | 0.129 |
| liver | current default | 200 | 0.04 | 0.75 | 0.286 (≈ baseline) |
| liver | tightest tested (top1_frac≥1%, conf≥0.5) | 39 | 0.21 | 0.67 | 0.290 |

Every rule we tried left accuracy essentially unchanged from baseline, even the tightest,
which cut kept classes by a factor of 3 to 5 and roughly doubled precision. Only the oracle
mask, at precision 1.0 on the exact true set, produced the real gain shown above.

The reason is that the handful of classes doing the damage are not random noise with low,
prunable confidence. They are the model's genuinely confusable siblings of real types, and
they carry the same mass and confidence signature as real rare types, because the same
underlying classifier that cannot tell them apart also cannot be used to detect that
confusion. A detector built from a classifier's own output inherits that classifier's blind
spots. Diffuse, no-evidence classes prune away easily, which is most of the reduction we do
see, but they were never the source of the error.

## Bottom line

- `refine_to_query` is safe to use by default. In every test here it never made accuracy
  worse, it protects real types well through high recall, and it is free, needing no
  retrain, no extra data and under a second. Use it as a light pruning pass with no
  downside.
- It is not a fix for the large-reference accuracy gap. It will not close the gap from 0.13
  to 0.52 on lung or 0.29 to 0.45 on liver, which requires retraining.
- The reliable way to get that gain today is retraining on a narrower, focused reference,
  using `examples/build_reference.py` in actinn-jax with your own labeled data or a
  hand-picked subset of the census-wide reference for your tissue and expected types. That
  is a validated win rather than a heuristic.
- One lever is not explored here: more cells per type in the underlying reference. The
  census-wide pull deliberately capped each type at 15 to 40 cells to keep the reference
  small, and both the coarse-routing accuracy of 0.62 to 0.63 and the within-group fine
  accuracy of 0.21 to 0.46 are plausibly data-starved rather than merely crowded. A rebuild
  with a higher per-type cap, whether through fewer total types or a bigger reference, is
  the natural next experiment.
