# Large-reference vs. narrowed-reference: how bad is it, and can it be fixed at runtime?

**Question:** the shipped `broad_human_v1` reference covers ~800 cell types across the
whole body. Any one dataset realistically contains a few dozen. Does the crowd of
irrelevant types hurt fine-grained accuracy, and if so, can we narrow the reference to
just the types present in a *specific* query — automatically, without ground truth, and
without ever throwing away a type that's genuinely there?

**Setup:** the shipped `broad_human_v1` model (798 types, 28 coarse groups) against two
real, ground-truth queries: the Krasnow lung atlas (65,662 cells, 46 true types) and a
CELLxGENE liver pull with zonation labels (5,566 cells, 12 true types). Scripts:
`benchmark/explore/refine_experiment.py`, `tune_refine_threshold.py`. Full numbers:
`docs/results_refine.csv`, `docs/results_refine_threshold_sweep.csv`.

## How bad is it?

| query | baseline (798 types) exact-CL | ontology-concordant |
|---|---|---|
| lung (46 types) | **0.13** | 0.52 |
| liver (12 types) | **0.29** | 0.46 |

Exact-name/CL accuracy is low; ontology-concordant (same lineage, e.g. a macrophage
subtype for a macrophage) is much better — most of the "error" is fine-grained sibling
confusion, not wild misclassification. See the [timing notebook](../../actinn-jax/examples/annotate_with_timing.ipynb)
and [liver zonation notebook](../../actinn-jax/examples/liver_zonation.ipynb) for that
distinction in context.

## What's actually causing it? Two separate problems, not one

Breaking the baseline error down by pipeline stage (only over cells whose true type is
in the 798-type vocabulary — 100% for both queries):

| query | coarse-routing accuracy | fine accuracy *given correct coarse group* |
|---|---|---|
| lung | 0.63 | 0.21 |
| liver | 0.62 | 0.46 |

**~37-38% of cells never reach the right coarse group at all** — a 28-way routing
decision that is itself imperfect, because each coarse classifier is trained on the same
sparse per-type data (~15-40 cells/type at this reference's scale). No amount of
fine-level narrowing can recover a cell that was routed to the wrong bucket in the first
place; that's a training-data/model-capacity problem, not a candidate-set problem.

**Even within the correct group, fine accuracy is only 0.21-0.46.** This is the part
narrowing *can* address — too many biologically similar sibling types compete in one
softmax, each with only a handful of training cells.

## How much can narrowing help? (ceiling, using ground truth)

| query | method | exact-CL | ontology | classes used |
|---|---|---|---|---|
| lung | baseline | 0.134 | 0.524 | 798 |
| lung | **oracle-mask** (mask to the true 46, no retrain) | 0.299 | 0.447 | 46 |
| lung | **oracle-retrain** (retrain on the true 46) | **0.522** | **0.668** | 46 |
| liver | baseline | 0.286 | 0.461 | 798 |
| liver | **oracle-mask** | 0.398 | 0.470 | 12 |
| liver | **oracle-retrain** | **0.453** | **0.505** | 12 |

Two things stand out:

1. **Masking alone recovers real accuracy** (0.13→0.30 lung, 0.29→0.40 liver) just by
   removing implausible competitors from the softmax — no retraining, no extra data.
2. **Retraining does meaningfully better than masking** (0.30→0.52 lung, 0.40→0.45
   liver). Masking restricts a *frozen* classifier's candidate set; retraining reshapes
   the decision boundary itself using only the relevant classes — a materially different
   (and better) function, not just a restricted view of the old one. The coarse
   classifier also becomes an easier few-way (not 28-way) problem when retrained.
3. **Curiously, oracle-masking's *ontology*-concordance on lung is slightly *worse* than
   baseline** (0.447 vs 0.524) despite exact-match improving. Forcing the model to choose
   among only the 46 true types removes the option to hedge onto a generic ancestor label
   (e.g. plain "macrophage") that would have counted as ontology-correct — masking trades
   some hedged-but-lineage-correct calls for specific-but-wrong ones. Worth knowing if you
   care about ontology-level correctness more than exact labels.

## Is there a reliable way to do this without ground truth?

We shipped `actinn_jax.refine_to_query(model, adata)`: it reads the model's *own*
predictions on your query (probability mass per class, argmax win-counts, confidence) and
masks out classes with no supporting evidence — no ground truth, no retraining, no extra
data, using the same mask-and-renormalize mechanism validated above.

**Recall (not dropping real types) works well.** On lung it recovered 44/46 true types
(missed 2); on liver it recovered 9/12 (missed `B cell`, `natural killer cell`,
`erythroid lineage cell`). We checked those three misses directly: the model's own
fine classifier assigns them **literally zero argmax wins** across 500+500+66 real cells
of those exact types — the evidence a detector could act on simply isn't there. No
threshold can recover a class the underlying classifier never once favors; that's a
retraining problem (see oracle-retrain above), not a detection-threshold problem.

**Precision (not admitting absent types) does not work well**, and this is the important,
honest finding. We swept six detection rules — absolute mass/count thresholds, per-group
relative thresholds, confidence floors, and a per-group cumulative-coverage ("elbow")
rule:

| query | rule | classes kept (of 798) | precision vs. oracle | recall vs. oracle | resulting exact-CL |
|---|---|---|---|---|---|
| lung | current default | 460 | 0.10 | 0.96 | 0.134 (≈ baseline) |
| lung | tightest tested (top1_frac≥1%, conf≥0.5) | 130 | 0.22 | 0.61 | 0.129 |
| liver | current default | 200 | 0.04 | 0.75 | 0.286 (≈ baseline) |
| liver | tightest tested (top1_frac≥1%, conf≥0.5) | 39 | 0.21 | 0.67 | 0.290 |

**Every rule we tried left accuracy essentially unchanged from baseline** — even the
tightest one, which cut kept-classes by 3-5× and roughly doubled precision. Only the
*oracle* mask (precision = 1.0, the exact true set) produced the real gain shown above.
Why: the handful of classes doing the actual damage are not random noise with low,
prunable confidence — they are the model's genuinely-confusable siblings of real types,
and they carry the *same* mass/confidence signature as real rare types, because the same
underlying classifier that's confused about telling them apart also can't be used to
detect that confusion. A detector built from a classifier's own output inherits that
classifier's blind spots. Diffuse, no-evidence classes prune away easily (that's most of
the reduction we do see) but they were never the source of the error to begin with.

## Bottom line

- `refine_to_query` is safe to use by default: in every test here it never made
  accuracy *worse*, it protects real types well (high recall), and it's free (no retrain,
  no extra data, sub-second). Use it as a light, no-downside pruning pass.
- It is **not** a fix for the large-reference accuracy gap. Don't expect it to close the
  0.13→0.52 (lung) or 0.29→0.45 (liver) gap — that requires retraining.
- **The reliable way to get that gain today is retraining on a narrower, focused
  reference** — `examples/build_reference.py` in actinn-jax, using your own labeled data
  or a hand-picked subset of the census-wide reference for your tissue/expected types.
  That's a real, validated win, not a heuristic.
- The other lever, not explored here: more cells per type in the underlying reference.
  The census-wide pull deliberately capped ~15-40 cells/type to keep the reference small;
  the coarse-routing accuracy (0.62-0.63) and within-group fine accuracy (0.21-0.46) are
  both very plausibly data-starved, not just crowded. A future rebuild with a higher
  per-type cap (fewer total types, or a bigger reference) is the natural next experiment.
