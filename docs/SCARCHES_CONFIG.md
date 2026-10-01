# How scArches is configured, and why the first configuration was wrong

scArches appears in the benchmark as a deep reference-mapping baseline: train scANVI on the
reference, then run the surgery that adapts that model to a query it has never seen. The
first version of this benchmark ran it on scvi-tools defaults and read labels off scANVI's
own classifier head. That is a defensible reading of "run scArches", and it is the wrong
one. This note records what changed, what it cost, and how we know.

## What prompted the check

[ArchMap](https://doi.org/10.1038/s41588-026-02756-y) (Lotfollahi, Bright, Skorobogat et
al., *Nature Genetics* 2026) is a web platform whose backbone is scArches, and whose authors
wrote scArches. To choose a default for their users they benchmarked three label-transfer
heads over the same reference embedding, on three atlases, with 80-20 splits. Their
Extended Data Fig. 3 reports F1 per cell type with accuracy, macro average and weighted
average:

| atlas (label level) | weighted kNN | XGBoost | native head |
|---------------------------|------:|------:|------:|
| HLCA (`ann_level_3`) accuracy | 0.98 | 0.98 | 0.64 |
| HLCA macro avg F1 | 0.97 | 0.95 | **0.18** |
| pancreas (`cell_type`) macro avg F1 | 0.95 | 0.95 | 0.73 |
| HEOCA (`level_2`) accuracy | 0.99 | 0.98 | 0.82 |
| HEOCA macro avg F1 | 0.99 | 0.97 | 0.62 |

The native head is scANVI's on the first two rows and scPoli's on the third. It is the
weakest of the three on every atlas, and on the HLCA it collapses: 0.18 macro-F1 against
0.97. The head this benchmark was reading.

The benchmark's own `lung_cross` split is HLCA to krasnow, and scArches scored 0.354
accuracy with 0.178 macro-F1 there. That resemblance is what prompted this check, and it
turned out to mean nothing: see the lung_cross section below. Two macro-F1 values near
0.18 on unrelated evaluations are not evidence of a shared cause.

## What changed

Four things, in `benchmark/adapters/scvi_adapter.py`:

**Label transfer.** Labels now come from a weighted kNN over the latent space rather than
from scANVI's classifier head. Each query cell takes a Gaussian-weighted vote from its 15
nearest reference cells; the winning label's share of the weight is returned as its
probability. This is the transfer scArches itself ships in `scarches.utils.knn`, and the
one the HLCA paper and ArchMap both use. The implementation is checked against that
function directly: identical labels, and probabilities agreeing to 2.2e-16.

ArchMap defines its Euclidean uncertainty as 1 minus this probability and labels a cell
`unknown` above 0.5, reasoning that below a half no label is a stronger prediction than its
alternatives. actinn-jax's `min_prob=0.5` is the same number on a different quantity, a
softmax output rather than a share of neighbour weight. Equal thresholds on quantities that
are not on a common scale are not matched coverage or matched calibration, and nothing here
licenses reading one method's abstention against the other's at a shared cutoff.

**A batch covariate.** The reference model is set up with one, picked from the first usable
column of `batch`, `dataset`, `STUDY`, `study`, `donor_id`, `sample` or `assay`. The order
takes the coarsest technical source first, because an integrated atlas is integrated across
datasets rather than donors and that is the key its own builders used. It also keeps the
levels populated: the HLCA subsampled to 300 cells per label has 14 datasets but 107 donors,
leaving about 134 cells per donor to learn a batch embedding from. Without it the
surgery has nothing to extend: scArches works by adding embeddings for the query's batches
while the shared weights stay frozen, so a model with no batch covariate has nowhere to put
them. Query batches are prefixed so they are always unseen categories, which is the
situation a user mapping new data is in, including on the within-dataset splits.

**The surgery architecture.** Layer normalisation in place of batch normalisation,
covariates fed to the encoder, two hidden layers and a 30-dimensional latent, rather than
scvi-tools defaults. Batch statistics do not survive a change of dataset, which is why the
scArches recipe replaces them.

**Training budget.** 200 reference epochs and 100 scANVI epochs with early stopping, then
100 query epochs fixed. The old budget was 40/20/40. ArchMap fixes query fine-tuning at 100
after sweeping 1 to 1,000 epochs against reconstruction loss, scIB metrics and UMAPs on
70,000 healthy and 118,000 disease cells mapped to the HLCA, and finding the loss flat past
about 100 (their Extended Data Fig. 2). scvi-tools' own default heuristic would have asked
for 400 epochs at these reference sizes, so 40 was well short of any published
recommendation.

The old configuration is still registered, as `scarches-native`, so the difference is
measurable in one run instead of across two states of the repository.

## What it costs and what it buys

pbmc3k, 8 types, three repeats, Apple MPS:

| | accuracy | macro-F1 | kappa | fit (s) | predict (s) | peak mem (MB) |
|---------------------------|------:|------:|------:|------:|------:|------:|
| paper run, as published | 0.940 | 0.888 | 0.918 | 12.5 | 2.4 | 1057 |
| `scarches-native`, rebuilt | 0.932 | 0.887 | 0.908 | 22.0 | 5.8 | 849 |
| `scarches`, corrected | 0.942 | **0.934** | 0.921 | 74.1 | 13.6 | 778 |

The rebuilt `scarches-native` reproduces the published macro-F1 to 0.001, which is what
makes the third row readable: the refactor did not move the old path, so the 4.7-point
macro-F1 gain belongs to the four changes together. It does not say which one earned it.
All four moved at once, so this is a comparison of two configurations, not an ablation, and
no claim about the label-transfer head on its own is supported by it. Accuracy moves 0.9
points. A gain that lands on macro-F1 and not on accuracy is a gain on the rare classes.

Fit time triples and predict time doubles. That is the honest price of the configuration
that works, and it is paid in the column where scArches was already slowest.

Config: [`configs/scarches_fix_pbmc.yaml`](../configs/scarches_fix_pbmc.yaml). Results:
[`results/scarches_fix_pbmc/results.csv`](../results/scarches_fix_pbmc/results.csv).

## lung_cross, where the premise for this check failed

The split that prompted the whole check does not move. Three repeats each:

| | accuracy | macro-F1 | ontology | fit (s) | predict (s) |
|---------------------------|------:|------:|------:|------:|------:|
| paper run, as published | 0.355 | 0.179 | 0.754 | 92 | 57 |
| `scarches-native`, rebuilt | 0.353 | 0.179 | 0.758 | 104 | 86 |
| `scarches`, corrected | 0.359 | 0.182 | 0.761 | 656 | 219 |

The reason is the label space, not the classifier. HLCA and krasnow share 20 of the 46
labels in the query, so 50.3% of query cells carry a label that no method can predict and
accuracy is capped at 0.497 before a cell is classified. Restricted to the 20 reachable
classes, macro-F1 is 0.758 corrected and 0.747 native, with 17 to 18 of 20 classes scoring
above zero. Nothing was collapsing.

ArchMap's 0.18 comes from an 80-20 split inside one atlas, where every label is reachable
by construction. The two numbers are not measuring the same thing. `PAPER.md` already
treats this split's exact-match score as a vocabulary artifact and excludes it from the
accuracy mean, which is the correct handling and was so before this check.

One observation that does survive, and is about the weighted kNN rather than about
lung_cross: it returns a mean probability of 0.967, with 99.3% of query cells above
ArchMap's 0.5 cutoff, while 64% of its labels are wrong. On a cross-dataset query that
cutoff abstains on 0.7% of cells. ArchMap validates the Euclidean uncertainty within an
atlas and does not claim it for this regime, so this is a limit of transfer, not a defect.

## The shared-vocabulary splits

These are the splits where the test means something, each corrected against native on the
identical inputs and the same code path, three repeats:

| split | types | accuracy gain | macro-F1 gain |
|------------------|------:|------:|------:|
| pbmc | 8 | +0.009 | +0.047 |
| lung_intra | 46 | +0.011 | +0.011 |
| liver_intra | 38 | +0.021 | +0.020 |

On lung_intra the native variant reproduces the published run to 0.0004 on both metrics,
the third independent time the control has matched (after pbmc and lung_cross). The liver
splits are rebuilt from a 38-type HLiCA against the paper's 36, so they carry an internal
comparison only; see `make_bench_datasets.py` for what is and is not established about
that input. Config: [`configs/scarches_fix_rest.yaml`](../configs/scarches_fix_rest.yaml).

The direction is consistent across every shared-vocabulary split. The size is not, and
with four factors changed together there is nothing here that attributes it.

## What this does not change

scANVI keeps its own classifier head. For scArches the head is a free choice, and both the
method's authors and its users make it differently from the default; for scANVI the head is
the method. Substituting a kNN there would be benchmarking something else.

## Citation

Lotfollahi, M., Bright, C., Skorobogat, A. et al. **ArchMap is a web-based platform for
reference-based analysis of single-cell datasets.** *Nature Genetics* (2026).
https://doi.org/10.1038/s41588-026-02756-y
