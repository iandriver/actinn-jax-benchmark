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
accuracy with 0.178 macro-F1 there. A macro-F1 of 0.18 on the HLCA, arrived at
independently, is not a coincidence worth defending.

## What changed

Four things, in `benchmark/adapters/scvi_adapter.py`:

**Label transfer.** Labels now come from a weighted kNN over the latent space rather than
from scANVI's classifier head. Each query cell takes a Gaussian-weighted vote from its 15
nearest reference cells; the winning label's share of the weight is returned as its
probability. This is the transfer scArches itself ships in `scarches.utils.knn`, and the
one the HLCA paper and ArchMap both use. The implementation is checked against that
function directly: identical labels, and probabilities agreeing to 2.2e-16.

The probability has a second use. ArchMap defines its Euclidean uncertainty as 1 minus this
quantity and labels a cell `unknown` above 0.5, reasoning that below a half no label is a
stronger prediction than its alternatives. That is the same operating point as actinn-jax's
`min_prob=0.5`, reached from a different mechanism, so the abstention results in
[`PAPER.md`](PAPER.md) §3.6 now compare like with like.

**A batch covariate.** The reference model is set up with one, picked from the first usable
column of `batch`, `donor_id`, `STUDY`, `dataset`, `sample` or `assay`. Without it the
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
macro-F1 gain belongs to the corrections. Accuracy moves 0.9 points. A gain that lands on
macro-F1 and not on accuracy is a gain on the rare classes, which is where ArchMap's figure
says the native head loses.

Fit time triples and predict time doubles. That is the honest price of the configuration
that works, and it is paid in the column where scArches was already slowest.

Config: [`configs/scarches_fix_pbmc.yaml`](../configs/scarches_fix_pbmc.yaml). Results:
[`results/scarches_fix_pbmc/results.csv`](../results/scarches_fix_pbmc/results.csv).

## What this does not change

scANVI keeps its own classifier head. For scArches the head is a free choice, and both the
method's authors and its users make it differently from the default; for scANVI the head is
the method. Substituting a kNN there would be benchmarking something else.

## Citation

Lotfollahi, M., Bright, C., Skorobogat, A. et al. **ArchMap is a web-based platform for
reference-based analysis of single-cell datasets.** *Nature Genetics* (2026).
https://doi.org/10.1038/s41588-026-02756-y
