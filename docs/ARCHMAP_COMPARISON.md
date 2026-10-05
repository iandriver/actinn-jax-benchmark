# ACTINN-JAX and ArchMap

ArchMap makes reference mapping and downstream inspection accessible through a web platform.
ACTINN-JAX supplies a compact label predictor and reference-building tools. Both support
reference-based annotation, but their mapping operations produce different outputs. The
ACTINN-JAX benchmark does not establish a speed or accuracy ranking against ArchMap.

The source is Lotfollahi, Bright, Skorobogat et al., [ArchMap is a web-based platform for
reference-based analysis of single-cell datasets](https://doi.org/10.1038/s41588-026-02756-y),
Nature Genetics, published 30 September 2026. The supplied PDF contains the main text,
Methods, four extended-data figures and the reporting summary. Separate Supplementary
Notes 1 to 3 were not included in that PDF; this comparison uses the included material.

## What each operation does

| question | ACTINN-JAX in this study | ArchMap in the published paper |
|---|---|---|
| What is fitted? | A classifier on reference expression and labels | An integrated reference model, with query-specific scArches adaptation |
| What does mapping return? | Labels, confidence and screening outputs | A joint reference-query embedding, transferred labels and mapping diagnostics |
| What can be reused? | Saved references; fitted linear baselines can also be reused | Pretrained atlas models, reference embeddings and label classifiers |
| Can users bring a reference? | A labeled reference can be trained for a focused label set | Custom atlases can be uploaded and evaluated before public or private use |
| How is uncertainty inspected? | Maximum model confidence, held-out-type threshold sweeps and low-confidence clusters | Weighted-neighbor and Mahalanobis uncertainty, anchors and reference-state presence |
| Where can it run? | Local CPU prediction, with some reference-building routes using accelerators | Hosted service, self-hosted application or local mapping notebook |

A label classifier asks which reference label best fits a cell. Query adaptation also asks
how that cell should be represented beside reference cells after accounting for dataset
differences. Training or adaptation time is part of the second task. Calling both operations
“prediction” makes a runtime ratio hard to interpret unless their outputs and boundaries are
made explicit.

## Evidence in the ArchMap paper

The main demonstration maps idiopathic pulmonary fibrosis data to HLCA. It reports 96.85%
reference-gene overlap, 53.96% of query cells with anchors, 8.99% labeled unknown and a
cluster-preservation score of 5/5. These are mapping diagnostics, not cell-type accuracy
estimates. High-uncertainty adventitial fibroblasts show an association with diseased cells
and marker expression, including CTHRC1. That association supports biological inspection;
it does not by itself establish a causal driver gene.

The Methods describe 100 epochs of query adaptation. Extended Data Figure 2 evaluates epoch
choice on healthy and disease lung queries using reconstruction loss, integration metrics
and embeddings. Extended Data Figure 3 compares KNN, XGBoost and native classifiers in
atlas embeddings under an 80–20 split; it is not a matched cross-study test against the
ACTINN-JAX panel. Extended Data Figure 4 evaluates separate versus combined batch mapping.
The platform's 200,000-cell query limit is a reported service constraint, with batch-wise
mapping supplied as a workaround; it should not be interpreted as an algorithmic scaling
limit. The reference-gene overlap recommendation is specific to their tested mapping setup.

ArchMap explicitly warns that residual batch effects and low-quality cells can cause high
uncertainty. The same distinction applies to ACTINN-JAX's low-confidence novelty screen.
Expression markers, quality checks and independent study evidence are needed to interpret
candidate populations biologically.

## Agreement and disagreement

| ArchMap statement or interpretation | Our position | Evidence and boundary |
|---|---|---|
| Published and custom atlases enable reusable reference-based analysis | Agree | Focused-reference annotation improves lineage concordance on a held-out liver study. This capability is already present in ArchMap. |
| High uncertainty can indicate disease-associated or unfamiliar states, but also batch effects or poor quality | Agree | Held-out-type screening supports inspection of uncertain cells. Neither paper treats uncertainty alone as proof of novelty. |
| A classifier can be chosen using the percentage of query cells labeled unknown (Figure 1 caption) | Qualify the selection criterion | Unknown fraction measures abstention, not accuracy. Our thresholds produce different coverage and false-rejection behavior. We have not directly evaluated ArchMap's rule. |
| Mapping requires substantial expertise and can require GPU compute | Separate the requested outputs | Frozen label prediction can be inexpensive on CPU. Integrated mapping does additional work; our results do not establish that its computation is unnecessary. |
| Integrated atlas embeddings retain biological structure and correct batch effects | Unmeasured here | Label accuracy does not test batch mixing or biological conservation. No disagreement with the reported integration benchmarks is established. |
| The hosted interface makes mapping accessible and supports collaboration | Unmeasured here | We have not tested usability, service runtime, collaboration or access barriers. Local execution and custom references are not unique to ACTINN-JAX. |

## Matched annotation evidence

scArches with weighted-neighbor label transfer, the linear baseline and ACTINN-JAX are
compared on identical query cells within eight splits (72 scored repeats). scArches has
higher accuracy and macro-F1 on PBMC, within-dataset lung and both brain tasks; the linear
baseline has higher scores on both liver splits and blood/gut. At 151 brain clusters,
accuracy is 0.860 for scArches, 0.816 for the linear baseline and 0.741 for ACTINN-JAX.
scArches takes longer but uses less peak memory. Its 15-neighbor weighted transfer and
100 query-adaptation epochs follow ArchMap; this is not a replication of ArchMap's atlases,
reference training or hosted execution.

A faster native-head setting is a secondary configuration (Supplementary Table S5).
Label transfer, batch handling, architecture and training budget differ jointly; their
individual effects are not isolated. The eleven-method and matched three-method panels
use different cohorts and do not support paired cross-panel rankings. Lung cross-dataset
transfer is evaluated using Cell Ontology concordance; its exact-name metrics are not used.


### Direct local annotation comparison

We ran the local ArchMap HLCA annotation pipeline on 53,904 Duong lung nuclei using
its pretrained checkpoint and published KNN and XGBoost classifiers. Source manual
annotations mapped to Cell Ontology terms covered 52,401 cells (97.2%). Concordance
credits the same term or an ancestor–descendant relationship; it does not establish
fine subtype identity, including the distinction between general capillaries and aerocytes.
Across three seeds, full-coverage concordance was 0.847 for ArchMap KNN, 0.851 for
ArchMap XGBoost and 0.843 for ACTINN-JAX. At 90% retained coverage, concordance was
0.888, 0.885 and 0.896, respectively. Thus, relative performance depended on coverage.
Confidence ranking used no truth labels; each method retained its own subset of cells.

Baselines used the same 584,944 reference cells and ann_level_5 target, with 1,992
measured genes shared with the 2,000-gene pretrained model. The linear baseline reached
0.768 concordance at full coverage after convergence. This gene budget differs from
our main linear benchmark. Duong has no supplied reference cell-ID or study overlap,
but was used for HLCA uncertainty-threshold calibration [Sikkema 2023]; it is not an
untouched external test. Resolved local software versions differ from the published
environment. These results evaluate annotation agreement, not integrated mapping quality.

In this local pipeline, the KNN and XGBoost classifiers share weighted-neighbor
uncertainty and therefore have identical unknown fractions despite different predicted
labels. Unknown fraction alone cannot distinguish their annotation accuracy.

