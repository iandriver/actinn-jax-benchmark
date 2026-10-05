# Accuracy and computational cost of reference-based cell-type annotation on commodity hardware

*Condensed version. The extended report carries the full protocols, the per-dataset results,
and the scaling and abstention studies summarized here.*

## Abstract

Reference-based cell-type annotation requires a choice of label resolution, reference and computational budget. We compared eleven reference-trained methods across eight splits of seven human datasets and evaluated two pretrained annotators where their vocabularies permitted comparison. We also used the Open Problems label-projection workflow. A matched comparison evaluated scArches with weighted-neighbor label transfer, a linear baseline and actinn-jax on identical query cells within eight splits. scArches gave the highest accuracy among these methods on PBMC, within-dataset lung and both brain tasks; the linear baseline led on liver and blood/gut. At 151 brain clusters, scArches scored 0.860 versus 0.816 for the linear baseline and 0.741 for actinn-jax, with complete-job times of 563, 6.3 and 20.4 s. scArches used less peak memory than either baseline. A saved actinn-jax reference annotated 524,699 cells in 31 s on the tested laptop. A focused liver reference improved ontology concordance from 0.580 to 0.858 on a study excluded from focused training. Distillation increased throughput with losses on withheld atlases; agreement identified higher-concordance subsets but did not improve the tested consensus over the best annotator. Our findings support ArchMap's emphasis on reference quality and uncertainty inspection. Unknown-cell percentage alone does not establish annotation accuracy. Label prediction and integrated atlas mapping require different evaluations; we did not assess ArchMap's integration quality or hosted performance. Code and pretrained references are publicly available.

## Key Points

- scArches has higher accuracy at fine brain resolution; the linear baseline has higher scores on liver and blood/gut. Time and memory move in different directions.
- A saved classifier can label large queries cheaply. Fitting and query adaptation must be reported separately from frozen prediction.
- Reference refinement and uncertainty inspection agree with ArchMap's motivation. Unknown-cell percentage alone cannot establish classifier accuracy.
- These experiments measure labels and screening, not ArchMap's integration quality or platform performance.

## Introduction

Reference-based annotation assigns labels to new single-cell RNA-seq data using a labeled
reference or a pretrained model. Previous comparisons measured annotation accuracy
[Abdelaal 2019], [Fu 2024]. Analysts also need to know the label resolution a method can
support and the time and memory needed for the intended workload.

ArchMap [Lotfollahi et al. 2026] combines scArches query adaptation with label transfer,
uncertainty scores and interactive inspection of published or custom atlases. It produces
a shared reference-query embedding as well as labels. Pan-human Azimuth [Sarkar et al. 2026]
supplies a pretrained hierarchical human annotator. We share their motivation to make
reference-based analysis accessible. Our measurements address annotation accuracy and
resource use, which are only part of an integrated mapping workflow.

We introduce actinn-jax, a JAX implementation of ACTINN [Ma & Pellegrini 2020] with sparse
preprocessing, saved references and chunked prediction. We compare eleven methods across
human datasets and evaluate scArches with weighted-neighbor label transfer against two
baselines on matched query cohorts. We separate reference fitting and query adaptation from
prediction using saved models. Focused-reference, distillation and agreement studies examine
how reference choice and repeated annotation affect the result.

The central question is which computation is justified by the required output. A cell-label
task can use a frozen classifier; placing cells in a shared atlas embedding additionally
requires evaluation of batch mixing and biological conservation. We measure the former.
We also test confidence-based screening, where coverage and errors must be considered
together. These distinctions define where our findings support or qualify interpretations
of the ArchMap paper.


## Materials and methods

*actinn-jax.* ACTINN [Ma & Pellegrini 2020] is a fully connected network with 100/50/25
hidden units, ReLU activations, softmax output and Adam training. The JAX implementation
replaces TensorFlow-1.x graph and session code, uses sparse preprocessing, saves fitted
references and predicts in chunks. The extended report compares this implementation with
TensorFlow ACTINN.

*Methods and datasets.* Eleven reference-trained methods run eight splits of seven human
datasets; two pretrained annotators are evaluated where their vocabularies permit scoring
(Supplementary Tables S1 and S2). The splits cover lung within-dataset and cross-dataset,
liver within-dataset and cross-study, blood and gut, PBMC, and Allen middle temporal gyrus
nuclei at subclass and cluster resolution. The brain tasks use the Allen taxonomy at two resolutions with separately sampled inputs. SVM, kNN
and CellTypist use untuned settings; the linear pipeline follows the Souza and Mehta recipe
[Souza & Mehta 2026]. This is not a comparison under equal tuning budgets.

*scArches configuration.* scArches uses scANVI reference mapping, a 30-dimensional latent space, two hidden layers,
layer normalization and a batch covariate. Reference scVI and scANVI training use up to 200
and 100 epochs with early stopping, respectively. Query adaptation uses 100 fixed epochs;
labels are transferred by Gaussian-weighted voting among 15 reference neighbors. The
weighted-neighbor transfer and query-adaptation budget follow ArchMap [Lotfollahi et al. 2026]; reference training, atlas inputs and hardware are specific to this benchmark.
Supplementary Table S5 describes a faster native-head configuration used in the eleven-method
panel and screening experiments. It has a smaller training budget and is reported separately
from the weighted-neighbor comparator.

*Metrics.* We report exact label-string accuracy, macro-F1 and ontology concordance.
Concordance credits the same term or an ancestor-descendant relation, allowing comparison
across vocabularies while also crediting some errors in specificity. Brain's supplied
ontology annotations map 24 subclasses to 18 terms and do not distinguish its 151 clusters;
brain is scored by exact match and macro-F1. This coarse crosswalk is a property of the
dataset annotation. The separate full-study liver demonstration reports exact ontology-term
match and lineage concordance, rather than label-string accuracy.

*Execution and accounting.* Methods run in separate environments on shared partitions,
with three repeats of each fixed partition. Repeats characterize training and runtime
variation, not donor or study generalization. The in-house panel uses Apple Silicon CPU
and MPS. Open Problems runs use AWS CPU instances through its Nextflow workflow. Stage
timings follow the adapters: scANVI trains jointly on reference and query in `predict`, and
scArches adapts to the query there. Compare fit plus predict for complete annotation cost
(Supplementary Table S4). Both actinn-jax and the linear baseline retain fitted components
for later queries. Shared-instance cost ratios remain sensitive to contention and threading.

*Agreement and screening.* The census model, Pan-human Azimuth and its distilled student
are run on identical query cells. Calls agree when their CL mappings are identical or
ancestor-descendant related. The teacher and student are dependent annotators. Confidence
thresholds are evaluated by withholding cell types from reference training. Low-confidence
clusters and markers are reported for inspection; uncertainty alone does not establish novelty.

*Scaling and workflows.* The runtime sweep holds the query fixed when varying reference
size. The atlas accuracy sweeps cap cells per type and then split the sample, so both
partitions change. The large-query sweep holds the reference fixed. Focused liver validation
excludes the query study, but source-study exclusion from broad census corpora is not
established. The supplement supplies scaling, abstention and per-class results; the extended
report provides full protocols.

## Results

The eleven-method panel includes scArches with its native classifier head. The matched
comparison in Figure 1 uses weighted-neighbor label transfer and the training settings
described in Methods.


### Accuracy and cost depend on the task

The top four mean accuracies span 0.008 across five shared-vocabulary non-brain splits
(Table 1). Their stage timings differ substantially, but the 58.7 s scANVI predict stage
includes training and the 0.24 s actinn-jax stage uses a fitted model. Their complete-job
times are 58.7 and 10.54 s, respectively. Fixed-partition repeats do not establish that
the methods are equivalent. actinn-jax and scANVI share the highest rounded ontology
concordance in this aggregate, 0.811.

![Matched annotation accuracy and resource use](figures/fig_matched_annotation.png)

**Figure 1.** Annotation agreement and cost. A: exact accuracy on shared-vocabulary tasks;
Cell Ontology concordance for lung across datasets (shaded row). Concordance credits identical
terms or an ancestor–descendant relationship, rather than fine-type accuracy. B: macro-F1 on
shared-vocabulary tasks. C: fit-plus-predict time (log scale). D: peak process memory. Methods
share query cells within each row. Points and bars show means and ranges of three repeats,
not donor confidence intervals. ArchMap platform and integration quality were not evaluated.

| method | acc | macro-F1 | ontology | fit (s) | predict (s) | peak mem (MB) |
|----------|----:|----:|----:|----:|----:|----:|
| **linear-anova-pca** | **0.839** | 0.803 | 0.808 | **2.5** | **0.31** | 4341 |
| scANVI | 0.836 | **0.808** | **0.811** | 0.0* | 58.7 | 2079 |
| scArches (native head) | 0.831 | 0.802 | 0.807 | 37.9 | 11.9 | 1774 |
| **actinn-jax** | 0.831 | 0.785 | **0.811** | 10.3 | **0.24** | 2407 |
| CellTypist | 0.823 | 0.792 | 0.805 | 44.0 | **0.52** | 1569 |
| SVM | 0.806 | 0.774 | 0.794 | 40.5 | **0.06** | 1416 |
| **ProtoCloud** | 0.790 | 0.742 | 0.778 | 118.6 | 1.32 | 1888 |
| SingleR | 0.770 | 0.746 | 0.750 | 0.2 | 34.7 | 3623 |
| kNN | 0.769 | 0.712 | 0.768 | 0.3 | **0.14** | 1489 |
| **scTOP** | 0.739 | 0.711 | 0.703 | 0.9 | 0.98 | 1927 |
| scmap-cluster | 0.646 | 0.626 | 0.771 | 0.2 | 7.33 | 8637 |

**Table 1.** Accuracy and macro-F1 average five shared-vocabulary non-brain splits; ontology concordance averages four splits. Brain: Supplementary Table S3. scArches uses its native head. *scANVI trains and scArches adapts queries in predict. Complete-job times: Supplementary Table S4.

Rankings differ between the in-house and Open Problems panels and across atlas samples
(Supplementary Figure S6). These comparisons change datasets, feature budgets or partitions,
and therefore do not isolate a single cause of the reversal.

Label resolution also matters. At Allen's 24 subclasses, ten methods fall within 0.017 of
each other. In the eleven-method panel at 151 clusters, SingleR leads at 0.845 and actinn-jax ranks tenth at 0.741. The weighted-neighbor scArches comparison uses a separately sampled cohort (Table 3).
Cluster labels derive from expression-space clustering, a possible explanation for the
strong correlation-method results; its causal contribution was not isolated.

The zero-shot scPRINT label head scores 0.201 ontology concordance on the lung query versus
0.917 for reference-trained actinn-jax. Their training information and vocabularies differ,
so this does not isolate architecture or assess every use of foundation-model embeddings.
The extended report separately describes results under the published Open Problems protocol
and the same-instance CPU comparison.

In the six-method CPU runtime sweep, prediction remains sub-second, and several fitted
models have nearly flat prediction cost as reference size increases (Supplementary Figure
S7). This behavior is not unique to actinn-jax. With a fixed reference and a large query,
actinn-jax labels 524,699 cells in 31 s versus 67 s for the linear pipeline; both have
similar whole-process peak memory at that query size (Supplementary Figure S8).

### Broad annotation followed by focused annotation

That budget pays for a workflow instead of a single call (Figure 2). A roughly 800-type census
reference provides labels over the types it covers without a supplied tissue; resolving those calls
through the reference's per-class tissue map identifies the tissue; a small focused reference
for that tissue then re-annotates the same cells at full resolution. On 56,545 `Andrews_2022` cells excluded from focused-reference training, the broad model
scores 0.231 by exact ontology-term match and 0.580 by lineage concordance; the 38-type
focused validation model scores 0.728 and 0.858. This full-study experiment differs from
Figure 2's capped, 3,396-cell comparison using a 36-type reference. The released liver
weights are trained on all studies, separately from these validation models.

The two passes hand off. Substituting a stronger broad model lifts the broad call but changes
nothing downstream, and using it to narrow the focused pass's classes makes the result worse,
from 0.731 to 0.708, because a wrong mask discards the correct class outright. Once the
focused reference covers the tissue, the broad model's value is routing rather than
resolution.

![workflow](figures/fig_workflow_umap_ondata.png)

**Figure 2.** The workflow on a withheld liver study (3,396 cells), same embedding
throughout. The census reference spreads 144 of its 798 labels over the query (concordance
0.34); resolving those calls to tissue gives 76% liver against 4% for the next candidate,
which selects the reference to load; the 36-type liver reference then re-annotates the same
cells at 0.73. Rightmost panel is the study's own labels. Only the focused reference has
a verified source-study holdout; the broad census corpus may include the query's study.

### Broad references and distillation

Building a broad reference from the census requires a foundation model on a GPU to discover
its hierarchy. A pretrained pan-human annotator [Sarkar et al. 2026] already publishes one, so
labeling a corpus with it and training on those labels transfers both vocabulary and
structure, using only raw counts, no GPU and no labeled input, in under ten minutes of CPU.
On this liver query the student scores 0.406 versus the teacher's 0.408 and the census
model's 0.338, at six to nine times the teacher's throughput (Table 2). On withheld-atlas
comparisons, student-teacher concordance gaps are 1.5 points on lung and 3.0 on liver.

| broad-pass model | classes | ontology | cells/s |
|------------------------------|------:|------:|--------:|
| census-built, ours | 798 | 0.338 | 2,962 |
| Pan-human Azimuth (teacher) | 382 | **0.408** | 1,076-1,563 |
| **distilled from Azimuth, ours** | 324 | 0.406 | **8,937-10,021** |

**Table 2.** Broad-pass entry points on 3,396 withheld cross-study liver cells, all three
scored on identical cells through one script. The distilled student needs only raw human
counts to build and inherits the teacher's vocabulary and hierarchy. It does not inherit the
teacher's calibrated abstention, which remains a limitation. The similar concordance on this query does not establish equivalence across tissues or
exclude source-study overlap with the census corpus.

The census hierarchy can also be built without a GPU. Deriving the coarse hierarchy from Cell
Ontology lineage rather than from a foundation-model embedding beats both no hierarchy at all
(0.616 against 0.547) and a same-sized random grouping (0.539), consistent with useful lineage structure. The experiment uses one random grouping and
cell-level intervals that omit donor dependence and training variation. Using the same lineage procedure, that route also
reaches a second organism: a pan-mouse reference of 453 types across 85 tissues reaches 0.638
ontology concordance on two withheld datasets after 17 s of CPU.

### Agreement identifies subsets with higher concordance

We ran the census model, teacher and student over identical query cells. The teacher and
student are dependent because the student learns the teacher's labels. Scored
on identical cells and compared in the Cell Ontology, they partition each query by agreement
(Figure 3). We ran this on three tissues: withheld cross-study liver (3,396 cells, 34 truth
types), the Krasnow lung atlas (65,662 cells, 46 types), and the Allen human middle temporal
gyrus (156,285 cells, 18 types).

One result holds everywhere. Cells on which the references disagree are annotated far less
reliably than cells on which they concur, for every reference and in every tissue: the census
model falls from 0.690 to 0.212 across the liver partition, 0.934 to 0.059 across lung, and
0.839 to 0.117 across brain, with the same direction for the other two. Agreement can be computed without truth labels, while its association with correctness
requires labeled evaluation. Correlated teacher-student errors and shared ontology mappings
may contribute to that association. Computing it costs three annotation runs, including
the teacher's runtime.

What varies, and varies enormously, is how much of a query the agreeing set covers: 23% on
liver, 48% on lung, 94% on brain. The brain figure is the instructive one. That query's Cell
Ontology annotation uses 18 terms for a region whose own working taxonomy, `CCN201908210`,
defines 154 cell sets, and 55% of its cells fall in a single class. The coarse truth crosswalk credits within-lineage confusions. Agreement coverage also
depends on the annotators and their output mappings; we did not isolate which factors
explain the coverage difference across tissues.

The tested consensus rule does not outperform the best single annotator in any tissue. Taking the most specific call the
agreeing references support scores 0.365 on liver against the best single reference's 0.408,
0.828 on lung against 0.831, and 0.837 on brain against 0.987. That last gap is the largest
because choosing the deepest agreeing call lets one reference's confident, lineage-compatible
but wrong specificity override two correct coarser calls. This rule did not improve on the best single annotator. Agreement provides a signal for
inspection rather than a validated guarantee of reliability.

Agreement is defined by the Cell Ontology, and the comparison was possible only because all
three references carry Cell Ontology terms. The Common Cell type Nomenclature [Miller 2020]
answers the same problem differently, giving each taxonomy's cell sets stable accessions and a
curated alias layer rather than a shared coordinate system. The two are complements rather
than substitutes: in the published human MTG taxonomy `CCN201908210` the 154 cell sets carry
an anatomical tag but no cell-type ontology id, and the field that matches cell sets across
taxonomies is filled for 23 of them. CL supplies an ancestor-descendant relation for mapped terms; CCN supplies the provenance CL cannot, recording which
taxonomy and publication each label came from. A workflow that compares annotations across
references wants both.

![reference agreement](figures/fig_consensus.png)

**Figure 3.** Ontology concordance within each agreement tier for three broad references, on
withheld cross-study liver (*A*), the Krasnow lung atlas (*B*) and the Allen human middle
temporal gyrus (*C*); dashed lines are the same models over the whole query. Agreement is
evaluated in the Cell Ontology because the three references answer in different vocabularies
(798, 382 and 324 classes). All-three-agree subsets score higher in each tissue, but their coverage differs. The
distilled model learns from Azimuth, so these are not three independent references.

### Confidence screening for withheld cell types

Withholding 9 of 36 cell types from the liver reference creates a query containing
1,350 cells of types absent from supervised training. Five of the eight methods with a
confidence score trade coverage for accuracy across the sweep; three show little movement
or discard most cells at the tested thresholds (Supplementary Figures S9 and S10).

At p ≥ 0.9, actinn-jax retains 66% of cells at 0.969 accuracy and flags 73% of held-out-type
cells; native-head scArches retains 61% at 0.983 accuracy and flags 71%. These are different coverage
points and do not establish equivalent abstention quality. Scores also have different
scales and meanings across methods. Thresholds should be assessed on accuracy-coverage
curves and known-type false rejection, rather than by a shared numerical cutoff.
Low confidence can reflect technical variation or low cell quality as well as novelty.

### scArches and frozen-classifier comparison

scArches had higher accuracy and macro-F1 than the two baselines on PBMC, within-dataset
lung and both brain tasks. The linear baseline had higher scores on both liver splits and
blood/gut. At 151 brain clusters, scArches reached 0.860 accuracy, compared with 0.816 for
the linear baseline and 0.741 for ACTINN-JAX. Complete-job times were 563, 6.3 and 20.4 s,
respectively. scArches took 68 to 90 times as long as the linear baseline over the seven
shared-vocabulary splits, while using less peak memory: 0.85–2.71 GB versus 1.34–7.20 GB.
Thus the preferred method depends on label resolution and the available time and memory.

Lung cross-dataset transfer is scored through Cell Ontology mappings because the atlases
use different label names and depths. Concordance is 0.761 for scArches, 0.743 for the linear
baseline and 0.752 for ACTINN-JAX. This lineage measure is reported separately from exact
accuracy and macro-F1 on shared-vocabulary tasks.

| split | accuracy / CL concordance†: scArches / linear / ACTINN | macro-F1: scArches / linear / ACTINN | time (s): scArches / linear / ACTINN | peak memory (MB): scArches / linear / ACTINN |
|---|---:|---:|---:|---:|
| PBMC | 0.944 / 0.913 / 0.913 | 0.930 / 0.829 / 0.795 | 81.8 / 0.9 / 6.4 | 854 / 1340 / 1078 |
| Lung within | 0.918 / 0.898 / 0.894 | 0.926 / 0.904 / 0.901 | 270.4 / 3.7 / 16.1 | 1575 / 5045 / 2150 |
| Liver within | 0.809 / 0.817 / 0.794 | 0.808 / 0.815 / 0.790 | 156.9 / 1.9 / 9.6 | 1391 / 3075 / 1757 |
| Blood/gut | 0.886 / 0.908 / 0.864 | 0.886 / 0.908 / 0.865 | 264.5 / 3.8 / 15.9 | 1803 / 5448 / 2692 |
| Liver across studies | 0.673 / 0.692 / 0.676 | 0.585 / 0.587 / 0.579 | 220.1 / 2.8 / 11.7 | 1478 / 3290 / 2431 |
| Brain subclasses | 0.992 / 0.986 / 0.986 | 0.992 / 0.986 / 0.986 | 179.1 / 2.6 / 9.3 | 1685 / 3307 / 2138 |
| Brain clusters | 0.860 / 0.816 / 0.741 | 0.860 / 0.816 / 0.737 | 562.7 / 6.3 / 20.4 | 2714 / 7196 / 3857 |
| Lung across datasets† | 0.761 / 0.743 / 0.752 | — | 875.1 / 7.7 / 24.9 | 2200 / 7895 / 3818 |

**Table 3.** Means of three repeats, ordered scArches / linear-anova-pca / ACTINN-JAX. Time includes fit plus predict; memory is peak process MB.

† Lung across datasets uses Cell Ontology concordance (same term or ancestor–descendant relationship), not exact label accuracy. Macro-F1 is not reported for that row.

## Discussion

The matched-query comparison shows a task-dependent trade-off between accuracy and computational cost.
scArches has higher accuracy at fine brain resolution, where its accuracy is 0.860 against
0.816 for the linear baseline and 0.741 for ACTINN-JAX. The linear baseline instead has
higher accuracy and macro-F1 on liver and blood/gut. It uses less time but more peak memory.
A narrow aggregate accuracy range does not establish that methods are interchangeable
at every label resolution.

### Agreement with ArchMap

We agree with ArchMap [Lotfollahi et al. 2026] that reference quality, reusable atlas models
and inspection of uncertain cells matter. The focused liver study supports the value of a
relevant reference: ontology concordance rises from 0.580 to 0.858 on a study excluded from
focused training. This credits lineage-related calls and should not be read as exact fine-type
accuracy. ArchMap already supports custom atlases, local execution and interactive inspection;
we claim none of those capabilities as unique to ACTINN-JAX. Our contribution is an
implementation and measured annotation workflows on commodity hardware.

ArchMap also warns that residual batch effects and poor-quality cells can produce high
uncertainty. Our held-out-type experiments support treating low confidence as a screening
signal followed by quality checks and biological inspection. Distillation and model agreement
provide additional annotation views, but a teacher and its student are dependent sources.
The tested consensus does not improve on the best annotator. Agreement is useful for
finding cells to inspect, not for counting independent confirmations.

### Where our evidence qualifies the interpretation

ArchMap's Figure 1 caption describes choosing a classifier using the percentage of query
cells labeled unknown. That percentage records how often a system abstains; by itself it
does not establish which labels are more accurate. In our screening experiment, methods have
different score scales and accuracy–coverage curves. Equal numerical thresholds do not retain
the same cells or reject the same fraction of known types. We therefore require accuracy at
matched coverage and false rejection of known types before using an unknown-cell percentage
to rank classifiers. Our direct local comparison below tests how this criterion relates
to annotation agreement in one pipeline configuration.

Our results also limit broad claims about the computation needed for annotation. A saved
ACTINN-JAX model labels 524,699 cells in 31 s on the tested laptop; the fitted linear pipeline
takes 67 s for that query. Both reuse trained components. Query adaptation can nevertheless
improve label accuracy, as the fine-brain scArches comparison shows. Its added time and lower
peak memory should be reported with that gain. A ratio charging one method for adaptation
and another only for frozen prediction answers a different question from inference speed.

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

### Comparisons that remain open

![Outputs of annotation and integrated mapping](figures/fig_mapping_outputs.png)

**Figure 4.** Outputs determine the comparison. A saved classifier returns labels and
confidence. An ArchMap-style mapping workflow also adapts the query into a reference
embedding. Both can reuse pretrained references. Label accuracy, coverage, time and memory
are shared evaluation targets; integrated mapping additionally requires batch-mixing and
biological-conservation measurements. The diagram describes outputs, not measured platform
performance.

We did not evaluate ArchMap's hosted service, accessibility, collaboration features or
integrated embedding quality. The local annotation comparison does not establish a
platform-wide ranking or contradict its integration results. A test of integrated mapping
would additionally require batch-mixing and biological-conservation measurements.


The study is limited to fixed partitions, unequal tuning effort and one hardware family for the in-house comparison. The panels use separately sampled cohorts; cell-level intervals do not account for donor dependence. Generalization requires independent studies.
