---
title: "A benchmark of cell-type annotation methods for single-cell data: cost, not accuracy, distinguishes them"
author: "Ian Driver"
date: ""
abstract: |
  Cell-type annotation by reference mapping is one of the most repeated operations in single-cell analysis, yet comparisons report accuracy far more often than the time and memory that decide what a scientist can actually run. We benchmarked **thirteen methods**, classical through foundation-model, across **eight splits of seven datasets** (8–151 cell types) on commodity hardware, and on Open Problems `label_projection`, whose datasets and metrics we did not choose. Accuracy among the leading methods is tightly clustered — the top four span **0.008** — while their inference cost differs by two orders of magnitude and their peak memory by 2.5×. No method leads everywhere, and no ordering is stable — it inverts with reference size, feature budget and label granularity. With several methods both accurate and cheap, the useful question is not which is best but what cheap annotation makes possible. Using **actinn-jax**, a JAX reimplementation of ACTINN whose cached reference maps a query in under a second, we show annotation becoming a **multi-pass workflow** rather than a single decision: a shipped ~800-type reference routes a query to tissue and hands off to a focused one (cross-study liver 0.23/0.58 to 0.72/0.86, exact/ontology); a pan-human annotator distills into an interchangeable entry point from raw counts alone — no GPU, no labels — matching its teacher at six to nine times the throughput; and three broad references over the same cells partition it by agreement. In liver, lung and brain every reference is far more accurate where all three concur, though a consensus *label* beats none of them: the value is knowing which calls to trust. We release the reimplementation, the harness and the pre-trained references.
---

*Independent Researcher, Detroit, MI, USA*  ·  *Correspondence: driver.ian@gmail.com*

# Key Points

- Among leading annotation methods, accuracy differences are small (top four within 0.008)
 while predict time differs by ~205× and peak memory by 2.5×; cost, not accuracy, is what
 distinguishes them in practice.
- Rankings are not stable: a prototype VAE moves from worst to best as the reference grows
 from 3k to 49k cells, and a tuned linear pipeline that fits faster than a gene-space MLP on
 one panel costs 2.7× more on another with a narrower feature budget.
- Predict is sub-second for every CPU method tested and flat in reference size for all of
 them; what makes chaining stages practical is the ratio, a sub-second call recurring against
 a fit of 19–123 s paid once.
- A pretrained pan-human annotator can be distilled into a fast reference using only raw
 counts — no GPU, no labels — matching the teacher's concordance and beating a census-built
 reference, at 6–9× the teacher's throughput.
- Agreement between independent broad references is a label-free confidence signal,
 replicated on three tissues: where three references concur, every one of them is far more
 accurate than where they disagree. How much of a query that covers varies from 23% to 94%
 and tracks the resolution of the query's own annotation. A consensus *label* beats the best
 single reference in none of the three.

# Introduction

Annotating cells by mapping to a labeled reference is run constantly and rarely reported as a
cost. Existing comparisons [Abdelaal 2019, Fu 2024] emphasize accuracy; the axis that decides
what runs on a laptop — wall-clock and memory without a GPU — is usually absent. Foundation
models [\[Kalfon 2025\]](https://doi.org/10.1038/s41467-025-58699-1) raise accuracy in some settings but need accelerators, and their
zero-shot label predictions underperform small models trained on curated references.

That reporting gap matters more than a missing column, because annotation in practice is not
one decision. A working analysis asks what broad compartments are present, which tissue the
query resembles, how the cells resolve against a reference matched to that tissue, whether
anything is present that no reference describes, and how much of the answer to believe. Each
of those is a reference call. When a call costs minutes, the workflow collapses to a single
pass against a single reference chosen in advance; when it costs a fraction of a second,
running several references and comparing them becomes ordinary. The benchmark below is
therefore a means rather than an end: it establishes that the accuracy differences among the
leading methods are small and the cost differences are not, which is what licenses spending
the budget on more passes instead of on a better single pass.

Concurrent work sharpens the question rather than settling it. **Pan-human Azimuth**
[\[Sarkar et al. 2026\]](https://doi.org/10.64898/2026.07.16.738997) ships a supervised hierarchical classifier over a harmonized
organism-wide typology — 8 levels, 382 leaf types, ~7M parameters over a fixed 5,055-gene
panel, trained on 9.7M curated cells, with abstention *learned* rather than thresholded
(expected calibration error 0.0044) — and runs on a laptop. It is better resourced than any
reference we could build, and its authors reach a conclusion parallel to ours: training-data
quality and organization matter as much as architecture or scale, with accuracy saturating
past ~5M training cells. A purpose-built pan-human model is therefore the right thing to
*start* from; the open question is what to do next, since no fixed typology can re-annotate
into a user's own label set or resolve states below its own leaves.

Because a benchmark written by a method's own author has a known failure mode, the comparison
was constrained by construction: every method runs on **every** dataset through one harness on
identical splits; the panel was chosen to include the baselines most likely to beat a small
MLP, and each of them does beat it somewhere; our own classical tier is left untuned while the
linear baseline is tuned; and the external validation uses a benchmark designed by others.

# Materials and methods

**actinn-jax.** A dependency-light JAX reimplementation of ACTINN [\[Ma & Pellegrini 2020\]](https://doi.org/10.1093/bioinformatics/btz592) — a
4-layer network (100/50/25 hidden units, ReLU, softmax, Adam) — replacing TensorFlow-1.x
graph/session code that no longer installs on current Python and ML environments. Preprocessing is
sparse-aware; a fitted reference is cached and reused across queries; prediction is chunked
for atlas-scale inputs. Accuracy matches the original within repeat noise.

**Panel and datasets.** Thirteen methods (Supplementary Table S1) across eight splits of seven
datasets (Supplementary Table S2): lung within-dataset and cross-dataset, liver within-dataset
and cross-**study**, an 86-type blood+gut set, PBMC, and Allen middle temporal gyrus nuclei at
two levels of one taxonomy — 8 to 151 cell types, spanning three generalization regimes.

**Metrics.** Accuracy, macro-F1, and **ontology-aware concordance**, which credits a call that
is the same node, an ancestor or a descendant of the truth in the Cell Ontology. The last is
required because vocabularies disagree about granularity: on our cross-dataset lung split
reference and query share only 20 of 46 type names, so exact-match accuracy (~0.35 for every method)
measures vocabulary mismatch rather than transfer. Concordance is reported only where both
sides carry ontology ids.

**Agreement between references.** Three broad annotators answer in three different
vocabularies, so agreement is defined in the Cell Ontology, where all three map: two calls
agree when they are the same term or one is an ancestor of the other — the same relation the
concordance metric uses. Cells are partitioned by how many of the three mutually agree.

**Execution.** Each method runs in its own environment as a separate process, because the
dependency sets are mutually unsatisfiable; one driver builds each split once from a fixed
seed and hands the identical pair to every method. Three repeats. Hardware: Apple Silicon,
CPU for classical/linear/correlation tiers, Apple MPS for deep and foundation tiers.
External validation runs on AWS `r7i.8xlarge` through Open Problems' own Nextflow pipeline.
Because wall-clock on a shared machine depends on co-scheduled load, cost is reported from the
fastest of three runs on an otherwise idle machine, and external cost as a ratio to a method
present in every run. Environments are pinned and the Cell Ontology release is recorded.

**Supplementary material** (separate document) carries the scaling studies (Figures S5–S7),
the abstention sweep (Figures S8–S9), confusion matrices and per-class recall (Figures S1–S4),
and Tables S1–S3.

**Workflow components.** A broad reference built from the CELLxGENE Census [\[CZI Census 2025\]](https://chanzuckerberg.github.io/cellxgene-census/);
a coarse→fine hierarchy obtained either from foundation-model embeddings or from Cell Ontology
lineage; confidence-threshold abstention calibrated per reference by withholding 10% of cell
types; masking-based refinement to a query's own supported classes or to a tissue; and a
cluster-level novelty screen. Protocols for each are documented in the benchmark
repository.

# Results

## Accuracy is clustered; cost is not

The top of the accuracy table is a four-way cluster spanning **0.008**, led by a tuned linear
pipeline rather than by a deep model (Table 1; Figure 1 draws the whole panel at once). Those same four differ by **~205× in predict
time** (0.33 s to 67 s) and **2.5× in peak memory**. Order within the cluster is not a result:
the stochastic methods move by more than 0.008 between identical reruns, scANVI by up to
0.056, so the four are best read as tied on accuracy and separated by cost. actinn-jax holds
the best ontology-aware concordance (0.811), likewise a margin inside repeat noise.

![accuracy and cost across every split](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_cost_accuracy_ranges.png)

**Figure 1.** Eleven methods over seven splits. *A:* accuracy as the gap to whichever method
leads that split — raw accuracy is dominated by split difficulty (0.94 on pbmc against 0.36 on
cross-dataset lung), so a range over it would measure the datasets rather than the methods;
`leads N` counts splits won. *B:* fit + predict on the same splits. Diamonds are means, dots
the individual splits, lines run worst to best. Six methods sit within **0.035** of the
per-split leader while spanning 3.4 s to 57 s. Every method's spread across splits is wider
than the distance between the leading methods, which is the caution that belongs with any
panel-mean ranking. Repeat-to-repeat variation is far smaller — five methods are exactly
deterministic, and only scANVI (0.056) and scArches (0.020) move more than 0.009 — so the
range drawn here is across datasets. Per-split detail: Supplementary Figure S11.

| method | acc | macro-F1 | ontology | fit (s) | predict (s) | peak mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| **linear-anova-pca** | **0.839** | 0.699 | 0.808 | **3.4** | **0.33** | 4386 |
| scArches | 0.833 | **0.701** | 0.804 | 48.7 | 17.2 | 1773 |
| scANVI | 0.832 | 0.698 | 0.808 | 0.0* | 66.7 | 2087 |
| **actinn-jax** | 0.831 | 0.683 | **0.811** | 24.2 | **0.54** | 2391 |
| CellTypist | 0.823 | 0.690 | 0.805 | 54.4 | **0.61** | 1569 |
| SVM | 0.808 | 0.675 | 0.796 | 55.2 | **0.07** | 1419 |
| ProtoCloud | 0.790 | 0.649 | 0.778 | 221.5 | 2.07 | 1907 |
| SingleR | 0.770 | 0.652 | 0.750 | 0.3 | 48.2 | 3612 |
| kNN | 0.770 | 0.623 | 0.768 | 0.4 | **0.19** | 1483 |
| scTOP | 0.739 | 0.619 | 0.703 | 1.1 | 1.21 | 1926 |
| scmap-cluster | 0.646 | 0.550 | 0.771 | 0.3 | 9.30 | 8609 |

**Table 1.** Accuracy and cost. Accuracy is the mean over the five shared-vocabulary
datasets, macro-F1 over all six, and ontology concordance over the four that carry Cell
Ontology ids; bold marks the best value in a column. The two brain splits are held out of
these means — they were measured after this pass — and reported per dataset instead;
including both would move actinn-jax from fourth to fifth, behind CellTypist. *scANVI does most of its work in one train+predict pass, attributed to predict.
Per-dataset scores: Supplementary Table S3.

Neither ordering is stable. Carrying four methods from a 3k-cell reference to a full atlas
reverses the accuracy ranking — a prototype VAE moves from worst to best — and the external
Open Problems panel, whose 1,000-gene budget is narrower than ours, reverses the cost ranking,
with the tuned linear pipeline costing 2.7× more than it does here (Supplementary Figure S6,
and the extended report).

Granularity moves it too, and against us. The same Allen middle temporal gyrus nuclei scored at
two levels of one taxonomy behave like two different benchmarks: at Allen's 24 subclasses ten
of the eleven methods land within 0.017 of each other, and at the 151 cell sets the taxonomy
enumerates the ranking inverts — SingleR, eighth overall, leads at **0.845**, while actinn-jax
falls to tenth at **0.741**. Correlation-to-centroid methods gain exactly where the trained
classifiers lose, helped by the fact that cluster-level labels are themselves the output of
clustering the same expression space. One split is a signal, not a characterization, but it is
the one split in this panel at the resolution a cortical taxonomy actually works at.

Prediction is where the annotation budget is actually spent, and it is small: sub-second for
every CPU method tested, and flat in reference size for all of them, since a cached model's
inference does not depend on how many cells trained it. What makes chaining stages practical
is therefore not a uniquely flat predict but the ratio — a sub-second call recurring against a
fit of 19–123 s that is paid once (Supplementary Figure S7). With the reference held fixed,
annotating a whole 525,000-cell atlas takes **41 s** (Supplementary Figure S8).

## One query, two passes: routing and then resolution

That budget buys a workflow rather than a call (Figure 2). A shipped ~800-type census
reference annotates any query without being told what tissue it is; resolving those calls
through the reference's per-class tissue map identifies the tissue; a small focused reference
for that tissue then re-annotates the same cells at full resolution. On withheld cross-study
liver cells the broad pass scores **0.23 exact / 0.58 ontology** and the focused reference
reaches **0.72 / 0.86**.

The two passes hand off; they do not combine. Substituting a stronger broad model lifts the
broad call but changes nothing downstream, and using it to narrow the focused pass's classes
makes the result **worse** (0.731 → 0.708), because a wrong mask discards the correct class
outright. Once the focused reference covers the tissue, the broad model's value is routing,
not resolution.

![workflow](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_workflow_umap_ondata.png)

**Figure 2.** The workflow on a withheld liver study (3,396 cells), same embedding
throughout. The census reference spreads 144 of its 798 labels over the query (concordance
0.34); resolving those calls to tissue gives **76% liver** against 4% for the next candidate,
which selects the reference to load; the 36-type liver reference then re-annotates the same
cells at **0.73**, tracking the clusters. Rightmost panel is the study's own labels.

## The broad entry point is interchangeable, and one can be distilled without a GPU

Building a broad reference from the census requires a foundation model on a GPU to discover
its hierarchy. A pretrained pan-human annotator [\[Sarkar et al. 2026\]](https://doi.org/10.64898/2026.07.16.738997) already publishes one, so
labeling a corpus with it and training on those labels transfers both vocabulary and
structure — using **only raw counts**, no GPU and no labeled input, in under ten minutes of
CPU. The result matches its teacher and beats our census-built reference, at six to nine times
the teacher's throughput (Table 2).

| broad-pass model | classes | ontology | cells/s |
|------------------------------|------:|------:|--------:|
| census-built, ours | 798 | 0.338 | 2,962 |
| Pan-human Azimuth (teacher) | 382 | **0.408** | 1,076–1,563 |
| **distilled from Azimuth, ours** | 324 | 0.406 | **8,937–10,021** |

**Table 2.** Broad-pass entry points on 3,396 withheld cross-study liver cells, all three
scored on identical cells through one script. The distilled student needs only raw human
counts to build and inherits the teacher's vocabulary and hierarchy. It does not inherit the
teacher's calibrated abstention, which remains a limitation. Student and teacher are level
here rather than separated: read the distillation as preserving the teacher's annotations at a
fraction of its cost, not as improving on them.

The GPU can be removed from the census route as well. Deriving the coarse hierarchy from Cell
Ontology lineage rather than from a foundation-model embedding beats both no hierarchy at all
(0.616 vs 0.547) and a same-sized random grouping (0.539), so it is the lineage structure and
not the mere act of splitting that helps. Being species-independent, that route also reaches a
second organism: a pan-mouse reference of 453 types across 85 tissues reaches 0.638 ontology
concordance on two withheld datasets after 17 s of CPU.

## Where independent references agree, all of them are right more often

Three interchangeable entry points invite a question that is only affordable when calls are
cheap: what if a user runs all of them? Scored on identical cells and compared in the Cell
Ontology, they partition each query by agreement (Figure 3). We ran this on three tissues:
withheld cross-study **liver** (3,396 cells, 34 truth types), the Krasnow **lung** atlas
(65,662 cells, 46 types), and the Allen human **middle temporal gyrus** (156,285 cells, 18
types).

One result holds everywhere. Cells on which the references disagree are annotated far less
reliably than cells on which they concur, for every reference and in every tissue: the census
model falls from 0.690 to 0.212 across the liver partition, 0.934 to 0.059 across lung, and
0.839 to 0.117 across brain, with the same direction for the other two. Because all three
improve together, agreement is selecting cells that are unambiguous rather than cells that one
model happens to get right — and unlike accuracy it is computable on a query whose answers are
unknown, for three sub-second calls and no labels.

What varies, and varies enormously, is how much of a query the agreeing set covers: **23% on
liver, 48% on lung, 94% on brain**. The brain figure is the instructive one. That query's Cell
Ontology annotation uses 18 terms for a region whose own working taxonomy, `CCN201908210`,
defines 154 cell sets, and 55% of its cells fall in a single class. High agreement there
reflects a coarse truth vocabulary rather than an easy tissue, and it shows the partition
reporting the resolution of the annotation it is scored against — which is a property of the
query, not of the method.

A consensus *label* is not the prize in any of the three. Taking the most specific call the
agreeing references support scores 0.365 on liver against the best single reference's 0.408,
0.828 on lung against 0.831, and 0.837 on brain against 0.987 — the last badly worse, because
choosing the deepest agreeing call lets one reference's confident, lineage-compatible but
wrong specificity override two correct coarser calls. Running several references does not
produce a better annotation; it tells you which annotations to trust.

What counts as agreement is the ontology's judgment rather than ours, and the comparison was
possible only because all three references carry Cell Ontology terms. The Common Cell type
Nomenclature [\[Miller 2020\]](https://doi.org/10.7554/eLife.59928) answers the same problem differently, giving each taxonomy's cell
sets stable accessions and a curated alias layer rather than a shared coordinate system. The
two are complements, not substitutes: in the published human MTG taxonomy `CCN201908210` the
154 cell sets carry an anatomical tag but no cell-type ontology id, and the field that matches
cell sets across taxonomies is filled for 23 of them. CL supplies the *total* subsumption
relation that makes an agreement partition computable; CCN supplies the provenance CL cannot,
recording which taxonomy and publication each label came from. A workflow that compares
annotations across references wants both.

![reference agreement](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_consensus.png)

**Figure 3.** Ontology concordance within each agreement tier for three broad references, on
withheld cross-study liver (*A*), the Krasnow lung atlas (*B*) and the Allen human middle
temporal gyrus (*C*); dashed lines are the same models over the whole query. Agreement is
evaluated in the Cell Ontology because the three references answer in different vocabularies
(798, 382 and 324 classes). The ordering is the same in all three tissues; the fraction of
cells on which the references agree is not, and tracks how finely the query's own annotation
is resolved.

## Abstention is tunable; zero-shot labels are not competitive

Withholding 9 of 36 cell types entirely and sweeping a confidence threshold over the eight
methods that return a per-cell confidence, five trade coverage for accuracy across the range
and three do not: CellTypist's probabilities are saturated and give one operating point
instead of a curve, scTOP's projection score keeps only 6% of the query at p ≥ 0.5, and
ProtoCloud's ambiguity flag is flat until 0.9 (Supplementary Figures S8–S9). Among the five
that work, abstain quality does not separate them — actinn-jax and scArches are effectively
tied (0.969 accuracy on 66% of cells with 73% of novel cells flagged, against 0.983 on 61%
with 71%) — but cost does, at 0.54 s of predict time against 17.2 s. This is a second and
complementary route to the same end as reference agreement: a calibrated threshold within one
reference, or concurrence across several with no calibration at all.

Separately, a foundation model run zero-shot scored **0.201** ontology concordance against
0.917 for a reference-trained model on the same cells, taking ~280× longer — reproduced
independently on Open Problems, where zero-shot entries sit at the bottom of the leaderboard.

# Discussion

Across two independent benchmarks the same pattern holds: methods separated by less than a
percentage point of accuracy are separated by two orders of magnitude in cost, and neither
ordering is stable across reference size or feature budget. Reporting accuracy alone therefore
under-determines the choice of method, and reporting it from a single reference size can
invert the recommendation.

The consequence we think matters most is not a ranking but a change in what an annotation
pipeline can afford to do. When a reference call costs a fraction of a second, the pipeline
stops being one classification and becomes a sequence of cheap ones: route the query to a
tissue, re-annotate against a reference matched to it, resolve states below the label, and ask
several independent references where they agree. Each step is unremarkable alone; together
they are what a user actually wants, and their feasibility is a cost property rather than an
accuracy property.

The agreement result is the clearest case, and also the most honest about its limits. It buys
no better label — the consensus call loses to the best single reference — but it identifies,
with no ground truth at all, the fraction of a query on which independent references concur
and on which every one of them is markedly more often right. That is available to anyone
willing to run three annotators instead of one, which is only a reasonable thing to ask when
each takes under a second.

We are explicit about what is not ours. ProtoCloud provides uncertainty, attribution and a
retraining-based refinement path, and becomes the most accurate method at atlas scale. A
purpose-built pan-human annotator is better resourced than our census-built reference, and our
distilled model matches rather than improves on it; we claim not better annotations but
cheaper ones that preserve its vocabulary and structure. What remains distinct is the hand-off
no fixed typology performs — re-annotation into a user's own label set, resolution below a
typology's leaves, and screening for what none of them claims — at a cost that keeps every
stage on the machine already on the desk.

The main limitations are that the cross-method comparison is human-only and single
hardware-family; that our classical tier is untuned while the linear baseline is tuned, which
biases against our own method; that the distilled reference inherits a vocabulary but not its
teacher's calibrated abstention; that the agreement result covers three tissues with three
references, so its direction is replicated but the size of the agreeing fraction is strongly
dependent on how finely the query is annotated; and that agreement is defined in the Cell
Ontology, whose subsumption judgments decide what counts as agreeing and whose resolution in
cortex is roughly a tenth of the working taxonomy's. Full limitations are in the extended
report.

# References {-}

1. 10x Genomics. 3k PBMCs from a healthy donor, Cell Ranger 1.1.0 (2016). Distributed with scanpy as `pbmc3k`. [10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0).
2. Abdelaal T, et al. A comparison of automatic cell identification methods for single-cell RNA sequencing data. *Genome Biology* 20:194 (2019). [doi:10.1186/s13059-019-1795-z](https://doi.org/10.1186/s13059-019-1795-z).
3. Alegbe T, Harris BT, Fachal L, et al. Cell-type-resolved genetic variation shapes inflammatory bowel disease risk (IBDverse). *Nature* (2026). [doi:10.1038/s41586-026-10627-z](https://doi.org/10.1038/s41586-026-10627-z).
4. Aran D, et al. Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage (SingleR). *Nature Immunology* 20:163-172 (2019). [doi:10.1038/s41590-018-0276-y](https://doi.org/10.1038/s41590-018-0276-y).
5. Bradbury J, et al. JAX: composable transformations of Python+NumPy programs (2018). [github.com/jax-ml/jax](https://github.com/jax-ml/jax).
6. CZI Cell Science Program. CZ CELLxGENE Discover Census, LTS release 2025-11-08. [chanzuckerberg.github.io/cellxgene-census](https://chanzuckerberg.github.io/cellxgene-census/).
7. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *KDD* 785-794 (2016). [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
8. Domínguez Conde C, et al. Cross-tissue immune cell analysis reveals tissue-specific features in humans (CellTypist). *Science* 376:eabl5197 (2022). [doi:10.1126/science.abl5197](https://doi.org/10.1126/science.abl5197).
9. Edgar RD, Portman JR, Hu H, et al. HLiCA: an integrated cell atlas of the healthy human liver. *bioRxiv* (2026). [doi:10.64898/2026.06.30.735539](https://doi.org/10.64898/2026.06.30.735539).
10. Fu Q, Dong C, Liu Y, et al. A comparison of scRNA-seq annotation methods based on experimentally labeled immune cell subtype dataset. *Briefings in Bioinformatics* 25(5):bbae392 (2024). [doi:10.1093/bib/bbae392](https://doi.org/10.1093/bib/bbae392).
11. Guo K, Ding J. ProtoCloud: a prototypical self-explaining model for single-cell analysis. *Cell Genomics* 6(6):101217 (2026). [doi:10.1016/j.xgen.2026.101217](https://doi.org/10.1016/j.xgen.2026.101217).
12. Kalfon J, Samaran J, Peyré G, Cantini L. scPRINT: pre-training on 50 million cells allows robust gene network predictions. *Nature Communications* 16:3607 (2025). [doi:10.1038/s41467-025-58699-1](https://doi.org/10.1038/s41467-025-58699-1).
13. Kiselev VY, Yiu A, Hemberg M. scmap: projection of single-cell RNA-seq data across data sets. *Nature Methods* 15:359-362 (2018). [doi:10.1038/nmeth.4644](https://doi.org/10.1038/nmeth.4644).
14. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2). *Science* 379:1123-1130 (2023). [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574).
15. Lotfollahi M, et al. Mapping single-cell data to reference atlases by transfer learning (scArches). *Nature Biotechnology* 40:121-130 (2022). [doi:10.1038/s41587-021-01001-7](https://doi.org/10.1038/s41587-021-01001-7).
16. Ma F, Pellegrini M. ACTINN: automated identification of cell types in single cell RNA sequencing. *Bioinformatics* 36(2):533-538 (2020). [doi:10.1093/bioinformatics/btz592](https://doi.org/10.1093/bioinformatics/btz592).
17. Miller JA, Gouwens NW, Tasic B, et al. Common cell type nomenclature for the mammalian brain. *eLife* 9:e59928 (2020). [doi:10.7554/eLife.59928](https://doi.org/10.7554/eLife.59928).
18. Open Problems for Single-Cell Analysis Consortium. Open Problems: a living benchmark for single-cell analysis (2024). [openproblems.bio](https://openproblems.bio).
19. Pedregosa F, et al. Scikit-learn: machine learning in Python. *JMLR* 12:2825-2830 (2011). [jmlr.org/papers/v12/pedregosa11a.html](https://www.jmlr.org/papers/v12/pedregosa11a.html).
20. Rosen Y, et al. Universal cell embedding provides a foundation model for cell biology (UCE). *Nature* (2026). [doi:10.1038/s41586-026-10689-z](https://doi.org/10.1038/s41586-026-10689-z).
21. Sarkar S, Li Z, Molla G, et al. Organism-scale annotation with Pan-human Azimuth. *bioRxiv* (2026). [doi:10.64898/2026.07.16.738997](https://doi.org/10.64898/2026.07.16.738997).
22. Sikkema L, et al. An integrated cell atlas of the lung in health and disease (HLCA). *Nature Medicine* 29:1563-1577 (2023). [doi:10.1038/s41591-023-02327-2](https://doi.org/10.1038/s41591-023-02327-2).
23. Souza H, Mehta P. Parameter-free representations outperform single-cell foundation models on downstream benchmarks. *bioRxiv* (2026). [doi:10.64898/2026.02.11.705358](https://doi.org/10.64898/2026.02.11.705358).
24. Travaglini KJ, Nabhan AN, Penland L, et al. A molecular cell atlas of the human lung from single-cell RNA sequencing. *Nature* 587(7835):619-625 (2020). [doi:10.1038/s41586-020-2922-4](https://doi.org/10.1038/s41586-020-2922-4).
25. Xu C, et al. Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models (scANVI). *Molecular Systems Biology* 17:e9620 (2021). [doi:10.15252/msb.20209620](https://doi.org/10.15252/msb.20209620).
26. Yampolskaya M, Herriges MJ, Ikonomou L, Kotton DN, Mehta P. scTOP: physics-inspired order parameters for cellular identification and visualization. *Development* 150(21):dev201873 (2023). [doi:10.1242/dev.201873](https://doi.org/10.1242/dev.201873).
27. Munroe R. Standards. *xkcd* 927. [xkcd.com/927](https://xkcd.com/927/).

[10x Genomics 2016]: https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0
[Abdelaal 2019]: https://doi.org/10.1186/s13059-019-1795-z
[Alegbe 2026]: https://doi.org/10.1038/s41586-026-10627-z
[Aran 2019]: https://doi.org/10.1038/s41590-018-0276-y
[Bradbury 2018]: https://github.com/jax-ml/jax
[CZI Census 2025]: https://chanzuckerberg.github.io/cellxgene-census/
[Chen & Guestrin 2016]: https://doi.org/10.1145/2939672.2939785
[Domínguez Conde 2022]: https://doi.org/10.1126/science.abl5197
[Edgar 2026]: https://doi.org/10.64898/2026.06.30.735539
[Fu 2024]: https://doi.org/10.1093/bib/bbae392
[Guo & Ding 2026]: https://doi.org/10.1016/j.xgen.2026.101217
[Kalfon 2025]: https://doi.org/10.1038/s41467-025-58699-1
[Kiselev 2018]: https://doi.org/10.1038/nmeth.4644
[Lin 2023]: https://doi.org/10.1126/science.ade2574
[Lotfollahi 2022]: https://doi.org/10.1038/s41587-021-01001-7
[Ma & Pellegrini 2020]: https://doi.org/10.1093/bioinformatics/btz592
[Miller 2020]: https://doi.org/10.7554/eLife.59928
[Open Problems 2024]: https://openproblems.bio
[Pedregosa 2011]: https://www.jmlr.org/papers/v12/pedregosa11a.html
[Rosen 2026]: https://doi.org/10.1038/s41586-026-10689-z
[Sarkar et al. 2026]: https://doi.org/10.64898/2026.07.16.738997
[Sikkema 2023]: https://doi.org/10.1038/s41591-023-02327-2
[Souza & Mehta 2026]: https://doi.org/10.64898/2026.02.11.705358
[Travaglini 2020]: https://doi.org/10.1038/s41586-020-2922-4
[Xu 2021]: https://doi.org/10.15252/msb.20209620
[Yampolskaya 2023]: https://doi.org/10.1242/dev.201873
[xkcd 927]: https://xkcd.com/927/
