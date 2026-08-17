---
title: "Supplementary material"
author: "Ian Driver"
date: ""
abstract: |
  
---

*Independent Researcher, Detroit, MI, USA*  ·  *Correspondence: driver.ian@gmail.com*


*Supplement to "A benchmark of cell-type annotation methods for single-cell data". Figures and tables referenced from
the main text as Supplementary Figure S1–S9 and Supplementary Table S1–S3.*

# Supplementary figures

![confusion matrices](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_confusion.png)

**Figure S1.** Confusion matrices for the broad and focused passes on the withheld
cross-study liver query. Each cell is the fraction of a true type receiving that label;
outlined cells are off-diagonal calls that are an ancestor or descendant of the truth in the
Cell Ontology, i.e. error under exact match and credit under ontology-aware concordance. The
scores below each panel are means over the twelve truth types drawn, not over the whole
query, so they run lower than the query-wide figures quoted in the main text. The broad pass
scores 0.12 exact and recovers +0.22 through the ontology — its mistakes are the right lineage
at the wrong depth (hepatocyte → midzonal or centrilobular hepatocyte; NK cell → hepatic pit
cell, the Cell Ontology term for a liver NK cell). The focused pass scores 0.63 exact, +0.05:
it is already right most of the time, so the ontology has little left to recover.

![per-class recall, blood and gut](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_blood_gut_intra.png)

**Figure S2.** Per-class recall for eleven methods on the 86-type blood+gut split, ordered by
how much the methods disagree (best minus worst recall, right-hand strip). The best and worst
method differ by a median of 0.30 recall per class and up to 0.73; mean pairwise Spearman
between methods' per-class recall is 0.58. Class size is not the explanation, and on this
split it cannot be: capping per label leaves **84 of the 86 classes holding exactly 30 test
cells**, with the remaining two at 29 and 14. There is no size variation left for rarity to
act through, so the disagreement is about which types a method finds hard, not how many
examples it saw.

![per-class recall, lung](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_lung_intra.png)

**Figure S3.** Per-class recall on the 46-type lung split, drawn as in Figure S2. Methods
agree more here than on blood+gut: mean pairwise Spearman 0.65, median best-minus-worst 0.16.

![per-class recall, liver](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_liver_intra.png)

**Figure S4.** Per-class recall on the 36-type liver split, drawn as in Figure S2. Mean
pairwise Spearman is 0.66, but the median best-minus-worst gap is 0.43 — the widest
disagreement of the three splits, and a reminder that a high rank correlation between methods
still leaves large per-class differences.

# Supplementary tables

| method | source | tier | engine | rejection | runtime |
|-----------|------------|-----------|--------------------|----------|------------|
| **actinn-jax** | [Ma & Pellegrini 2020] | classical | JAX MLP | confidence threshold | JAX |
| SVM | [Pedregosa 2011] | classical | linear SVM (SGD) | — | scikit-learn |
| kNN | [Pedregosa 2011] | classical | k-nearest neighbors | — | scikit-learn |
| CellTypist | [Domínguez Conde 2022] | linear | L2 logistic regression | prob threshold | scikit-learn |
| **linear-anova-pca** | [Pedregosa 2011] † | linear | normalize → ANOVA → PCA(220) → logreg | prob | scikit-learn |
| **scTOP** | [Yampolskaya 2023] | parameter-free | rank z-score class-average projection | — | NumPy |
| SingleR | [Aran 2019] | correlation | Spearman + fine-tuning | — | R |
| scmap-cluster | [Kiselev 2018] | correlation | centroid cosine | yes (unassigned) | R |
| scANVI | [Xu 2021] | deep | scVI semi-supervised VAE | prob | scVI |
| scArches | [Lotfollahi 2022] | deep | scANVI reference surgery | prob | scVI |
| **ProtoCloud** | [Guo & Ding 2026] | deep | prototype VAE + LRP attribution | ambiguity flag | PyTorch |
| scPRINT | [Kalfon 2025] | foundation | pretrained transformer, zero-shot | — | PyTorch |
| **Pan-human Azimuth** | [Sarkar et al. 2026] | pretrained | 8-level hierarchical NN, fixed 382-leaf typology | trained `Unassigned` | TensorFlow |

**Table S1.** The benchmarked methods: model family (*tier*), the engine each one runs, whether
it can decline to call a cell (*rejection*), and the runtime stack it was run on. Bold marks
the methods added to this panel here.

† `linear-anova-pca` has no method paper: it is a baseline assembled here from scikit-learn
components, tuned deliberately to be the strongest simple competitor we could build.

| dataset | source | split | tissue | #types | genes | notes |
|-------------|----------|-------|-----------|------|-------|-----------|
| lung_intra | [Travaglini 2020] | within-dataset | lung (Krasnow) | 46 | Ensembl | 300 cells/type ref |
| lung_cross | [Sikkema 2023] → [Travaglini 2020] | cross-dataset | lung (HLCA → Krasnow) | 46 | Ensembl | different lab/protocol |
| liver_intra | [Edgar 2026] | within-dataset | liver (HLiCA) | 36 | Ensembl | 150 cells/type |
| liver_cross | [Edgar 2026] | cross-**study** | liver (HLiCA) | 34 | Ensembl | train 6 studies → test withheld study |
| blood_gut_intra | [Alegbe 2026] | within-dataset | blood + gut (IBDverse) | 86 | Ensembl | high cardinality; no CL ids |
| pbmc | [10x Genomics 2016] | within-dataset | PBMC (pbmc3k) | 8 | symbols | small-n; scPRINT skips (symbols) |

**Table S2.** The six benchmark datasets. *split* separates within-dataset holdouts from
cross-dataset and cross-study transfer; *genes* records whether the matrix is keyed by
Ensembl ids or gene symbols.

| dataset | actinn-jax | best method (acc) |
|---|---|---|
| lung_intra (46 types) | 0.894 | ProtoCloud 0.932 |
| lung_cross (cross-dataset)† | 0.358 exact / 0.752 ontology | ProtoCloud 0.374 / **0.791 ontology** |
| liver_intra (36 types) | 0.802 | linear-anova-pca 0.804 |
| liver_cross (cross-study) | **0.686 exact / 0.731 ontology** | actinn-jax |
| blood_gut (86 types) | 0.860 | linear-anova-pca 0.902 |
| pbmc (8 types) | 0.913 | scArches 0.940 |

**Table S3.** Per-dataset accuracy: actinn-jax against whichever method leads that dataset.
† on lung_cross the exact-match score is a vocabulary artifact, so ontology concordance is
the meaningful column.

# Cost and scaling

These studies establish the cost claims the main text summarizes in one paragraph each. They
are here rather than in the main text because the workflow argument depends on inference being
cheap, not on the shape of the curve that makes it cheap.

![accuracy and memory against reference size](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_atlas_scaling.png)

**Figure S5.** Accuracy and peak memory against reference size, four methods carried to full
atlas scale on two datasets. *A:* lung, 3k → 49k reference cells. *B:* the HLiCA liver atlas,
2.7k → 47k, an independent replication of the same reversal — a prototype VAE moves from worst
to best as the reference grows. *C:* peak memory over both sweeps. Error bars on *A* and *B*
are 95% binomial intervals on the query, which grows with the reference (1,035 to 16,398 cells
on lung), so they narrow from left to right; each point is a single run, so they cover sampling
of the query and not run-to-run variation.

![fit and predict scaling](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_scaling.png)

**Figure S6.** Fit and predict time against reference size and label cardinality, for the six
CPU methods. Fit grows on both axes for every trained method, steeply for CellTypist and the
SVM. Predict is flat in reference size for all six, so that flatness is a property of cached
inference generally rather than of any one method; cardinality is the axis that separates them,
where actinn-jax rises 3.2× from 5 to 86 types against the tuned linear pipeline's 11.9×. SVM
and kNN predict faster than either throughout. scTOP's smallest-reference point carries
one-time import cost and is not a scaling effect.

![annotating an atlas](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_query_scaling.png)

**Figure S7.** Cost against query size with the reference fixed at 17,753 cells. *A:*
wall-clock to annotate the query. *B:* throughput, which declines 31% for actinn-jax and holds
flat for the linear pipeline, narrowing the advantage from 4.2× to 3.1×. *C:* peak memory,
which does not distinguish them — on this axis it measures holding the query, not running the
method. Three runs per point on a shared laptop: *A* and *B* report the fastest run, since
contention can only add time, and *C* the largest peak, since resident-set size understates a
run the OS has partly evicted.

# Abstention

![what a threshold does to each method](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_abstain_grid.png)

**Figure S8.** What a confidence threshold does to each of the eight methods that return a
per-cell confidence, with 9 of 36 cell types held out of the liver reference so 1,350 query
cells are out-of-distribution. Every quantity is a fraction, so all three share one axis. The
five that work share a shape — accuracy and novelty rising as coverage falls — and the three
that do not each fail visibly and differently.

![the abstain trade-off](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_abstain.png)

**Figure S9.** The same sweep read as a trade-off, which is what compares methods to each other
rather than to themselves. *Left:* accuracy on kept cells against the fraction kept — the five
usable methods lie along one band, which is the basis for calling their abstain quality tied.
*Right:* the share of held-out-type cells flagged as the threshold rises.
