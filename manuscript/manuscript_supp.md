---
title: |
  Supplementary material
author:
  - Ian Driver$^{\ast}$
date: ""
geometry: margin=1in
fontsize: 11pt
linkcolor: RoyalBlue
urlcolor: RoyalBlue
header-includes:
  # long unbreakable code tokens (paths, dotted names) otherwise run off the page;
  # emergencystretch lets TeX loosen a problem paragraph and move them to the next
  # line, without hyphenating inside a filename and inventing a name that does not exist
  - \setlength{\emergencystretch}{3em}
  - \usepackage{authblk}
  - \renewcommand\Authands{, }
  - \usepackage{newunicodechar}
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\footnotesize}
  - \newunicodechar{μ}{\ensuremath{\mu}}
  - \newunicodechar{σ}{\ensuremath{\sigma}}
  - \newunicodechar{≈}{\ensuremath{\approx}}
  - \newunicodechar{×}{\ensuremath{\times}}
  - \newunicodechar{→}{\ensuremath{\rightarrow}}
  - \newunicodechar{≥}{\ensuremath{\ge}}
  - \newunicodechar{≤}{\ensuremath{\le}}
  - \newunicodechar{±}{\ensuremath{\pm}}
  - \newunicodechar{≠}{\ensuremath{\neq}}
  - \newunicodechar{≫}{\ensuremath{\gg}}
  - \newunicodechar{²}{\textsuperscript{2}}
---

\begin{center}\small Independent Researcher, Detroit, MI, USA \\ $^{\ast}$ Correspondence: driver.ian@gmail.com\end{center}


*Supplement to "Accuracy and computational cost of reference-based cell-type annotation on commodity hardware". Figures S1 to S14 and Tables S1 to S6 support the main manuscript.*

# Supplementary figures

```{=latex}
\begin{minipage}{\linewidth}
```

![confusion matrices](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_confusion.png)

**Figure S1.** Confusion matrices for broad and focused annotation of the held-out liver study. Entries are fractions within each true type; outlined off-diagonal cells receive lineage credit in the Cell Ontology. The displayed twelve types are a subset of the query, so their average scores differ from query-wide concordance. The broad reference scores 0.12 exact match plus 0.22 lineage credit; the focused reference scores 0.63 plus 0.05. Lineage credit includes related labels at different depths and does not establish fine-type identity.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![per-class recall, blood and gut](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_blood_gut_intra.png)

**Figure S2.** Per-class recall for eleven methods on the 86-type blood+gut split, ordered by
how much the methods disagree (best minus worst recall, right-hand strip). The best and worst
method differ by a median of 0.30 recall per class and up to 0.73; mean pairwise Spearman
between methods' per-class recall is 0.58. Class size is not the explanation, and on this
split it cannot be: capping per label leaves **84 of the 86 classes holding exactly 30 test
cells**, with the remaining two at 29 and 14. Near-equal query counts limit variation in test precision due to class size. They do not
eliminate differences in reference abundance, donor diversity or biological separability.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![per-class recall, lung](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_lung_intra.png)

**Figure S3.** Per-class recall on the 46-type lung split, drawn as in Figure S2. Methods
agree more here than on blood+gut: mean pairwise Spearman 0.65, median best-minus-worst 0.16.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![per-class recall, liver](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_liver_intra.png)

**Figure S4.** Per-class recall on the 36-type liver split, drawn as in Figure S2. Mean
pairwise Spearman is 0.66, but the median best-minus-worst gap is 0.43, the widest
disagreement of the three splits, and a reminder that a high rank correlation between methods
still leaves large per-class differences.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![per-class recall, brain clusters](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_perclass_brain_cluster_intra.png)

**Figure S5.** Per-class recall on the 151-type brain cluster split, drawn as in Figure S2.
This is the split where the panel's ranking inverts (§3.1): the two correlation methods,
weakest almost everywhere else, sit among the leaders, and actinn-jax is tenth of eleven.
Mean pairwise Spearman is 0.61 and the median best-minus-worst gap is 0.32, rising to 0.88.
The disagreement is concentrated in the `L2/3 IT`, `Sst` and `Vip` families. Query counts
are nearly equal after capping:
capping at 100 cells per cluster leaves 130 of the 151 classes tied at 25 test cells.

```{=latex}
\end{minipage}
```

# Supplementary tables

| method | source | tier | engine | rejection | runtime |
|-----------|------------|-----------|--------------------|----------|------------|
| **actinn-jax** | [Ma & Pellegrini 2020] | classical | JAX MLP | confidence threshold | JAX |
| SVM | [Pedregosa 2011] | classical | linear SVM (SGD) | none | scikit-learn |
| kNN | [Pedregosa 2011] | classical | k-nearest neighbors | none | scikit-learn |
| CellTypist | [Domínguez Conde 2022] | linear | L2 logistic regression | prob threshold | scikit-learn |
| **linear-anova-pca** | [Pedregosa 2011] † | linear | normalize, ANOVA, PCA(220), logreg | prob | scikit-learn |
| **scTOP** | [Yampolskaya 2023] | parameter-free | rank z-score class-average projection | none | NumPy |
| SingleR | [Aran 2019] | correlation | Spearman + fine-tuning | none | R |
| scmap-cluster | [Kiselev 2018] | correlation | centroid cosine | yes (unassigned) | R |
| scANVI | [Xu 2021] | deep | scVI semi-supervised VAE | prob | scVI |
| scArches | [Lotfollahi 2022] | deep | scANVI reference surgery | prob | scVI |
| **ProtoCloud** | [Guo & Ding 2026] | deep | prototype VAE + LRP attribution | ambiguity flag | PyTorch |
| scPRINT | [Kalfon 2025] | foundation | pretrained transformer, zero-shot | none | PyTorch |
| **Pan-human Azimuth** | [Sarkar et al. 2026] | pretrained | 8-level hierarchical NN, fixed 382-leaf typology | trained `Unassigned` | TensorFlow |

**Table S1.** The benchmarked methods: model family (*tier*), the engine each one runs, whether
it can decline to call a cell (*rejection*), and the runtime stack it was run on. Bold marks
the methods added to this panel here.

† `linear-anova-pca` follows the Souza and Mehta recipe (doi:10.\allowbreak{}64898/\allowbreak{}2026.\allowbreak{}02.\allowbreak{}11.\allowbreak{}705358).
The adapter fits up to 20,000 selected genes, 220 PCs and C = 1.0 on the reference.
Other classical baselines use untuned settings; tuning effort is unequal.

| dataset | source | split | tissue | #types | genes | notes |
|-------------|----------|-------|-----------|------|-------|-----------|
| lung_intra | [Travaglini 2020] | within-dataset | lung (Krasnow) | 46 | Ensembl | 300 cells/type ref |
| lung_cross | [Sikkema 2023] to [Travaglini 2020] | cross-dataset | lung (HLCA to Krasnow) | 46 | Ensembl | query study present in reference |
| liver_intra | [Edgar 2026] | within-dataset | liver (HLiCA) | 36 | Ensembl | 150 cells/type |
| liver_cross | [Edgar 2026] | cross-**study** | liver (HLiCA) | 34 | Ensembl | train 6 studies, test withheld study |
| blood_gut_intra | [Alegbe 2026] | within-dataset | blood + gut (IBDverse) | 86 | Ensembl | high cardinality; no CL ids |
| pbmc | [10x Genomics 2016] | within-dataset | PBMC (pbmc3k) | 8 | symbols | small-n; scPRINT skips (symbols) |
| brain_intra | [Jorstad 2023] | within-dataset | brain, MTG (Allen) | 24 | Ensembl | single nuclei; Allen subclasses |
| brain_cluster | [Jorstad 2023] | within-dataset | brain, MTG (Allen) | 151 | Ensembl | Allen source, CCN cell sets |

**Table S2.** The eight benchmark splits over seven datasets. *split* separates within-dataset holdouts from
cross-dataset and cross-study transfer; *genes* records whether the matrix is keyed by
Ensembl ids or gene symbols. Brain is labelled with Allen's `Subclass` rather than with Cell
Ontology terms. The dataset's supplied ontology crosswalk maps its 24 subclasses to 18
terms and does not distinguish the 151 clusters, so it is excluded from ontology aggregates.
This does not imply that Cell Ontology itself lacks the finer terms.

| dataset | actinn-jax | best method (acc) |
|---|---|---|
| lung_intra (46 types) | 0.894 | ProtoCloud 0.932 |
| lung_cross (cross-dataset)† | 0.752 ontology | **ProtoCloud 0.791 ontology** |
| liver_intra (36 types) | 0.802 | linear-anova-pca 0.804 |
| liver_cross (cross-study) | **0.686 exact / 0.731 ontology** | actinn-jax |
| blood_gut (86 types) | 0.860 | linear-anova-pca 0.902 |
| pbmc (8 types) | 0.913 | scArches 0.929 |
| brain, subclass (24 types) | 0.986 | ProtoCloud 0.991 |
| brain, cluster (151 types) | 0.741 | SingleR 0.845 |

**Table S3.** Eleven-method panel using native-head scArches. Per-dataset accuracy: actinn-jax against whichever method leads that dataset.
† Lung across datasets is scored by Cell Ontology concordance; other rows use exact label accuracy.

# Cost and scaling

These studies report how measured accuracy, runtime and memory vary with the sampled
reference, label cardinality and query size.

```{=latex}
\begin{minipage}{\linewidth}
```

![accuracy and memory against reference size](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_atlas_scaling.png)

**Figure S6.** Accuracy and peak memory against reference size, four methods carried to atlas
scale on three tissues. *A:* lung, 3.1k to 49k reference cells. *B:* the HLiCA liver atlas,
2.7k to 47k. *C:* Allen brain (MTG), 1.8k to 46k. Rankings change across samples, with ProtoCloud leading the largest samples. *D:* peak memory over all three sweeps. Each cap is followed by a 25% query split, so query size and composition change with
reference size. Error bars on *A* to *C* are approximate 95% binomial intervals treating
cells as independent; they omit donor dependence and training variation. Each point is one run.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![fit and predict scaling](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_scaling.png)

**Figure S7.** Fit and predict time against reference size and label cardinality, for the six
CPU methods. Fit grows on both axes for every trained method, steeply for CellTypist and the
SVM. Prediction varies little with reference size in this sweep; this behavior is shared across
methods, although methods retaining reference cells can have reference-size-dependent complexity; cardinality is the axis that separates them,
where actinn-jax rises 3.1 times from 5 to 86 types against the tuned linear pipeline's 11.0. SVM
and kNN predict faster than either throughout. scTOP's smallest-reference point carries
one-time import cost and is not a scaling effect.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![annotating an atlas](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_query_scaling.png)

**Figure S8.** Cost against query size, 50,000 to 524,699 cells, with the reference fixed at
17,753 cells; every query in the main accuracy panel is smaller than the leftmost point here.
*A:* wall-clock to annotate the query. *B:* throughput, which declines 21% for actinn-jax and
10% for the linear pipeline, narrowing the advantage from 2.4 times to 2.1. *C:* peak memory,
which does not distinguish them, since on this axis it measures holding the query rather than
running the method. Three runs per point on a shared laptop: *A* and *B* report the fastest run, since
contention can only add time, and *C* the largest peak, since resident-set size understates a
run the OS has partly evicted.

```{=latex}
\end{minipage}
```

# Abstention

```{=latex}
\begin{minipage}{\linewidth}
```

![what a threshold does to each method](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_abstain_grid.png)

**Figure S9.** What a confidence threshold does to each of the eight methods that return a
per-cell confidence, with native-head scArches and 9 of 36 cell types held out of the liver reference so 1,350 query
cells are out-of-distribution. Every quantity is a fraction, so all three share one axis. Five methods trade coverage for accuracy across the sweep. The other three have saturated
or differently scaled scores, so the tested thresholds provide few useful operating points.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![the abstain trade-off](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_abstain.png)

**Figure S10.** Accuracy and coverage from the same held-out-type threshold sweep. Left: accuracy on retained cells versus fraction retained. Right: fraction of held-out-type cells flagged. Different score scales mean that a common numerical threshold is not matched calibration. scArches uses the native-head setting. The curves show the cost of rejecting cells; no matched-coverage equivalence test was performed.

```{=latex}
\end{minipage}
```

# Per-split accuracy and cost

```{=latex}
\begin{minipage}{\linewidth}
```

![accuracy against cost on every split](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_pareto_facets.png)

**Figure S11.** The eleven-method data supporting Figure S14 without the gap-to-leader normalization: annotation agreement
against fit + predict time on each of the eight splits, error bars the range over three
repeats on both axes. Axis limits are per panel, so vertical position compares methods within
a split and not across them. Scores use exact accuracy on shared-vocabulary tasks and Cell Ontology concordance for
lung across datasets. Figure S14 includes only shared-vocabulary tasks. Most
error bars are invisible because most methods are deterministic given a fixed split; the ones
that show are scArches, SVM and scANVI, whose repeats differ by up to 0.024, 0.023 and 0.015.
† The lung cross-dataset panel uses Cell Ontology concordance (same term or ancestor–descendant relationship), rather than exact label accuracy.

```{=latex}
\end{minipage}
```

# Gene budget on the Open Problems panel

```{=latex}
\begin{minipage}{\linewidth}
```

![gene budget curve](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/gene_budget_curve.png)

**Figure S12.** actinn-jax accuracy and macro-F1 against input gene budget across all six Open Problems
datasets. More genes help most datasets but regress the fine-grained,
domain-shifted tabula_sapiens.

```{=latex}
\end{minipage}
```

```{=latex}
\begin{minipage}{\linewidth}
```

![gene budget signals](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/gene_budget_signals.png)

**Figure S13.** Label-free signals for setting the gene budget without test labels. Held-out
reference cross-validation and query-cells-per-class both single out tabula_sapiens, the one
dataset where more genes cost real accuracy, about 10 points. hypomap drifts down as well,
but from a saturated 0.998, and neither signal flags it.

```{=latex}
\end{minipage}
```


# Timing boundaries and statistical interpretation

| method | fit stage (s) | predict stage (s) | sum (s) | predict-stage work |
|---|---:|---:|---:|---|
| linear-anova-pca | 2.5 | 0.31 | 2.81 | inference using fitted transforms and classifier |
| actinn-jax | 10.3 | 0.24 | 10.54 | inference using fitted network |
| scArches | 37.9 | 11.9 | 49.8 | query adaptation followed by prediction |
| scANVI | 0.0 | 58.7 | 58.7 | reference-query training followed by prediction |

**Table S4.** Eleven-method panel using native-head scArches.  Stage boundaries for the four highest mean accuracies in the five-split
non-brain aggregate. Values reproduce the main table; sums use its rounded stage values.
The sum measures a complete annotation job on this protocol. It excludes repeated use of
a previously fitted model. The approximately 250-fold ratio between scANVI's and
actinn-jax's predict-stage times compares training plus prediction with inference alone.
Both actinn-jax and the linear adapter reuse fitted components for later queries.

Three repeats of each fixed partition characterize optimization and runtime variation.
They do not measure generalization across donors or studies and do not establish paired
accuracy equivalence. Cell-level binomial intervals in the atlas and hierarchy studies
omit within-donor dependence and training variation. The hierarchy control uses one random
grouping, and atlas accuracy sweeps change query composition alongside reference size.

The full-study focused liver demonstration uses 56,545 `Andrews_2022` cells and a 38-type
validation reference trained on the other studies. It reports exact ontology-term match
(0.231 broad versus 0.728 focused) and lineage concordance (0.580 versus 0.858).
The main workflow figure instead uses 3,396 cells and a 36-type reference, at 0.338 broad
and 0.731 focused concordance. The distributed liver models use all studies; they are not
the held-out validation weights. Source-study exclusion from broad census corpora is not
established.

The agreement analysis compares a census model with Pan-human Azimuth and a student
trained from Azimuth's predictions. This dependence precludes treating the three models as
independent confirmations. Ontology concordance can credit overly broad or overly specific
predictions. Agreement should guide inspection, and low-confidence clusters require marker,
quality and study-level checks before they are interpreted as novel biology.

# Relation to ArchMap

ArchMap (Lotfollahi et al., Nature Genetics, 2026;
[doi:10.\allowbreak{}1038/\allowbreak{}s41588-026-02756-y](https://doi.org/10.1038/s41588-026-02756-y)) provides a
platform for atlas mapping, label transfer, uncertainty and reference-state inspection.
It supports custom atlases and local execution as well as the hosted application. Its
mapping protocol adapts queries using scArches and evaluates a joint embedding, whereas
this study primarily evaluates annotation labels and resource use. The benchmark's
scArches adapter is not an evaluation of the ArchMap platform. We did not run a matched
end-to-end comparison and make no performance ranking between ArchMap and actinn-jax.

# scArches configurations

```{=latex}
\begin{minipage}{\linewidth}
```

| split | accuracy / CL concordance†, native / WKNN | macro-F1, native / WKNN | fit + predict (s), native / WKNN |
|---|---:|---:|---:|
| PBMC | 0.933 / 0.944 | 0.885 / 0.930 | 24.3 / 81.8 |
| Lung within | 0.907 / 0.918 | 0.915 / 0.926 | 77.0 / 270.4 |
| Liver within | 0.786 / 0.809 | 0.786 / 0.808 | 42.7 / 156.9 |
| Blood/gut | 0.877 / 0.886 | 0.877 / 0.886 | 88.3 / 264.5 |
| Liver across studies | 0.650 / 0.673 | 0.570 / 0.585 | 63.5 / 220.1 |
| Brain subclasses | 0.990 / 0.992 | 0.990 / 0.992 | 53.7 / 179.1 |
| Brain clusters | 0.819 / 0.860 | 0.817 / 0.860 | 95.3 / 562.7 |
| Lung across datasets† | 0.758 / 0.761 | — | 189.9 / 875.1 |

**Table S5.** scArches configurations (native classifier head / weighted nearest neighbors, WKNN) on identical inputs; means of three repeats. The native-head setting uses 40/20/40 reference-scVI/reference-scANVI/query epochs, default architecture and no batch covariate. The weighted-neighbor setting uses 200/100/100 epochs, the mapping architecture and batch handling described below. Time includes fit plus predict.

† Lung across datasets uses Cell Ontology concordance (same term or ancestor–descendant relationship), not exact label accuracy. Macro-F1 is not reported for that row.

```{=latex}
\end{minipage}
```

scArches uses scANVI reference mapping, a 30-dimensional latent space, two hidden layers,
layer normalization and a batch covariate. Reference scVI and scANVI training use up to 200
and 100 epochs with early stopping, respectively. Query adaptation uses 100 fixed epochs;
labels are transferred by Gaussian-weighted voting among 15 reference neighbors. The
weighted-neighbor transfer and query-adaptation budget follow ArchMap [Lotfollahi et al. 2026]; reference training, atlas inputs and hardware are specific to this benchmark.

Across seven shared-vocabulary splits, the weighted-neighbor configuration had 0.0173
higher mean accuracy and 0.0209 higher macro-F1 than the native-head configuration, with
3.0 to 5.9 times the complete-job time. The largest accuracy difference was at 151 brain
clusters (0.0414); the smallest was at 24 subclasses (0.0021). Label transfer, batch handling,
architecture and training budget differ jointly; this is a comparison of configurations,
not an ablation of individual settings.

# Matched annotation cohorts

The three-method comparison comprises 72 scored runs with identical query cell IDs and
truth labels within each split. The native-head configuration contributes 24 additional
runs to Table S5. Repeats characterize training variation on fixed partitions.

The eleven-method liver panel uses 36 labels; the matched panel uses 38. The blood/gut
panels have the same sample counts but different sampled cells. Brain panels have the
same split sizes, but their cell identities are not verified as identical. Consequently,
these panels support within-panel comparisons and do not support paired cross-panel
rankings, including a comparison of SingleR's 0.845 with scArches' 0.860 brain-cluster score.

Lung cross-dataset transfer uses Cell Ontology concordance and is excluded from exact-label
accuracy and macro-F1 means. Results, predictions and manifests are provided
in the repository. Label accuracy does not evaluate ArchMap's hosted service or integrated
embedding quality.

```{=latex}
\begin{minipage}{\linewidth}
```

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

**Table S6.** scArches, linear-anova-pca and ACTINN-JAX on the matched cohorts. Entries follow that order in every column. Means of three repeats; time includes fit plus predict, and memory is peak process MB.

† Lung across datasets uses Cell Ontology concordance (same term or ancestor–descendant relationship), not exact label accuracy. Macro-F1 is not reported for that row.

```{=latex}
\end{minipage}
```

# Eleven-method panel overview

```{=latex}
\begin{minipage}{\linewidth}
```

![Eleven-method accuracy and cost ranges](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_cost_accuracy_ranges.png)

**Figure S14.** Eleven-method panel using native-head scArches. Accuracy is shown as the gap to the leading method on each split; time is fit plus predict. Points and ranges summarize different splits, not donor uncertainty. The weighted-neighbor scArches comparison uses separate cohorts (Table S6).

```{=latex}
\end{minipage}
```


## Direct local annotation comparison

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

```{=latex}
\begin{minipage}{\linewidth}
```

![Duong annotation agreement at equal coverage](/Users/iandriver/Downloads/actinn-jax-benchmark/docs/figures/fig_archmap_direct_coverage.png)

**Figure S15.** Ontology concordance on the mapped Duong cohort as a function of retained
coverage. Confidence ranks are computed independently for each method; ties receive
fractional weights to match coverage exactly. Shading is the range of three seeded runs,
not a donor or study confidence interval. The linear configuration has one converged fit.

```{=latex}
\end{minipage}
```
