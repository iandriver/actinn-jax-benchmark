# Supplementary material

*Supplement to "A benchmark of cell-type annotation methods for single-cell data". Figures and tables referenced from
the main text as Supplementary Figure S1–S4 and Supplementary Table S1–S3.*

## Supplementary figures

![confusion matrices](figures/fig_confusion.png)

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

![per-class recall, blood and gut](figures/fig_perclass_blood_gut_intra.png)

**Figure S2.** Per-class recall for eleven methods on the 86-type blood+gut split, ordered by
how much the methods disagree (best minus worst recall, right-hand strip). The best and worst
method differ by a median of 0.30 recall per class and up to 0.73; mean pairwise Spearman
between methods' per-class recall is 0.58. Class size is not the explanation — the test split
is capped per label at 14–30 cells for all 86 classes, and the ten smallest classes score
slightly higher than the ten largest (0.865 vs 0.827).

![per-class recall, lung](figures/fig_perclass_lung_intra.png)

**Figure S3.** Per-class recall on the 46-type lung split, drawn as in Figure S2. Methods
agree more here than on blood+gut: mean pairwise Spearman 0.65, median best-minus-worst 0.16.

![per-class recall, liver](figures/fig_perclass_liver_intra.png)

**Figure S4.** Per-class recall on the 36-type liver split, drawn as in Figure S2. Mean
pairwise Spearman is 0.66, but the median best-minus-worst gap is 0.43 — the widest
disagreement of the three splits, and a reminder that a high rank correlation between methods
still leaves large per-class differences.

## Supplementary tables

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
