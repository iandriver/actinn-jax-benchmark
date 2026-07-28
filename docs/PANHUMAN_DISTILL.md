# Distilling Pan-human Azimuth into an actinn-jax reference

**Question:** can Pan-human Azimuth stand in for scPRINT as the source of a pretrained
actinn-jax reference — same harmonized vocabulary, actinn-jax's cost profile?

**Answer: yes, on the data you distill on, and that is the whole constraint.** On held-out
cells from the distillation corpus the student reproduces the teacher's label on **85.6%**
of cells (89.6% ontology-equivalent) and **matches its accuracy against ground truth**
(0.666 vs 0.662) while predicting **~13× faster**. On a tissue withheld from the corpus
entirely it falls to 0.407 against the teacher's 0.512. A distilled model is as broad as
its distillation corpus and no broader — so a genuinely pan-human student needs a
pan-human corpus, not a better recipe.

Scripts: [`distill_dump.py`](../benchmark/explore/distill_dump.py) (teacher, `.venv-panhuman`)
→ [`distill_train.py`](../benchmark/explore/distill_train.py) (student, core `.venv`).
Numbers: [`results_panhuman_distill.csv`](results_panhuman_distill.csv). Background on the
teacher: [PAN_HUMAN_AZIMUTH.md](PAN_HUMAN_AZIMUTH.md).

## Why this is worth doing

The shipped `broad_human_v1` gets its coarse→fine hierarchy from clustering scPRINT
embeddings ([UPDATE_BROAD_REFERENCE.md](UPDATE_BROAD_REFERENCE.md), stage 2) — the one step
in the build that wants a GPU and an hour. Pan-human Azimuth already *has* a hierarchy: 8
levels, every node mapped to a Cell Ontology term. Distilling it takes both the labels and
the structure from the teacher, which removes the foundation model from the build:

| | `broad_human_v1` | distilled |
|---|---|---|
| labels | CELLxGENE `cell_type`, as-is | Pan-human Azimuth's harmonized typology |
| hierarchy | Ward clustering of scPRINT centroids | the teacher's own broad level |
| accelerator | GPU/MPS for the embed stage | **none** |
| labeled input | required | **not required** — the teacher labels raw counts |
| vocabulary | inherits CELLxGENE's fragmentation | one-to-one CL crosswalk |

The last two matter most. Distillation needs only *unlabeled human counts*, so any h5ad on
disk can extend the corpus, and the result speaks a vocabulary with a published ontology
mapping rather than 798 census strings of uneven granularity.

## Setup

Corpus: three local atlases, capped at 600 cells per label each, teacher-labeled at
1,137–1,405 cells/s.

| atlas | cells | truth types | teacher fine labels | `Unassigned` |
|---|---:|---:|---:|---:|
| krasnow lung | 18,551 | 46 | 81 | 0.0% |
| HLiCA liver | 5,400 | 36 | 99 | 0.1% |
| blood + gut | 10,255 | 86 | 115 | 0.3% |

Concatenated on the shared Ensembl gene space: **34,063 cells**, 111 teacher labels after
dropping classes with fewer than 8 cells, trained on a 4,000-gene HVG panel. The atlases'
own labels are never used for training — only for scoring.

Two arms, because they answer different questions:

- **in-corpus** — held-out cells from the same atlases. *Does the student reproduce the
  teacher?*
- **held-out liver** — the entire liver atlas withheld. *Does the student generalize to
  tissue the corpus never covered?*

## Results

| arm | train | test | classes | student≡teacher (exact) | student≡teacher (ontology) | student vs truth | teacher vs truth |
|---|---:|---:|---:|---:|---:|---:|---:|
| in-corpus | 25,551 | 8,512 | 111 | **0.856** | **0.896** | **0.666** | 0.662 |
| held-out liver | 28,721 | 5,342 | 102 | 0.529 | 0.607 | 0.407 | 0.512 |

Cost, same machine, same cells:

| | teacher | student |
|---|---:|---:|
| predict throughput | 1,137–1,405 cells/s | **14,134–17,809 cells/s** |
| train time | — (pretrained) | 15 s |
| model size | ~7.0M params + TF 2.17 / Keras 3 runtime | **11.9 MB**, pure JAX |

Three readings:

1. **Distillation is nearly lossless in-distribution.** 85.6% exact label agreement across
   111 classes, and where the student and teacher disagree they usually disagree by a
   sibling: ontology-equivalent agreement is 89.6%. Accuracy against the atlases' own
   labels is **the same to within noise** (0.666 vs 0.662) — the student is not a
   degraded copy, it is the teacher's decision boundary in a smaller model.
2. **The speedup is the point.** ~13× faster prediction with no accuracy cost, in an
   environment that does not need TensorFlow. That is the same trade the paper makes
   against every other baseline, applied to the strongest published broad annotator.
3. **Breadth does not come for free.** Withhold liver and the student loses 10 points to
   the teacher (0.407 vs 0.512) and only reproduces its call on 53% of cells. The teacher
   saw 9.7M cells across 23 tissues; a 34k-cell corpus from three atlases cannot stand in
   for that. **The distillation corpus, not the distillation method, is the binding
   constraint.**

The in-corpus arm should not be over-read: the student trained on cells from those atlases
(never on their labels), so it can absorb atlas-specific structure the teacher does not
use. It measures faithful reproduction of the teacher on a known distribution — which is
what a distilled reference is for — not independent biological generalization.

## Building a pan-human student

The path follows directly from the held-out arm: distill on a corpus with census-wide
breadth. That is stage 1 of the existing pipeline, and **stage 2 drops out entirely**:

```bash
# 1. breadth (hours, network) -- the same census pull the shipped reference uses
ACTINN_REF_WORK=/tmp/actinn_ref_build PER_TYPE=60 CENSUS_VERSION=2025-11-08 \
  .venv-scprint/bin/python benchmark/explore/fetch_census_wide.py

# 2. teacher labels the corpus (no GPU; ~1,200 cells/s)
.venv-panhuman/bin/python benchmark/explore/distill_dump.py --cap 600

# 3. student (CPU, minutes)
.venv/bin/python benchmark/explore/distill_train.py --out /tmp/panhuman_distill_v1
```

`distill_dump.py` picks the census pull up automatically once it exists (it is the
`census` entry in `CORPORA`, skipped silently when absent); `--only census` restricts the
run to it. Expect the class count to rise well above 111 — the three local atlases only
exercise 111 of the teacher's 382 leaves — and the held-out gap to narrow in proportion to
the tissue coverage added.

One trap worth naming: the census pull must carry `feature_name`. Pan-human Azimuth keys
its 5,055-gene panel on **symbols**, while census data is Ensembl-keyed, so a pull without
the symbol column produces a corpus the teacher cannot score. `fetch_census_wide.py` now
requests it.

## Licensing

`panhumanpy` is **MIT** (v1.0.0), which permits training on its outputs and redistributing
the result. Before shipping a distilled reference in the package, confirm the terms on the
**weights** specifically ([Zenodo](https://doi.org/10.5281/zenodo.20401417)) — the package
license covers the code, and the weights are downloaded separately at first use. Attribute
Sarkar et al. either way; a distilled model is a derivative of their labeling.

## Limitations

- **One teacher, no ensemble.** Every student error the teacher also makes is invisible to
  these numbers except in the `*_vs_truth` columns.
- **Three atlases, two tissues with CL ids.** blood+gut carries no ontology ids, so it
  contributes breadth to training but nothing to the concordance columns.
- **One held-out arm.** Liver is withheld; whether the 10-point drop is representative of
  other tissues is untested.
- **The teacher's `Unassigned` class survives distillation but is barely exercised** —
  0.0–0.3% of the corpus. Its quality-control behaviour is inherited in name; it is not
  measured here.
- **Hard targets only.** actinn-jax trains on labels, so the teacher's calibrated
  probabilities — the part of a distillation that usually carries the most information,
  especially for classes with few cells — are discarded. Soft-target training would need a
  loss change in the package, and would likely close part of the held-out gap.
- **Three of eight levels used.** The student takes `azimuth_broad` as its hierarchy and
  `azimuth_fine` as its leaves. The teacher's deeper levels (up to 382 classes) are
  available in the same dump and would give a finer student at the cost of more cells per
  class.
