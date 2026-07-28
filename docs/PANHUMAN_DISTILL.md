# Distilling Pan-human Azimuth into an actinn-jax reference

**Question:** can Pan-human Azimuth stand in for scPRINT as the source of a pretrained
actinn-jax reference — same harmonized vocabulary, actinn-jax's cost profile?

**Answer: yes.** Distilled on 85k cells — three local atlases plus a census-wide pull — a
324-class student reproduces the teacher and, on a liver study none of the actinn-jax
models trained on, **scores above the reference we ship and level with the teacher itself**:

| broad annotator, same 3,396 cells | classes | ontology | throughput |
|---|---:|---:|---:|
| actinn-jax `broad_human_v1` (shipped) | 798 | 0.338 | 2,962 cells/s |
| **actinn-jax distilled from Pan-human Azimuth** | 324 | **0.406** | **10,021 cells/s** |
| Pan-human Azimuth (the teacher) | 47 | 0.380 | ~1,000–1,500 cells/s |

The distilled model is smaller (17 MB), 3.4× faster than the shipped reference, needs no
GPU and no labels to build — and answers in a vocabulary with a published CL crosswalk
instead of 798 census strings of uneven granularity.

**The distillation corpus is what bounds it.** From three atlases alone the student trailed
its teacher by 10 points on withheld liver (0.407 vs 0.512); adding the census pull closed
that to 3 (0.481 vs 0.511), and on withheld lung the gap is **1.5** (0.695 vs 0.710).
Breadth came from data, not from a better recipe.

It ships as `actinn_jax.bundled_reference("panhuman_distill_v1")`. The teacher's weights are
CC BY 4.0, so **attribution is a licence condition** — see below; the notice travels inside
the model's `build_info.json`.

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

Corpus: three local atlases plus a census-wide pull, capped per label, teacher-labeled at
699–1,405 cells/s.

| source | cells | truth types | teacher fine labels | `Unassigned` |
|---|---:|---:|---:|---:|
| krasnow lung | 18,551 | 46 | 81 | 0.0% |
| HLiCA liver | 5,400 | 36 | 99 | 0.1% |
| blood + gut | 10,255 | 86 | 115 | 0.3% |
| **CELLxGENE census** (2025-11-08, ≤60/type) | **51,346** | **867** | **408** | 0.5% |

The census pull alone exercises **408** of the teacher's labels — the three local atlases
manage 111 between them. Concatenated on the shared Ensembl gene space: **85,256 cells**
and 324 teacher labels after dropping classes with fewer than 8 cells, trained on a
4,000-gene HVG panel. The atlases' own labels are never used for training — only for
scoring.

Two arms, because they answer different questions:

- **in-corpus** — held-out cells from the same atlases. *Does the student reproduce the
  teacher?*
- **held-out liver** — the entire liver atlas withheld. *Does the student generalize to
  tissue the corpus never covered?*

## Results

Both runs, so the effect of adding breadth is visible:

| corpus | arm | classes | student≡teacher (exact) | (ontology) | student vs truth | teacher vs truth |
|---|---|---:|---:|---:|---:|---:|
| 3 atlases, 34k | in-corpus | 111 | 0.856 | 0.896 | 0.666 | 0.662 |
| 3 atlases, 34k | held-out liver | 102 | 0.529 | 0.607 | 0.407 | 0.512 |
| **+ census, 85k** | in-corpus | **324** | 0.757 | 0.822 | 0.513 | 0.521 |
| **+ census, 85k** | held-out liver | **324** | **0.723** | **0.785** | **0.481** | 0.511 |
| **+ census, 85k** | held-out lung | **324** | **0.836** | **0.878** | **0.695** | 0.710 |

Only the **held-out liver** rows compare across corpora — the in-corpus test population
changes when census cells enter it, which is why both student *and teacher* accuracy fall
there (a harder, broader evaluation, not a worse model). On the fixed liver arm, adding
census breadth moves agreement 0.529 → **0.723** and closes the accuracy gap from 10.5
points to **3.0**.

**A second withheld atlas agrees, and more strongly.** Holding out krasnow lung entirely
(18,550 cells) the student tracks its teacher to **1.5 points** — 0.695 vs 0.710 — at 0.836
exact agreement. So the liver result is not a one-atlas fluke: once the corpus covers the
territory, the student is a faithful stand-in for the teacher on data it never saw.

Cost, same machine:

| | teacher | student (census-scale) |
|---|---:|---:|
| predict throughput | 699–1,405 cells/s | **12,039–22,393 cells/s** |
| train time | — (pretrained) | 33 s (full corpus, 324 classes) |
| model size | ~7.0M params + TF 2.17 / Keras 3 runtime | **17.1 MB**, pure JAX |

Three readings:

1. **Distillation reproduces the teacher, and the residual disagreement is mostly
   sibling-level.** 72–86% exact agreement, and ontology-equivalent agreement runs 6
   points higher in every arm — the student and teacher usually land on neighbouring nodes
   rather than different lineages. Accuracy against the atlases' own labels tracks the
   teacher within ~1 point in-corpus.
2. **The speedup is the point.** ~10–20× faster prediction at matched accuracy, in an
   environment that does not need TensorFlow. That is the trade the paper makes against
   every other baseline, applied to the strongest published broad annotator.
3. **Breadth comes from the corpus.** The whole improvement between the two runs is data;
   the recipe is unchanged. **The distillation corpus, not the distillation method, is the
   binding constraint** — which is exactly why the census pull was worth 5.3 hours.

## Against the reference we ship

The arms above ask "did distillation work". This asks "is the result better than
`broad_human_v1`". Query: the withheld HLiCA liver study (3,396 cells, 34 truth types, all
CL-annotated) — **not** part of any distillation corpus.
[`results_broad_head_to_head.csv`](results_broad_head_to_head.csv), from
[`distill_compare_broad.py`](../benchmark/explore/distill_compare_broad.py).

| model | classes | ontology | predict | cells/s |
|---|---:|---:|---:|---:|
| actinn-jax `broad_human_v1` (shipped) | 798 | 0.338 | 1.15 s | 2,962 |
| **actinn-jax distilled from PHA** | 324 | **0.406** | **0.34 s** | **10,021** |
| Pan-human Azimuth (teacher) | 47 | 0.380 | — | — |

The distilled reference is **7 points better than the one we ship, at 3.4× its speed**, with
less than half the classes. Fewer, better-harmonized, ontology-mapped classes beat more
classes inherited from a fragmented vocabulary.

**Do not read the 0.406 vs 0.380 as beating the teacher.** Both actinn-jax models draw on a
census sample that may include cells from these same HLiCA studies, so liver exposure cannot
be ruled out for either of them; the teacher has no such exposure. The comparison that is
clean is **shipped vs distilled** — both census-derived, same possible exposure, 0.338 vs
0.406. Against the teacher, the honest statement is *level with it*.

The in-corpus arms should not be over-read either: the student trained on cells from those
atlases (never on their labels), so it can absorb atlas-specific structure the teacher does
not use. They measure faithful reproduction of the teacher on a known distribution — which
is what a distilled reference is for — not independent biological generalization.

## Reproducing it

Three commands. Stage 2 of the normal reference build — the scPRINT embedding — **drops out
entirely**, because the hierarchy comes from the teacher:

```bash
# 1. breadth: the same census pull the shipped reference uses (5.3 h, network-bound)
ACTINN_REF_WORK=/tmp/actinn_ref_build PER_TYPE=60 CENSUS_VERSION=2025-11-08 \
  .venv-scprint/bin/python benchmark/explore/fetch_census_wide.py

# 2. teacher labels the corpus (no GPU; 73 s for 51k cells)
.venv-panhuman/bin/python benchmark/explore/distill_dump.py --cap 600

# 3. student (CPU; 33 s to train, ~4 min including both scoring arms)
.venv/bin/python benchmark/explore/distill_train.py --out /tmp/panhuman_distill_census

# 4. how does it compare to what we ship, on a query neither trained on?
.venv/bin/python benchmark/explore/distill_compare_broad.py \
  --query /Volumes/IanSSD/hlica/liver_query_xstudy.h5ad \
  --student /tmp/panhuman_distill_census \
  --teacher-parquet /tmp/panhuman_tier1_liver_cross.parquet
```

`distill_dump.py` picks the census pull up automatically once it exists (the `census` entry
in `CORPORA`, skipped silently when absent); `--only census` restricts the run to it. Total
compute after the pull: **under 10 minutes**, all CPU.

One trap worth naming: the census pull must carry `feature_name`. Pan-human Azimuth keys
its 5,055-gene panel on **symbols**, while census data is Ensembl-keyed, so a pull without
the symbol column produces a corpus the teacher cannot score. `fetch_census_wide.py` now
requests it — caught before the 5-hour pull rather than after.

## Licensing and attribution

Both halves are cleared for this use:

| | license |
|---|---|
| `panhumanpy` v1.0.0 (code) | **MIT** |
| Pan-human Azimuth weights ([Zenodo](https://doi.org/10.5281/zenodo.20401417)) | **CC BY 4.0** |

CC BY permits deriving from and redistributing the model **provided credit is given**, and
a distilled reference is squarely a derivative of their labeling. So attribution is not
optional politeness here, it is the licence term — any distilled reference we ship must
carry it, and `distill_train.py` writes it into the model's `build_info.json` so the
artifact cannot be separated from its credit:

> Distilled from **Pan-human Azimuth** — Sarkar, Li, Molla, … Satija, *Organism-scale
> annotation with Pan-human Azimuth*, bioRxiv 2026,
> [doi:10.64898/2026.07.16.738997](https://doi.org/10.64898/2026.07.16.738997). Model
> weights © the authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), via
> [`panhumanpy`](https://github.com/satijalab/panhumanpy) (MIT) and
> [Zenodo](https://doi.org/10.5281/zenodo.20401417).

## Limitations

- **One teacher, no ensemble.** Every student error the teacher also makes is invisible to
  these numbers except in the `*_vs_truth` columns.
- **blood+gut carries no ontology ids**, so it contributes breadth to training but nothing
  to the concordance columns.
- **"Held-out liver" withholds an *atlas*, not a tissue.** Before the census pull it was
  genuinely tissue-held-out; the census sample spans 376 tissues, so the improved liver arm
  partly reflects liver cells entering the corpus from other studies. That is the intended
  effect — corpus coverage is the variable under test — but it is not evidence of
  generalization to biology the corpus never saw.
- **The two runs' test sets differ slightly** (5,342 vs 5,391 liver cells): the
  minimum-cells-per-class filter retains more cells once the corpus is larger.
- **Two held-out atlases, both from the local set.** Liver and lung agree, but neither is a
  tissue the census sample leaves uncovered — nothing here measures what happens on biology
  genuinely outside the corpus.
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
