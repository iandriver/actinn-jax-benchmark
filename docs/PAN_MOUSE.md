# A pan-mouse reference, and a hierarchy that needs no GPU

`actinn_jax.bundled_reference("broad_mouse_v1")` — 453 cell types across 85 tissues,
38 MB, built in **17 seconds of CPU** from a census pull. On 12,646 cells from two mouse
datasets held out of the reference entirely, it reaches **0.638** ontology concordance
(**0.718** on the 71% of cells it keeps at `min_prob=0.5`) at **9,712 cells/s**.

Getting there required removing the GPU from the build, which turned out to improve the
human reference too.

## Why the human recipe doesn't transfer

Two routes exist for building a broad reference, and neither worked for mouse as-is:

- **Distillation** ([PANHUMAN_DISTILL.md](PANHUMAN_DISTILL.md)) takes labels *and* the
  coarse→fine hierarchy from a pretrained annotator. Pan-human Azimuth is **human-only**,
  so there is no teacher.
- **The census route** ([UPDATE_BROAD_REFERENCE.md](UPDATE_BROAD_REFERENCE.md)) discovers
  the hierarchy by clustering scPRINT embeddings — a GPU hour, and scPRINT's mouse support
  is untested here.

## The Cell Ontology is the hierarchy

The census already labels every cell with a Cell Ontology term, and CL encodes exactly the
relation the embedding clustering is trying to recover: which cell types are kinds of the
same thing. So describe each type by the set of CL terms it descends from and cluster
*those* — same Ward linkage, ontology indicator vectors in place of embeddings
([`ontology_hierarchy.py`](../benchmark/explore/ontology_hierarchy.py)).

That is worth a control, since "any grouping might help". Built from one corpus (51,346
human cells / 867 types) and scored on krasnow lung —
[`results_hierarchy_ablation.csv`](results_hierarchy_ablation.csv):

| hierarchy | ontology | note |
|---|---:|---|
| **Cell Ontology lineage** | **0.616** | |
| random | 0.539 | same group sizes, types shuffled |
| flat | 0.547 | no hierarchy at all |

Random lands on flat, so it is the *structure* doing the work rather than the mere presence
of groups. The ontology hierarchy is free, deterministic, and species-independent — which is
what makes the mouse reference buildable, and it also beats the shipped human reference's
0.538 on the same query (a different corpus, so not a controlled comparison).

## `broad_mouse_v1`

| | |
|---|---|
| corpus | CELLxGENE census 2025-11-08, `mus_musculus`, primary cells |
| sampling | ≤60 cells per type → **27,026 cells / 453 types / 85 tissues** from 47 datasets |
| held out | 2 datasets (`98e5ea9f…`, `812fa7bd…`) excluded from the reference entirely |
| hierarchy | 305 CL terms → **21 coarse groups** |
| build | 17 s, CPU only, no GPU and no foundation model |
| size | 38.1 MB |

**Abstain calibration** (45 cell types withheld as out-of-distribution, plus a within-type
test split):

| `min_prob` | accuracy (kept) | coverage | OOD flagged |
|---:|---:|---:|---:|
| 0.0 | 0.791 | 100% | 0% |
| 0.5 | 0.876 | 82% | 49% |
| 0.7 | 0.914 | 68% | 69% |
| 0.9 | 0.948 | 46% | 88% |

**Held-out test** — 12,646 cells (≤100/type) from the two withheld datasets, 137 truth
types, 41 tissues, none of it seen during training:

| metric | value |
|---|---:|
| exact label match | 0.394 |
| ontology concordance | **0.638** |
| ontology at `p ≥ 0.5` | **0.718** |
| coverage at `p ≥ 0.5` | 0.713 |
| throughput | 9,712 cells/s |

For scale, `broad_human_v1` on its held-out lung atlas scores 0.538 / 0.638 at 38% coverage.
Those are different queries and not comparable directly, but the mouse reference is clearly
in the same regime and keeps roughly twice as many cells at the same threshold — mouse
census has fewer near-duplicate subtypes than human's ~800-way vocabulary.

End to end, with no arguments to choose:

```bash
actinn-jax annotate mouse_data.h5ad --min-prob 0.5 -o annotations.csv
# query: (2000, 53384)
# organism mus_musculus -> reference broad_mouse_v1
# abstained at min_prob=0.5: 327 cells (16.4%)
```

## The bug this run produced, and the assertions that now catch it

The first mouse build was **entirely invalid** and looked completely healthy. Adding
`ORGANISM` to the fetch changed the `get_obs` call but left `get_anndata(census,
"homo_sapiens", …)` hardcoded. soma joinids are per-experiment, so mouse-selected ids were
pulled from the *human* experiment, which returned the human cells holding those numbers:
the right cell **count**, the wrong species. The build then produced 489 types with a
*better*-looking calibration table than the real thing.

Two printed numbers contradicted it and were read past: the pull reported **489 types where
the plan said 453** (you cannot obtain more types than you selected), and **243 tissues when
mouse census has 101**. What actually exposed it was checking the query's dataset ids —
nine datasets, none of the two requested — and then the gene space: `ENSG…`, human.

`fetch_census_wide.py` now fails the pull outright rather than writing a plausible file:

- gene ids must match the organism (`ENSMUSG` for mouse, `ENSG` for human)
- with `ONLY_DATASETS`, every returned cell must come from a requested dataset

The lesson generalizes past this script: a pull that is filtered, sampled and re-assembled
should assert something about the *content* it returns, not just complete without raising.
Cell counts matched the plan exactly through the entire failure.

## Limitations

- **One organism pair, one held-out split.** Two withheld datasets is a real test, but a
  narrow one; both come from the same census release the reference was drawn from.
- **The ontology hierarchy was validated on human, applied to mouse.** The ablation above
  is human; nothing here shows CL lineage groups mouse types as well as it groups human
  ones, though CL is species-neutral by construction.
- **Mouse census is shallow in datasets** — 51 total, of which one embryo atlas holds 11.4M
  of 18.4M cells. Breadth across *tissues* is good (85 in the reference), breadth across
  *labs and protocols* much less so than human's 487 datasets.
- **No distilled mouse counterpart.** If a pan-mouse pretrained annotator appears, the
  distillation route is worth re-running: on human it produced a better reference than the
  census route.
