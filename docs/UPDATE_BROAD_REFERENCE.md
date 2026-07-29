# Updating the shipped broad reference

`actinn_jax.bundled_reference("broad_human_v1")` is a pretrained census-wide model — 798
cell types, 28 coarse groups, ~50 MB — archived at
[doi:10.5281/zenodo.21688151](https://doi.org/10.5281/zenodo.21688151) and downloaded on
first use. This is how it gets rebuilt.

One step beyond the stages below: after a rebuild, re-archive it
(`python tools/package_references.py` in the package repo, which rewrites the checksum the
shipped code verifies against) and publish a new Zenodo version — otherwise users keep
downloading the old model, or the new checksum makes the published archive un-installable.
The concept DOI [10.5281/zenodo.21688150](https://doi.org/10.5281/zenodo.21688150) always
resolves to the newest version.

One command:

```bash
benchmark/explore/update_broad_reference.sh
```

It runs four stages through one work directory (`$ACTINN_REF_WORK`, default
`/tmp/actinn_ref_build`), skips stages whose output already exists, and ends by scoring the
installed reference on an atlas that was never part of it.

| stage | environment | output | cost |
|---|---|---|---|
| 1 `fetch` | `.venv-scprint` | `census_wide_ref.h5ad` + `census_release.json` | network-bound; ~30 s per 10-dataset batch |
| 2 `embed` | `.venv-scprint` | `census_wide_emb.npz` | the only accelerated step; scPRINT runs ~22 ms/cell on MPS |
| 3 `build` | actinn-jax `.venv` | the shipped reference + `build_info.json` | minutes, CPU |
| 4 `verify` | actinn-jax `.venv` | a score you can compare against the previous build | ~1 min |

Run a subset with `STAGES`:

```bash
STAGES="build verify" benchmark/explore/update_broad_reference.sh
```

## What each stage does

**1. Fetch** ([`fetch_census_wide.py`](../benchmark/explore/fetch_census_wide.py)) samples
CELLxGENE census — all primary human cells — stratified by `cell_type` at `PER_TYPE=40`
cells per type, dropping types with fewer than 12 cells and the uninformative labels
(`unknown`, `native cell`, `eukaryotic cell`, `animal cell`). The pull is checkpointed per
batch of 10 datasets and retried, because transient TileDB/S3 read errors are normal at
this scale; re-running resumes from `census_parts/`.

**2. Embed** ([`embed_broad.py`](../benchmark/explore/embed_broad.py)) runs scPRINT
(`medium-v1.5`) over the reference in 4,000-cell chunks. scPRINT's QC **drops cells**, so
the embedding is not positionally aligned to the input; the label is carried through the
model as `carry_label` and saved beside the vectors. Stage 3 relies on that pairing.

**3. Build** ([`build_census_model.py`](../benchmark/explore/build_census_model.py))
clusters per-type centroids in scPRINT space into `max(8, round(sqrt(n_types)))` coarse
groups, then trains a coarse classifier plus one fine classifier per group on a 4,000-gene
HVG panel. Before shipping it runs the abstain calibration: 10% of cell types are withheld
*entirely* as out-of-distribution, plus a within-type test split, and `min_prob` is swept
to trade coverage against accuracy. That table is what the package README quotes.

**4. Verify** ([`verify_reference.py`](../benchmark/explore/verify_reference.py)) loads the
*installed* reference and annotates a held-out atlas (default: the krasnow lung atlas, 50
cells per type), reporting ontology-aware concordance, coverage at `min_prob`, and
throughput.

## Reference numbers for the current build

`broad_human_v1`, scored on krasnow lung (2,161 cells, 46 truth types):

| metric | value |
|---|---:|
| exact label match | 0.177 |
| ontology concordance, all cells | 0.538 |
| ontology concordance, `p ≥ 0.5` | 0.638 |
| coverage at `p ≥ 0.5` | 0.384 |
| throughput | ~1,850 cells/s |

Exact match is low and that is expected: the census vocabulary and an atlas's vocabulary
name the same cell differently (`periportal region hepatocyte` vs `Hepatocyte`). Ontology
concordance is the number to compare across rebuilds. A rebuild that lands materially below
0.538 is a regression regardless of how healthy the build log looked.

## Things that have gone wrong here

- **The stages did not agree on filenames.** Stage 2's default output (`/tmp/broad_emb.npz`)
  was not what stage 3 read (`/tmp/census_wide_emb.npz`), so a hand-run pipeline could train
  against a *stale* embedding with no error anywhere. Every path now derives from
  `$ACTINN_REF_WORK`.
- **`stable` is a moving pointer.** The same command a month apart builds a different
  reference. Stage 1 now records the resolved release to `census_release.json` and stage 3
  copies it into the shipped `build_info.json`, so a reference can say what it came from.
  Pin explicitly with `CENSUS_VERSION=2025-11-08` for a reproducible rebuild.
- **A rebuild can train cleanly and annotate worse.** Stage 3 backs the previous reference
  up to `<name>.bak.<timestamp>` before overwriting, and stage 4 exists to catch this.
  Restore by moving the backup back over the reference directory.

## Knobs

| variable | default | effect |
|---|---|---|
| `ACTINN_REF_WORK` | `/tmp/actinn_ref_build` | work directory for all intermediates |
| `ACTINN_JAX_REPO` | `~/Downloads/actinn-jax` | package checkout the reference is written into |
| `CENSUS_VERSION` | `stable` | census release to pull |
| `PER_TYPE` | `40` | cells per cell type — drives both breadth and total size |
| `N_HVG` | `4000` | genes in the trained panel |
| `REF_NAME` | `broad_human_v1` | reference directory name; set it to build a variant side by side |
| `STAGES` | all four | subset of `fetch embed build verify` |

## Building a different reference

The same three stages build any reference; only the input changes. For a
tissue-specific one, skip stages 1–2 and point stage 3 at your own labeled data:

```bash
REF_H5AD=/path/to/labeled.h5ad REF_NAME=my_reference \
  STAGES="build verify" benchmark/explore/update_broad_reference.sh
```

That still wants an embedding for the hierarchy. Two ways to avoid the GPU entirely:
`examples/build_reference.py` in the package (discovers the hierarchy from scPRINT, so it
does need one), or distillation from a pretrained teacher —
[PANHUMAN_DISTILL.md](PANHUMAN_DISTILL.md) builds a reference whose hierarchy comes from
Pan-human Azimuth's own levels, with no foundation model in the loop.
