"""Stage 1/3 of the broad-reference build: pull a CENSUS-WIDE reference.

Samples across ALL CELLxGENE datasets for one organism to cover most of the Cell Ontology
(~hundreds-1000+ cell types), capped per type. This is the broad reference scPRINT
itself was trained on the breadth of. Run in .venv-scprint.

``ORGANISM`` selects the species (``homo_sapiens`` default, ``mus_musculus`` for the
pan-mouse reference). ``EXCLUDE_DATASETS`` / ``ONLY_DATASETS`` (comma-separated dataset
ids) carve a held-out test set out of the census itself: build the reference with datasets
excluded, then re-run with ``ONLY_DATASETS`` set to those same ids to pull a query the
reference has never seen. Without that, "held-out" would only mean held-out *cells* from
datasets the reference already covers.

Stratified by cell_type across all primary human cells; cells are scattered across many
datasets so the get_anndata pull is the slow step (bounded by the low per-type cap).

Driven by ``update_broad_reference.sh``; see docs/UPDATE_BROAD_REFERENCE.md. Every path
comes from ``$ACTINN_REF_WORK`` so the three stages cannot disagree about filenames --
they did before, and stage 3 then silently read a stale embedding.

``CENSUS_VERSION`` defaults to ``stable``, which is a moving pointer: the same command a
month apart builds a different reference. The resolved release is written to
``census_release.json`` so the shipped model can say which census it came from.
"""
import json, os, glob, warnings, time; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np, anndata as ad, scanpy as sc

WORK = os.environ.get("ACTINN_REF_WORK", "/tmp/actinn_ref_build")
os.makedirs(WORK, exist_ok=True)
ORGANISM = os.environ.get("ORGANISM", "homo_sapiens")
OUT_NAME = os.environ.get("OUT_NAME", "census_wide_ref.h5ad")
# Separate checkpoint dirs per output, or a query pull would resume from the reference's
# half-finished batches and silently mix the two.
PARTS = f"{WORK}/census_parts_{OUT_NAME.replace('.h5ad', '')}"
os.makedirs(PARTS, exist_ok=True)
OUT = f"{WORK}/{OUT_NAME}"
CENSUS_VERSION = os.environ.get("CENSUS_VERSION", "stable")


def _ids(name):
    return {s.strip() for s in os.environ.get(name, "").split(",") if s.strip()}


EXCLUDE_DATASETS = _ids("EXCLUDE_DATASETS")
ONLY_DATASETS = _ids("ONLY_DATASETS")

PER_TYPE = int(os.environ.get("PER_TYPE", 40))   # cells per cell type (bounded total)
MIN_CELLS = 12       # drop ultra-rare types that can't train
DROP = {"unknown", "native cell", "eukaryotic cell", "animal cell"}
# `feature_name` costs nothing to carry and is required by any symbol-keyed consumer --
# Pan-human Azimuth keys its 5,055-gene panel on symbols, and a census pull without it is
# unusable for distillation (docs/PANHUMAN_DISTILL.md).
COLS = {"obs": ["cell_type", "cell_type_ontology_term_id", "tissue", "assay", "dataset_id"],
        "var": ["feature_id", "feature_name"]}

try:
    desc = cellxgene_census.get_census_version_description(CENSUS_VERSION)
except Exception as e:                                  # offline / API change
    desc = {"error": str(e)[:200]}
desc.update({"requested_version": CENSUS_VERSION, "per_type_cap": PER_TYPE,
             "min_cells": MIN_CELLS, "organism": ORGANISM,
             "exclude_datasets": sorted(EXCLUDE_DATASETS),
             "only_datasets": sorted(ONLY_DATASETS)})
with open(f"{WORK}/census_release_{OUT_NAME.replace('.h5ad', '')}.json", "w") as fh:
    json.dump(desc, fh, indent=2)
if OUT_NAME == "census_wide_ref.h5ad":       # the reference build stage 3 reads
    with open(f"{WORK}/census_release.json", "w") as fh:
        json.dump(desc, fh, indent=2)
print(f"census {CENSUS_VERSION} -> {desc.get('release_build', desc)} | {ORGANISM}", flush=True)

with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    t = time.time()
    obs = cellxgene_census.get_obs(census, ORGANISM,
        value_filter="is_primary_data == True",
        column_names=["soma_joinid", "cell_type", "dataset_id"])
    print(f"primary {ORGANISM} obs: {len(obs):,} cells, {obs.cell_type.nunique()} types "
          f"in {time.time()-t:.0f}s", flush=True)

    if ONLY_DATASETS:
        obs = obs[obs.dataset_id.isin(ONLY_DATASETS)]
        print(f"restricted to {len(ONLY_DATASETS)} dataset(s): {len(obs):,} cells, "
              f"{obs.cell_type.nunique()} types", flush=True)
    if EXCLUDE_DATASETS:
        obs = obs[~obs.dataset_id.isin(EXCLUDE_DATASETS)]
        print(f"excluded {len(EXCLUDE_DATASETS)} dataset(s): {len(obs):,} cells remain, "
              f"{obs.cell_type.nunique()} types", flush=True)

    rng = np.random.default_rng(0); keep = []; kept_types = 0
    for c, g in obs.groupby("cell_type", observed=True):
        if c in DROP or len(g) < MIN_CELLS:
            continue
        j = g.soma_joinid.values
        keep.append(rng.choice(j, min(PER_TYPE, len(j)), replace=False))
        kept_types += 1
    jid = np.sort(np.concatenate(keep))
    chosen = obs.set_index("soma_joinid").loc[jid]
    by_ds = chosen.groupby("dataset_id", observed=True).groups   # dataset -> joinids
    print(f"plan: {len(jid):,} cells across {kept_types} types / {len(by_ds)} datasets "
          f"(cap {PER_TYPE}/type, >= {MIN_CELLS})", flush=True)

    # batch consecutive datasets (adjacent joinid ranges) into one localized read to
    # cut per-call overhead; checkpoint per batch (resumable) with retries (transient
    # S3/TileDB read errors are common).
    B = 10
    order = sorted(by_ds.items(), key=lambda kv: int(kv[1].min()))
    batches = [order[k:k + B] for k in range(0, len(order), B)]
    t = time.time(); done = skipped = 0
    for k, batch in enumerate(batches):
        fp = f"{PARTS}/batch_{k:04d}.h5ad"
        if os.path.exists(fp):
            done += len(batch); continue
        coords = sorted(int(x) for _, idx in batch for x in idx.values)
        for attempt in range(4):
            try:
                a = cellxgene_census.get_anndata(census, "homo_sapiens",
                    obs_coords=coords, column_names=COLS)
                a.write_h5ad(fp); done += len(batch); break
            except Exception as e:
                if attempt == 3:
                    print(f"  SKIP batch {k}: {str(e)[:70]}", flush=True); skipped += len(batch)
                else:
                    time.sleep(5 * (attempt + 1))
        if (k + 1) % 5 == 0 or k + 1 == len(batches):
            print(f"  batch {k+1}/{len(batches)} ({done} datasets done, {skipped} skipped) "
                  f"({time.time()-t:.0f}s)", flush=True)

parts = [sc.read_h5ad(f) for f in sorted(glob.glob(f"{PARTS}/batch_*.h5ad"))]
a = ad.concat(parts, join="outer", merge="first")
a.var_names = a.var["feature_id"].astype(str).values
print(f"pulled {a.shape} / {a.obs.cell_type.nunique()} types / {a.obs.tissue.nunique()} "
      f"tissues in {time.time()-t:.0f}s", flush=True)
a.obs_names_make_unique()
a.write_h5ad(OUT)
print(f"CENSUS_WIDE_DONE {a.shape} {a.obs.cell_type.nunique()}", flush=True)
