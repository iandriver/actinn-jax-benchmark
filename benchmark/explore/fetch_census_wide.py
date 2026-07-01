"""Pull a CENSUS-WIDE human reference: sample across ALL CELLxGENE human datasets to
cover most of the Cell Ontology (~hundreds-1000+ cell types), capped per type. This is
the broad reference scPRINT itself was trained on the breadth of. Run in .venv-scprint.

Stratified by cell_type across all primary human cells; cells are scattered across many
datasets so the get_anndata pull is the slow step (bounded by the low per-type cap).
"""
import os, glob, warnings, time; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np, anndata as ad, scanpy as sc

PARTS = "/tmp/census_parts"; os.makedirs(PARTS, exist_ok=True)

PER_TYPE = 40        # cells per cell type (low cap -> bounded total)
MIN_CELLS = 12       # drop ultra-rare types that can't train
DROP = {"unknown", "native cell", "eukaryotic cell", "animal cell"}
COLS = {"obs": ["cell_type", "cell_type_ontology_term_id", "tissue", "assay", "dataset_id"],
        "var": ["feature_id"]}

with cellxgene_census.open_soma(census_version="stable") as census:
    t = time.time()
    obs = cellxgene_census.get_obs(census, "homo_sapiens",
        value_filter="is_primary_data == True",
        column_names=["soma_joinid", "cell_type", "dataset_id"])
    print(f"primary human obs: {len(obs):,} cells, {obs.cell_type.nunique()} types "
          f"in {time.time()-t:.0f}s", flush=True)

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
a.write_h5ad("/tmp/census_wide_ref.h5ad")
print(f"CENSUS_WIDE_DONE {a.shape} {a.obs.cell_type.nunique()}", flush=True)
