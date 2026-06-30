"""Pull a broad human reference from Tabula Sapiens 'All Cells' (~170 cell types,
~25 organs) for the shipped actinn-jax annotator. Capped via a within-dataset
stratified joinid subsample (one contiguous block -> fast). Run in .venv-scprint.
"""
import warnings, time; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np

DID = "53d208b0-2cfd-4366-9866-c3c6114081bc"  # Tabula Sapiens - All Cells (1.14M)
PER_TYPE, CAP = 110, 22000

with cellxgene_census.open_soma(census_version="stable") as census:
    t = time.time()
    obs = cellxgene_census.get_obs(census, "homo_sapiens",
        value_filter=f"dataset_id == '{DID}'", column_names=["soma_joinid", "cell_type"])
    print(f"All-Cells obs: {len(obs)} cells, {obs.cell_type.nunique()} types in {time.time()-t:.0f}s", flush=True)
    rng = np.random.default_rng(0); keep = []
    for c, g in obs.groupby("cell_type"):
        j = g.soma_joinid.values
        keep.append(rng.choice(j, min(PER_TYPE, len(j)), replace=False))
    jid = np.sort(np.concatenate(keep))
    if len(jid) > CAP:
        jid = np.sort(rng.choice(jid, CAP, replace=False))
    print(f"pulling {len(jid)} cells...", flush=True)
    t = time.time()
    a = cellxgene_census.get_anndata(census, "homo_sapiens", obs_coords=jid.tolist(),
        column_names={"obs": ["cell_type", "cell_type_ontology_term_id", "tissue"],
                      "var": ["feature_id"]})
    a.var_names = a.var["feature_id"].astype(str).values
print(f"pulled {a.shape} / {a.obs.cell_type.nunique()} types / {a.obs.tissue.nunique()} tissues "
      f"in {time.time()-t:.0f}s", flush=True)
a.obs_names_make_unique()
a.write_h5ad("/tmp/broad_human_ref.h5ad")
print(f"BROAD_REF_DONE {a.shape}", flush=True)
