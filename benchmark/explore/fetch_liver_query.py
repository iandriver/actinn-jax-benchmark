"""Pull a human liver query with zonation ground truth (periportal / midzonal /
centrilobular hepatocytes + other liver cells) for the actinn-jax zonation example.
Single dataset -> localized read. Run in .venv-scprint.
"""
import warnings, time; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np

DID = "ddb22b3d-a75c-4dd1-9730-dff7fc8ca530"   # best-balanced liver atlas (all 3 zones)
CAP = 500                                        # cells per cell type

with cellxgene_census.open_soma(census_version="stable") as census:
    obs = cellxgene_census.get_obs(census, "homo_sapiens",
        value_filter=f"is_primary_data == True and dataset_id == '{DID}'",
        column_names=["soma_joinid", "cell_type"])
    print(f"liver dataset: {len(obs):,} cells, {obs.cell_type.nunique()} types", flush=True)
    rng = np.random.default_rng(1); keep = []
    for c, g in obs.groupby("cell_type", observed=True):
        j = g.soma_joinid.values
        keep.append(rng.choice(j, min(CAP, len(j)), replace=False))
    jid = np.sort(np.concatenate(keep))
    print(f"pulling {len(jid):,} cells...", flush=True)
    t = time.time()
    a = cellxgene_census.get_anndata(census, "homo_sapiens", obs_coords=jid.tolist(),
        column_names={"obs": ["cell_type", "cell_type_ontology_term_id", "tissue"],
                      "var": ["feature_id"]})
    a.var_names = a.var["feature_id"].astype(str).values
print(f"pulled {a.shape} / {a.obs.cell_type.nunique()} types in {time.time()-t:.0f}s", flush=True)
a.obs_names_make_unique()
a.write_h5ad("/tmp/liver_zonation_query.h5ad")
print("LIVER_QUERY_DONE", a.obs["cell_type"].value_counts().head(8).to_dict(), flush=True)
