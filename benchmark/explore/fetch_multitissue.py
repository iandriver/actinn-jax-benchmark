"""Build a diverse multi-organ reference: the named organs (fat, heart, kidney, liver)
plus a few more, pulled from CZ CELLxGENE census.

Each organ is capped via a within-dataset stratified subsample of soma_joinids (fast:
the dataset's cells are one contiguous block, so this avoids the across-census
scattered-read slowness). Kidney/adipose come from dedicated atlases (not in TS).
Run in .venv-scprint.
"""
import warnings, time; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np, anndata as ad

SOURCES = {  # organ -> dataset_id
    "Heart": "e6a11140-2545-46bc-929e-da243eed2cae",         # Tabula Sapiens
    "Liver": "6d41668c-168c-4500-b06a-4674ccf3e19d",         # Tabula Sapiens
    "Fat": "5e5e7a2f-8f1c-42ac-90dc-b4f80f38e84c",           # Tabula Sapiens (adipose)
    "Kidney": "0b75c598-0893-4216-afe8-5414cab7739d",        # KPMP kidney atlas (not in TS)
    "Pancreas": "ff45e623-7f5f-46e3-b47d-56be0341f66b",      # Tabula Sapiens
    "Skin": "0041b9c3-6a49-4bf7-8514-9bc7190067a7",          # Tabula Sapiens
    "Stomach": "9ba03780-4b13-44bc-a7d3-ce532ea0a856",       # Tabula Sapiens
    "Bone_Marrow": "4f1555bc-4664-46c3-a606-78d34dd10d92",   # Tabula Sapiens
}
PER_TYPE, CAP = 60, 2200


def strat_jids(obs, per, cap, seed=0):
    rng = np.random.default_rng(seed); keep = []
    for c, g in obs.groupby("cell_type"):
        j = g.soma_joinid.values
        keep.append(rng.choice(j, min(per, len(j)), replace=False))
    j = np.sort(np.concatenate(keep))
    return j if len(j) <= cap else np.sort(rng.choice(j, cap, replace=False))


parts = []
with cellxgene_census.open_soma(census_version="stable") as census:
    for organ, did in SOURCES.items():
        t = time.time()
        obs = cellxgene_census.get_obs(census, "homo_sapiens",
            value_filter=f"dataset_id == '{did}'", column_names=["soma_joinid", "cell_type"])
        jid = strat_jids(obs, PER_TYPE, CAP).tolist()
        a = cellxgene_census.get_anndata(census, "homo_sapiens", obs_coords=jid,
            column_names={"obs": ["cell_type", "cell_type_ontology_term_id"], "var": ["feature_id"]})
        a.var_names = a.var["feature_id"].astype(str).values
        a.obs["organ"] = organ
        parts.append(a)
        print(f"  {organ}: {a.n_obs} cells / {a.obs.cell_type.nunique()} types in {time.time()-t:.0f}s", flush=True)

comb = ad.concat(parts, join="inner"); comb.obs_names_make_unique()
comb.write_h5ad("/tmp/multitissue.h5ad")
print(f"MULTITISSUE_DONE {comb.shape} | {comb.obs.cell_type.nunique()} types | "
      f"{comb.obs.organ.nunique()} organs", flush=True)
