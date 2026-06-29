"""Fetch a diverse multi-organ Tabula Sapiens slice from CZ CELLxGENE census.

Pulls several organs CONTIGUOUSLY by dataset_id (fast — avoids the scattered-read
slowness), stratified-subsamples each organ by cell_type, and concatenates into one
multi-organ reference with raw counts + CL ids + ensembl genes. Run in .venv-scprint.
"""
import warnings, time; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np, anndata as ad

ORGANS = {  # 8 diverse organs (small-to-medium, mostly non-overlapping cell types)
    "Pancreas": "ff45e623-7f5f-46e3-b47d-56be0341f66b",
    "Skin": "0041b9c3-6a49-4bf7-8514-9bc7190067a7",
    "Liver": "6d41668c-168c-4500-b06a-4674ccf3e19d",
    "Trachea": "d8732da6-8d1d-42d9-b625-f2416c30054b",
    "Heart": "e6a11140-2545-46bc-929e-da243eed2cae",
    "Bone_Marrow": "4f1555bc-4664-46c3-a606-78d34dd10d92",
    "Stomach": "9ba03780-4b13-44bc-a7d3-ce532ea0a856",
    "Eye": "a0754256-f44b-4c4a-962c-a552e47d3fdc",
}
PER_TYPE, PER_ORGAN = 80, 2500


def strat(labels, per, cap, seed=0):
    rng = np.random.default_rng(seed); keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep += list(idx if len(idx) <= per else rng.choice(idx, per, replace=False))
    keep = np.array(sorted(keep))
    return keep if len(keep) <= cap else np.sort(rng.choice(keep, cap, replace=False))


parts = []
with cellxgene_census.open_soma(census_version="stable") as census:
    for organ, did in ORGANS.items():
        t = time.time()
        a = cellxgene_census.get_anndata(
            census, "homo_sapiens", obs_value_filter=f"dataset_id == '{did}'",
            column_names={"obs": ["cell_type", "cell_type_ontology_term_id", "tissue"],
                          "var": ["feature_id", "feature_name"]})
        a.var_names = a.var["feature_id"].astype(str).values
        sel = strat(a.obs["cell_type"].astype(str).to_numpy(), PER_TYPE, PER_ORGAN)
        a = a[sel].copy(); a.obs["organ"] = organ
        parts.append(a)
        print(f"  {organ}: {a.n_obs} cells / {a.obs.cell_type.nunique()} types "
              f"(of {len(sel)}) in {time.time()-t:.0f}s", flush=True)

comb = ad.concat(parts, join="inner")
comb.obs_names_make_unique()
comb.write_h5ad("/tmp/ts_sapiens.h5ad")
print(f"TS_FETCH_DONE {comb.shape} | {comb.obs.cell_type.nunique()} types | "
      f"{comb.obs.organ.nunique()} organs", flush=True)
