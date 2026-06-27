# Pull a curated, ontology-labeled PBMC/blood reference from CZ CELLxGENE census
# (the same underlying data scPRINT/lamindb use) -> raw counts + cell_type labels.
import warnings; warnings.filterwarnings("ignore")
import cellxgene_census, numpy as np
ORG = "homo_sapiens"
with cellxgene_census.open_soma(census_version="stable") as census:
    obs = cellxgene_census.get_obs(census, ORG,
        value_filter="tissue_general == 'blood' and is_primary_data == True and disease == 'normal'",
        column_names=["soma_joinid", "cell_type"])
    print("blood primary-normal cells:", len(obs), "| types:", obs.cell_type.nunique(), flush=True)
    vc = obs.cell_type.value_counts()
    keep = vc[vc >= 800].index
    obs = obs[obs.cell_type.isin(keep)]
    rng = np.random.default_rng(0); idx = []
    for ct, g in obs.groupby("cell_type"):
        j = g.soma_joinid.values
        idx.append(rng.choice(j, min(120, len(j)), replace=False))
    joinids = np.sort(np.concatenate(idx)).tolist()
    print("sampling", len(joinids), "cells across", len(keep), "types", flush=True)
    adata = cellxgene_census.get_anndata(census, ORG, obs_coords=joinids,
        column_names={"obs": ["cell_type", "cell_type_ontology_term_id", "assay", "dataset_id"]})
adata.var_names_make_unique()
adata.write_h5ad("/tmp/census_blood.h5ad")
print("CENSUS_SAVED", adata.shape, "| genes(ensembl):", list(adata.var_names[:2]), flush=True)
