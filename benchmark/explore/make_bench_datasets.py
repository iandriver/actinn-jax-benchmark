"""Prepare benchmark dataset files that need pre-splitting (HLiCA liver).

Writes small, subsampled h5ads (counts in .X, Ensembl var_names, cell_type +
cell_type_ontology_term_id in obs) to IanSSD:
  - liver_intra.h5ad         : ~150 cells/type from all studies (driver does the split)
  - liver_ref_xstudy.h5ad    : non-Andrews_2022 studies (cross-study reference)
  - liver_query_xstudy.h5ad  : Andrews_2022 only (cross-study query)

Backed read + row-slice keeps peak memory low despite the 5.3 GB source.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc

SRC = "/Volumes/IanSSD/hlica/all_cells.h5ad"
OUT = "/Volumes/IanSSD/hlica"
PER_TYPE = 150
rng = np.random.default_rng(0)

a = sc.read_h5ad(SRC, backed="r")
obs = a.obs[["cell_type", "STUDY"]].copy()
obs["cell_type"] = obs["cell_type"].astype(str)
print(f"source {a.shape}, {obs.cell_type.nunique()} types, studies: {obs.STUDY.value_counts().to_dict()}", flush=True)


def strat(mask, per=PER_TYPE):
    idx = []
    sub = obs[mask]
    for c, g in sub.groupby("cell_type", observed=True):
        pos = g.index.map(lambda x: obs.index.get_loc(x)).to_numpy()
        idx.append(rng.choice(pos, min(per, len(pos)), replace=False))
    return np.sort(np.concatenate(idx))


def dump(mask, path, per=PER_TYPE):
    idx = strat(mask, per)
    sub = a[idx].to_memory()
    # put counts in .X (methods expect counts or normalize themselves); keep raw too
    if sub.raw is not None:
        sub.X = sub.raw.X.copy()
    sub.write_h5ad(path)
    print(f"wrote {path}: {sub.shape}, raw={'yes' if sub.raw is not None else 'no'}, "
          f"{sub.obs.cell_type.nunique()} types", flush=True)


allmask = np.ones(a.n_obs, dtype=bool)
dump(allmask, f"{OUT}/liver_intra.h5ad")
dump((obs.STUDY != "Andrews_2022").to_numpy(), f"{OUT}/liver_ref_xstudy.h5ad")
dump((obs.STUDY == "Andrews_2022").to_numpy(), f"{OUT}/liver_query_xstudy.h5ad")
print("BENCH_DATASETS_DONE", flush=True)
