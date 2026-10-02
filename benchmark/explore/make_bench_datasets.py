"""Prepare benchmark dataset files that need pre-splitting (HLiCA liver).

Writes small, subsampled h5ads (counts in .X, Ensembl var_names, cell_type +
cell_type_ontology_term_id in obs) to IanSSD:
  - liver_intra.h5ad         : ~150 cells/type from all studies (driver does the split)
  - liver_ref_xstudy.h5ad    : non-Andrews_2022 studies (cross-study reference)
  - liver_query_xstudy.h5ad  : Andrews_2022 only (cross-study query)

Backed read + row-slice keeps peak memory low despite the 5.3 GB source.

The source is worth naming explicitly, because the splits it produces have changed. The
paper matrix (run 2026-08-16) recorded 5400/5400/3396 cells. This script, unchanged since
its only commit, now produces 5700/5700/3696 from either published HLiCA version
(2026-07-20 and 2026-09-25), both of which carry 38 cell types. What the historical run
read is UNRESOLVED. The counts are consistent with a 36-type sample, and so is the
"36-type" description in configs/scaling_memory_hlica.yaml, but neither establishes what
the raw file contained, which labels were excluded, or which cells were drawn: the
historical input, its hashes and its cell ids are not on this machine, and the original
panel saved no per-cell predictions to recover them from. Treat it as unresolved sample
and source provenance, and check the counts against the run you are comparing to. Override SRC and SUFFIX to build
from a second source alongside the current one rather than over it:

    HLICA_SRC=.../other.h5ad HLICA_SUFFIX=_alt python make_bench_datasets.py
"""
import os
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc

# Redownload the source with:
#   curl -fL -o /Volumes/IanSSD/hlica/all_cells.h5ad \
#     https://datasets.cellxgene.cziscience.com/537e75ec-90e5-479d-84e9-d3f8793a3414.h5ad
# That is HLiCA all-cells, CELLxGENE collection 059202e1-1f1b-483f-9151-f3a25a380c39,
# dataset ef3055e1-0cad-40a3-961d-7110a323dc4f, version of 2026-09-25, 5.73 GB. Keep it:
# the derived splits are enough to rerun a method, but not to rebuild the panel.
SRC = os.environ.get("HLICA_SRC", "/Volumes/IanSSD/hlica/all_cells.h5ad")
OUT = os.environ.get("HLICA_OUT", "/Volumes/IanSSD/hlica")
SUFFIX = os.environ.get("HLICA_SUFFIX", "")
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
dump(allmask, f"{OUT}/liver_intra{SUFFIX}.h5ad")
dump((obs.STUDY != "Andrews_2022").to_numpy(), f"{OUT}/liver_ref_xstudy{SUFFIX}.h5ad")
dump((obs.STUDY == "Andrews_2022").to_numpy(), f"{OUT}/liver_query_xstudy{SUFFIX}.h5ad")
print("BENCH_DATASETS_DONE", flush=True)
