"""Prepare the brain benchmark dataset (Allen human MTG) for the paper matrix.

Writes two subsampled h5ads (counts in .X, Ensembl var_names, Subclass + Cluster +
cell_type + cell_type_ontology_term_id in obs) to IanSSD:
  - brain_intra.h5ad         : ~300 cells per Allen subclass (24 types)
  - brain_cluster_intra.h5ad : ~100 cells per Allen cluster (151 types)
Both keep 10x 3' v3 nuclei only.

Two choices worth stating, since both narrow the source:

*Assay.* The source mixes 141,782 10x 3' v3 nuclei with 14,503 Smart-seq v4 ones.
A within-dataset split over both would put a full-length protocol on one side of a
comparison meant to measure annotation, not platform transfer, so only 10x is kept.

*Label.* Two levels, because they turn out to measure different things. ``Subclass``
(24 types) is the standard annotation level, and every method scores above 0.97 on it --
cortical subclasses are separable enough that the panel saturates. ``Cluster`` (151 types)
is the level the taxonomy's CCN cell sets enumerate, and it is where the difficulty in
cortex actually lives. The CELLxGENE ``cell_type`` column is coarser than either -- 18 Cell
Ontology terms, because CL has no way to say L4 IT rather than L2/3 IT -- so it is carried
along for reference but is not a label the methods are scored on.

Backed read + row-slice keeps peak memory low despite the 3.9 GB source.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc

SRC = "/Volumes/IanSSD/allen_mtg/human_mtg_great_apes.h5ad"
OUT = "/Volumes/IanSSD/allen_mtg"
# 100 per cluster keeps the fine split (~11k cells) the same size as lung_intra; the
# smallest of the 151 clusters holds 34 nuclei, so no class is left below a usable test fold.
LEVELS = {"brain_intra": ("Subclass", 300), "brain_cluster_intra": ("Cluster", 100)}
KEEP = ["Subclass", "Cluster", "Neighborhood", "cell_type",
        "cell_type_ontology_term_id", "donor_id", "assay", "sex"]
rng = np.random.default_rng(0)

a = sc.read_h5ad(SRC, backed="r")
obs = a.obs
print(f"source {a.shape}, {obs.Subclass.nunique()} subclasses, "
      f"{obs.cell_type.nunique()} CL terms, assays {obs.assay.value_counts().to_dict()}",
      flush=True)

tenx = (obs.assay.astype(str) == "10x 3' v3").to_numpy()

for name, (key, per) in LEVELS.items():
    lab = obs[key].astype(str).to_numpy()
    idx = []
    for c in np.unique(lab):
        pos = np.where(tenx & (lab == c))[0]
        idx.append(rng.choice(pos, min(per, len(pos)), replace=False))
    idx = np.sort(np.concatenate(idx))

    sub = a[idx].to_memory()
    sub.X = sub.raw.X.copy()      # CELLxGENE keeps raw counts in .raw
    sub.raw = None
    sub.obs = sub.obs[KEEP].copy()
    for c in ("Subclass", "Cluster"):
        sub.obs[c] = sub.obs[c].astype(str)
    path = f"{OUT}/{name}.h5ad"
    sub.write_h5ad(path)
    print(f"wrote {path}: {sub.shape}, {sub.obs[key].nunique()} {key} classes, "
          f"min class {sub.obs[key].value_counts().min()}, "
          f"X max {sub.X.max():.0f} (counts)", flush=True)
print("BRAIN_DATASET_DONE", flush=True)
