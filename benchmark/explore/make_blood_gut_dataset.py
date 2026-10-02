"""Prepare the blood+gut benchmark split from the IBDverse multi-tissue atlas.

Writes blood_gut_intra.h5ad: 120 cells per label from the 1.85M-cell source, counts in
.X, Ensembl var_names, label in cell_type. The driver does the train/test split.

Two things about this source are worth stating, because neither is obvious:

The label column is `leiden`, despite the name. It holds 86 cell-type names, not cluster
numbers. `Cell_type` is the same partition with one class left null: the 6,829 cells
`leiden` calls Platelet+RBC. Taking `Cell_type` and casting to string would therefore
produce an 86th class literally named "nan", so `leiden` is the column to use.

The cap of 120 is not arbitrary. It is the value that reproduces the paper matrix's split
exactly, 10,255 cells over 86 labels, splitting 7,692 / 2,563 at the driver's test_frac of
0.25. No build script for this split survived, so the cap was recovered by matching those
counts; 119 gives 10,171 and 125 gives 10,675. The cells drawn are not the paper's, since
the original seed is unknown, but the shape of the split is.

The source carries no Cell Ontology ids, which is why this split is scored on accuracy and
macro-F1 only.

    python benchmark/explore/make_blood_gut_dataset.py
"""
import os
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc

# Source: IBDverse multi-tissue atlas (Alegbe et al. 2026, doi 10.1038/s41586-026-10627-z),
# distributed via ibdverse.info. 15.13 GB, 1,852,681 x 33,538, Ensembl var_names, integer
# counts in .X, no raw and no Cell Ontology ids. There is no scripted fetch: download it
# from the portal, which the project owner reports takes about 20 minutes. The file is not
# kept on disk between runs, since the derived split is 0.30 GB against its 15.13 GB.
SRC = os.environ.get("IBDVERSE_SRC",
                     os.path.expanduser("~/Downloads/multi_tissue-atlas.h5ad"))
OUT = os.environ.get("IBDVERSE_OUT", "/Volumes/IanSSD/hlica/blood_gut_intra.h5ad")
LABEL_COL = "leiden"
PER_TYPE = 120
rng = np.random.default_rng(0)

a = sc.read_h5ad(SRC, backed="r")
obs = a.obs[[LABEL_COL]].copy()
obs[LABEL_COL] = obs[LABEL_COL].astype(str)
print(f"source {a.shape}, {obs[LABEL_COL].nunique()} labels", flush=True)

idx = []
for c, g in obs.groupby(LABEL_COL, observed=True):
    pos = np.flatnonzero((obs[LABEL_COL] == c).to_numpy())
    idx.append(rng.choice(pos, min(PER_TYPE, len(pos)), replace=False))
idx = np.sort(np.concatenate(idx))

sub = a[idx].to_memory()
sub.obs["cell_type"] = sub.obs[LABEL_COL].astype(str)
sub.write_h5ad(OUT)
print(f"wrote {OUT}: {sub.shape}, {sub.obs.cell_type.nunique()} types", flush=True)
print("BLOOD_GUT_DATASET_DONE", flush=True)
