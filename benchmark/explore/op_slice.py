"""Pre-slice the Open Problems label_projection h5ads to their 1000 task HVGs.

Every ported component's first act is `adata[:, adata.var['hvg']]`, so slicing up front is
the same input -- it just avoids holding a 19 GB atlas in memory to throw 96% of it away.

The files carry three matrices (X, layers/counts, layers/normalized) and the components read
only `layers['counts']`, so this reads that one element instead of the whole object: ~3x less
memory before the subset even happens.

    python op_slice.py <dataset>
"""
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

SRC = "/Volumes/IanSSD/op_label_projection"
OUT = "/Volumes/IanSSD/op_label_projection_hvg"


def slim(path, out_path):
    with h5py.File(path, "r") as f:
        var = read_elem(f["var"])
        hvg = var["hvg"].values.astype(bool) if "hvg" in var else np.ones(len(var), bool)
        counts = read_elem(f["layers/counts"])[:, hvg]
        obs = read_elem(f["obs"])
        uns = read_elem(f["uns"]) if "uns" in f else {}
    a = ad.AnnData(X=counts.copy(), obs=obs, var=var.loc[hvg].copy(), uns=uns)
    a.layers["counts"] = a.X.copy()
    # `hvg` must stay True so the component's own subset is a no-op rather than empty.
    a.var["hvg"] = True
    a.write_h5ad(out_path)
    print(f"  {os.path.basename(path)}: {counts.shape} -> {out_path}", flush=True)
    return a.shape


def main(ds):
    os.makedirs(f"{OUT}/{ds}", exist_ok=True)
    for part in ("train", "test"):
        src, dst = f"{SRC}/{ds}/{part}.h5ad", f"{OUT}/{ds}/{part}.h5ad"
        if os.path.exists(dst):
            print(f"  {part}: already sliced", flush=True)
            continue
        slim(src, dst)
    # solution carries only labels; copy the two columns we score against
    dst = f"{OUT}/{ds}/solution.parquet"
    if not os.path.exists(dst):
        with h5py.File(f"{SRC}/{ds}/solution.h5ad", "r") as f:
            obs = read_elem(f["obs"])
        pd.DataFrame({"label": obs["label"].astype(str).values},
                     index=obs.index.astype(str)).to_parquet(dst)
        print(f"  solution: {len(obs)} labels", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
