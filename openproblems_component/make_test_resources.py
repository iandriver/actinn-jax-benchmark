"""Create a small OP-format train/test pair so components can be run standalone.

The Open Problems `label_projection` API hands a method an AnnData with raw counts in
``layers['counts']``, log-CP10k values in ``X``, the training labels in ``obs['label']``,
an HVG mask in ``var['hvg']``, and dataset/normalization ids in ``uns``. This builds a
tiny synthetic instance of exactly that shape so each component's ``VIASH START`` block
runs end-to-end locally -- a pre-flight check before paying for a cloud run.

    python openproblems_component/make_test_resources.py
    python openproblems_component/<name>/script.py
"""

import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

OUT = pathlib.Path("resources_test/task_label_projection/cxg_immune_cell_atlas")
N_TRAIN, N_TEST, N_GENES, N_TYPES = 400, 150, 600, 6


def build(n_cells, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_TYPES, n_cells)
    # Each type over-expresses its own contiguous block of marker genes.
    lam = np.full((n_cells, N_GENES), 0.4)
    block = N_GENES // N_TYPES
    for t in range(N_TYPES):
        lam[y == t, t * block:(t + 1) * block] += 6.0
    counts = sp.csr_matrix(rng.poisson(lam).astype(np.float32))

    a = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(
            {"label": [f"celltype_{i}" for i in y]},
            index=[f"cell_{seed}_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(N_GENES)]),
    )
    a.layers["counts"] = counts
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # HVG mask: the marker blocks plus a few extras, mimicking OP's 1000-HVG budget.
    hvg = np.zeros(N_GENES, dtype=bool)
    hvg[: block * N_TYPES] = True
    a.var["hvg"] = hvg
    a.uns["dataset_id"] = "cxg_immune_cell_atlas"
    a.uns["normalization_id"] = "log_cp10k"
    return a


OUT.mkdir(parents=True, exist_ok=True)
build(N_TRAIN, 0).write_h5ad(OUT / "train.h5ad")
build(N_TEST, 1).write_h5ad(OUT / "test.h5ad")
print(f"wrote {OUT}/train.h5ad and test.h5ad "
      f"({N_TRAIN}/{N_TEST} cells, {N_GENES} genes, {N_TYPES} types)")
