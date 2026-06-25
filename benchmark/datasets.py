"""Dataset loading, synthetic data, and reference/query split construction."""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def load(path):
    return sc.read_h5ad(path)


def make_synthetic(n_per_type=300, n_genes=1200, n_types=6, seed=0):
    """Linearly separable synthetic raw-count AnnData (for smoke tests)."""
    rng = np.random.default_rng(seed)
    block = n_genes // (n_types + 1)
    rows, labels = [], []
    for t in range(n_types):
        base = rng.poisson(0.2, size=(n_per_type, n_genes)).astype(np.float32)
        base[:, t * block:(t + 1) * block] += rng.poisson(
            8.0, size=(n_per_type, block)
        ).astype(np.float32)
        rows.append(base)
        labels += [f"type_{t}"] * n_per_type
    X = np.vstack(rows)
    perm = rng.permutation(X.shape[0])
    X, labels = X[perm], list(np.array(labels)[perm])
    return AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"cell_type": labels},
                         index=[f"cell{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )


def stratified_subsample(labels, max_per_label, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_label:
            idx = rng.choice(idx, max_per_label, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def intra_split(adata, label_key, test_frac=0.25, seed=0):
    """Stratified train/test split within one dataset -> (ref, query)."""
    labels = np.asarray(adata.obs[label_key].values)
    rng = np.random.default_rng(seed)
    test = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        test[rng.choice(idx, max(1, int(len(idx) * test_frac)), replace=False)] = True
    return adata[~test].copy(), adata[test].copy()
