"""Shared preprocessing helpers used by multiple method adapters.

Each adapter is free to preprocess however its method expects; these helpers just
remove duplicated boilerplate (raw-count extraction, log-normalization, gene
alignment between a reference and a query).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def counts_adata(adata, use_raw="auto"):
    """Return an AnnData holding raw counts in ``X`` with upper-cased gene names.

    Mirrors actinn-jax's convention: CELLxGENE objects keep normalized values in
    ``.X`` and raw integer counts in ``.raw``; ``use_raw='auto'`` picks raw.
    """
    if use_raw == "auto":
        use_raw = adata.raw is not None
    if use_raw and adata.raw is not None:
        X, var_names = adata.raw.X, adata.raw.var_names
    else:
        X, var_names = adata.X, adata.var_names
    out = AnnData(
        X=sp.csr_matrix(X).astype(np.float32),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(var_names).str.upper()),
    )
    out.var_names_make_unique()
    return out


def lognorm(adata, target_sum=1e4):
    """In-place library-size normalize to ``target_sum`` then ``log1p``."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_hvg(adata, n_top_genes=2000):
    """Return the names of the top-N highly variable genes (Seurat flavor)."""
    a = adata.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_top_genes, a.n_vars))
    return list(a.var_names[a.var["highly_variable"].values])


def dense_aligned(adata, genes):
    """Dense ``(n_cells, len(genes))`` aligned to ``genes``; missing genes -> 0."""
    pos = pd.Index(adata.var_names).get_indexer(pd.Index(genes))
    out = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    present = pos >= 0
    if present.any():
        sub = adata[:, pos[present]].X
        out[:, present] = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return out
