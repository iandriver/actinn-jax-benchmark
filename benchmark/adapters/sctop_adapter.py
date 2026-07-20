"""Adapter for scTOP (Yampolskaya, Souza & Mehta) — a parameter-free baseline.

scTOP represents each cell by a rank-based z-scored expression vector and classifies by
projecting it onto a per-cell-type *basis* built the same way ("order parameters"); the
predicted type is the largest projection. There is no training and no fitted parameters
beyond the reference class averages, which makes it the natural fast/simple counterpart
to actinn-jax — and Souza & Mehta report it *beating* single-cell foundation models on
out-of-distribution transfer, so it is the most important baseline to have here.

Installed with ``uv pip install sctop`` (pure numpy/pandas; CPU).
The reported "probability" is scTOP's top order parameter — a similarity score, not a
calibrated probability.
"""

import numpy as np
import pandas as pd

from ..prep import counts_adata, dense_aligned
from .base import AnnotationMethod, Predictions, register


def _genes_by_cells(X, genes):
    """Dense ``(n_genes, n_cells)`` DataFrame — the orientation scTOP expects."""
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return pd.DataFrame(np.asarray(arr, dtype=np.float32).T, index=list(genes))


@register
class ScTOP(AnnotationMethod):
    name = "sctop"
    tier = "classical"          # parameter-free projection
    device = "cpu"

    def __init__(self, min_expr_frac=0.10, chunk_size=2000):
        # scTOP is a rank-based method, so it is very sensitive to the long tail of
        # near-all-zero genes: on raw pbmc3k (33k genes, 97% zeros) the ranks are
        # dominated by ties at zero, the per-type bases go degenerate and everything
        # collapses onto the rarest type (acc 0.18). Filtering to genes expressed in a
        # fraction of reference cells restores it (0.91). A *fraction* is used rather
        # than a cell count so the default is scale-free across datasets rather than
        # tuned per dataset.
        self.min_expr_frac = min_expr_frac
        self.chunk_size = chunk_size

    def fit(self, ref, label_key):
        """Build the basis: one averaged, processed profile per cell type."""
        from sctop.processing import process

        a = counts_adata(ref)
        y = ref.obs[label_key].astype(str).to_numpy()
        expr_frac = np.asarray((a.X > 0).sum(axis=0)).ravel() / max(1, a.n_obs)
        keep = expr_frac >= self.min_expr_frac
        if keep.sum() < 50:                          # degenerate filter -> keep all
            keep = np.ones(a.n_vars, dtype=bool)
        self.genes = list(np.asarray(a.var_names)[keep])
        cols = {}
        for t in pd.unique(y):
            sub = a.X[y == t][:, keep]              # densified per type -> bounded
            cols[str(t)] = process(_genes_by_cells(sub, self.genes),
                                   average=True).iloc[:, 0]
        self.basis = pd.DataFrame(cols)

    def predict(self, query):
        from sctop.processing import process, score

        a = counts_adata(query)
        X = dense_aligned(a, self.genes)            # (n_cells, n_genes), missing -> 0
        sample = process(_genes_by_cells(X, self.genes), chunk_size=self.chunk_size)
        proj = score(self.basis, sample, chunk_size=self.chunk_size)
        return Predictions(
            cell_ids=list(query.obs_names),
            labels=proj.idxmax(axis=0).to_numpy().astype(str),
            probabilities=proj.max(axis=0).to_numpy(dtype=np.float32),
        )
