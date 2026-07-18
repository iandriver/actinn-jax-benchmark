"""Adapter for ProtoCloud (Guo & Ding, Cell Genomics 2026).

A prototype-based interpretable VAE for cell-type annotation with built-in
uncertainty (cell-prototype similarity) and gene-level explainability. Runs in the
``.venv-protocloud`` environment (PyTorch + the vendored ``ProtoCloud`` package);
point the method's ``python:`` there in the config. ProtoCloud only uses CUDA-or-CPU
(no Apple MPS), so on a Mac it runs on CPU.

Vendored under ``benchmark/vendor/ProtoCloud`` (github.com/Ding-Group/ProtoCloud).
Its uncertainty flag (``pc_certainty == 'ambiguous'``) is surfaced as
``Predictions.unassigned`` so it can be compared to actinn-jax's abstain.
"""

import numpy as np

from ..prep import counts_adata
from .base import AnnotationMethod, Predictions, register


def _as_protocloud_adata(adata, labels=None, genes=None):
    """counts_adata with a ``counts`` layer, ``var['gene_name']``, and optional labels.

    ``genes`` restricts to a fixed gene panel (the reference HVGs); the API path of
    ProtoCloud does no gene selection itself, so we select HVGs here to match how the
    model is used in its paper (and the other deep adapters in this harness).
    """
    a = counts_adata(adata)
    if genes is not None:
        import pandas as pd
        pos = pd.Index(a.var_names).get_indexer(pd.Index(genes))
        import scipy.sparse as sp
        X = np.zeros((a.n_obs, len(genes)), dtype="float32")
        present = pos >= 0
        X[:, present] = a.X[:, pos[present]].toarray()
        import anndata as ad
        a = ad.AnnData(X=sp.csr_matrix(X), obs=a.obs.copy(),
                       var=pd.DataFrame(index=list(genes)))
    a.layers["counts"] = a.X.copy()
    a.var["gene_name"] = np.asarray(a.var_names, dtype=object)
    if labels is not None:
        a.obs["_label"] = np.asarray(labels).astype(str)
    return a


def _hvg_genes(adata, n_hvg):
    """Top-``n_hvg`` highly variable genes (seurat_v3 on raw counts)."""
    import scanpy as sc
    a = counts_adata(adata)
    a.layers["counts"] = a.X.copy()
    n = min(n_hvg, a.n_vars)
    sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat_v3", layer="counts")
    return list(a.var_names[a.var.highly_variable])


@register
class ProtoCloud(AnnotationMethod):
    name = "protocloud"
    tier = "deep"
    device = "cpu"                       # ProtoCloud uses cuda-or-cpu only (no MPS)

    def __init__(self, epochs=100, latent_dim=20, num_prototypes_per_class=6,
                 test_ratio=0.1, seed=7, n_hvg=2000):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.test_ratio = test_ratio
        self.seed = seed
        self.n_hvg = n_hvg

    def fit(self, ref, label_key):
        from ProtoCloud.api import ProtoCloudModel
        self._genes = _hvg_genes(ref, self.n_hvg)
        a = _as_protocloud_adata(ref, labels=ref.obs[label_key].to_numpy(),
                                 genes=self._genes)
        self.model = ProtoCloudModel(
            latent_dim=self.latent_dim,
            num_prototypes_per_class=self.num_prototypes_per_class,
        )
        self.model.fit_model(
            a, celltype_col="_label", count_layer="counts",
            epochs=self.epochs, test_ratio=self.test_ratio, seed=self.seed,
            validate=False,
        )

    def predict(self, query):
        a = _as_protocloud_adata(query, genes=self._genes)
        out = self.model.predict_model(a, count_layer="counts")
        labels = out.obs["pc_prediction"].to_numpy().astype(str)
        prob = (out.obs["pc_calibrated_certainty"].to_numpy(dtype=np.float32)
                if "pc_calibrated_certainty" in out.obs else None)
        unassigned = ((out.obs["pc_certainty"].to_numpy() == "ambiguous")
                      if "pc_certainty" in out.obs else None)
        return Predictions(
            cell_ids=list(query.obs_names),
            labels=labels,
            probabilities=prob,
            unassigned=unassigned,
        )
