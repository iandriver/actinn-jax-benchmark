"""Deep reference-mapping adapters built on scvi-tools (scANVI, scArches).

These run in the ``.venv-scvi`` environment (PyTorch + scvi-tools); point the method's
``python:`` there in the config. scvi is imported lazily so the registry still loads
in the core JAX env. Apple MPS is used when available (set ``device`` via the adapter).

scANVI couples reference + query (semi-supervised), so the work is attributed to
``predict``; ``fit`` only stashes the reference. Compare by total (fit + predict) time.
"""

import numpy as np

from ..prep import counts_adata
from .base import AnnotationMethod, Predictions, register

UNKNOWN = "Unknown"


def _accelerator():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _prep_combined(ref, ref_labels, query, n_hvg=2000):
    """Concatenate ref (labeled) + query (Unknown) on shared genes, pick HVGs."""
    import anndata as ad
    import scanpy as sc
    r = counts_adata(ref)
    q = counts_adata(query)
    r.obs["_label"] = np.asarray(ref_labels).astype(str)
    q.obs["_label"] = UNKNOWN
    comb = ad.concat([r, q], join="inner", label="_split", keys=["ref", "query"])
    comb.layers["counts"] = comb.X.copy()
    sc.pp.highly_variable_genes(comb, n_top_genes=min(n_hvg, comb.n_vars),
                                flavor="seurat_v3", layer="counts")
    return comb[:, comb.var.highly_variable].copy()


@register
class ScANVI(AnnotationMethod):
    name = "scanvi"
    tier = "deep"
    device = "mps"

    def __init__(self, scvi_epochs=40, scanvi_epochs=20, n_hvg=2000):
        self.scvi_epochs = scvi_epochs
        self.scanvi_epochs = scanvi_epochs
        self.n_hvg = n_hvg

    def fit(self, ref, label_key):
        # scANVI is semi-supervised over ref+query together; defer to predict.
        self._ref = ref
        self._labels = ref.obs[label_key].to_numpy()

    def predict(self, query):
        import scvi
        comb = _prep_combined(self._ref, self._labels, query, self.n_hvg)
        acc = _accelerator()
        scvi.model.SCVI.setup_anndata(comb, labels_key="_label", layer="counts")
        m = scvi.model.SCVI(comb)
        m.train(max_epochs=self.scvi_epochs, accelerator=acc, enable_progress_bar=False)
        sca = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category=UNKNOWN)
        sca.train(max_epochs=self.scanvi_epochs, accelerator=acc,
                  enable_progress_bar=False)
        pred = np.asarray(sca.predict())
        qmask = (comb.obs["_split"] == "query").to_numpy()
        return Predictions(cell_ids=list(query.obs_names), labels=pred[qmask])


@register
class ScArches(AnnotationMethod):
    """scArches: train scANVI on the reference, then surgically map the query."""

    name = "scarches"
    tier = "deep"
    device = "mps"

    def __init__(self, scvi_epochs=40, scanvi_epochs=20, query_epochs=40, n_hvg=2000):
        self.scvi_epochs = scvi_epochs
        self.scanvi_epochs = scanvi_epochs
        self.query_epochs = query_epochs
        self.n_hvg = n_hvg

    def fit(self, ref, label_key):
        import scanpy as sc
        import scvi
        r = counts_adata(ref)
        r.obs["_label"] = ref.obs[label_key].astype(str).to_numpy()
        r.layers["counts"] = r.X.copy()
        sc.pp.highly_variable_genes(r, n_top_genes=min(self.n_hvg, r.n_vars),
                                    flavor="seurat_v3", layer="counts")
        r = r[:, r.var.highly_variable].copy()
        self._genes = list(r.var_names)
        acc = _accelerator()
        scvi.model.SCVI.setup_anndata(r, labels_key="_label", layer="counts")
        m = scvi.model.SCVI(r)
        m.train(max_epochs=self.scvi_epochs, accelerator=acc, enable_progress_bar=False)
        self._model = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category=UNKNOWN)
        self._model.train(max_epochs=self.scanvi_epochs, accelerator=acc,
                          enable_progress_bar=False)

    def predict(self, query):
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp
        import scvi
        q = counts_adata(query)
        # Align query onto the reference HVG space (missing genes -> 0).
        pos = pd.Index(q.var_names).get_indexer(pd.Index(self._genes))
        out = np.zeros((q.n_obs, len(self._genes)), dtype="float32")
        present = pos >= 0
        out[:, present] = q.X[:, pos[present]].toarray()
        qa = ad.AnnData(X=sp.csr_matrix(out), obs=q.obs.copy(),
                        var=pd.DataFrame(index=self._genes))
        qa.obs["_label"] = UNKNOWN
        qa.layers["counts"] = qa.X.copy()
        scvi.model.SCANVI.prepare_query_anndata(qa, self._model)
        qm = scvi.model.SCANVI.load_query_data(qa, self._model)
        qm.train(max_epochs=self.query_epochs, accelerator=_accelerator(),
                 enable_progress_bar=False, plan_kwargs={"weight_decay": 0.0})
        return Predictions(cell_ids=list(query.obs_names), labels=np.asarray(qm.predict()))
