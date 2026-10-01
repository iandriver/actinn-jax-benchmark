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

# The architecture the scArches surgery is designed around: layer norm in place of
# batch norm (batch statistics do not survive a change of dataset) and covariates fed
# to the encoder so the surgery has somewhere to attach the query's new batches.
ARCHES_ARCH = dict(use_layer_norm="both", use_batch_norm="none", encode_covariates=True,
                   dropout_rate=0.2, n_layers=2, n_latent=30)

# obs columns to try as the batch covariate, coarsest technical source first. An
# integrated atlas is integrated across datasets, not donors, and that is the batch key
# its own builders use; donor and sample come next for single-study references that have
# no dataset column. Preferring the coarse key also keeps the levels well populated: the
# HLCA subsampled to 300 cells per label has 14 datasets but 107 donors, which is about
# 134 cells per donor to learn a batch embedding from.
BATCH_CANDIDATES = ("batch", "dataset", "STUDY", "study", "donor_id", "sample", "assay")


def _accelerator():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _pick_batch_key(obs, candidates=BATCH_CANDIDATES, max_levels=200):
    """First usable batch column in ``obs``, or None when the reference is one batch."""
    for c in candidates:
        if c in obs.columns and 2 <= obs[c].astype(str).nunique() <= max_levels:
            return c
    return None


def _weighted_knn(ref_latent, ref_labels, query_latent, n_neighbors=15):
    """Label transfer by weighted kNN over the latent space.

    This is the transfer scArches itself ships (``scarches.utils.knn``) and the one the
    HLCA and ArchMap use: Gaussian-weighted votes from the k nearest reference cells,
    with the winning label's weight share returned as its probability. Returns
    ``(labels, probabilities)``; 1 - probability is ArchMap's Euclidean uncertainty.
    """
    from sklearn.neighbors import NearestNeighbors
    ref_labels = np.asarray(ref_labels).astype(str)
    k = min(n_neighbors, ref_latent.shape[0])
    dist, idx = NearestNeighbors(n_neighbors=k).fit(ref_latent).kneighbors(query_latent)
    stds = np.std(dist, axis=1)
    stds = (2.0 / np.maximum(stds, 1e-8)) ** 2
    tilda = np.exp(-dist / stds[:, None])
    denom = tilda.sum(axis=1, keepdims=True)
    # All-zero rows happen when every neighbour is far enough to underflow; fall back
    # to an unweighted vote rather than dividing by zero.
    tilda = np.where(denom > 0, tilda, 1.0)
    weights = tilda / tilda.sum(axis=1, keepdims=True)
    classes = np.unique(ref_labels)
    nbr = ref_labels[idx]
    probs = np.stack([(weights * (nbr == c)).sum(axis=1) for c in classes], axis=1)
    win = probs.argmax(axis=1)
    return classes[win], probs[np.arange(len(win)), win].astype(np.float32)


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
        soft = sca.predict(soft=True)
        pred = np.asarray(soft.idxmax(axis=1))
        conf = soft.to_numpy().max(axis=1).astype(np.float32)
        qmask = (comb.obs["_split"] == "query").to_numpy()
        return Predictions(cell_ids=list(query.obs_names), labels=pred[qmask],
                           probabilities=conf[qmask])


@register
class ScArches(AnnotationMethod):
    """scArches: train scANVI on the reference, then surgically map the query.

    Configured the way the method is used in practice rather than on scvi-tools
    defaults. Three choices matter and all three are measured in the ArchMap paper
    [Lotfollahi 2026]: the surgery architecture, a batch covariate for the surgery to
    extend, and weighted-kNN label transfer over the latent space. The last dominates:
    scANVI's own classifier head scores 0.18 macro-F1 on the HLCA against the
    weighted-kNN's 0.97 (their Extended Data Fig. 3a). ``label_transfer="native"``
    restores the head for comparison.
    """

    name = "scarches"
    tier = "deep"
    device = "mps"

    def __init__(self, scvi_epochs=200, scanvi_epochs=100, query_epochs=100,
                 n_hvg=2000, batch_key="auto", label_transfer="wknn", n_neighbors=15,
                 arch=None, early_stopping=True):
        self.scvi_epochs = scvi_epochs
        self.scanvi_epochs = scanvi_epochs
        self.query_epochs = query_epochs
        self.n_hvg = n_hvg
        self.batch_key = batch_key
        self.label_transfer = label_transfer
        self.n_neighbors = n_neighbors
        self.arch = ARCHES_ARCH if arch is None else arch
        self.early_stopping = early_stopping

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
        if self.batch_key is False:
            self._batch_key, self._use_batch = None, False
        else:
            key = (_pick_batch_key(r.obs) if self.batch_key == "auto"
                   else self.batch_key)
            self._batch_key = key if (key and key in r.obs.columns) else None
            self._use_batch = True
            # A single-batch reference still needs the covariate to exist, so that the
            # query arrives as a second batch and the surgery has something to learn.
            r.obs["_batch"] = (r.obs[self._batch_key].astype(str) if self._batch_key
                               else "reference")
        acc = _accelerator()
        scvi.model.SCVI.setup_anndata(r, labels_key="_label", layer="counts",
                                      **({"batch_key": "_batch"} if self._use_batch
                                         else {}))
        m = scvi.model.SCVI(r, **self.arch)
        m.train(max_epochs=self.scvi_epochs, accelerator=acc,
                early_stopping=self.early_stopping, enable_progress_bar=False)
        self._model = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category=UNKNOWN)
        self._model.train(max_epochs=self.scanvi_epochs, accelerator=acc,
                          early_stopping=self.early_stopping,
                          enable_progress_bar=False)
        self._ref_latent = self._model.get_latent_representation()
        self._ref_labels = r.obs["_label"].to_numpy()

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
        # The query's batches are unseen by construction; that is what the surgery is
        # for. Prefixing keeps a shared level name (e.g. a donor in both) from being
        # silently treated as a batch the reference model already knows.
        if self._use_batch:
            qa.obs["_batch"] = ("query:" + qa.obs[self._batch_key].astype(str)
                                if self._batch_key and self._batch_key in qa.obs.columns
                                else "query")
        qa.layers["counts"] = qa.X.copy()
        scvi.model.SCANVI.prepare_query_anndata(qa, self._model)
        qm = scvi.model.SCANVI.load_query_data(qa, self._model)
        # ArchMap fixes query fine-tuning at 100 epochs; no early stopping, so the
        # budget is the same for every query.
        qm.train(max_epochs=self.query_epochs, accelerator=_accelerator(),
                 enable_progress_bar=False, plan_kwargs={"weight_decay": 0.0})
        if self.label_transfer == "native":
            soft = qm.predict(soft=True)
            labels = np.asarray(soft.idxmax(axis=1))
            probs = soft.to_numpy().max(axis=1).astype(np.float32)
        else:
            labels, probs = _weighted_knn(self._ref_latent, self._ref_labels,
                                          qm.get_latent_representation(qa),
                                          self.n_neighbors)
        return Predictions(cell_ids=list(query.obs_names), labels=labels,
                           probabilities=probs)


@register
class ScArchesNative(ScArches):
    """scArches on scvi-tools defaults with scANVI's own classifier head.

    The configuration the first version of this benchmark used, kept so the effect of
    the corrections in :class:`ScArches` is measurable in a single run.
    """

    name = "scarches-native"

    def __init__(self, scvi_epochs=40, scanvi_epochs=20, query_epochs=40, n_hvg=2000):
        super().__init__(scvi_epochs=scvi_epochs, scanvi_epochs=scanvi_epochs,
                         query_epochs=query_epochs, n_hvg=n_hvg, batch_key=False,
                         label_transfer="native", arch={}, early_stopping=False)
