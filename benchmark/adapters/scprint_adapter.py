"""Adapter for the scPRINT foundation model (zero-shot cell-type annotation).

Runs in the ``.venv-scprint`` environment (point the method's ``python:`` there).
scPRINT is **zero-shot**: it ignores the reference and predicts Cell-Ontology ids
from its pretrained classifier head, so ``fit`` only loads the checkpoint and
``predict`` runs inference. Predictions are returned as ``label_cl`` (CL ids) and the
driver scores them in CL-id space (ontology concordance), since they don't map to the
dataset's label vocabulary.

Requirements (see docs/SCPRINT_AND_CPU_CLASSIFIER.md):
- lamindb bionty populated for human + mouse (``populate_my_ontology``).
- checkpoint ``medium-v1.5.ckpt`` at the repo root (set ``SCPRINT_CKPT`` to override).
- Apple MPS works only with an autocast bypass (applied here); CPU is ~30x slower.
"""

import os

import numpy as np

from .base import AnnotationMethod, Predictions, register

_CKPT = os.environ.get("SCPRINT_CKPT", "medium-v1.5.ckpt")


def _patch_autocast():
    """Make torch.autocast a no-op on MPS (unsupported by PyTorch); fp32 instead."""
    import contextlib
    import torch
    if getattr(torch.autocast, "_scprint_patched", False):
        return
    orig = torch.autocast

    class _AC:
        _scprint_patched = True

        def __init__(self, device_type, **kw):
            self._cm = (orig(device_type, **kw)
                        if device_type in ("cuda", "cpu", "xpu")
                        else contextlib.nullcontext())

        def __enter__(self):
            return self._cm.__enter__()

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

    torch.autocast = _AC


def _to_scprint_input(adata):
    """Raw counts in X, Ensembl var_names, human organism — what scPRINT expects."""
    import anndata as ad
    import scipy.sparse as sp
    if adata.raw is not None:
        X, var = adata.raw.X, adata.raw.var.copy()
    else:
        X, var = adata.X, adata.var.copy()
    a = ad.AnnData(X=sp.csr_matrix(X).astype("float32"), obs=adata.obs.copy(), var=var)
    if "organism_ontology_term_id" not in a.obs:
        a.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    return a


@register
class ScPrint(AnnotationMethod):
    name = "scprint"
    tier = "foundation"
    device = "mps"

    def __init__(self, ckpt=_CKPT, device="mps", max_len=2000, batch_size=32):
        self.ckpt = ckpt
        self._dev = device
        self.max_len = max_len
        self.batch_size = batch_size

    def fit(self, ref, label_key):
        # Zero-shot: the reference is ignored; just load the pretrained model.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        _patch_autocast()
        import torch
        from scprint import scPrint
        model = scPrint.load_from_checkpoint(self.ckpt, precpt_gene_emb=None)
        dev = self._dev
        if dev == "mps" and not torch.backends.mps.is_available():
            dev = "cpu"
        self.device = dev
        self._model = model.to(dev)

    def predict(self, query):
        import torch
        from scdataloader import Preprocessor
        from scprint.tasks import Embedder
        try:
            from scdataloader.preprocess import additional_preprocess
        except Exception:
            additional_preprocess = None

        a = _to_scprint_input(query)
        pp = Preprocessor(
            do_postp=False, force_preprocess=True,
            **({"additional_preprocess": additional_preprocess} if additional_preprocess else {}),
        )
        a = pp(a)
        emb = Embedder(doclass=True, precision="32", dtype=torch.float32,
                       batch_size=self.batch_size, num_workers=0, doplot=False,
                       max_len=self.max_len)
        res = emb(self._model, a)
        out = res[0] if isinstance(res, (tuple, list)) else res

        # The Preprocessor replaces obs_names (new UUIDs) but preserves cell count and
        # order, so align positionally. If cells were dropped, pad the tail as unknown.
        col = out.obs["pred_cell_type_ontology_term_id"].astype(str).to_numpy()
        cl = np.full(len(query.obs_names), "unknown", dtype=object)
        cl[:len(col)] = col[:len(cl)]
        return Predictions(cell_ids=list(query.obs_names), labels=cl, label_cl=cl)
