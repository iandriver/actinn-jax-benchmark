"""Adapter for CellTypist (logistic-regression annotation)."""

import numpy as np

from ..prep import counts_adata, lognorm
from .base import AnnotationMethod, Predictions, register


@register
class CellTypistMethod(AnnotationMethod):
    name = "celltypist"
    tier = "classical"
    device = "cpu"

    def _prep(self, adata):
        # CellTypist expects log1p of CP10k-normalized counts, genes in var_names.
        return lognorm(counts_adata(adata))

    def fit(self, ref, label_key):
        import celltypist
        a = self._prep(ref)
        self.model = celltypist.train(
            a, labels=label_key, feature_selection=True, check_expression=False
        )

    def predict(self, query):
        import celltypist
        a = self._prep(query)
        res = celltypist.annotate(a, model=self.model, majority_voting=False)
        pred = res.predicted_labels["predicted_labels"].to_numpy()
        proba = None
        if hasattr(res, "probability_matrix") and res.probability_matrix is not None:
            proba = res.probability_matrix.to_numpy().max(axis=1).astype(np.float32)
        return Predictions(
            cell_ids=list(query.obs_names), labels=pred, probabilities=proba
        )
