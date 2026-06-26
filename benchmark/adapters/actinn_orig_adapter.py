"""Adapter for the ORIGINAL TensorFlow ACTINN (the baseline actinn-jax replaced).

The original is vendored under ``benchmark/vendor/actinn_original`` and runs in its
own TensorFlow environment (point this method's ``python:`` at ``.venv-tf`` in the
config). TensorFlow and the vendored code are imported lazily inside the methods so
this adapter still registers cleanly in the JAX environment.

Note: the original `celltype_predict_actinn` couples training and prediction in a
single call (joint normalization over train+query, then train, then predict), so the
work is attributed to `predict`; `fit` only stages the reference. Compare methods by
total (fit + predict) time. The original also densifies via `adata.to_df()` — part of
what makes it slow and memory-heavy, which is the point of the comparison.
"""

import os
import tempfile

from ..prep import counts_adata
from .base import AnnotationMethod, Predictions, register


@register
class ActinnOriginal(AnnotationMethod):
    name = "actinn-orig"
    tier = "classical"
    device = "cpu"

    def fit(self, ref, label_key):
        # Original expects raw counts in .X and a path to the reference h5ad.
        self._label = label_key
        self._dir = tempfile.mkdtemp(prefix="actinn_orig_")
        self._ref_path = os.path.join(self._dir, "ref.h5ad")
        counts_adata(ref).write_h5ad(self._ref_path)

    def predict(self, query):
        from ..vendor.actinn_original import actinn_predict as orig
        q = counts_adata(query)  # raw counts in .X, matching gene-name format
        out, _ = orig.celltype_predict_actinn(
            q, self._ref_path, self._dir,
            train_label_name=self._label,
            output_label_name="pred",
            output_h5ad=False,
        )
        return Predictions(
            cell_ids=list(query.obs_names),
            labels=out.obs["pred"].to_numpy(),
        )
