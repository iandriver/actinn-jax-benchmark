"""Adapter for actinn-jax (this project's method under test)."""

import numpy as np

from .base import AnnotationMethod, Predictions, register


@register
class ActinnJax(AnnotationMethod):
    name = "actinn-jax"
    tier = "classical"
    device = "cpu"

    def fit(self, ref, label_key):
        import actinn_jax as aj
        self.model = aj.train_reference(
            ref, train_label_name=label_key, print_cost=False
        )

    def predict(self, query):
        frame, _ = self.model.predict_frame(query)
        return Predictions(
            cell_ids=list(query.obs_names),
            labels=frame["celltype"].to_numpy(),
            probabilities=frame["celltype_probability"].to_numpy(dtype=np.float32),
        )
