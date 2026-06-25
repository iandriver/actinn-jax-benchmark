"""Adapter for a linear SVM classifier (a strong, repeatedly top-ranked baseline).

Uses an SGD-trained linear SVM (hinge loss). liblinear's `LinearSVC` is the classic
"SVM" in annotation benchmarks but its TRON/coordinate-descent solver does not scale
to atlas-sized multiclass problems (tens of thousands of cells x thousands of genes x
dozens of classes), running many minutes per fit. SGD-hinge is the same linear-SVM
decision rule trained online, giving comparable accuracy at a fraction of the time —
the appropriate choice for a runtime-focused benchmark.

Optionally rejects low-confidence calls via a decision-margin threshold, mirroring the
rejection option used in benchmarks like Abdelaal et al. 2019.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ..prep import counts_adata, dense_aligned, lognorm, select_hvg
from .base import AnnotationMethod, Predictions, register


@register
class SVMMethod(AnnotationMethod):
    name = "svm"
    tier = "classical"
    device = "cpu"

    def __init__(self, n_hvg=2000, reject_margin=None):
        self.n_hvg = n_hvg
        self.reject_margin = reject_margin  # None disables rejection

    def fit(self, ref, label_key):
        a = lognorm(counts_adata(ref))
        self.genes = select_hvg(a, self.n_hvg)
        X = dense_aligned(a, self.genes)
        self.scaler = StandardScaler().fit(X)
        # Linear SVM (hinge loss) trained by SGD — scales to atlas-sized data.
        self.clf = SGDClassifier(loss="hinge").fit(
            self.scaler.transform(X), ref.obs[label_key].to_numpy()
        )

    def predict(self, query):
        a = lognorm(counts_adata(query))
        X = self.scaler.transform(dense_aligned(a, self.genes))
        scores = self.clf.decision_function(X)
        idx = np.argmax(scores, axis=1) if scores.ndim > 1 else (scores > 0).astype(int)
        labels = self.clf.classes_[idx]
        unassigned = None
        if self.reject_margin is not None and scores.ndim > 1:
            margin = np.max(scores, axis=1)
            unassigned = margin < self.reject_margin
        return Predictions(
            cell_ids=list(query.obs_names), labels=labels, unassigned=unassigned
        )
