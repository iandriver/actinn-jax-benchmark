"""Adapter for a kNN-on-PCA classifier (a simple lower-bound baseline)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ..prep import counts_adata, dense_aligned, lognorm, select_hvg
from .base import AnnotationMethod, Predictions, register


@register
class KNNMethod(AnnotationMethod):
    name = "knn"
    tier = "classical"
    device = "cpu"

    def __init__(self, n_hvg=2000, n_pcs=50, k=30):
        self.n_hvg = n_hvg
        self.n_pcs = n_pcs
        self.k = k

    def fit(self, ref, label_key):
        a = lognorm(counts_adata(ref))
        self.genes = select_hvg(a, self.n_hvg)
        X = dense_aligned(a, self.genes)
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=min(self.n_pcs, X.shape[1])).fit(
            self.scaler.transform(X)
        )
        Z = self.pca.transform(self.scaler.transform(X))
        self.clf = KNeighborsClassifier(n_neighbors=self.k).fit(
            Z, ref.obs[label_key].to_numpy()
        )

    def predict(self, query):
        a = lognorm(counts_adata(query))
        X = self.scaler.transform(dense_aligned(a, self.genes))
        Z = self.pca.transform(X)
        proba = self.clf.predict_proba(Z).max(axis=1).astype(np.float32)
        return Predictions(
            cell_ids=list(query.obs_names),
            labels=self.clf.predict(Z),
            probabilities=proba,
        )
