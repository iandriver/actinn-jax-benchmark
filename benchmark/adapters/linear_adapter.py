"""Adapter for the tuned linear pipeline of Souza & Mehta (bioRxiv 2026).

Their Tabula Sapiens 2.0 recipe, which matches single-cell foundation models
(macro-F1 0.899 vs 0.907-0.910 for TranscriptFormer):

    per-cell normalization -> ANOVA gene selection -> standardization
        -> PCA -> multinomial logistic regression

This is a deliberately *strong* classical baseline: our other classical adapters
(SVM/kNN/CellTypist) are untuned, so without this the classical tier is easy to beat.
Everything is fit on the reference only and replayed on the query (no leakage).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..prep import counts_adata, dense_aligned, lognorm
from .base import AnnotationMethod, Predictions, register


@register
class LinearAnovaPCA(AnnotationMethod):
    """normalize -> ANOVA(k genes) -> standardize -> PCA(n_pcs) -> logistic regression."""

    name = "linear-anova-pca"
    tier = "classical"
    device = "cpu"

    def __init__(self, n_genes=20000, n_pcs=220, C=1.0, max_iter=1000):
        self.n_genes = n_genes      # paper: 20000 (effectively "all informative genes")
        self.n_pcs = n_pcs          # paper: 220 components
        self.C = C
        self.max_iter = max_iter

    def fit(self, ref, label_key):
        a = lognorm(counts_adata(ref))
        y = ref.obs[label_key].astype(str).to_numpy()
        all_genes = list(a.var_names)

        # ANOVA F-test on the sparse matrix, then densify only the selected genes.
        k = int(min(self.n_genes, a.n_vars))
        sel = SelectKBest(f_classif, k=k).fit(a.X, y)
        keep = sel.get_support()
        self.genes = list(np.asarray(all_genes)[keep])

        X = dense_aligned(a, self.genes).astype(np.float32, copy=False)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        n_comp = int(min(self.n_pcs, Xs.shape[0], Xs.shape[1]))
        self.pca = PCA(n_components=n_comp, svd_solver="randomized",
                       random_state=0).fit(Xs)
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                      n_jobs=-1).fit(self.pca.transform(Xs), y)

    def predict(self, query):
        a = lognorm(counts_adata(query))
        X = dense_aligned(a, self.genes).astype(np.float32, copy=False)
        Z = self.pca.transform(self.scaler.transform(X))
        return Predictions(
            cell_ids=list(query.obs_names),
            labels=self.clf.predict(Z),
            probabilities=self.clf.predict_proba(Z).max(axis=1).astype(np.float32),
        )
