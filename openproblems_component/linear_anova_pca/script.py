import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "n_hvg": 1000,
    "n_genes": 20000,
    "n_pcs": 220,
}
meta = {"name": "linear_anova_pca"}
## VIASH END

print("Load input data", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

# Same input budget as the other gene-space methods in this task.
if par.get("n_hvg") and "hvg" in input_train.var:
    hvg = input_train.var["hvg"].values.astype(bool)
    input_train = input_train[:, hvg].copy()
    input_test = input_test[:, hvg].copy()


def lognorm_counts(adata):
    """CP10k + log1p starting from raw counts, matching the benchmark's `lognorm`."""
    out = ad.AnnData(
        X=sp.csr_matrix(adata.layers["counts"]).astype(np.float32),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(adata.var_names).str.upper()),
    )
    out.var_names_make_unique()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def dense_aligned(adata, genes):
    """Dense (n_cells, len(genes)) aligned to `genes`; missing genes -> 0."""
    pos = pd.Index(adata.var_names).get_indexer(pd.Index(genes))
    out = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    present = pos >= 0
    if present.any():
        sub = adata[:, pos[present]].X
        out[:, present] = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return out


print("Fit normalize -> ANOVA -> standardize -> PCA -> logistic regression", flush=True)
train = lognorm_counts(input_train)
y = input_train.obs["label"].astype(str).to_numpy()

# ANOVA F-test on the sparse matrix, then densify only the selected genes.
k = int(min(par["n_genes"], train.n_vars))
keep = SelectKBest(f_classif, k=k).fit(train.X, y).get_support()
genes = list(np.asarray(list(train.var_names))[keep])

X = dense_aligned(train, genes)
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
n_comp = int(min(par["n_pcs"], Xs.shape[0], Xs.shape[1]))
pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0).fit(Xs)
clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1).fit(pca.transform(Xs), y)

print("Predict on test data", flush=True)
test = lognorm_counts(input_test)
Z = pca.transform(scaler.transform(dense_aligned(test, genes)))
label_pred = clf.predict(Z)

print("Create output data", flush=True)
output = ad.AnnData(
    # Force plain object dtype: newer pandas hands back nullable StringArrays,
    # which anndata refuses to write ("allow_write_nullable_strings is False").
    obs=pd.DataFrame(
        {"label_pred": np.asarray(label_pred, dtype=object)},
        index=pd.Index([str(i) for i in input_test.obs_names], dtype=object),
    ),
    uns={
        "method_id": meta["name"],
        "dataset_id": input_test.uns["dataset_id"],
        "normalization_id": input_test.uns["normalization_id"],
    },
)

print("Write output data", flush=True)
output.write_h5ad(par["output"], compression="gzip")
