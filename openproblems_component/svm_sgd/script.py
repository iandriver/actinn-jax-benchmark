import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "n_hvg": 1000,
}
meta = {"name": "svm_sgd"}
## VIASH END

print("Load input data", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

if par.get("n_hvg") and "hvg" in input_train.var:
    hvg = input_train.var["hvg"].values.astype(bool)
    input_train = input_train[:, hvg].copy()
    input_test = input_test[:, hvg].copy()


def lognorm_counts(adata):
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
    pos = pd.Index(adata.var_names).get_indexer(pd.Index(genes))
    out = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    present = pos >= 0
    if present.any():
        sub = adata[:, pos[present]].X
        out[:, present] = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return out


print("Train the linear SVM", flush=True)
train = lognorm_counts(input_train)
genes = list(train.var_names)
X = dense_aligned(train, genes)
scaler = StandardScaler().fit(X)
clf = SGDClassifier(loss="hinge").fit(
    scaler.transform(X), input_train.obs["label"].astype(str).to_numpy()
)

print("Predict on test data", flush=True)
test = lognorm_counts(input_test)
scores = clf.decision_function(scaler.transform(dense_aligned(test, genes)))
idx = np.argmax(scores, axis=1) if scores.ndim > 1 else (scores > 0).astype(int)
label_pred = clf.classes_[idx]

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
