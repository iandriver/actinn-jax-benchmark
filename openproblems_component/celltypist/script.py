import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "n_hvg": 1000,
}
meta = {"name": "celltypist"}
## VIASH END

print("Load input data", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

if par.get("n_hvg") and "hvg" in input_train.var:
    hvg = input_train.var["hvg"].values.astype(bool)
    input_train = input_train[:, hvg].copy()
    input_test = input_test[:, hvg].copy()


def lognorm_counts(adata, label=None):
    """CellTypist expects log1p of CP10k-normalized counts with genes in var_names."""
    out = ad.AnnData(
        X=sp.csr_matrix(adata.layers["counts"]).astype(np.float32),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(adata.var_names).str.upper()),
    )
    out.var_names_make_unique()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    if label is not None:
        out.obs["label"] = adata.obs[label].astype(str).values
    return out


print("Train CellTypist", flush=True)
train = lognorm_counts(input_train, label="label")
model = celltypist.train(
    train, labels="label", use_SGD=True, feature_selection=False,
    check_expression=False,
)

print("Predict on test data", flush=True)
test = lognorm_counts(input_test)
res = celltypist.annotate(test, model=model, majority_voting=False)
label_pred = res.predicted_labels["predicted_labels"].to_numpy()

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
