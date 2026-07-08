import anndata as ad
import numpy as np
import pandas as pd
import actinn_jax as aj

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "n_hvg": 1000,
}
meta = {"name": "actinn_jax"}
## VIASH END

print("Load input data", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

# actinn-jax is a gene-space method: it trains on raw counts with its own CP10k+log2
# normalization and gene filter. Optionally restrict to the task-provided HVGs (the same
# feature set the PCA-based baselines use) to keep atlas-scale training tractable.
if par.get("n_hvg") and "hvg" in input_train.var:
    hvg = input_train.var["hvg"].values.astype(bool)
    input_train = input_train[:, hvg].copy()
    input_test = input_test[:, hvg].copy()


def counts_adata(adata, label=None):
    b = ad.AnnData(X=adata.layers["counts"].copy(),
                   obs=adata.obs.copy(), var=adata.var.copy())
    b.var_names = adata.var_names
    if label is not None:
        b.obs["label"] = adata.obs[label].astype(str).values
    return b


print("Train actinn-jax on the reference", flush=True)
ref = counts_adata(input_train, label="label")
model = aj.train_reference(ref, train_label_name="label")

print("Predict on test data", flush=True)
query = counts_adata(input_test)
frame, _ = model.predict_frame(query, use_raw=False)
label_pred = frame.loc[list(input_test.obs_names), "celltype"].to_numpy()

print("Create output data", flush=True)
output = ad.AnnData(
    obs=pd.DataFrame({"label_pred": label_pred}, index=input_test.obs.index),
    uns={
        "method_id": meta["name"],
        "dataset_id": input_test.uns["dataset_id"],
        "normalization_id": input_test.uns["normalization_id"],
    },
)

print("Write output data", flush=True)
output.write_h5ad(par["output"], compression="gzip")
