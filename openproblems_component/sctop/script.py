import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sctop.processing import process, score

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "n_hvg": 1000,
    "min_expr_frac": 0.10,
}
meta = {"name": "sctop"}
## VIASH END

print("Load input data", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

if par.get("n_hvg") and "hvg" in input_train.var:
    hvg = input_train.var["hvg"].values.astype(bool)
    input_train = input_train[:, hvg].copy()
    input_test = input_test[:, hvg].copy()


def counts_matrix(adata):
    """Raw counts as CSR with upper-cased gene names (scTOP does its own ranking)."""
    out = ad.AnnData(
        X=sp.csr_matrix(adata.layers["counts"]).astype(np.float32),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(adata.var_names).str.upper()),
    )
    out.var_names_make_unique()
    return out


def genes_by_cells(X, genes):
    """Dense (n_genes, n_cells) DataFrame -- the orientation scTOP expects."""
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return pd.DataFrame(np.asarray(arr, dtype=np.float32).T, index=list(genes))


def dense_aligned(adata, genes):
    pos = pd.Index(adata.var_names).get_indexer(pd.Index(genes))
    out = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    present = pos >= 0
    if present.any():
        sub = adata[:, pos[present]].X
        out[:, present] = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return out


print("Build the per-cell-type basis", flush=True)
train = counts_matrix(input_train)
y = input_train.obs["label"].astype(str).to_numpy()

# scTOP is rank-based, so an all-but-zero gene tail makes the bases degenerate.
expr_frac = np.asarray((train.X > 0).sum(axis=0)).ravel() / max(1, train.n_obs)
keep = expr_frac >= par["min_expr_frac"]
if keep.sum() < 50:  # degenerate filter -> keep everything
    keep = np.ones(train.n_vars, dtype=bool)
genes = list(np.asarray(list(train.var_names))[keep])
print(f"  kept {len(genes)} of {train.n_vars} genes", flush=True)

basis = pd.DataFrame(
    {
        str(t): process(genes_by_cells(train.X[y == t][:, keep], genes),
                        average=True).iloc[:, 0]
        for t in pd.unique(y)
    }
)

print("Project the test cells onto the basis", flush=True)
test = counts_matrix(input_test)
sample = process(genes_by_cells(dense_aligned(test, genes), genes), chunk_size=2000)
proj = score(basis, sample, chunk_size=2000)

# A test cell with no counts in any retained gene has no ranks to z-score, so its whole
# projection column is NaN and a plain idxmax raises "Encountered all NA values" -- losing
# the entire dataset over a handful of cells (185 of 11,508 on gtex_v9, 1 of 284 on
# tabula_sapiens, none elsewhere). scTOP genuinely cannot place those cells, so they are
# labelled `unassigned`, which scores as wrong: the run completes and what it reports is a
# lower bound rather than a number obtained by dropping the inconvenient cells.
unscorable = proj.isna().all(axis=0).to_numpy()
label_pred = proj.fillna(-np.inf).idxmax(axis=0).to_numpy().astype(str)
label_pred[unscorable] = "unassigned"
if unscorable.any():
    print(f"  {int(unscorable.sum())} of {len(unscorable)} test cells have no signal in the "
          f"retained genes -> unassigned", flush=True)

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
