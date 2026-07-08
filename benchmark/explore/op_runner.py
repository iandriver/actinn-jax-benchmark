"""Run actinn-jax through the Open Problems label_projection contract and score it with
OP's metrics, so it can be slotted onto their published leaderboard.

Each OP dataset dir has train.h5ad / test.h5ad / solution.h5ad with:
  layers['counts'], layers['normalized']; obs['label','batch']; obsm['X_pca'];
  uns['dataset_id','normalization_id']. test omits label; solution has the true label.

actinn-jax is a gene-space method (it does its own CP10k+log2 + gene filtering), so it
trains on layers['counts'] + obs['label'] and predicts test -- exactly what a real OP
component would do. We report accuracy + macro-F1 (== OP's 'accuracy' and 'f1' metrics),
plus fit/predict wall time (actinn-jax's differentiator).

    python op_runner.py <dataset_dir> [out_csv]
"""
import sys, time, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax as aj
from sklearn.metrics import accuracy_score, f1_score

ds_dir = sys.argv[1]
out_csv = sys.argv[2] if len(sys.argv) > 2 else None
name = os.path.basename(ds_dir.rstrip("/"))


def load_counts(path):
    """Load ONLY the counts layer + obs/var (skip the normalized layer & X_pca) so the
    18 GB atlases don't blow up memory. Backed read -> pull counts sparse to memory."""
    a = sc.read_h5ad(path, backed="r")
    X = a.layers["counts"][:]
    obs = a.obs.copy(); var = a.var.copy(); names = list(a.obs_names)
    try:
        a.file.close()
    except Exception:
        pass
    b = ad.AnnData(X=X, obs=obs, var=var)
    b.obs_names = names
    return b


t0 = time.time()
tr = load_counts(f"{ds_dir}/train.h5ad")
tr.obs["label"] = tr.obs["label"].astype(str)
te = load_counts(f"{ds_dir}/test.h5ad")
sol = sc.read_h5ad(f"{ds_dir}/solution.h5ad", backed="r")

# Restrict to the OP-provided highly-variable genes (var['hvg']) -- the same feature set
# the framework's PCA is built from. Keeps atlas-scale training (up to 482k x 56k) tractable
# and makes actinn-jax's gene-space input consistent with what the other methods consume.
if "hvg" in tr.var.columns:
    mask = tr.var["hvg"].astype(bool).to_numpy()
    tr = tr[:, mask].copy(); te = te[:, mask].copy()
    print(f"restricted to {int(mask.sum())} HVGs", flush=True)
print(f"{name}: train {tr.n_obs}x{tr.n_vars}, test {te.n_obs}, "
      f"{tr.obs['label'].nunique()} labels (loaded {time.time()-t0:.0f}s)", flush=True)

t = time.time(); model = aj.train_reference(tr, train_label_name="label"); fit_s = time.time() - t
t = time.time(); frame, _ = model.predict_frame(te, use_raw=False); pred_s = time.time() - t
pred = frame["celltype"].to_numpy()
true = sol.obs.loc[list(te.obs_names), "label"].astype(str).to_numpy()

acc = accuracy_score(true, pred)
f1m = f1_score(true, pred, average="macro")
row = {"dataset": name, "method": "actinn-jax", "accuracy": round(acc, 4),
       "f1_macro": round(f1m, 4), "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 2),
       "n_train": tr.n_obs, "n_test": te.n_obs, "n_labels": tr.obs["label"].nunique()}
print("RESULT", row, flush=True)
if out_csv:
    hdr = not os.path.exists(out_csv)
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=hdr, index=False)
print(f"{name}: accuracy {acc:.3f} | macro-F1 {f1m:.3f} | fit {fit_s:.0f}s predict {pred_s:.1f}s",
      flush=True)
print("OP_RUN_DONE", flush=True)
