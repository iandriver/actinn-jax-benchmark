"""scANVI + scArches (OP recipe) on CPU, HVG-restricted, for a per-cell prediction.

Faithful to OP's src/methods/scanvi_scarches: SCVI -> SCANVI.from_scvi_model ->
SCANVI.load_query_data (scArches surgery) -> predict. Adapted for local CPU: restrict to
the task's 1000 HVGs (the same features actinn-jax uses -> a fairer head-to-head) and
subsample the reference. Reports accuracy/macro-F1 and writes per-cell predictions.

    python run_scarches.py <dataset_dir> <out_prefix> [ref_cells]
"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
import torch, scvi
from sklearn.metrics import accuracy_score, f1_score

torch.set_num_threads(os.cpu_count())
scvi.settings.seed = 0
scvi.settings.num_threads = os.cpu_count()

d = sys.argv[1]; out = sys.argv[2]; name = os.path.basename(d.rstrip("/"))
REF_CELLS = int(sys.argv[3]) if len(sys.argv) > 3 else 30000

def load(p):
    a = sc.read_h5ad(p, backed="r")
    hvg = a.var["hvg"].astype(bool).to_numpy()
    X = sp.csr_matrix(a.layers["counts"][:, hvg])
    obs = a.obs.copy(); var = a.var[hvg].copy(); n = list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=X, obs=obs, var=var); b.obs_names = n
    b.layers["counts"] = b.X.copy()
    return b

t0 = time.time()
print(f"{name}: loading (HVG)", flush=True)
tr = load(f"{d}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
te = load(f"{d}/test.h5ad")
sol = sc.read_h5ad(f"{d}/solution.h5ad", backed="r")
true = sol.obs.loc[list(te.obs_names), "label"].astype(str).to_numpy()

# subsample reference for CPU feasibility (stratify by batch to preserve structure)
if tr.n_obs > REF_CELLS:
    rng = np.random.default_rng(0)
    idx = rng.choice(tr.n_obs, REF_CELLS, replace=False)
    tr = tr[np.sort(idx)].copy()
tr.obs["batch"] = tr.obs["batch"].astype(str); te.obs["batch"] = te.obs["batch"].astype(str)
print(f"ref {tr.n_obs}x{tr.n_vars} ({tr.obs['label'].nunique()} labels, {tr.obs['batch'].nunique()} batches); query {te.n_obs}", flush=True)

arches = dict(use_layer_norm="both", use_batch_norm="none", encode_covariates=True,
              dropout_rate=0.2, n_hidden=128, n_layers=2, n_latent=30)
tk = dict(train_size=0.9, early_stopping=True, accelerator="cpu")

print("SCVI train", flush=True)
scvi.model.SCVI.setup_anndata(tr, batch_key="batch", labels_key="label")
m = scvi.model.SCVI(tr, **arches); m.train(max_epochs=200, **tk)
print(f"  [{time.time()-t0:.0f}s]", flush=True)
print("SCANVI train", flush=True)
sca = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category="Unknown"); sca.train(max_epochs=100, **tk)
print(f"  [{time.time()-t0:.0f}s]", flush=True)
print("scArches surgery + query train", flush=True)
q = scvi.model.SCANVI.load_query_data(te, sca)
q.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0), early_stopping=True, accelerator="cpu")
pred = np.asarray(q.predict(te))
print(f"  [{time.time()-t0:.0f}s]", flush=True)

acc = accuracy_score(true, pred); f1 = f1_score(true, pred, average="macro")
print(f"RESULT {name}: scArches(HVG,cpu) acc={acc:.4f} f1={f1:.4f}  ref={tr.n_obs}", flush=True)
pd.DataFrame({"cell": list(te.obs_names), "true": true, "scarches_pred": pred}).to_csv(out + "_pred.csv", index=False)
with open(out + "_metrics.txt", "w") as fh:
    fh.write(f"{name}\tscarches_hvg_cpu\tacc={acc:.4f}\tf1={f1:.4f}\tref={tr.n_obs}\ttotal_s={time.time()-t0:.0f}\n")
print("SCARCHES_DONE", flush=True)
