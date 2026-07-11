"""Prototype: UCE-style protein-embedding gene featurization for actinn-jax.

Represent each cell as the expression-weighted average of its genes' ESM2 protein
embeddings (UCE's core idea), giving a fixed 1280-d cell vector from ALL embeddable genes.
Train actinn-jax's own MLP on it (standardized) and compare to the raw-gene runs. Because
the feature dimension is fixed regardless of #genes, it can't overfit the way raw-gene
expansion did -- the direct test on tabula_sapiens (raw 5000 genes collapsed 0.405->0.303).

    python protein_embed_probe.py <dataset_dir> <esm2_pt> <out_csv>
"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax.actinn_utils as au
from actinn_jax.actinn_predict import _normalize, _encode_labels
from sklearn.metrics import accuracy_score, f1_score

d, esm2_npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
name = os.path.basename(d.rstrip("/"))

# --- load ESM2 gene->embedding table (keys are UPPERCASE symbols) ---
z = np.load(esm2_npz, allow_pickle=True)
E = z["E"].astype(np.float32); genes = [str(g).upper() for g in z["genes"]]
emb = {g: i for i, g in enumerate(genes)}
DIM = E.shape[1]
print(f"ESM2 table: {len(emb)} genes x {DIM}", flush=True)

def load(p):
    a = sc.read_h5ad(p, backed="r"); X = sp.csr_matrix(a.layers["counts"][:])
    obs, var, n = a.obs.copy(), a.var.copy(), list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=X, obs=obs, var=var); b.obs_names = n; return b

tr = load(f"{d}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
te = load(f"{d}/test.h5ad")
sol = sc.read_h5ad(f"{d}/solution.h5ad", backed="r")
true = sol.obs.loc[list(te.obs_names), "label"].astype(str).to_numpy()

# genes present in both dataset and ESM2 table (match on UPPERCASE gene symbol)
symcol = "feature_name" if "feature_name" in tr.var.columns else None
gsym = pd.Index((tr.var[symcol] if symcol else tr.var_names).astype(str).str.upper())
hit = np.array([g in emb for g in gsym])
G = np.stack([E[emb[g]] for g in gsym[hit]])       # (n_hit, DIM)
print(f"{name}: {int(hit.sum())}/{tr.n_vars} genes have ESM2 embeddings", flush=True)

def protein_features(adata, chunk=20000):
    """Expression-weighted mean of gene protein embeddings -> (cells, DIM)."""
    Xn = _normalize(adata.X)[:, hit].tocsr()       # log-norm, embeddable genes
    n = Xn.shape[0]; out = np.empty((n, DIM), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = Xn[s:e]
        wsum = np.asarray(blk.sum(1)).ravel(); wsum[wsum == 0] = 1.0
        out[s:e] = (blk @ G) / wsum[:, None]
    return out

t = time.time()
Ftr = protein_features(tr); Fte = protein_features(te)
# standardize on reference (frozen)
mu = Ftr.mean(0); sd = Ftr.std(0); sd[sd == 0] = 1.0
Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
yint, classes = _encode_labels(tr.obs["label"].to_numpy())
params = au.train(Ftr, au.one_hot(yint, len(classes)), num_epochs=au.DEFAULT_NUM_EPOCHS,
                  seed=au.DEFAULT_SEED, print_cost=False)
pred = np.array([classes[i] for i in au.predict_proba(params, Fte).argmax(1)])
acc = accuracy_score(true, pred); f1 = f1_score(true, pred, average="macro")
row = {"dataset": name, "method": "actinn-protein-esm2", "dim": DIM,
       "n_genes_used": int(hit.sum()), "accuracy": round(acc, 4),
       "f1_macro": round(f1, 4), "fit_s": round(time.time() - t, 1)}
print("RESULT", row, flush=True)
pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
print("PROT_DONE", name, flush=True)
