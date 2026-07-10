"""Label-free signals for picking the gene budget (no test labels used).

For each dataset at gene budgets N, compute from (reference + UNLABELED query) only:
  - ref_cv_f1 : 5-fold macro-F1 on the reference (in-distribution ceiling; logreg proxy).
  - domain_auc: 5-fold AUC of a ref-vs-query classifier in the N-gene space (domain shift;
                ~0.5 = ref and query indistinguishable, ->1.0 = big shift).
  - n_test, n_classes, cells_per_test_class (cheap red flags).
The hypothesis: ref_cv_f1 rises with genes for ALL datasets (so ref-only would always say
"add genes"), while domain_auc flags the datasets where more genes overfit (tabula_sapiens).

    python probe_signals.py <dataset_dir> <out_csv>
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
from actinn_jax.actinn_predict import _normalize

d = sys.argv[1]; out = sys.argv[2]; name = os.path.basename(d.rstrip("/"))
rng = np.random.default_rng(0)

def load(p):
    a = sc.read_h5ad(p, backed="r"); hvg = a.var["hvg"].astype(bool).to_numpy()
    X = sp.csr_matrix(a.layers["counts"][:]); obs = a.obs.copy(); var = a.var.copy(); n = list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=X, obs=obs, var=var); b.obs_names = n; return b

tr = load(f"{d}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
te = load(f"{d}/test.h5ad")
sol = sc.read_h5ad(f"{d}/solution.h5ad", backed="r")
true = sol.obs.loc[list(te.obs_names), "label"].astype(str).to_numpy()
op_hvg = tr.var["hvg"].astype(bool).to_numpy()

# seurat_v3 ranking for >1000 budgets
hv = sc.pp.highly_variable_genes(tr, flavor="seurat_v3", n_top_genes=min(6000, tr.n_vars),
                                 inplace=False)  # tr.X is raw counts
rank = hv["highly_variable_rank"].to_numpy()
def mask_for(N):
    if N == 1000: return op_hvg
    m = np.zeros(tr.n_vars, dtype=bool); m[np.argsort(np.where(np.isnan(rank), np.inf, rank))[:N]] = True; return m

def subsample(X, y, k):
    if X.shape[0] <= k: return X, y
    idx = rng.choice(X.shape[0], k, replace=False); return X[idx], y[idx]

n_classes = tr.obs["label"].nunique()
rows = []
for N in [1000, 5000]:
    m = mask_for(N)
    Xr = _normalize(tr.X)[:, m]; Xq = _normalize(te.X)[:, m]
    yr = tr.obs["label"].to_numpy()

    # ref in-distribution CV (macro-F1), subsampled+stratified
    Xrs, yrs = subsample(Xr.tocsr(), yr, 6000)
    # keep only classes with >=5 cells for CV
    vc = pd.Series(yrs).value_counts(); keep = vc[vc >= 5].index
    mkeep = np.isin(yrs, keep)
    lr = LogisticRegression(max_iter=200, n_jobs=-1)
    cvf1 = cross_val_score(lr, Xrs[mkeep].toarray(), yrs[mkeep],
                           cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="f1_macro").mean()

    # domain classifier: ref(0) vs query(1), balanced
    k = min(3000, te.n_obs, tr.n_obs)
    Xr_s, _ = subsample(Xr.tocsr(), yr, k); Xq_s = Xq.tocsr()[rng.choice(te.n_obs, min(k, te.n_obs), replace=False)]
    Xd = sp.vstack([Xr_s, Xq_s]).toarray()
    yd = np.r_[np.zeros(Xr_s.shape[0]), np.ones(Xq_s.shape[0])]
    auc = cross_val_score(LogisticRegression(max_iter=200, n_jobs=-1), Xd, yd,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()

    r = {"dataset": name, "n_genes": N, "ref_cv_f1": round(cvf1, 4), "domain_auc": round(auc, 4),
         "n_test": int(te.n_obs), "n_classes": int(n_classes),
         "test_per_class": round(te.n_obs / n_classes, 1)}
    rows.append(r); print("  PROBE", r, flush=True)

pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
print("PROBE_DONE", name, flush=True)
