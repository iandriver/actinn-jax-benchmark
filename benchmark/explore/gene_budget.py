"""Does actinn-jax improve with more genes? Sweep the input gene budget.
    python gene_budget.py <dataset_dir> <out_csv>
Budgets: 1000 (OP hvg) | 2000 | 5000 | all, using seurat_v3 HVG for the >1000 tiers.
standardize=True throughout. Reports accuracy, macro-F1, #genes after actinn's own filter,
fit time, peak RSS.
"""
import sys, os, time, threading, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp, psutil
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax as aj
from sklearn.metrics import accuracy_score, f1_score

d = sys.argv[1]; out = sys.argv[2]; name = os.path.basename(d.rstrip("/"))

def load(p):
    a = sc.read_h5ad(p, backed="r"); X = sp.csr_matrix(a.layers["counts"][:])
    obs, var, n = a.obs.copy(), a.var.copy(), list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=X, obs=obs, var=var); b.obs_names = n; b.layers["counts"] = b.X.copy(); return b

print(f"{name}: loading all genes", flush=True)
tr = load(f"{d}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
te = load(f"{d}/test.h5ad")
sol = sc.read_h5ad(f"{d}/solution.h5ad", backed="r")
true = sol.obs.loc[list(te.obs_names), "label"].astype(str).to_numpy()
op_hvg = tr.var["hvg"].astype(bool).to_numpy()
print(f"train {tr.shape}, test {te.n_obs}", flush=True)

# precompute seurat_v3 HVG ranking on the training counts (once)
hv = sc.pp.highly_variable_genes(tr, flavor="seurat_v3", n_top_genes=min(8000, tr.n_vars),
                                 layer="counts", inplace=False)
rank = hv["highly_variable_rank"].to_numpy()  # 0=most variable; NaN=not selected

def mask_for(budget):
    if budget == "1000_op": return op_hvg
    if budget == "all": return np.ones(tr.n_vars, dtype=bool)
    N = int(budget)
    m = np.zeros(tr.n_vars, dtype=bool)
    order = np.argsort(np.where(np.isnan(rank), np.inf, rank))[:N]
    m[order] = True; return m

def peak_sampler(peak, run):
    p = psutil.Process(os.getpid())
    while run[0]:
        try: peak[0] = max(peak[0], p.memory_info().rss)
        except Exception: pass
        time.sleep(0.1)

rows = []
for budget in ["1000_op", "2000", "5000", "all"]:
    m = mask_for(budget)
    trm = tr[:, m].copy(); tem = te[:, m].copy()
    peak = [0.0]; run = [True]; th = threading.Thread(target=peak_sampler, args=(peak, run), daemon=True); th.start()
    t = time.time()
    mdl = aj.train_reference(trm, train_label_name="label", standardize=True, print_cost=False)
    fit = time.time() - t
    pred = mdl.predict_frame(tem, use_raw=False)[0]["celltype"].to_numpy()
    run[0] = False
    acc = accuracy_score(true, pred); f1 = f1_score(true, pred, average="macro")
    r = {"dataset": name, "budget": budget, "n_input_genes": int(m.sum()),
         "n_model_genes": int(mdl.select_idx.size), "accuracy": round(acc, 4),
         "f1_macro": round(f1, 4), "fit_s": round(fit, 1), "peak_gb": round(peak[0]/1e9, 1)}
    rows.append(r); print("  RESULT", r, flush=True)

pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
print("GENE_BUDGET_DONE", name, flush=True)
