"""Scaling curves for the CPU-classical tier (the Pareto-relevant comparison):
runtime + peak memory vs #reference cells (fixed #types) and vs #cell types (fixed
cells/type). Methods: actinn-jax, svm, knn, celltypist -- all core-venv, same machine.

Run AFTER the main matrix (it saturates the CPU). Outputs:
  docs/results_scaling_cells.csv, docs/results_scaling_types.csv
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import adapters, datasets
from benchmark.resources import ResourceMonitor

METHODS = ["actinn-jax", "svm", "knn", "celltypist"]
LUNG = "/Users/iandriver/Downloads/krasnow_lung_atlas_10x.h5ad"
BLOODGUT = "/Volumes/IanSSD/hlica/blood_gut_intra.h5ad"
LABEL = "cell_type"


def time_method(name, ref, query):
    m = adapters.get(name)
    with ResourceMonitor() as rf:
        m.fit(ref, LABEL)
    with ResourceMonitor() as rp:
        m.predict(query)
    return {"method": name, "fit_s": rf.elapsed, "predict_s": rp.elapsed,
            "peak_mem_mb": max(rf.peak_mb, rp.peak_mb)}


# ---- A) vs #reference cells (fixed type set) ----
print("=== scaling vs #cells (lung) ===", flush=True)
lung = sc.read_h5ad(LUNG)
labels = lung.obs[LABEL].astype(str).to_numpy()
rng = np.random.default_rng(0)
q_idx = datasets.stratified_subsample(labels, 90)          # fixed ~4k query
q_mask = np.zeros(lung.n_obs, bool); q_mask[q_idx] = True
query = lung[q_mask].copy()
pool = lung[~q_mask]
pool_lab = pool.obs[LABEL].astype(str).to_numpy()
rows_c = []
for N in [1000, 2000, 5000, 10000, 20000, 40000]:
    per = max(1, N // len(set(pool_lab)))
    idx = datasets.stratified_subsample(pool_lab, per)
    ref = pool[idx].copy()
    for name in METHODS:
        r = time_method(name, ref, query)
        r.update({"n_ref": ref.n_obs, "n_types": ref.obs[LABEL].nunique()})
        rows_c.append(r)
        print(f"  N={ref.n_obs:6d} {name:11s} fit={r['fit_s']:6.1f}s pred={r['predict_s']:5.2f}s "
              f"mem={r['peak_mem_mb']:.0f}MB", flush=True)
pd.DataFrame(rows_c).to_csv("docs/results_scaling_cells.csv", index=False)

# ---- B) vs #cell types (fixed cells/type) ----
print("=== scaling vs #types (blood+gut) ===", flush=True)
bg = sc.read_h5ad(BLOODGUT)
bg_lab = bg.obs[LABEL].astype(str)
all_types = sorted(bg_lab.unique())
rows_t = []
for K in [5, 10, 20, 40, 86]:
    keep = set(all_types[:K])
    sub = bg[bg_lab.isin(keep)].copy()
    ref, query = datasets.intra_split(sub, LABEL, 0.25)
    for name in METHODS:
        r = time_method(name, ref, query)
        r.update({"n_types": K, "n_ref": ref.n_obs})
        rows_t.append(r)
        print(f"  K={K:3d} {name:11s} fit={r['fit_s']:6.1f}s pred={r['predict_s']:5.2f}s "
              f"mem={r['peak_mem_mb']:.0f}MB", flush=True)
pd.DataFrame(rows_t).to_csv("docs/results_scaling_types.csv", index=False)
print("SCALING_DONE", flush=True)
