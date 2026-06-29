"""Cross-DATASET hepatocyte zonation transfer: train on one human liver study, predict
zonation in another (different lab/protocol/patients). The hardest generalization test.

  GSE158723  <->  GSE136103 (Ramachandran, healthy)
Zones in each were derived independently from the same landmark-gene axis; we train on
one set's zones and evaluate against the other's. Genes aligned by Ensembl id. Core env.
"""
import os, warnings; warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, numpy as np, scanpy as sc, pandas as pd
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import metrics
import actinn_jax as aj

ZORD = {"portal": 0, "mid": 1, "central": 2}
A = sc.read_h5ad("/tmp/liver_zonation_ref.h5ad"); A.obs["zone"] = A.obs["zone"].astype(str)
B = sc.read_h5ad("/tmp/gse136103_zonation_ref.h5ad"); B.obs["zone"] = B.obs["zone"].astype(str)
common = A.var_names.intersection(B.var_names)
A = A[:, common].copy(); B = B[:, common].copy()
print(f"GSE158723: {A.n_obs} hep | GSE136103: {B.n_obs} hep | shared genes: {len(common)}\n", flush=True)
rows = []


def transfer(name, ref, qry):
    m = aj.train_reference(ref, train_label_name="zone", print_cost=False)
    out, _ = aj.predict(qry, m, output_label_name="pred")
    t = np.array([ZORD[z] for z in out.obs["zone"]]); p = np.array([ZORD.get(z, 1) for z in out.obs["pred"]])
    acc = float((t == p).mean()); w1 = float((np.abs(t - p) <= 1).mean())
    mf1 = metrics.compute(out.obs["zone"].to_numpy(), out.obs["pred"].to_numpy())["macro_f1"]
    # portal<->central confusion (the gross error)
    gross = float(((t == 0) & (p == 2)).sum() + ((t == 2) & (p == 0)).sum()) / max((t != 1).sum(), 1)
    rows.append({"transfer": name, "n_ref": ref.n_obs, "n_qry": qry.n_obs,
                 "exact_acc": round(acc, 3), "within1_acc": round(w1, 3),
                 "macro_f1": round(mf1, 3), "portal_central_confusion": round(gross, 3)})
    print(f"  {name:28} exact {acc:.3f} within1 {w1:.3f} mF1 {mf1:.3f} "
          f"portal<->central {gross:.3f}", flush=True)


transfer("GSE158723 -> GSE136103", A, B)
transfer("GSE136103 -> GSE158723", B, A)
print()
df = pd.DataFrame(rows); df.to_csv("/tmp/cross_zonation_results.csv", index=False)
print(df.to_string(index=False)); print("CROSS_ZONATION_DONE", flush=True)
