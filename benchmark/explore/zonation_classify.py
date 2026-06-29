"""Can actinn-jax learn hepatocyte zonation (portal/mid/central) — and generalize?

Zone labels were *derived* from landmark genes, so the sharp tests are:
  (B) drop the landmark genes -> does the model still predict zone? (broad encoding)
  (C) hold out whole donors/samples -> does zonation transfer across patients?
Reports exact and within-1-zone accuracy (zonation is ordinal; portal<->central is the
real error). Run in the core env.
"""
import os, warnings; warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, numpy as np, scanpy as sc, pandas as pd
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import datasets, metrics
import actinn_jax as aj

CENTRAL = ["GLUL","CYP2E1","CYP1A2","CYP3A4","ADH1B","ADH4","OAT","CYP2C8"]
PORTAL = ["ASS1","ASL","SDS","HAL","CPS1","PCK1","ARG1","AGT","GLS2","SLC7A2"]
ZORD = {"portal": 0, "mid": 1, "central": 2}

ad = sc.read_h5ad("/tmp/liver_zonation_ref.h5ad")
ad.obs["zone"] = ad.obs["zone"].astype(str)
marker_ens = ad.var_names[ad.var["symbol"].isin(CENTRAL + PORTAL)]
print(f"{ad.n_obs} hepatocytes | {len(marker_ens)} landmark genes | samples: "
      f"{list(ad.obs['sample'].unique())}", flush=True)
rows = []


def evaluate(name, ref, qry):
    model = aj.train_reference(ref, train_label_name="zone", print_cost=False)
    out, _ = aj.predict(qry, model, output_label_name="pred")
    t = np.array([ZORD[z] for z in out.obs["zone"]])
    p = np.array([ZORD.get(z, 1) for z in out.obs["pred"]])
    acc = float((t == p).mean()); within1 = float((np.abs(t - p) <= 1).mean())
    mf1 = metrics.compute(out.obs["zone"].to_numpy(), out.obs["pred"].to_numpy())["macro_f1"]
    rows.append({"test": name, "n_genes": ref.n_vars, "n_ref": ref.n_obs, "n_qry": qry.n_obs,
                 "exact_acc": round(acc, 3), "within1_acc": round(within1, 3), "macro_f1": round(mf1, 3)})
    print(f"  {name:32} genes {ref.n_vars:5} | exact {acc:.3f} within1 {within1:.3f} mF1 {mf1:.3f}", flush=True)


no_markers = ad[:, ~ad.var_names.isin(marker_ens)].copy()

# A/B: random split, all genes vs landmark genes removed
ref, qry = datasets.intra_split(ad, "zone", 0.25)
evaluate("A random-split, all genes", ref, qry)
ref2, qry2 = datasets.intra_split(no_markers, "zone", 0.25)
evaluate("B random-split, NO landmark genes", ref2, qry2)

# C: hold out 2 whole samples (cross-donor) — all genes and no-markers
samples = list(ad.obs["sample"].unique())
held = samples[-2:]
def by_sample(a):
    return a[~a.obs["sample"].isin(held)].copy(), a[a.obs["sample"].isin(held)].copy()
evaluate(f"C held-out samples {held}, all genes", *by_sample(ad))
evaluate(f"C held-out samples, NO landmark genes", *by_sample(no_markers))

print()
df = pd.DataFrame(rows); df.to_csv("/tmp/zonation_results.csv", index=False)
print(df.to_string(index=False)); print("ZONATION_CLF_DONE", flush=True)
