"""Rejection / abstain: can a method flag cells whose true type is NOT in the reference,
and what accuracy-coverage tradeoff does its confidence give on in-distribution cells?

Setup (liver_intra, HLiCA): hold out ~25% of cell types ENTIRELY from the reference so
their cells are genuinely out-of-distribution (OOD) in the query. Train each method with
per-cell confidence (actinn-jax fine-label probability; CellTypist max class probability),
sweep a confidence threshold, and report:
  - in-distribution accuracy on KEPT cells vs. coverage (fraction kept)
  - OOD-flag rate: fraction of OOD cells correctly sent below threshold

scmap-cluster has a single native "unassigned" operating point (not a sweep) and is noted
qualitatively in the writeup. Core-venv methods only. Run AFTER the main matrix.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import adapters, datasets

SRC = "/Volumes/IanSSD/hlica/liver_intra.h5ad"
LABEL = "cell_type"
OOD_FRAC = 0.25
THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9]

a = sc.read_h5ad(SRC)
types = np.array(sorted(a.obs[LABEL].astype(str).unique()))
rng = np.random.default_rng(0)
ood = set(rng.choice(types, max(1, int(len(types) * OOD_FRAC)), replace=False))
lab = a.obs[LABEL].astype(str).to_numpy()
is_ood = np.array([t in ood for t in lab])

# reference = in-distribution types only; query = held-out split of in-dist + ALL ood cells
ind = a[~is_ood].copy()
ref, ind_query = datasets.intra_split(ind, LABEL, 0.3)
query = sc.concat([ind_query, a[is_ood].copy()])
q_true = query.obs[LABEL].astype(str).to_numpy()
q_is_ood = np.array([t in ood for t in q_true])
print(f"{len(types)} types: {len(ood)} held out OOD | ref {ref.n_obs} / query {query.n_obs} "
      f"({q_is_ood.sum()} OOD cells)", flush=True)

rows = []
for name in ["actinn-jax", "celltypist"]:
    m = adapters.get(name)
    m.fit(ref, LABEL)
    pred = m.predict(query)
    conf = pred.probabilities
    labels = np.asarray(pred.labels)
    if conf is None:
        print(f"{name}: no probabilities, skipping"); continue
    for thr in THRESHOLDS:
        kept = (conf >= thr) & (~q_is_ood)      # in-dist cells above threshold
        indist = ~q_is_ood
        cov = kept.sum() / max(indist.sum(), 1)
        acc = (labels[kept] == q_true[kept]).mean() if kept.sum() else float("nan")
        ood_flagged = (conf[q_is_ood] < thr).mean() if q_is_ood.sum() else float("nan")
        rows.append({"method": name, "min_prob": thr, "indist_acc_kept": round(acc, 3),
                     "coverage": round(cov, 3), "ood_flagged": round(ood_flagged, 3)})
        print(f"  {name:11s} thr={thr}: acc(kept)={acc:.3f} cov={cov:.3f} "
              f"ood_flagged={ood_flagged:.3f}", flush=True)

pd.DataFrame(rows).to_csv("docs/results_rejection.csv", index=False)
print("REJECTION_DONE", flush=True)
