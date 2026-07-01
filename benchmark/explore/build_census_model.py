"""Build the shipped census-wide HierarchicalReferenceModel + calibrate abstain.

The scPRINT embedding (QC-filtered subset, carrying cell_type) gives per-type centroids
-> coarse hierarchy. actinn-jax is trained SEPARATELY on the full reference by label +
that hierarchy (no per-cell alignment needed). Core .venv (actinn_jax).

Calibration holds out whole cell types (OOD) + a within-type test split, and sweeps
min_prob to show in-distribution accuracy vs OOD-flag rate. Then ships the full model to
references/broad_human_v1.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax as aj

REF = "/tmp/census_wide_ref.h5ad"
EMB = "/tmp/census_wide_emb.npz"
OUT = "/Users/iandriver/Downloads/actinn-jax/actinn_jax/references/broad_human_v1"
N_HVG = 4000

ref = sc.read_h5ad(REF)
z = np.load(EMB, allow_pickle=True)
emb, emb_ct = z["emb"], z["cell_type"].astype(str)           # QC-filtered survivors + labels
labels = ref.obs["cell_type"].astype(str).to_numpy()
types = np.array(sorted(set(labels)))
N_GROUPS = max(8, int(round(np.sqrt(len(types)))))
print(f"ref {ref.shape} | {len(types)} types | emb {emb.shape} ({len(set(emb_ct))} embedded types) "
      f"| G={N_GROUPS}", flush=True)

# hierarchy from embedded per-type centroids (covers embedded types; rest -> fallback)
grp = aj.discover_hierarchy(emb, emb_ct, n_groups=N_GROUPS)


def hvg_subset(ad_train, n=N_HVG):
    raw = ad_train.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=min(n, raw.n_vars))
    return ad_train[:, raw.var["highly_variable"].values].copy()


# ---- calibration: OOD whole types + within-type test split ----
rng = np.random.default_rng(0)
ood_types = set(rng.choice(types, max(1, int(len(types) * 0.10)), replace=False))
is_ood = np.array([t in ood_types for t in labels])
test = np.zeros(ref.n_obs, dtype=bool)
for c in types:
    if c in ood_types:
        continue
    idx = np.where(labels == c)[0]
    if len(idx) >= 5:
        test[rng.choice(idx, max(1, int(len(idx) * 0.2)), replace=False)] = True
tr = ~is_ood & ~test
grp_ind = {t: g for t, g in grp.items() if t not in ood_types}
print(f"calibration: {tr.sum()} train / {test.sum()} in-dist test / {is_ood.sum()} OOD cells "
      f"({len(ood_types)} OOD types)", flush=True)

cal = aj.build_hierarchical_reference(hvg_subset(ref[tr].copy()), "cell_type",
                                      hierarchy=grp_ind, print_cost=False)
pf_ind = cal.predict_frame(ref[test].copy())[0]
pf_ood = cal.predict_frame(ref[is_ood].copy())[0]
p_ind = pf_ind["celltype_probability"].values
lab_ind, true_ind = pf_ind["celltype"].values, labels[test]
p_ood = pf_ood["celltype_probability"].values
print("min_prob | in-dist acc(kept) | in-dist kept | OOD flagged", flush=True)
for thr in (0.0, 0.3, 0.5, 0.7, 0.9):
    kept = p_ind >= thr
    acc = float((lab_ind[kept] == true_ind[kept]).mean()) if kept.sum() else float("nan")
    print(f"  {thr:>4} | {acc:.3f} | {kept.mean():.3f} | {float((p_ood < thr).mean()):.3f}", flush=True)

# ---- ship: full reference, full hierarchy ----
t = time.time()
model = aj.build_hierarchical_reference(hvg_subset(ref.copy()), "cell_type",
                                        hierarchy=grp, print_cost=False)
ng = len(set(model.type_to_group.values()))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model.save(OUT)
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
print(f"shipped {len(model.classes)} types / {ng} coarse groups / {sz:.1f}MB in {time.time()-t:.0f}s",
      flush=True)
print("CENSUS_MODEL_DONE", flush=True)
