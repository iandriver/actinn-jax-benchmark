"""Stage 3/3 of the broad-reference build: train, calibrate, and ship the model.

The scPRINT embedding (QC-filtered subset, carrying cell_type) gives per-type centroids
-> coarse hierarchy. actinn-jax is trained SEPARATELY on the full reference by label +
that hierarchy (no per-cell alignment needed). Core .venv (actinn_jax).

Calibration holds out whole cell types (OOD) + a within-type test split, and sweeps
min_prob to show in-distribution accuracy vs OOD-flag rate. Then ships the full model to
references/broad_human_v1 alongside a ``build_info.json`` recording the census release,
the sizes and the calibration table -- so a shipped reference can answer "what am I built
from?" without anyone having to remember. See docs/UPDATE_BROAD_REFERENCE.md.
"""
import json, os, subprocess, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc

WORK = os.environ.get("ACTINN_REF_WORK", "/tmp/actinn_ref_build")
PKG = os.environ.get("ACTINN_JAX_REPO", os.path.expanduser("~/Downloads/actinn-jax"))
sys.path.insert(0, PKG)
import actinn_jax as aj

REF = os.environ.get("REF_H5AD", f"{WORK}/census_wide_ref.h5ad")
EMB = os.environ.get("REF_EMB", f"{WORK}/census_wide_emb.npz")
NAME = os.environ.get("REF_NAME", "broad_human_v1")
OUT = os.environ.get("REF_OUT", f"{PKG}/actinn_jax/references/{NAME}")
N_HVG = int(os.environ.get("N_HVG", 4000))
HIERARCHY = os.environ.get("HIERARCHY", "scprint")     # scprint | ontology
OBO = os.environ.get("ONTOLOGY_OBO", "/tmp/cl-basic.obo")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ref = sc.read_h5ad(REF)
labels = ref.obs["cell_type"].astype(str).to_numpy()
types = np.array(sorted(set(labels)))
N_GROUPS = max(8, int(round(np.sqrt(len(types)))))

# HIERARCHY=scprint (default) clusters scPRINT embeddings of per-type centroids -- the one
# GPU step of the build. HIERARCHY=ontology clusters Cell Ontology lineage instead: free,
# deterministic, and species-independent, which is what makes a pan-mouse reference
# buildable at all (see ontology_hierarchy.py and docs/PAN_MOUSE.md).
if HIERARCHY == "ontology":
    from ontology_hierarchy import ontology_hierarchy
    grp, info = ontology_hierarchy(ref.obs["cell_type_ontology_term_id"], labels,
                                   n_groups=N_GROUPS, obo=OBO)
    print(f"ref {ref.shape} | {len(types)} types | ontology hierarchy from "
          f"{info['n_cl_terms']} CL terms over {info['n_types_with_cl']} types "
          f"| G={N_GROUPS}", flush=True)
else:
    z = np.load(EMB, allow_pickle=True)
    emb, emb_ct = z["emb"], z["cell_type"].astype(str)       # QC-filtered survivors + labels
    print(f"ref {ref.shape} | {len(types)} types | emb {emb.shape} "
          f"({len(set(emb_ct))} embedded types) | G={N_GROUPS}", flush=True)
    # covers embedded types; the rest fall into build_hierarchical_reference's catch-all
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
                                      hierarchy=grp_ind, ontology_key="cell_type_ontology_term_id", print_cost=False)
pf_ind = cal.predict_frame(ref[test].copy())[0]
pf_ood = cal.predict_frame(ref[is_ood].copy())[0]
p_ind = pf_ind["celltype_probability"].values
lab_ind, true_ind = pf_ind["celltype"].values, labels[test]
p_ood = pf_ood["celltype_probability"].values
print("min_prob | in-dist acc(kept) | in-dist kept | OOD flagged", flush=True)
calibration = []
for thr in (0.0, 0.3, 0.5, 0.7, 0.9):
    kept = p_ind >= thr
    acc = float((lab_ind[kept] == true_ind[kept]).mean()) if kept.sum() else float("nan")
    calibration.append({"min_prob": thr, "accuracy_kept": round(acc, 4),
                        "coverage": round(float(kept.mean()), 4),
                        "ood_flagged": round(float((p_ood < thr).mean()), 4)})
    print(f"  {thr:>4} | {acc:.3f} | {kept.mean():.3f} | {float((p_ood < thr).mean()):.3f}", flush=True)

# ---- ship: full reference, full hierarchy ----
t = time.time()
model = aj.build_hierarchical_reference(hvg_subset(ref.copy()), "cell_type",
                                        hierarchy=grp, ontology_key="cell_type_ontology_term_id", print_cost=False)
ng = len(set(model.type_to_group.values()))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model.save(OUT)
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6


def _git_sha(repo):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


release = {}
if os.path.exists(f"{WORK}/census_release.json"):
    with open(f"{WORK}/census_release.json") as fh:
        release = json.load(fh)

with open(os.path.join(OUT, "build_info.json"), "w") as fh:
    json.dump({
        "name": NAME,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "census_release": release,
        "reference_h5ad": REF,
        "n_cells": int(ref.n_obs), "n_types": len(model.classes),
        "n_coarse_groups": ng, "n_tissues": int(ref.obs["tissue"].nunique())
        if "tissue" in ref.obs else None,
        "n_hvg": N_HVG, "size_mb": round(sz, 1),
        "hierarchy_source": HIERARCHY,
        "organism": release.get("organism", "homo_sapiens"),
        "calibration": calibration,
        "benchmark_sha": _git_sha(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        "actinn_jax_sha": _git_sha(PKG),
        "actinn_jax_version": getattr(aj, "__version__", None),
    }, fh, indent=2)

print(f"shipped {len(model.classes)} types / {ng} coarse groups / {sz:.1f}MB in {time.time()-t:.0f}s",
      flush=True)
print(f"wrote {OUT}/build_info.json", flush=True)
print("CENSUS_MODEL_DONE", flush=True)
