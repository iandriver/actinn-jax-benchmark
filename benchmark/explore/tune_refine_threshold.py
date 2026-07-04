"""Sweep presence-detection rules against ground truth (lung + liver) to find a
threshold that trades precision/recall sensibly, then re-score end-to-end accuracy.
Evidence is computed once per query (impossible thresholds -> pure evidence, "kept"
ignored) and every rule is then evaluated offline from that table -- one forward pass
per query, not per rule.
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark/benchmark")
import numpy as np, pandas as pd, scanpy as sc
import actinn_jax as aj
from actinn_jax.hierarchy import detect_present_classes
from metrics import load_cl_ancestors, _ontology

OBO = "/tmp/cl-basic.obo"
QUERIES = {
    "lung": "/Users/iandriver/Downloads/krasnow_lung_atlas_10x.h5ad",
    "liver": "/tmp/liver_zonation_query.h5ad",
}
model = aj.bundled_reference("broad_human_v1")
anc = load_cl_ancestors(OBO)


def score(pred_name, true_cl):
    pred_cl = np.array([model.class_to_cl.get(p, "") for p in pred_name])
    keep = np.array([bool(c) for c in true_cl])
    return float((pred_cl == true_cl)[keep].mean()), _ontology(true_cl, pred_cl, anc, keep)


def allowed_from_kept(kept_names, group_of):
    allowed_classes = {}
    for grp_id, fm in model.fine.items():
        members = {fm} if isinstance(fm, str) else set(fm.classes)
        k = members & kept_names
        if k:
            allowed_classes[grp_id] = k
    return set(allowed_classes), allowed_classes


RULES = {
    "current (mass>=1 or top1>=1@conf>=0.3)":
        lambda ev, g: set(ev.loc[(ev.mass >= 1.0) | ((ev.top1_count >= 1) & (ev.max_prob >= 0.3)), "class"]),
    "mass_frac>=2% of group":
        lambda ev, g: set(ev.loc[ev.mass / ev.group.map(g) >= 0.02, "class"]),
    "top1_frac>=1% of group @conf>=0.5":
        lambda ev, g: set(ev.loc[(ev.top1_count / ev.group.map(g) >= 0.01) & (ev.max_prob >= 0.5), "class"]),
    "top1_frac>=1% (no conf floor)":
        lambda ev, g: set(ev.loc[ev.top1_count / ev.group.map(g) >= 0.01, "class"]),
    "mass_frac>=2% AND top1_frac>=0.5%":
        lambda ev, g: set(ev.loc[(ev.mass / ev.group.map(g) >= 0.02) & (ev.top1_count / ev.group.map(g) >= 0.005), "class"]),
    "top-90%-coverage per group (elbow)": None,  # handled specially below
}


def elbow_90(ev, group_sizes):
    kept = []
    for g, sub in ev.groupby("group"):
        sub = sub.sort_values("top1_count", ascending=False)
        cum = sub["top1_count"].cumsum()
        thresh = 0.90 * group_sizes[g]
        n_keep = int((cum < thresh).sum()) + 1
        kept.extend(sub["class"].iloc[:n_keep].tolist())
    return set(kept)


all_rows = []
for qname, path in QUERIES.items():
    q = sc.read_h5ad(path)
    true_cl = q.obs["cell_type_ontology_term_id"].astype(str).values
    true_names = set(q.obs["cell_type"].astype(str).unique())
    # oracle set by CL id (consistent with refine_experiment.py)
    cl_to_names = {}
    for name, cl in model.class_to_cl.items():
        cl_to_names.setdefault(cl, set()).add(name)
    oracle_names = set()
    for cl in set(true_cl) - {"unknown", ""}:
        oracle_names |= cl_to_names.get(cl, set())

    _, _, ev = detect_present_classes(model, q, min_mass=1e18, min_top1=10**9, top1_conf=1.1)
    group_sizes = ev.groupby("group")["top1_count"].sum().to_dict()  # cells routed per group

    for rule_name, fn in RULES.items():
        kept = elbow_90(ev, group_sizes) if fn is None else fn(ev, group_sizes)
        tp = len(kept & oracle_names)
        precision = tp / max(len(kept), 1)
        recall = tp / max(len(oracle_names), 1)
        ag, ac = allowed_from_kept(kept, None)
        frame, _ = model.predict_frame(q, allowed_groups=ag, allowed_classes=ac)
        ex, on = score(frame["celltype"].values, true_cl)
        all_rows.append({"query": qname, "rule": rule_name, "n_kept": len(kept),
                         "n_true": len(oracle_names), "precision": precision, "recall": recall,
                         "exact_cl": ex, "ontology": on})
        print(f"{qname:6s} | {rule_name:42s} | kept {len(kept):4d}/{len(oracle_names):3d} true | "
              f"P {precision:.2f} R {recall:.2f} | exact {ex:.3f} onto {on:.3f}", flush=True)

pd.DataFrame(all_rows).to_csv(
    "/Users/iandriver/Downloads/actinn-jax-benchmark/docs/results_refine_threshold_sweep.csv", index=False)
print("\nTHRESHOLD_SWEEP_DONE")
