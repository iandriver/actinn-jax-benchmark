"""Diagnose the large-reference (798-type) vs. narrowed-reference gap, and evaluate
refine_to_query as the practical, no-ground-truth-needed fix.

Uses the shipped broad_human_v1 model + two real ground-truth queries (lung, liver).
Compares:
  baseline        - unrestricted 798-type model (what ships today)
  oracle-mask     - masked/renormalized to the TRUE type set (ceiling for masking alone,
                    needs ground truth -- not available to a real user)
  detected-refine - aj.refine_to_query's own presence detection (the real deliverable,
                    no ground truth used)
  oracle-retrain  - actually retrains the small MLP on just the TRUE type subset, reusing
                    the shipped model's own coarse grouping (fair ablation: does retraining
                    beat masking the same classifier?), from the cached labeled reference

Also breaks accuracy into coarse-routing vs. fine-given-correct-coarse, to separate
"wrong bucket" errors (unrecoverable by narrowing) from "right bucket, wrong sibling"
errors (exactly what narrowing fixes).
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark/benchmark")
import actinn_jax as aj
from metrics import load_cl_ancestors, _ontology

OBO = "/tmp/cl-basic.obo"
REF_H5AD = "/tmp/census_wide_ref.h5ad"  # cached labeled reference (not shipped) -- oracle-retrain only

QUERIES = {
    "lung (krasnow, 46 types)": "/Users/iandriver/Downloads/krasnow_lung_atlas_10x.h5ad",
    "liver (12 types incl. zonation)": "/tmp/liver_zonation_query.h5ad",
}

model = aj.bundled_reference("broad_human_v1")
anc = load_cl_ancestors(OBO)
cl_to_names = {}
for name, cl in model.class_to_cl.items():
    cl_to_names.setdefault(cl, set()).add(name)
cl_to_group = {}
for name, grp in model.type_to_group.items():
    cl = model.class_to_cl.get(name)
    if cl:
        cl_to_group.setdefault(cl, set()).add(grp)

print(f"reference: {len(model.classes)} types / {len(set(model.type_to_group.values()))} coarse groups\n")


def score(pred_name, true_cl):
    pred_cl = np.array([model.class_to_cl.get(p, "") for p in pred_name])
    keep = np.array([bool(c) for c in true_cl])
    exact_cl = (pred_cl == true_cl)[keep].mean() if keep.any() else float("nan")
    onto = _ontology(true_cl, pred_cl, anc, keep)
    return exact_cl, onto, keep.mean()


def build_allowed(true_cls):
    """Every model class whose CL id is among the query's true types -> oracle set."""
    allowed_names = set()
    for cl in true_cls:
        allowed_names |= cl_to_names.get(cl, set())
    allowed_classes = {}
    for grp_id, fm in model.fine.items():
        members = {fm} if isinstance(fm, str) else set(fm.classes)
        kept = members & allowed_names
        if kept:
            allowed_classes[grp_id] = kept
    return set(allowed_classes), allowed_classes, allowed_names


def oracle_retrain(ref, allowed_names):
    sub = ref[ref.obs["cell_type"].astype(str).isin(allowed_names)].copy()
    hier = {t: g for t, g in model.type_to_group.items() if t in allowed_names}
    return aj.build_hierarchical_reference(sub, "cell_type", hierarchy=hier,
                                           ontology_key="cell_type_ontology_term_id",
                                           print_cost=False)


ref = None
rows = []
for qname, path in QUERIES.items():
    print(f"=== {qname} ===")
    q = sc.read_h5ad(path)
    true_cl = q.obs["cell_type_ontology_term_id"].astype(str).values
    true_cl_set = set(true_cl) - {"unknown", ""}
    n_true_types = len(true_cl_set)

    # -- baseline (unrestricted) --
    frame, _ = model.predict_frame(q)
    ex, on, cov = score(frame["celltype"].values, true_cl)
    rows.append({"query": qname, "method": "baseline (798 types)", "exact_cl": ex,
                "ontology": on, "coverage": cov, "n_classes": len(model.classes)})

    # coarse-vs-fine diagnostic (only over cells whose true type IS in the 798 vocab)
    true_grp = np.array([next(iter(cl_to_group.get(c, {""})), "") for c in true_cl])
    known = true_grp != ""
    coarse_ok = (frame["coarse"].values == true_grp)
    coarse_acc = coarse_ok[known].mean()
    fine_given_right = (frame["celltype"].values[known & coarse_ok] ==
                        np.array([next(iter(cl_to_names.get(c, {""})), "") for c in true_cl])[known & coarse_ok])
    print(f"  known-to-reference: {known.mean():.2f} of cells | coarse-routing acc: {coarse_acc:.2f} "
          f"| fine-acc GIVEN correct coarse: {fine_given_right.mean():.2f}")

    # -- oracle-mask (ground truth -> mask, no retrain) --
    allowed_groups, allowed_classes, allowed_names = build_allowed(true_cl_set)
    oframe, _ = model.predict_frame(q, allowed_groups=allowed_groups, allowed_classes=allowed_classes)
    ex, on, cov = score(oframe["celltype"].values, true_cl)
    rows.append({"query": qname, "method": "oracle-mask (ground truth)", "exact_cl": ex,
                "ontology": on, "coverage": cov, "n_classes": len(allowed_names)})

    # -- detected-refine (the real deliverable; no ground truth) --
    refined = aj.refine_to_query(model, q)
    detected_names = set().union(*refined.allowed_classes.values()) if refined.allowed_classes else set()
    tp = len(detected_names & allowed_names)
    precision = tp / max(len(detected_names), 1)
    recall = tp / max(len(allowed_names), 1)
    dframe, _ = refined.predict_frame(q)
    ex, on, cov = score(dframe["celltype"].values, true_cl)
    rows.append({"query": qname, "method": "detected-refine (no ground truth)", "exact_cl": ex,
                "ontology": on, "coverage": cov, "n_classes": len(detected_names),
                "precision_vs_oracle": precision, "recall_vs_oracle": recall})
    missed = allowed_names - detected_names
    print(f"  detector vs oracle: precision {precision:.2f} recall {recall:.2f} "
          f"({len(detected_names)} kept vs {len(allowed_names)} true) | missed: {sorted(missed)[:5]}")

    # -- oracle-retrain (needs the cached labeled reference; heaviest, most "correct") --
    if ref is None:
        ref = sc.read_h5ad(REF_H5AD)
    ormodel = oracle_retrain(ref, allowed_names)
    rframe, _ = ormodel.predict_frame(q)
    ex, on, cov = score(rframe["celltype"].values, true_cl)
    rows.append({"query": qname, "method": "oracle-retrain (ground truth + cached ref)",
                "exact_cl": ex, "ontology": on, "coverage": cov, "n_classes": len(allowed_names)})
    print()

res = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(res.round(3).to_string(index=False))
res.to_csv("/Users/iandriver/Downloads/actinn-jax-benchmark/docs/results_refine.csv", index=False)
print("\nREFINE_EXPERIMENT_DONE")
