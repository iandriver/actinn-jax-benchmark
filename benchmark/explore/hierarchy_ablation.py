"""Is a Cell-Ontology hierarchy doing real work, or would any grouping do?

The census-built reference clusters scPRINT embeddings to get its coarse groups -- the one
GPU step of the build. `ontology_hierarchy.py` proposes clustering Cell Ontology lineage
instead, which is free and species-independent. Comparing that against the *shipped* model
is confounded: the shipped model was built from a smaller corpus. So compare arms built
from the **same** corpus:

  ontology   groups from CL lineage
  random     the same number of groups, types shuffled into them -- the control that
             separates "a hierarchy helps" from "this hierarchy helps"
  flat       one group; a plain classifier over every type

Scored on a held-out atlas by ontology-aware concordance.

    .venv/bin/python benchmark/explore/hierarchy_ablation.py [--ref ...] [--query ...]
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import actinn_jax as aj
from benchmark import metrics
from ontology_hierarchy import ontology_hierarchy


def hvg_subset(a, n):
    raw = a.copy()
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=min(n, raw.n_vars))
    return a[:, raw.var["highly_variable"].values].copy()


def concordance(true_cl, pred_cl, anc):
    ok = n = 0
    for t, p in zip(true_cl, pred_cl):
        if not t or t in ("unknown", "nan", ""):
            continue
        n += 1
        ok += bool(p and p not in ("unknown", "nan", "") and
                   (p == t or p in anc.get(t, ()) or t in anc.get(p, ())))
    return (ok / n if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/actinn_ref_build/census_wide_ref.h5ad")
    ap.add_argument("--query", default=os.path.expanduser(
        "~/Downloads/krasnow_lung_atlas_10x.h5ad"))
    ap.add_argument("--per-label", type=int, default=50)
    ap.add_argument("--n-hvg", type=int, default=4000)
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--csv", default="docs/results_hierarchy_ablation.csv")
    a = ap.parse_args()

    anc = metrics.load_cl_ancestors(a.obo)
    ref = sc.read_h5ad(a.ref)
    labels = ref.obs["cell_type"].astype(str).to_numpy()
    types = np.array(sorted(set(labels)))
    n_groups = max(8, int(round(np.sqrt(len(types)))))
    print(f"reference {ref.shape} | {len(types)} types | G={n_groups}")

    q = sc.read_h5ad(a.query)
    qlab = q.obs["cell_type"].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    keep = np.sort(np.concatenate([
        rng.choice(np.where(qlab == c)[0],
                   min(a.per_label, int((qlab == c).sum())), replace=False)
        for c in np.unique(qlab)]))
    q = q[keep].copy()
    true_cl = q.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
    print(f"query {q.shape} | {q.obs.cell_type.nunique()} truth types")

    onto, info = ontology_hierarchy(ref.obs["cell_type_ontology_term_id"], labels,
                                    n_groups=n_groups, obo=a.obo)
    shuffled = list(onto.values())
    rng.shuffle(shuffled)
    arms = {
        "ontology": onto,
        "random": dict(zip(onto.keys(), shuffled)),   # same group-size profile, no meaning
        "flat": {t: "0" for t in types},
    }

    ref_hvg = hvg_subset(ref, a.n_hvg)
    rows = []
    for name, grp in arms.items():
        t0 = time.time()
        model = aj.build_hierarchical_reference(
            ref_hvg, "cell_type", hierarchy=grp,
            ontology_key="cell_type_ontology_term_id", print_cost=False)
        fit_s = time.time() - t0
        t0 = time.time()
        frame = model.predict_frame(q)[0]
        pred_s = time.time() - t0
        cl_map = model.class_to_cl or {}
        pred_cl = np.array([cl_map.get(p, "unknown")
                            for p in frame["celltype"].to_numpy()])
        onto_score, n = concordance(true_cl, pred_cl, anc)
        rows.append({"hierarchy": name,
                     "n_groups": len(set(model.type_to_group.values())),
                     "n_classes": len(model.classes),
                     "ontology": round(onto_score, 4), "n_scored": n,
                     "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 2),
                     "cells_per_s": round(q.n_obs / max(pred_s, 1e-9))})
        print(f"  {name:<9} ontology={onto_score:.3f}  "
              f"groups={rows[-1]['n_groups']}  fit={fit_s:.0f}s")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    df.to_csv(a.csv, index=False)
    print(f"\n{df.to_string(index=False)}\nwrote {a.csv}")
    print("HIERARCHY_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
