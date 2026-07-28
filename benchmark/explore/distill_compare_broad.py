"""Three broad annotators, one set of cells: shipped, distilled, and the teacher.

`distill_train.py` scores the student against the teacher on its own held-out arms, which
answers "did distillation work" but not "is the distilled reference better than the one we
ship". This puts `broad_human_v1`, the distilled student, and Pan-human Azimuth's own calls
on **identical cells** with one metric.

The teacher's labels come from the parquet `distill_dump.py` already wrote, so no Keras
process is needed here.

    .venv/bin/python benchmark/explore/distill_compare_broad.py \
        --query /tmp/distill/liver_corpus.h5ad --student /tmp/panhuman_distill_census
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

import actinn_jax as aj
from benchmark import metrics


def concordance(true_cl, pred_cl, anc):
    ok = n = 0
    for t, p in zip(true_cl, pred_cl):
        if not t or t in ("unknown", "nan", ""):
            continue
        n += 1
        ok += bool(p and p not in ("unknown", "unmapped", "nan", "") and
                   (p == t or p in anc.get(t, ()) or t in anc.get(p, ())))
    return (ok / n if n else float("nan")), n


def score_model(model, q, truth_cl, anc, label):
    t0 = time.time()
    frame = model.predict_frame(q)[0]
    dt = time.time() - t0
    pred = frame["celltype"].to_numpy()
    cl_map = model.class_to_cl or {}
    pred_cl = np.array([cl_map.get(p, "unknown") for p in pred])
    onto, n = concordance(truth_cl, pred_cl, anc)
    return {"model": label, "n_classes": len(model.classes),
            "ontology": round(onto, 4), "n_scored": n,
            "mapped_to_cl": round(float((pred_cl != "unknown").mean()), 4),
            "predict_s": round(dt, 2), "cells_per_s": round(q.n_obs / max(dt, 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="/tmp/distill/liver_corpus.h5ad")
    ap.add_argument("--teacher-parquet", default="")
    ap.add_argument("--student", default="/tmp/panhuman_distill_census")
    ap.add_argument("--shipped", default="broad_human_v1")
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--csv", default="docs/results_broad_head_to_head.csv")
    a = ap.parse_args()

    anc = metrics.load_cl_ancestors(a.obo)
    q = sc.read_h5ad(a.query)
    # Accept either a dumped corpus (truth/truth_cl) or a raw atlas (CELLxGENE schema), so
    # the comparison can run on a query that was never part of the distillation corpus.
    cl_col = "truth_cl" if "truth_cl" in q.obs else "cell_type_ontology_term_id"
    lab_col = "truth" if "truth" in q.obs else "cell_type"
    truth_cl = q.obs[cl_col].astype(str).to_numpy()
    print(f"query {q.shape} | {q.obs[lab_col].nunique()} truth types "
          f"| {int((truth_cl != 'unknown').sum())} with CL ids")

    rows = [score_model(aj.bundled_reference(a.shipped), q, truth_cl, anc,
                        f"actinn-jax {a.shipped} (shipped)"),
            score_model(aj.HierarchicalReferenceModel.load(a.student), q, truth_cl, anc,
                        "actinn-jax distilled from PHA")]

    parq = a.teacher_parquet or a.query.replace("_corpus.h5ad", "_teacher.parquet")
    if parq.endswith(".parquet") and os.path.exists(parq):
        meta = pd.read_parquet(parq)
        # The teacher's rows may come from a harness-built split rather than this exact
        # file, so align on cell id instead of assuming the same rows in the same order --
        # a positional zip would silently score the teacher against the wrong cells.
        shared = [c for c in q.obs_names if c in set(meta.index)]
        if len(shared) < q.n_obs:
            print(f"teacher covers {len(shared)}/{q.n_obs} query cells; scoring the overlap")
        m = meta.loc[shared]
        tcl = pd.Series(truth_cl, index=list(q.obs_names)).loc[shared].to_numpy()
        onto, n = concordance(tcl, m["azimuth_fine_CL_ID"].astype(str).to_numpy(), anc)
        rows.append({"model": "Pan-human Azimuth (teacher)",
                     "n_classes": int(m["azimuth_fine"].nunique()),
                     "ontology": round(onto, 4), "n_scored": n,
                     "mapped_to_cl": round(float(
                         (m["azimuth_fine_CL_ID"].astype(str) != "unmapped").mean()), 4),
                     "predict_s": None, "cells_per_s": None})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    df.to_csv(a.csv, index=False)
    print(f"\n{df.to_string(index=False)}\n\nwrote {a.csv}")
    print("COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
