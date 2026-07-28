"""Post-build gate: does a bundled reference still annotate a held-out atlas sensibly?

A rebuilt reference can load fine, report a plausible class count, and still be worse than
the one it replaces -- a census release shifts the label vocabulary, a gene panel drifts,
a hierarchy degenerates. This scores the *installed* reference on an atlas that was never
part of it and prints numbers comparable across rebuilds, so "should I ship this?" has an
answer instead of a vibe.

Scored by ontology-aware concordance (same-node / ancestor / descendant), because a broad
census vocabulary and an atlas's own vocabulary disagree on names for the same cell. Exact
string match is reported too, but it is the vocabulary artifact, not the signal.

    .venv/bin/python benchmark/explore/verify_reference.py [--name broad_human_v1]
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import actinn_jax as aj
from benchmark import metrics

DEFAULT_QUERY = os.path.expanduser("~/Downloads/krasnow_lung_atlas_10x.h5ad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="broad_human_v1")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--per-label", type=int, default=50)
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--min-prob", type=float, default=0.5)
    a = ap.parse_args()

    model = aj.bundled_reference(a.name)
    info_path = os.path.join(os.path.dirname(aj.hierarchy.__file__), "references",
                             a.name, "build_info.json")
    if os.path.exists(info_path):
        with open(info_path) as fh:
            info = json.load(fh)
        print(f"{a.name}: built {info.get('built_utc')} from "
              f"{info.get('n_cells')} cells / {info.get('n_types')} types "
              f"(census {info.get('census_release', {}).get('release_build', '?')})")
    print(f"loaded: {len(model.classes)} classes, "
          f"{len(set(model.type_to_group.values()))} coarse groups")

    q = sc.read_h5ad(a.query)
    labels = q.obs["cell_type"].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    keep = np.sort(np.concatenate([
        rng.choice(np.where(labels == c)[0],
                   min(a.per_label, int((labels == c).sum())), replace=False)
        for c in np.unique(labels)]))
    q = q[keep].copy()
    print(f"query: {q.shape}, {q.obs.cell_type.nunique()} truth types")

    t = time.time()
    frame = model.predict_frame(q)[0]
    dt = time.time() - t

    pred = frame["celltype"].to_numpy()
    truth = q.obs["cell_type"].astype(str).to_numpy()
    prob = frame["celltype_probability"].to_numpy()
    exact = float((pred == truth).mean())

    onto = float("nan")
    if os.path.exists(a.obo) and "cell_type_ontology_term_id" in q.obs:
        anc = metrics.load_cl_ancestors(a.obo)
        cl_map = model.class_to_cl or {}
        true_cl = q.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
        pred_cl = np.array([cl_map.get(p, "unknown") for p in pred])
        ok = n = 0
        for tcl, pcl in zip(true_cl, pred_cl):
            if not tcl or tcl == "unknown":
                continue
            n += 1
            ok += bool(pcl and pcl != "unknown" and
                       (pcl == tcl or pcl in anc.get(tcl, ()) or tcl in anc.get(pcl, ())))
        onto = ok / n if n else float("nan")
        mapped = float((pred_cl != "unknown").mean())
        print(f"CL coverage of predictions: {mapped:.1%}")

    kept = prob >= a.min_prob
    kept_onto = float("nan")
    if kept.any() and os.path.exists(a.obo) and "cell_type_ontology_term_id" in q.obs:
        sub = np.where(kept)[0]
        ok = n = 0
        for i in sub:
            tcl = true_cl[i]
            if not tcl or tcl == "unknown":
                continue
            n += 1
            pcl = pred_cl[i]
            ok += bool(pcl and pcl != "unknown" and
                       (pcl == tcl or pcl in anc.get(tcl, ()) or tcl in anc.get(pcl, ())))
        kept_onto = ok / n if n else float("nan")

    print(f"\n{'metric':<34}{'value':>10}")
    print("-" * 44)
    print(f"{'exact label match':<34}{exact:>10.3f}")
    print(f"{'ontology concordance (all cells)':<34}{onto:>10.3f}")
    print(f"{f'ontology concordance (p>={a.min_prob})':<34}{kept_onto:>10.3f}")
    print(f"{f'coverage at p>={a.min_prob}':<34}{float(kept.mean()):>10.3f}")
    print(f"{'predict seconds':<34}{dt:>10.2f}")
    print(f"{'cells/s':<34}{q.n_obs / max(dt, 1e-9):>10.0f}")
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
