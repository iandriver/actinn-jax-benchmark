"""Stage 2 of Pan-human Azimuth distillation: train the actinn-jax student.

Reads the corpus + teacher labels written by ``distill_dump.py`` and trains a
``HierarchicalReferenceModel`` in which:

  * the **classes** are Pan-human Azimuth's refined fine labels (its harmonized,
    CL-mapped vocabulary, including its trained ``Unassigned`` quality-control class), and
  * the **hierarchy** is Pan-human Azimuth's own broad level -- not a Ward clustering of
    scPRINT centroids. The teacher already knows its coarse->fine structure, so the
    student inherits it and the build needs no foundation model and no GPU.

Two questions, two arms:

  in-corpus    held-out cells from the same atlases. Measures distillation fidelity:
               how often does the student reproduce the teacher it was trained on?
  held-out     an entire atlas withheld from training. Measures whether the student
               generalizes to tissue the distillation corpus never covered, which is the
               question that decides whether a distilled *pan-human* model is possible
               from a corpus smaller than the teacher's 9.7M cells.

Both arms also score student and teacher against the atlases' own labels by ontology-aware
concordance, so distillation loss is separated from teacher error.

    .venv/bin/python benchmark/explore/distill_train.py [--dump /tmp/distill]
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import actinn_jax as aj
from benchmark import metrics

TEACHER_LABEL = "azimuth_fine"
TEACHER_GROUP = "azimuth_broad"
TEACHER_CL = "azimuth_fine_CL_ID"


def load_corpus(dump):
    """Concatenate every dumped atlas on the shared Ensembl gene space."""
    parts = []
    for f in sorted(os.listdir(dump)):
        if not f.endswith("_corpus.h5ad"):
            continue
        name = f[: -len("_corpus.h5ad")]
        a = sc.read_h5ad(f"{dump}/{f}")
        t = pd.read_parquet(f"{dump}/{name}_teacher.parquet")
        assert list(t.index) == list(a.obs_names), f"{name}: teacher/corpus misaligned"
        for c in (TEACHER_LABEL, TEACHER_GROUP, TEACHER_CL):
            a.obs[c] = t[c].astype(str).values
        parts.append(a)
        print(f"  {name}: {a.shape}, {a.obs[TEACHER_LABEL].nunique()} teacher labels")
    corpus = ad.concat(parts, join="inner", merge="first")
    corpus.obs_names_make_unique()
    print(f"corpus {corpus.shape} on {corpus.n_vars} shared genes, "
          f"{corpus.obs[TEACHER_LABEL].nunique()} teacher labels, "
          f"{corpus.obs[TEACHER_GROUP].nunique()} teacher groups")
    return corpus


def hvg_subset(a, n):
    raw = a.copy()
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=min(n, raw.n_vars))
    return a[:, raw.var["highly_variable"].values].copy()


def drop_singletons(a, min_cells):
    """A class with one or two cells cannot be split into train and test; keeping it
    makes the fidelity number depend on which side of the split its cells landed."""
    vc = a.obs[TEACHER_LABEL].value_counts()
    keep = a.obs[TEACHER_LABEL].isin(vc[vc >= min_cells].index).to_numpy()
    if (~keep).any():
        print(f"  dropped {int((~keep).sum())} cells in "
              f"{int((vc < min_cells).sum())} teacher labels with < {min_cells} cells")
    return a[keep].copy()


def stratified_test(labels, frac, seed=0):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        test[rng.choice(idx, max(1, int(round(len(idx) * frac))), replace=False)] = True
    return test


def concordance(true_cl, pred_cl, anc):
    ok = n = 0
    for t, p in zip(true_cl, pred_cl):
        if not t or t in ("unknown", "nan", ""):
            continue
        n += 1
        ok += bool(p and p not in ("unknown", "unmapped", "nan", "") and
                   (p == t or p in anc.get(t, ()) or t in anc.get(p, ())))
    return ok / n if n else float("nan")


def train_and_score(train, test, n_hvg, anc, arm, cl_of):
    hierarchy = dict(zip(train.obs[TEACHER_LABEL].astype(str),
                         train.obs[TEACHER_GROUP].astype(str)))
    t0 = time.time()
    model = aj.build_hierarchical_reference(
        hvg_subset(train, n_hvg), TEACHER_LABEL, hierarchy=hierarchy,
        ontology_key=TEACHER_CL, print_cost=False)
    fit_s = time.time() - t0

    t0 = time.time()
    frame = model.predict_frame(test)[0]
    pred_s = time.time() - t0
    student = frame["celltype"].to_numpy()
    teacher = test.obs[TEACHER_LABEL].astype(str).to_numpy()

    truth_cl = test.obs["truth_cl"].astype(str).to_numpy()
    teacher_cl = test.obs[TEACHER_CL].astype(str).to_numpy()
    student_cl = np.array([cl_of.get(s, "unknown") for s in student])
    scored = ~pd.Series(truth_cl).isin(["unknown", "nan", ""]).to_numpy()

    row = {
        "arm": arm,
        "n_train": int(train.n_obs), "n_test": int(test.n_obs),
        "n_classes": len(model.classes),
        "n_groups": len(set(model.type_to_group.values())),
        "fidelity_exact": round(float((student == teacher).mean()), 4),
        "fidelity_ontology": round(concordance(teacher_cl, student_cl, anc), 4),
        "student_vs_truth": round(concordance(truth_cl[scored], student_cl[scored], anc), 4),
        "teacher_vs_truth": round(concordance(truth_cl[scored], teacher_cl[scored], anc), 4),
        "n_scored": int(scored.sum()),
        "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 2),
        "cells_per_s": round(test.n_obs / max(pred_s, 1e-9)),
    }
    return model, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/distill")
    ap.add_argument("--n-hvg", type=int, default=4000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--min-cells", type=int, default=8)
    ap.add_argument("--holdout", default="liver", help="corpus withheld for the second arm")
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--out", default="/tmp/panhuman_distill_v1",
                    help="where to save the model trained on everything")
    ap.add_argument("--csv", default="docs/results_panhuman_distill.csv")
    a = ap.parse_args()

    anc = metrics.load_cl_ancestors(a.obo)
    corpus = drop_singletons(load_corpus(a.dump), a.min_cells)
    cl_of = dict(zip(corpus.obs[TEACHER_LABEL].astype(str),
                     corpus.obs[TEACHER_CL].astype(str)))
    rows = []

    # arm 1 -- in-corpus fidelity
    test = stratified_test(corpus.obs[TEACHER_LABEL].astype(str).to_numpy(), a.test_frac)
    print(f"\n[in-corpus] train {int((~test).sum())} / test {int(test.sum())}")
    model, row = train_and_score(corpus[~test].copy(), corpus[test].copy(),
                                 a.n_hvg, anc, "in_corpus", cl_of)
    rows.append(row)
    print(json.dumps(row, indent=2))

    # arm 2 -- an entire atlas withheld
    if a.holdout and (corpus.obs["corpus"] == a.holdout).any():
        ho = (corpus.obs["corpus"] == a.holdout).to_numpy()
        print(f"\n[held-out {a.holdout}] train {int((~ho).sum())} / test {int(ho.sum())}")
        _, row = train_and_score(corpus[~ho].copy(), corpus[ho].copy(),
                                 a.n_hvg, anc, f"heldout_{a.holdout}", cl_of)
        rows.append(row)
        print(json.dumps(row, indent=2))

    # ship: retrain on everything. The arms above exist to decide whether to trust this
    # model, not to be it -- shipping the 75% arm would throw away a quarter of the corpus.
    print("\n[ship] training on the full corpus")
    hierarchy = dict(zip(corpus.obs[TEACHER_LABEL].astype(str),
                         corpus.obs[TEACHER_GROUP].astype(str)))
    t0 = time.time()
    model = aj.build_hierarchical_reference(
        hvg_subset(corpus, a.n_hvg), TEACHER_LABEL, hierarchy=hierarchy,
        ontology_key=TEACHER_CL, print_cost=False)
    print(f"  {len(model.classes)} classes / "
          f"{len(set(model.type_to_group.values()))} groups in {time.time()-t0:.0f}s")
    model.save(a.out)
    sz = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)
             if os.path.isfile(os.path.join(a.out, f))) / 1e6
    # The teacher's weights are CC BY 4.0, so credit is a licence term, not a courtesy.
    # It is written into the model directory itself: a reference that travels without its
    # build_info is a reference that travels without its attribution.
    with open(os.path.join(a.out, "build_info.json"), "w") as fh:
        json.dump({"name": os.path.basename(a.out),
                   "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   # Pan-human Azimuth is human-only, so any model distilled from it is.
                   # Recorded so the package can refuse a cross-species query outright.
                   "organism": "homo_sapiens",
                   "teacher": {
                       "model": "Pan-human Azimuth",
                       "citation": "Sarkar, Li, Molla, ... Satija. Organism-scale "
                                   "annotation with Pan-human Azimuth. bioRxiv 2026.",
                       "doi": "10.64898/2026.07.16.738997",
                       "package": "panhumanpy 1.0.0 (MIT)",
                       "weights": "https://doi.org/10.5281/zenodo.20401417",
                       "weights_license": "CC BY 4.0",
                       "attribution_required": True,
                   },
                   "teacher_label": TEACHER_LABEL, "teacher_hierarchy": TEACHER_GROUP,
                   "n_cells": int(corpus.n_obs), "n_classes": len(model.classes),
                   "n_hvg": a.n_hvg, "arms": rows}, fh, indent=2)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    df.to_csv(a.csv, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nsaved model to {a.out} ({sz:.1f} MB); wrote {a.csv}")
    print("DISTILL_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
