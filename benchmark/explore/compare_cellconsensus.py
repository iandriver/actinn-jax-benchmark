"""CellConsensus against the reference-trained model, on the same query cells.

CellConsensus (de Mathelin et al. 2026, doi:10.64898/2026.08.07.743503) assigns cell types
from a consensus corpus of marker genes rather than from a labelled reference, so it needs no
training data at all. That also means it answers in its own fixed vocabulary -- three
hierarchical levels, roughly 76 types at the finest -- which is the same situation as the
pretrained annotators in the paper: exact match against a dataset's own label strings would
measure vocabulary overlap, not accuracy.

It exposes `predict(output="cl_id")`, so it can be scored the way those are: ontology-aware
concordance against the query's own Cell Ontology ids, on the identical split the matrix uses.

Two things this script is careful about, both of which silently produce a plausible-looking
wrong answer:

  * **Gene identifiers.** The matrices are Ensembl-keyed and marker sets are symbols. Handing
    Ensembl ids to a marker method scores every marker as absent and yields one constant
    label. var_names are renamed to `feature_name` and the overlap is asserted, not assumed.
  * **Counts.** `fit` documents a raw count matrix and normalises in place, so the raw layer
    is used where the object carries one.

    .venv-cellconsensus/bin/python benchmark/explore/compare_cellconsensus.py --dataset liver_intra
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from benchmark import datasets, metrics

DATASETS = {
    "liver_intra": dict(path="/Volumes/IanSSD/hlica/liver_intra.h5ad",
                        subsample=None, test_frac=0.25),
    "lung_intra": dict(path=os.path.expanduser("~/Downloads/krasnow_lung_atlas_10x.h5ad"),
                       subsample=300, test_frac=0.25),
}
LABEL = "cell_type"
CL = "cell_type_ontology_term_id"


def to_symbols(ad):
    """Marker methods key on symbols; these matrices key on Ensembl ids."""
    if "feature_name" not in ad.var.columns:
        raise SystemExit("no feature_name column: cannot resolve symbols")
    sym = ad.var["feature_name"].astype(str)
    ad = ad[:, ~sym.duplicated().values].copy()
    ad.var_names = ad.var["feature_name"].astype(str).values
    ad.var_names_make_unique()
    return ad


def diagnose(cc, q, truth_cl, query, anc):
    """Is a wrong call the markers' fault or the hierarchy's?

    CellConsensus assigns a level-1 class and then refines only within that branch. A type
    with no level-1 ancestor is therefore unreachable no matter how good its markers are.
    Scoring every level-3 type directly and taking a flat argmax removes the routing while
    holding the marker sets fixed, which separates the two explanations.
    """
    import collections
    import numpy as np
    from cellconsensus.consensus import load_cell_type
    from cellconsensus.consensus.markers import load_cl_to_meta

    M = load_cl_to_meta()
    key2cl = {}
    for cl, rec in M["cl_to_meta"].items():
        k = rec.get("meta3")
        if k and rec.get("name") == M["meta3_groups"].get(k):
            key2cl.setdefault(k, cl)
    for k in M["meta3_groups"]:
        if k not in key2cl:
            for cl, rec in M["cl_to_meta"].items():
                if rec.get("meta3") == k:
                    key2cl[k] = cl
                    break

    keys = [k for k in load_cell_type(level=3) if k in key2cl]
    S = cc.predict_score(keys, level=3)
    flat_key = np.asarray(S.columns[np.asarray(S).argmax(1)]).astype(str)
    flat_cl = np.array([key2cl[k] for k in flat_key])
    ones = np.ones(len(flat_cl), bool)
    conc = float(metrics._ontology(truth_cl, flat_cl, anc, ones))
    print(f"  flat argmax over {len(keys)} level-3 types (no routing): ontology {conc:.3f}",
          flush=True)

    truth = query.obs[LABEL].astype(str).to_numpy()
    hier = np.asarray(cc.predict(level=3, output="name")).astype(str)
    # compare like with like: predict() returns display names, predict_score() internal keys,
    # so "nk" against "natural killer cell" would otherwise read as a change
    flat_key = np.array([M["meta3_groups"].get(k, k) for k in flat_key])
    moved = []
    for t_ in sorted(set(truth)):
        sel = truth == t_
        if sel.sum() < 20:
            continue
        h = collections.Counter(hier[sel]).most_common(1)[0]
        f = collections.Counter(flat_key[sel]).most_common(1)[0]
        if h[0] != f[0]:
            moved.append((t_, int(sel.sum()), h[0], h[1], f[0], f[1]))
    for t_, n, h0, hn, f0, fn in moved:
        print(f"    {t_} (n={n}): hierarchy -> {h0} ({hn}), flat -> {f0} ({fn})", flush=True)
    return [{"dataset": query.uns.get("name", ""), "level": "flat-argmax",
             "ontology": round(conc, 4), "n_labels": int(len(set(flat_key))),
             "coverage": 1.0, "n_query": int(q.n_obs)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="liver_intra", choices=sorted(DATASETS))
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diagnose", action="store_true",
                    help="also score a flat argmax over every level-3 type, which isolates "
                         "the hierarchy from the marker sets")
    a = ap.parse_args()
    cfg = DATASETS[a.dataset]

    ad = sc.read_h5ad(cfg["path"])
    if cfg["subsample"]:
        keep = datasets.stratified_subsample(
            ad.obs[LABEL].astype(str).to_numpy(), cfg["subsample"], seed=0)
        ad = ad[keep].copy()
    _, query = datasets.intra_split(ad, LABEL, cfg["test_frac"], seed=0)
    print(f"{a.dataset}: query {query.n_obs:,} cells, "
          f"{query.obs[LABEL].astype(str).nunique()} truth types", flush=True)

    truth_cl = query.obs[CL].astype(str).to_numpy()
    # raw counts if the object carries them: fit() documents raw and normalises in place
    q = query.raw.to_adata() if query.raw is not None else query.copy()
    q.obs = query.obs.copy()
    q = to_symbols(q)

    from cellconsensus import CellConsensus
    cc = CellConsensus()
    # a marker method handed the wrong identifier space fails silently, so check first
    from cellconsensus.consensus import load_consensus
    ref_genes = set()
    for lvl in (1, 2, 3):
        for genes in load_consensus(level=lvl)["consensus"].values():
            ref_genes |= set(genes)
    overlap = len(ref_genes & set(map(str, q.var_names)))
    frac = overlap / max(len(ref_genes), 1)
    print(f"marker-gene overlap: {overlap}/{len(ref_genes)} ({frac:.0%})", flush=True)
    if frac < 0.2:
        raise SystemExit("marker overlap too low -- identifier mismatch, results would be junk")

    t = time.time()
    cc.fit(q, verbose=False)
    fit_s = time.time() - t

    anc = metrics.load_cl_ancestors(a.obo)
    rows = []
    for lvl in (1, 2, 3):
        t = time.time()
        pred_cl = np.asarray(cc.predict(level=lvl, output="cl_id")).astype(str)
        dt = time.time() - t
        name = np.asarray(cc.predict(level=lvl, output="name")).astype(str)
        keep = np.array([c not in ("", "nan", "None") for c in pred_cl])
        conc = metrics._ontology(truth_cl, pred_cl, anc, keep)
        rows.append({"dataset": a.dataset, "level": lvl, "ontology": round(float(conc), 4),
                     "n_labels": int(len(set(name))), "coverage": round(float(keep.mean()), 3),
                     "fit_s": round(fit_s, 1), "predict_s": round(dt, 2),
                     "n_query": int(q.n_obs)})
        top = ", ".join(f"{k} {v}" for k, v in
                        __import__("collections").Counter(name).most_common(4))
        print(f"  level {lvl}: ontology {conc:.3f} | {len(set(name))} distinct labels | "
              f"mapped {keep.mean():.2f} | top: {top}", flush=True)

    if a.diagnose:
        rows += diagnose(cc, q, truth_cl, query, anc)

    out = a.out or f"/tmp/cellconsensus_{a.dataset}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"fit {fit_s:.1f}s on {q.n_obs:,} cells -> {out}")
    print("CELLCONSENSUS_DONE", flush=True)


if __name__ == "__main__":
    main()
