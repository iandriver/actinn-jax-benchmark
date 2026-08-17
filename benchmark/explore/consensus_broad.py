"""Do several broad references agree, and is agreeing worth anything?

The workflow argument in the paper is that cheap reference calling changes what is
practical: if a broad pass costs a fraction of a second, a user can run *several* references
over the same cells rather than choosing one in advance. That argument has been asserted and
never tested. This tests it.

Three broad annotators answer over the identical 3,396 withheld cross-study liver cells --
the shipped census reference (798 classes), the model distilled from Pan-human Azimuth (324),
and Azimuth itself (382) -- in three different vocabularies. They are only comparable in the
Cell Ontology, where all three map, so agreement is defined there: two calls agree when they
are the same term or one is an ancestor of the other, the same relation the paper's
concordance metric uses.

Two questions, and the second is the one that matters:

  1. Does a consensus call beat the best single reference?
  2. Does *agreement itself* predict correctness? If cells where the references agree are
     reliably right and cells where they disagree are not, then agreement is a confidence
     signal that costs no ground truth -- which is the thing a user actually needs and
     cannot otherwise get from an unlabelled query.

Run in two passes, because Azimuth is Keras and lives in its own environment:

    .venv-panhuman/bin/python benchmark/explore/consensus_broad.py --dump-teacher
    .venv/bin/python benchmark/explore/consensus_broad.py
"""

import argparse
import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from benchmark import metrics

QUERY = "/Volumes/IanSSD/hlica/liver_query_xstudy.h5ad"
TEACHER_DUMP = "/tmp/consensus_teacher_cl.parquet"
BAD = {"", "unknown", "unmapped", "nan", "None"}


def clean(x):
    x = str(x)
    return "" if x in BAD else x


def compatible(a, b, anc):
    """Same term, or one is an ancestor of the other -- the paper's concordance relation."""
    if not a or not b:
        return False
    return a == b or a in anc.get(b, ()) or b in anc.get(a, ())


def consensus(calls, anc, fallback=None):
    """Largest mutually-compatible subset; its most specific member is the call.

    Specificity is ancestor count, so the consensus keeps the most informative label the
    agreeing references support rather than retreating to their common ancestor -- retreating
    would make agreement look good for the trivial reason that vaguer calls are easier to be
    consistent with.

    ``fallback`` is the index of the reference to trust when nothing agrees. Without it the
    rule would default to whichever reference happens to be listed first, which would report a
    consensus that is worse than the best single model for a reason that is an artefact of
    argument order rather than of the idea.
    """
    named = [(i, c) for i, c in enumerate(calls) if c]
    if not named:
        return "", 0
    best = []
    for k in range(len(named), 0, -1):
        for sub in itertools.combinations(named, k):
            if all(compatible(x, y, anc) for (_, x), (_, y) in itertools.combinations(sub, 2)):
                best = list(sub)
                break
        if best:
            break
    if len(best) == 1 and fallback is not None:
        pick = dict(named).get(fallback)
        if pick:
            return pick, 1
    return max(best, key=lambda ic: len(anc.get(ic[1], ())))[1], len(best)


def dump_teacher(a):
    """Pan-human Azimuth's per-cell CL ids, so the core-venv pass can read them."""
    from benchmark import adapters
    q = sc.read_h5ad(a.query)
    m = adapters.get("panhuman-azimuth")
    m.fit(q, "cell_type")                      # warms weights only; does no training
    p = m.predict(q)
    cl = getattr(p, "label_cl", None)
    if cl is None:
        raise SystemExit("adapter returned no label_cl; cannot score in ontology space")
    df = pd.DataFrame({"cl": [clean(c) for c in cl]}, index=list(q.obs_names))
    df.to_parquet(a.teacher)
    print(f"wrote {a.teacher}: {len(df)} cells, "
          f"{(df.cl != '').mean():.1%} mapped to CL", flush=True)
    print("TEACHER_DUMP_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--teacher", default=TEACHER_DUMP)
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--out", default="docs/results_consensus_broad.csv")
    ap.add_argument("--dump-teacher", action="store_true")
    a = ap.parse_args()
    if a.dump_teacher:
        return dump_teacher(a)

    import actinn_jax as aj
    anc = metrics.load_cl_ancestors(a.obo)
    q = sc.read_h5ad(a.query)
    truth = np.array([clean(c) for c in q.obs["cell_type_ontology_term_id"].astype(str)])
    print(f"query {q.shape} | {int((truth != '').sum())} cells with a truth CL id", flush=True)

    preds = {}
    for label, ref in (("census (broad_human_v1)", "broad_human_v1"),
                       ("distilled (panhuman_distill_v1)", "panhuman_distill_v1")):
        model = aj.bundled_reference(ref)
        frame = model.predict_frame(q)[0]
        cmap = model.class_to_cl or {}
        preds[label] = np.array([clean(cmap.get(p, "")) for p in frame["celltype"]])
        print(f"  {label}: {len(model.classes)} classes, "
              f"{(preds[label] != '').mean():.1%} mapped", flush=True)

    if os.path.exists(a.teacher):
        t = pd.read_parquet(a.teacher)
        # align on cell id: a positional join would silently score the wrong cells
        shared = [c for c in q.obs_names if c in set(t.index)]
        assert len(shared) == q.n_obs, f"teacher covers {len(shared)}/{q.n_obs} cells"
        preds["Pan-human Azimuth"] = t.loc[list(q.obs_names), "cl"].fillna("").to_numpy()
        print(f"  Pan-human Azimuth: {(preds['Pan-human Azimuth'] != '').mean():.1%} mapped",
              flush=True)
    else:
        print(f"  (no teacher dump at {a.teacher}; running on two references)", flush=True)

    names = list(preds)
    keep = truth != ""
    rows = []
    for n in names:
        rows.append({"model": n, "subset": "all cells", "n": int(keep.sum()),
                     "coverage": 1.0,
                     "ontology": round(float(metrics._ontology(
                         truth, preds[n], anc, keep)), 4)})

    # Two rules: naive (deepest agreeing call, first reference when nothing agrees) and one
    # that falls back to the strongest single reference. Reporting only the naive rule would
    # blame the consensus idea for the tie-break.
    single = {n: float(metrics._ontology(truth, preds[n], anc, keep)) for n in names}
    fb = names.index(max(single, key=single.get))
    print(f"\nfallback reference when nothing agrees: {names[fb]} "
          f"({single[names[fb]]:.4f} alone)", flush=True)
    cons, level = zip(*[consensus([preds[n][i] for n in names], anc, fallback=fb)
                        for i in range(q.n_obs)])
    cons, level = np.array(cons), np.array(level)
    naive = np.array([consensus([preds[n][i] for n in names], anc)[0]
                      for i in range(q.n_obs)])
    for lab, arr in (("consensus", cons), ("consensus (naive tie-break)", naive)):
        rows.append({"model": lab, "subset": "all cells", "n": int(keep.sum()),
                     "coverage": 1.0,
                     "ontology": round(float(metrics._ontology(truth, arr, anc, keep)), 4)})

    # The signal that needs no ground truth: does agreement itself predict correctness?
    for lv in sorted(set(level), reverse=True):
        sel = keep & (level == lv)
        if not sel.any():
            continue
        tag = f"{lv} of {len(names)} references agree"
        rows.append({"model": "consensus", "subset": tag, "n": int(sel.sum()),
                     "coverage": round(float(sel.sum() / keep.sum()), 4),
                     "ontology": round(float(metrics._ontology(truth, cons, anc, sel)), 4)})
        for n in names:
            rows.append({"model": n, "subset": tag, "n": int(sel.sum()),
                         "coverage": round(float(sel.sum() / keep.sum()), 4),
                         "ontology": round(float(metrics._ontology(
                             truth, preds[n], anc, sel)), 4)})

    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d.to_csv(a.out, index=False)
    print("\n" + d.to_string(index=False))
    print(f"\nwrote {a.out}")
    print("CONSENSUS_DONE", flush=True)


if __name__ == "__main__":
    main()
