"""Gather the per-run query-scaling JSONs into one tidy CSV, one row per run.

Every (method, size, repeat) is a separate process, so the results arrive as separate files.
Repeats are kept as rows rather than averaged here: the plot needs the spread, and a mean
written to disk cannot be un-averaged later.

    .venv-protocloud/bin/python benchmark/explore/collect_query_scaling.py \
        --glob '/tmp/qs_reps/*_rep*.json' --out docs/results_query_scaling.csv
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

PAT = re.compile(r"(?P<method>.+)_(?P<n>\d+)_rep(?P<rep>\d+)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/tmp/qs_reps/*_rep*.json")
    ap.add_argument("--out", default="docs/results_query_scaling.csv")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(a.glob)):
        m = PAT.search(os.path.basename(path))
        if not m:                      # only the per-repeat runs, not stray json in the dir
            continue
        for r in json.load(open(path)):
            r["rep"] = int(m.group("rep"))
            rows.append(r)
    if not rows:
        raise SystemExit(f"no runs matched {a.glob}")

    d = pd.DataFrame(rows).sort_values(["method", "n_query", "rep"])
    cols = ["method", "n_query", "rep", "predict_s", "cells_per_s", "peak_gb", "fit_s",
            "n_ref", "acc_withheld_study", "n_withheld"]
    d = d[[c for c in cols if c in d.columns]]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d.to_csv(a.out, index=False)

    g = d.groupby(["method", "n_query"])
    print(f"{len(d)} runs -> {a.out}\n")
    for (meth, n), sub in g:
        print(f"{meth:<18} n={n:>7,}  reps={len(sub)}  "
              f"predict {sub.predict_s.mean():7.1f}s "
              f"(min {sub.predict_s.min():.1f}, max {sub.predict_s.max():.1f}, "
              f"spread {sub.predict_s.max()/max(sub.predict_s.min(), 1e-9):.2f}x)  "
              f"peak {sub.peak_gb.mean():5.1f} GB  acc {sub.acc_withheld_study.mean():.4f}")
    print("\nCOLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
