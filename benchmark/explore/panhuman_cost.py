"""Runtime and peak memory of Pan-human Azimuth vs actinn-jax, as query size grows.

Pan-human Azimuth reports ~1,000 cells/s on a MacBook Air (M2, 16 GB) with inference
time linear in dataset size (their Fig. 2F). This checks that on our hardware and puts
it beside actinn-jax, which is the cost claim PAPER.md section 3.4 turns on.

Each (method, size) runs in a fresh subprocess so peak RSS is attributable, using the
same ResourceMonitor the benchmark harness uses. Model load is charged to `fit` for both
methods, prediction to `predict`.

    .venv-protocloud/bin/python benchmark/explore/panhuman_cost.py          # driver
    <venv>/bin/python benchmark/explore/panhuman_cost.py --worker ...       # internal
"""

import argparse
import json
import os
import subprocess
import sys

REPO = "/Users/iandriver/Downloads/actinn-jax-benchmark"
sys.path.insert(0, REPO)

LUNG = os.path.expanduser("~/Downloads/krasnow_lung_atlas_10x.h5ad")
SIZES = [1000, 4000, 16000, 40000]
PYTHON = {
    "actinn-jax": f"{REPO}/.venv-protocloud/bin/python",
    "panhuman-azimuth": f"{REPO}/.venv-panhuman/bin/python",
}


def worker(method_name, n_cells):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import scanpy as sc
    sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
    from benchmark import adapters
    from benchmark.resources import ResourceMonitor

    adata = sc.read_h5ad(LUNG)
    rng = np.random.default_rng(0)
    # A fixed reference for fit; the query grows.
    ref_idx = rng.choice(adata.n_obs, 4000, replace=False)
    rest = np.setdiff1d(np.arange(adata.n_obs), ref_idx)
    q_idx = rng.choice(rest, min(n_cells, len(rest)), replace=False)
    ref, query = adata[ref_idx].copy(), adata[q_idx].copy()

    method = adapters.get(method_name)
    with ResourceMonitor() as m_fit:
        method.fit(ref, "cell_type")
    with ResourceMonitor() as m_pred:
        method.predict(query)

    print("RESULT " + json.dumps({
        "method": method_name, "n_query": int(query.n_obs),
        "fit_s": round(m_fit.elapsed, 2), "predict_s": round(m_pred.elapsed, 2),
        "cells_per_s": round(query.n_obs / max(m_pred.elapsed, 1e-9)),
        "peak_mem_mb": round(max(m_fit.peak_mb, m_pred.peak_mb)),
    }))


def main():
    rows = []
    for n in SIZES:
        for meth, py in PYTHON.items():
            r = subprocess.run(
                [py, __file__, "--worker", "--method", meth, "--n", str(n)],
                cwd=REPO, capture_output=True, text=True)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                print(f"FAILED {meth} n={n}: {r.stderr.strip().splitlines()[-3:]}")
                continue
            rows.append(json.loads(line[len("RESULT "):]))
            print(f"  {meth:18s} n={n:6d}  {rows[-1]['predict_s']:7.2f}s  "
                  f"{rows[-1]['cells_per_s']:6d} cells/s  {rows[-1]['peak_mem_mb']:6d} MB")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = f"{REPO}/docs/results_panhuman_cost.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}\n")
    for meth in PYTHON:
        d = df[df.method == meth]
        if len(d):
            print(f"{meth}:")
            print(d[["n_query", "fit_s", "predict_s", "cells_per_s",
                     "peak_mem_mb"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    worker(a.method, a.n) if a.worker else main()
