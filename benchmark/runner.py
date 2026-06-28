"""Run a single method on one (reference, query) pair under resource monitoring.

This is the isolated unit of work: the driver invokes it as a subprocess (so each
method can live in its own environment). It writes predictions (parquet) and timing
/ memory metrics (json); accuracy is computed later by the driver against truth.

    python -m benchmark.runner --method actinn-jax \
        --ref ref.h5ad --query query.h5ad --label cell_type \
        --out preds.parquet --metrics metrics.json
"""

import os

# Cap math-library threads BEFORE numpy/scanpy import. On macOS, Apple Accelerate
# oversubscribes worker threads in this subprocess context, causing pathological,
# nondeterministic slowdowns (many threads parked in cvwait at ~1 core of useful
# work). A modest fixed cap removes the thrash and makes timing reproducible. A
# config-driven cap from the driver still wins via setdefault.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json

import pandas as pd

from . import adapters, datasets
from .resources import ResourceMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    ref = datasets.load(args.ref)
    query = datasets.load(args.query)
    method = adapters.get(args.method)

    with ResourceMonitor() as m_fit:
        method.fit(ref, args.label)
    with ResourceMonitor() as m_pred:
        preds = method.predict(query)

    df = pd.DataFrame({"cell_id": preds.cell_ids, "pred_label": preds.labels})
    if preds.probabilities is not None:
        df["probability"] = preds.probabilities
    if preds.unassigned is not None:
        df["unassigned"] = preds.unassigned
    if preds.label_cl is not None:
        df["pred_cl"] = preds.label_cl
    df.to_parquet(args.out)

    with open(args.metrics, "w") as fh:
        json.dump({
            "method": method.name,
            "tier": method.tier,
            "device": method.device,
            "n_ref": int(ref.n_obs),
            "n_query": int(query.n_obs),
            "fit_s": m_fit.elapsed,
            "predict_s": m_pred.elapsed,
            "peak_mem_mb": max(m_fit.peak_mb, m_pred.peak_mb),
        }, fh)


if __name__ == "__main__":
    main()
