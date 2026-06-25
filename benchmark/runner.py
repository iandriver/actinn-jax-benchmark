"""Run a single method on one (reference, query) pair under resource monitoring.

This is the isolated unit of work: the driver invokes it as a subprocess (so each
method can live in its own environment). It writes predictions (parquet) and timing
/ memory metrics (json); accuracy is computed later by the driver against truth.

    python -m benchmark.runner --method actinn-jax \
        --ref ref.h5ad --query query.h5ad --label cell_type \
        --out preds.parquet --metrics metrics.json
"""

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
