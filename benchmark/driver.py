"""Config-driven benchmark driver.

Reads a YAML config describing datasets, splits, and methods, then for each
(dataset, method, repeat) runs the method in a subprocess (``benchmark.runner``),
collects predictions + timing/memory, scores accuracy against truth, and writes a
tidy results table (CSV + parquet).

    python -m benchmark.driver configs/smoke.yaml
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from . import datasets, metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL_ID = "cell_type_ontology_term_id"


def _expand(p):
    return os.path.expanduser(p) if p else p


def _load_maybe_subsampled(path, label, max_per_label):
    """Load an h5ad; if subsampling, do it in backed mode to bound memory."""
    path = _expand(path)
    if not max_per_label:
        return sc.read_h5ad(path)
    backed = sc.read_h5ad(path, backed="r")
    sel = datasets.stratified_subsample(
        np.asarray(backed.obs[label].values), max_per_label
    )
    sub = backed[sel].to_memory()
    backed.file.close()
    return sub


def build_pair(ds, label):
    """Return (ref, query) AnnData for a dataset spec."""
    kind = ds["type"]
    if kind == "synthetic":
        ref = datasets.make_synthetic(
            ds.get("n_per_type", 300), ds.get("n_genes", 1200),
            ds.get("n_types", 6), seed=1)
        query = datasets.make_synthetic(
            ds.get("n_per_type", 300), ds.get("n_genes", 1200),
            ds.get("n_types", 6), seed=2)
        return ref, query
    if kind == "intra":
        adata = _load_maybe_subsampled(ds["path"], label, ds.get("subsample_per_label"))
        return datasets.intra_split(adata, label, ds.get("test_frac", 0.25))
    if kind == "cross":
        ref = _load_maybe_subsampled(
            ds["ref_path"], label, ds.get("ref_subsample_per_label"))
        query = _load_maybe_subsampled(
            ds["query_path"], label, ds.get("query_subsample_per_label"))
        return ref, query
    raise ValueError(f"Unknown dataset type: {kind}")


def run(config_path):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    label = cfg.get("label", "cell_type")
    repeats = cfg.get("repeats", 1)
    out_dir = _expand(cfg.get("output", "results/run"))
    os.makedirs(out_dir, exist_ok=True)

    anc = None
    if cfg.get("ontology_obo"):
        anc = metrics.load_cl_ancestors(_expand(cfg["ontology_obo"]))

    rows = []
    for ds in cfg["datasets"]:
        print(f"\n=== dataset: {ds['name']} ({ds['type']}) ===", flush=True)
        ref, query = build_pair(ds, label)
        truth = np.asarray(query.obs[label].values)
        # Optional ontology mapping for cross-vocabulary concordance.
        truth_cl = pred_cl_map = None
        if anc is not None and CL_ID in ref.obs and CL_ID in query.obs:
            truth_cl = query.obs[CL_ID].astype(str).to_numpy()
            pred_cl_map = dict(zip(ref.obs[label].astype(str), ref.obs[CL_ID].astype(str)))

        with tempfile.TemporaryDirectory() as work:
            ref_p = os.path.join(work, "ref.h5ad")
            qry_p = os.path.join(work, "query.h5ad")
            ref.write_h5ad(ref_p)
            query.write_h5ad(qry_p)

            for m in cfg["methods"]:
                name = m["name"]
                python = m.get("python", sys.executable)
                for rep in range(repeats):
                    preds_p = os.path.join(work, f"{name}_{rep}.parquet")
                    met_p = os.path.join(work, f"{name}_{rep}.json")
                    cmd = [python, "-m", "benchmark.runner",
                           "--method", name, "--ref", ref_p, "--query", qry_p,
                           "--label", label, "--out", preds_p, "--metrics", met_p]
                    print(f"  run {name} (rep {rep})", flush=True)
                    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"    FAILED: {r.stderr.strip().splitlines()[-1:]}", flush=True)
                        rows.append({"dataset": ds["name"], "method": name, "repeat": rep,
                                     "error": r.stderr.strip()[-500:]})
                        continue

                    preds = pd.read_parquet(preds_p).set_index("cell_id").loc[list(query.obs_names)]
                    with open(met_p) as fh:
                        meta = json.load(fh)
                    pred_cl = (np.array([pred_cl_map.get(str(p), "") for p in preds["pred_label"]])
                               if pred_cl_map is not None else None)
                    acc = metrics.compute(
                        truth, preds["pred_label"].to_numpy(),
                        unassigned=preds["unassigned"].to_numpy() if "unassigned" in preds else None,
                        ontology=anc, truth_cl=truth_cl, pred_cl=pred_cl)
                    rows.append({"dataset": ds["name"], "repeat": rep, **meta, **acc})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    try:
        results.to_parquet(os.path.join(out_dir, "results.parquet"))
    except Exception:
        pass
    print(f"\nwrote {len(results)} rows to {out_dir}/results.csv")
    cols = [c for c in ["dataset", "method", "accuracy", "macro_f1",
                        "ontology_concordance", "fit_s", "predict_s", "peak_mem_mb"]
            if c in results.columns]
    if cols:
        print(results[cols].to_string(index=False))
    return results


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml")
