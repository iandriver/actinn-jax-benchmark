"""Run one Open Problems label_projection component on one dataset and score it.

The components are viash scripts: a `par` dict between ## VIASH START / ## VIASH END that
the framework replaces at build time. This substitutes the same dict and executes the script
unmodified, so what runs here is what ran on the OP harness.

Accuracy and macro-F1 do not depend on the machine, which is the whole reason this can
complete the coverage locally. Runtime and memory do, and are not reported from here.

    python op_run.py <method> <dataset> <out.json>
"""
import json
import os
import re
import runpy
import sys
import time

DATA = "/Volumes/IanSSD/op_label_projection_hvg"
COMPONENTS = "/Users/iandriver/Downloads/actinn-jax-benchmark/openproblems_component"


def run(method, dataset, work):
    src = open(f"{COMPONENTS}/{method}/script.py").read()
    par = {
        "input_train": f"{DATA}/{dataset}/train.h5ad",
        "input_test": f"{DATA}/{dataset}/test.h5ad",
        "output": os.path.join(work, "output.h5ad"),
        "n_hvg": 1000,
        # defaults carried from each component's own VIASH block
        "min_expr_frac": 0.10, "n_genes": 20000, "n_pcs": 220,
    }
    inject = f"par = {par!r}\nmeta = {{'name': {method!r}}}\n"
    patched = re.sub(r"## VIASH START.*?## VIASH END", inject, src, flags=re.S)
    path = os.path.join(work, "script.py")
    with open(path, "w") as fh:
        fh.write(patched)
    t0 = time.time()
    runpy.run_path(path, run_name="__main__")
    return time.time() - t0, par["output"]


def score(out_h5ad, dataset):
    import anndata as ad
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    pred = ad.read_h5ad(out_h5ad).obs["label_pred"].astype(str)
    truth = pd.read_parquet(f"{DATA}/{dataset}/solution.parquet")["label"].astype(str)
    pred.index = pred.index.astype(str)
    truth = truth.reindex(pred.index)
    assert truth.notna().all(), "prediction/solution cell ids do not line up"
    return {"accuracy": float(accuracy_score(truth, pred)),
            "f1_macro": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "n_test": int(len(pred)), "n_classes": int(truth.nunique())}


if __name__ == "__main__":
    method, dataset, out = sys.argv[1:4]
    work = os.path.join(os.path.dirname(out), f"work_{method}_{dataset}")
    os.makedirs(work, exist_ok=True)
    secs, h5 = run(method, dataset, work)
    res = {"method": method, "dataset": dataset, "local_runtime_s": round(secs, 1)}
    res.update(score(h5, dataset))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res), flush=True)
