"""How much does it cost to annotate a whole atlas, rather than to build one?

Every scaling result in the paper varies the *reference* and holds the query fixed. The
claim that inference is "chunked" and "memory-bounded" for atlas-scale *queries* is made
three times and measured nowhere. This measures it: one fixed reference, a query that grows
from 50k to the full 525k-cell HLiCA liver atlas.

Leakage: the reference is drawn from seven studies and the eighth (Andrews_2022) is withheld,
so accuracy is reported on the withheld study's cells alone. Cost is reported over the whole
query, since cost does not care whether a cell was seen in training.

One size per process, so `ru_maxrss` is that size's peak and not a running maximum. The
first version looped sizes in one process and was killed at 524,699 cells -- by its own
bookkeeping, not by the method: it held the whole atlas *and* a materialised copy of the
subset, each carrying `.raw`, so the largest query duplicated the atlas. The measurement
harness has to be cheaper than the thing it measures. Now the atlas is opened backed, only
the query rows are read into memory, and the model is fitted once and reloaded.

actinn-jax fits once and reloads a saved model; every other method fits in-process, since
the adapters have no on-disk form. Fit cost is reported either way. A method that the OS
kills at some query size is a result, not a failed run, so the caller wraps each size in
`/usr/bin/time -l` and records the resident peak the process reached before it died.

    .venv-protocloud/bin/python benchmark/explore/query_scaling.py --fit-only
    .venv-protocloud/bin/python benchmark/explore/query_scaling.py --n-query 524699
    .venv-protocloud/bin/python benchmark/explore/query_scaling.py \
        --method linear-anova-pca --n-query 250000
"""

import argparse
import json
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from benchmark import adapters

ATLAS = "/Volumes/IanSSD/hlica/all_cells.h5ad"
LABEL = "cell_type"
WITHHELD = "Andrews_2022"


def peak_gb():
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="actinn-jax")
    ap.add_argument("--n-query", type=int, default=524699)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--model", default="/tmp/query_scaling_model")
    ap.add_argument("--ref-per-type", type=int, default=500,
                    help="cells per type in the reference; held fixed across query sizes")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    atlas = sc.read_h5ad(ATLAS, backed="r")
    study = atlas.obs["STUDY"].astype(str).to_numpy()
    lab = atlas.obs[LABEL].astype(str).to_numpy()
    print(f"loaded {atlas.shape[0]:,} x {atlas.shape[1]:,} in {time.time()-t0:.0f}s "
          f"| peak {peak_gb():.1f} GB", flush=True)

    # reference: capped per type, drawn only from the seven non-withheld studies
    rng = np.random.default_rng(0)
    train_pool = np.where(study != WITHHELD)[0]
    keep = []
    for c in np.unique(lab[train_pool]):
        idx = train_pool[lab[train_pool] == c]
        keep.append(rng.choice(idx, min(len(idx), a.ref_per_type), replace=False))
    ref_idx = np.sort(np.concatenate(keep))
    import actinn_jax as aj

    if a.fit_only:
        ref = atlas[ref_idx].to_memory()
        print(f"reference {ref.n_obs:,} cells / {len(np.unique(lab[ref_idx]))} types",
              flush=True)
        t = time.time()
        model = aj.train_reference(ref, train_label_name=LABEL, print_cost=False)
        fit_s = time.time() - t
        model.save(a.model)
        json.dump({"fit_s": round(fit_s, 1), "n_ref": int(ref.n_obs)},
                  open(a.model + ".meta.json", "w"))
        print(f"fit {fit_s:.1f}s | peak {peak_gb():.1f} GB -> {a.model}", flush=True)
        print("QUERY_SCALING_DONE", flush=True)
        return

    n = min(a.n_query, atlas.n_obs)
    if a.method == "actinn-jax":
        meta = json.load(open(a.model + ".meta.json"))
        model = aj.ReferenceModel.load(a.model)
        adapter = None
    else:
        ref = atlas[ref_idx].to_memory()
        adapter = adapters.get(a.method)
        t = time.time()
        adapter.fit(ref, LABEL)
        meta = {"fit_s": round(time.time() - t, 1), "n_ref": int(ref.n_obs)}
        del ref
        print(f"fit {meta['fit_s']}s | peak {peak_gb():.1f} GB", flush=True)
    order = rng.permutation(atlas.n_obs)         # same seed -> nested subsets across runs
    q_idx = np.sort(order[:n])
    q = atlas[q_idx].to_memory()
    del atlas                                    # the harness must not outweigh the method
    print(f"query {q.n_obs:,} cells in memory | peak {peak_gb():.1f} GB", flush=True)

    t = time.time()
    if adapter is None:
        labels = model.predict_frame(q)[0]["celltype"].to_numpy()
    else:
        labels = np.asarray(adapter.predict(q).labels)
    dt = time.time() - t
    held = study[q_idx] == WITHHELD
    acc = float((labels[held] == lab[q_idx][held]).mean()) if held.any() else float("nan")
    rows = [{"method": a.method, "n_query": n, "predict_s": round(dt, 2),
             "cells_per_s": round(n / dt), "peak_gb": round(peak_gb(), 2),
             "fit_s": meta["fit_s"], "n_ref": meta["n_ref"],
             "acc_withheld_study": round(acc, 4), "n_withheld": int(held.sum())}]
    print(f"  n={n:>7,}  predict {dt:7.1f}s  {n/dt:>8,.0f} cells/s  "
          f"peak {peak_gb():5.1f} GB  acc(withheld)={acc:.3f} on {held.sum():,} cells",
          flush=True)

    out = a.out or f"/tmp/query_scaling_{a.method}_{n}.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {out}")
    print("QUERY_SCALING_DONE", flush=True)


if __name__ == "__main__":
    main()
