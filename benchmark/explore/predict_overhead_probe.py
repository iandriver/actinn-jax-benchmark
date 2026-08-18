"""How much of actinn-jax's small-query predict time is one-time cost, not per-cell work?

Table 3 reports 0.54 s of predict for actinn-jax against 0.33 s for the linear pipeline, while
Figure 4 has actinn-jax faster at every query size it measures. Part of that gap is a fixed
cost the harness charges to predict: the first call on a fresh model compiles, and the harness
times the first call because that is what a one-shot annotation costs. This measures the size
of it -- repeated calls on one model and query, then the same on a 4x larger query.

    .venv/bin/python benchmark/explore/predict_overhead_probe.py
"""
import warnings, sys, time; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
import numpy as np, scanpy as sc, actinn_jax as aj
from benchmark import datasets

a = sc.read_h5ad("/Volumes/IanSSD/hlica/liver_intra.h5ad")
ref, q = datasets.intra_split(a, "cell_type", 0.25)
m = aj.train_reference(ref, train_label_name="cell_type", print_cost=False)
print(f"query {q.n_obs} cells", flush=True)
for i in range(4):
    t = time.perf_counter(); m.predict_frame(q); dt = time.perf_counter() - t
    print(f"  call {i}: {dt:.3f} s  ({q.n_obs/dt:,.0f} cells/s)", flush=True)
# and on a 4x larger query built by tiling, to see the marginal per-cell cost
big = q.concatenate([q, q, q], index_unique=None)
for i in range(2):
    t = time.perf_counter(); m.predict_frame(big); dt = time.perf_counter() - t
    print(f"  {big.n_obs} cells call {i}: {dt:.3f} s  ({big.n_obs/dt:,.0f} cells/s)", flush=True)
