"""Run Pan-human Azimuth on the cross-study liver query and dump every level.

Pan-human Azimuth needs Keras/TF (.venv-panhuman) while actinn-jax needs JAX, so the
two stages cannot share a process. This script writes tier-1 predictions to disk for
``panhuman_tier1_refine.py`` to consume.

    .venv-panhuman/bin/python benchmark/explore/panhuman_tier1_dump.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")

import pandas as pd
import yaml

import benchmark.driver as drv
from benchmark.adapters.panhuman_adapter import _to_panhuman_input

OUT = "/tmp/panhuman_tier1_liver_cross.parquet"

cfg = yaml.safe_load(open("configs/panhuman_compare.yaml"))
ds = [d for d in cfg["datasets"] if d["name"] == "liver_cross"][0]
ref, query = drv.build_pair(ds, "cell_type")
print(f"query: {query.n_obs} cells, ref: {ref.n_obs} cells")

a, feature_col = _to_panhuman_input(query)

import panhumanpy as ph

az = ph.AzimuthNN(a, feature_names_col=feature_col, eval_batch_size=8192)
for col in ("azimuth_broad", "azimuth_medium", "azimuth_fine", "final_level_labels"):
    if col in az.cells_meta:
        # panhumanpy's mapper raises TypeError sorting a column that mixes None with
        # strings (it blanks unrefinable cells to None here, to bool False elsewhere).
        az.cells_meta[col] = az.cells_meta[col].astype(str)
        az.map_to_cell_ontology(col, include_cl_id=True)

meta = az.cells_meta.copy()
meta.index = list(query.obs_names)
keep = [c for c in meta.columns if c.startswith("azimuth_") or c.startswith("final_level")]
meta[keep].astype(str).to_parquet(OUT)
print(f"wrote {OUT}")
print(meta[keep].head(3).to_string()[:600])
