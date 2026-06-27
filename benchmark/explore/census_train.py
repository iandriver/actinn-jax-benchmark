# Train actinn-jax (fast, CPU) on the census-curated reference; held-out accuracy.
import os, warnings, time; warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v,"8")
import scanpy as sc, numpy as np, sys
sys.path.insert(0,"/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import datasets, metrics
from benchmark.resources import ResourceMonitor
import actinn_jax as aj
ad = sc.read_h5ad("/tmp/census_blood.h5ad")
print("census ref:", ad.shape, "| types:", ad.obs.cell_type.nunique(), flush=True)
ref, qry = datasets.intra_split(ad, "cell_type", 0.25)
with ResourceMonitor() as t: model = aj.train_reference(ref, train_label_name="cell_type", print_cost=False)
with ResourceMonitor() as t2: out,_ = aj.predict(qry, model, output_label_name="pred")
acc = (out.obs["pred"].values == out.obs["cell_type"].values).mean()
m = metrics.compute(out.obs["cell_type"].values, out.obs["pred"].values)
print(f"actinn-jax on census blood: train {t.elapsed:.1f}s ({ref.n_obs} cells) | "
      f"predict {t2.elapsed:.2f}s ({qry.n_obs} cells) | peak {max(t.peak_mb,t2.peak_mb):.0f}MB", flush=True)
print(f"  held-out accuracy {acc:.3f} | macro-F1 {m['macro_f1']:.3f}", flush=True)
print("CENSUS_TRAIN_DONE", flush=True)
