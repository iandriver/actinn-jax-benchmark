# Fast CPU cell-type classifier from curated, CL-labeled atlas data (HCLA).
# HCLA = Human Lung Cell Atlas: 49 integrated datasets, harmonized Cell Ontology
# labels — the same kind of curated "underlying data" lamindb/CELLxGENE manage.
import os, warnings, time; warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v,"8")
import scanpy as sc, numpy as np, sys
sys.path.insert(0,"/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import datasets, metrics
from benchmark.resources import ResourceMonitor
import actinn_jax as aj

PER = int(sys.argv[1]) if len(sys.argv)>1 else 300
print(f"loading HCLA backed, subsampling {PER}/cell_type...", flush=True)
backed = sc.read_h5ad("/Users/iandriver/Downloads/Sikkama_HCLA.h5ad", backed="r")
labels = np.asarray(backed.obs["cell_type"].values)
sel = datasets.stratified_subsample(labels, PER)
ad = backed[sel].to_memory(); backed.file.close()
print(f"reference: {ad.shape} | {ad.obs.cell_type.nunique()} cell types", flush=True)

ref, qry = datasets.intra_split(ad, "cell_type", 0.25)
obo = "/tmp/cl-basic.obo"
anc = metrics.load_cl_ancestors(obo) if os.path.exists(obo) else None
with ResourceMonitor() as t:
    model = aj.train_reference(ref, train_label_name="cell_type", print_cost=False)
with ResourceMonitor() as t2:
    out,_ = aj.predict(qry, model, output_label_name="pred")
truth = out.obs["cell_type"].astype(str).values
pred  = out.obs["pred"].astype(str).values
acc = (pred==truth).mean()
m = metrics.compute(truth, pred)
onto = None
if anc is not None and "cell_type_ontology_term_id" in out.obs:
    name2cl = dict(zip(ref.obs.cell_type.astype(str), ref.obs.cell_type_ontology_term_id.astype(str)))
    tcl = out.obs.cell_type_ontology_term_id.astype(str).values
    pcl = np.array([name2cl.get(p,"") for p in pred])
    onto = metrics._ontology(tcl, pcl, anc, np.ones(len(tcl),bool))
print(f"\nFAST CPU CLASSIFIER (actinn-jax on HCLA atlas):", flush=True)
print(f"  train {t.elapsed:.1f}s ({ref.n_obs} cells, {ref.obs.cell_type.nunique()} types) | "
      f"predict {t2.elapsed:.2f}s ({qry.n_obs} cells) | peak {max(t.peak_mb,t2.peak_mb):.0f}MB", flush=True)
print(f"  held-out accuracy {acc:.3f} | macro-F1 {m['macro_f1']:.3f}" +
      (f" | ontology {onto:.3f}" if onto is not None else ""), flush=True)
print("ATLAS_CLF_DONE", flush=True)
