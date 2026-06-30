"""Embed the broad reference with scPRINT (standalone; scPRINT venv, no actinn_jax/jax).
Saves /tmp/broad_emb.npz with the (n_cells, 256) embedding aligned to the h5ad order.
"""
import os, warnings, time, contextlib; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad, scipy.sparse as sp, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# patch torch.autocast('mps') -> no-op
_orig = torch.autocast
class _AC:
    def __init__(self, device_type, **kw):
        self._cm = _orig(device_type, **kw) if device_type in ("cuda","cpu","xpu") else contextlib.nullcontext()
    def __enter__(self): return self._cm.__enter__()
    def __exit__(self, *a): return self._cm.__exit__(*a)
torch.autocast = _AC

from scprint import scPrint
from scprint.tasks import Embedder
from scdataloader import Preprocessor
try:
    from scdataloader.preprocess import additional_preprocess
except Exception:
    additional_preprocess = None

REF = "/tmp/broad_human_ref.h5ad"
CKPT = "/Users/iandriver/Downloads/actinn-jax-benchmark/medium-v1.5.ckpt"
SCHEMA = ("cell_type_ontology_term_id","assay_ontology_term_id","disease_ontology_term_id",
          "self_reported_ethnicity_ontology_term_id","sex_ontology_term_id",
          "development_stage_ontology_term_id","tissue_ontology_term_id")

CHUNK = 4000   # bound peak memory: preprocess+embed cells in blocks

ref = sc.read_h5ad(REF)
print(f"ref {ref.shape} / {ref.obs.cell_type.nunique()} types", flush=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = scPrint.load_from_checkpoint(CKPT, precpt_gene_emb=None).to(dev)
pp = Preprocessor(do_postp=False, force_preprocess=True,
                  **({"additional_preprocess": additional_preprocess} if additional_preprocess else {}))

t = time.time(); parts = []
for start in range(0, ref.n_obs, CHUNK):
    sub = ref[start:start + CHUNK]
    a = ad.AnnData(X=sp.csr_matrix(sub.X).astype(np.float32), obs=sub.obs[[]].copy(), var=sub.var.copy())
    a.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    for c in SCHEMA: a.obs[c] = "unknown"
    a.obs["suspension_type"] = "cell"; a.obs["is_primary_data"] = True
    a = pp(a)
    emb = Embedder(doclass=False, precision="32", dtype=torch.float32, batch_size=32,
                   num_workers=0, doplot=False, max_len=2000)
    res = emb(model, a); out = res[0] if isinstance(res, (tuple, list)) else res
    assert out.n_obs == sub.n_obs, f"chunk {start}: {sub.n_obs}->{out.n_obs}"
    parts.append(np.asarray(out.obsm["scprint_emb"]))
    print(f"  chunk {start}-{start+sub.n_obs} done ({time.time()-t:.0f}s)", flush=True)
E = np.vstack(parts)
assert E.shape[0] == ref.n_obs
np.savez("/tmp/broad_emb.npz", emb=E, cell_type=ref.obs["cell_type"].astype(str).to_numpy())
print(f"embedded {E.shape} in {time.time()-t:.0f}s -> /tmp/broad_emb.npz", flush=True)
print("EMBED_DONE", flush=True)
