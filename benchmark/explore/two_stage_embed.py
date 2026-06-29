"""Stage-0 of the two-stage experiment: embed a krasnow ref/query split with scPRINT.

Caches, for the reference and query: raw counts + labels + scPRINT cell embedding
(obsm['scprint_emb']); for the query also scPRINT's zero-shot CL prediction. Everything
downstream (coreset, hierarchy, scoping, routing) runs fast on CPU off these caches.

Run in the scPRINT env: .venv-scprint/bin/python benchmark/explore/two_stage_embed.py
"""
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys, numpy as np, scanpy as sc, anndata as ad, scipy.sparse as sp, torch
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark.adapters.scprint_adapter import _patch_autocast
_patch_autocast()
from scprint import scPrint
from scprint.tasks import Embedder
from scdataloader import Preprocessor
try:
    from scdataloader.preprocess import additional_preprocess
except Exception:
    additional_preprocess = None

KRAS = "/Users/iandriver/Downloads/krasnow_lung_atlas_10x.h5ad"
CKPT = "/Users/iandriver/Downloads/actinn-jax-benchmark/medium-v1.5.ckpt"
REF_PER, QRY_PER = 300, 100


def raw_counts_adata(adata):
    X = adata.raw.X if adata.raw is not None else adata.X
    var = adata.raw.var.copy() if adata.raw is not None else adata.var.copy()
    a = ad.AnnData(X=sp.csr_matrix(X).astype(np.float32), obs=adata.obs.copy(), var=var)
    a.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    return a


def split(adata, label, ref_per, qry_per, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(adata.obs[label].values)
    ref_idx, qry_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        ref_idx += list(idx[:ref_per])
        qry_idx += list(idx[ref_per:ref_per + qry_per])
    return adata[np.sort(ref_idx)].copy(), adata[np.sort(qry_idx)].copy()


def embed(orig, model):
    """Return a copy of `orig` with obsm['scprint_emb'] and scPRINT pred cols attached."""
    a = orig.copy()
    pp = Preprocessor(do_postp=False, force_preprocess=True,
                      **({"additional_preprocess": additional_preprocess} if additional_preprocess else {}))
    a = pp(a)
    emb = Embedder(doclass=True, precision="32", dtype=torch.float32, batch_size=32,
                   num_workers=0, doplot=False, max_len=2000)
    res = emb(model, a)
    out = res[0] if isinstance(res, (tuple, list)) else res
    assert out.n_obs == orig.n_obs, f"cell count changed {orig.n_obs}->{out.n_obs}"
    orig.obsm["scprint_emb"] = np.asarray(out.obsm["scprint_emb"])
    orig.obs["scprint_pred_cl"] = out.obs["pred_cell_type_ontology_term_id"].astype(str).values
    return orig


print("loading krasnow + splitting...", flush=True)
kras = raw_counts_adata(sc.read_h5ad(KRAS))
ref, qry = split(kras, "cell_type", REF_PER, QRY_PER)
print(f"ref {ref.shape} ({ref.obs.cell_type.nunique()} types) | query {qry.shape}", flush=True)
model = scPrint.load_from_checkpoint(CKPT, precpt_gene_emb=None)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev); print(f"model on {dev}", flush=True)

import time
t = time.time(); ref = embed(ref, model); print(f"embedded ref in {time.time()-t:.0f}s", flush=True)
t = time.time(); qry = embed(qry, model); print(f"embedded query in {time.time()-t:.0f}s", flush=True)
ref.write_h5ad("/tmp/ts_ref.h5ad"); qry.write_h5ad("/tmp/ts_query.h5ad")
print(f"EMBED_DONE emb_dim={ref.obsm['scprint_emb'].shape[1]}", flush=True)
