"""Stage-0: embed a ref/query split of any atlas with scPRINT, cache for stage-1.

Parameterized via env:
  TS_DATA   h5ad path                        (default: krasnow lung)
  TS_LABEL  fine cell-type obs column        (default: cell_type)
  TS_EXTRA  comma-sep extra obs cols to keep (e.g. biological hierarchy: Lineage,Category)
  TS_REF_PER / TS_QRY_PER  cells per label   (default 300 / 100)
  TS_OUT    output prefix                    (default /tmp/ts)

Caches <prefix>_ref.h5ad and <prefix>_query.h5ad with raw counts + labels +
obsm['scprint_emb'] (+ query scPRINT CL prediction). Large atlases are subsampled in
backed mode before materializing. Run in .venv-scprint.
"""
import os, time, warnings; warnings.filterwarnings("ignore")
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

DATA = os.environ.get("TS_DATA", "/Users/iandriver/Downloads/krasnow_lung_atlas_10x.h5ad")
LABEL = os.environ.get("TS_LABEL", "cell_type")
EXTRA = [c for c in os.environ.get("TS_EXTRA", "").split(",") if c]
REF_PER = int(os.environ.get("TS_REF_PER", 300))
QRY_PER = int(os.environ.get("TS_QRY_PER", 100))
OUT = os.environ.get("TS_OUT", "/tmp/ts")
CKPT = "/Users/iandriver/Downloads/actinn-jax-benchmark/medium-v1.5.ckpt"


def stratified_split_idx(labels, ref_per, qry_per, seed=0):
    rng = np.random.default_rng(seed)
    ref_idx, qry_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        ref_idx += list(idx[:ref_per]); qry_idx += list(idx[ref_per:ref_per + qry_per])
    return np.sort(ref_idx), np.sort(qry_idx)


def raw_counts_adata(adata, keep):
    X = adata.raw.X if adata.raw is not None else adata.X
    var = adata.raw.var.copy() if adata.raw is not None else adata.var.copy()
    obs = adata.obs[[LABEL] + [c for c in keep if c in adata.obs]].copy()
    a = ad.AnnData(X=sp.csr_matrix(X).astype(np.float32), obs=obs, var=var)
    a.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    # scPRINT's Preprocessor expects the CELLxGENE obs schema; fill the rest as
    # 'unknown' (these are metadata, not used by zero-shot prediction).
    for c in ("cell_type_ontology_term_id", "assay_ontology_term_id",
              "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id",
              "sex_ontology_term_id", "development_stage_ontology_term_id",
              "tissue_ontology_term_id"):
        if c not in a.obs:
            a.obs[c] = "unknown"
    a.obs["suspension_type"] = "cell"
    a.obs["is_primary_data"] = True
    return a


DOCLASS = os.environ.get("TS_DOCLASS", "1") == "1"  # off for label-less atlases


def embed(orig, model):
    a = orig.copy()
    pp = Preprocessor(do_postp=False, force_preprocess=True,
                      **({"additional_preprocess": additional_preprocess} if additional_preprocess else {}))
    a = pp(a)

    def _run(doclass):
        aa = a.copy()  # Embedder mutates in place; fresh copy avoids duplicate columns
        e = Embedder(doclass=doclass, precision="32", dtype=torch.float32, batch_size=32,
                     num_workers=0, doplot=False, max_len=2000)
        r = e(model, aa)
        return r[0] if isinstance(r, (tuple, list)) else r

    try:
        out = _run(DOCLASS)
    except Exception as ex:  # e.g. a CL id not in scPRINT's class set
        print(f"  doclass={DOCLASS} failed ({type(ex).__name__}); retrying embeddings-only", flush=True)
        out = _run(False)
    assert out.n_obs == orig.n_obs, f"{orig.n_obs}->{out.n_obs}"
    orig.obsm["scprint_emb"] = np.asarray(out.obsm["scprint_emb"])
    col = "pred_cell_type_ontology_term_id"
    if col in out.obs and getattr(out.obs[col], "ndim", 1) == 1:
        orig.obs["scprint_pred_cl"] = np.asarray(out.obs[col].astype(str).values).ravel()
    else:
        orig.obs["scprint_pred_cl"] = np.array(["unknown"] * orig.n_obs)
    return orig


print(f"loading {DATA} (backed) label={LABEL}...", flush=True)
backed = sc.read_h5ad(DATA, backed="r")
labels = np.asarray(backed.obs[LABEL].astype(str).values)
ri, qi = stratified_split_idx(labels, REF_PER, QRY_PER)
ref = raw_counts_adata(backed[ri].to_memory(), EXTRA)
qry = raw_counts_adata(backed[qi].to_memory(), EXTRA)
backed.file.close()
print(f"ref {ref.shape} ({ref.obs[LABEL].nunique()} types) | query {qry.shape}", flush=True)

model = scPrint.load_from_checkpoint(CKPT, precpt_gene_emb=None)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev); print(f"model on {dev}", flush=True)
t = time.time(); ref = embed(ref, model); print(f"embedded ref {ref.n_obs} in {time.time()-t:.0f}s", flush=True)
t = time.time(); qry = embed(qry, model); print(f"embedded query {qry.n_obs} in {time.time()-t:.0f}s", flush=True)
ref.write_h5ad(f"{OUT}_ref.h5ad"); qry.write_h5ad(f"{OUT}_query.h5ad")
print(f"EMBED_DONE emb_dim={ref.obsm['scprint_emb'].shape[1]}", flush=True)
