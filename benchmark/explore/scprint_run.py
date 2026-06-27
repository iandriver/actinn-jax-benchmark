import os, time, warnings
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
warnings.filterwarnings("ignore")
import sys, torch, scanpy as sc, numpy as np
DEVICE = sys.argv[1] if len(sys.argv)>1 else "cpu"

import contextlib as _ctx
_orig_ac = torch.autocast
class _AC:
    def __init__(self, device_type, **kw):
        self._cm = _orig_ac(device_type, **kw) if device_type in ("cuda","cpu","xpu") else _ctx.nullcontext()
    def __enter__(self): return self._cm.__enter__()
    def __exit__(self,*a): return self._cm.__exit__(*a)
torch.autocast = _AC

from scprint import scPrint
from scprint.tasks import Embedder
from scdataloader import Preprocessor
try:
    from scdataloader.preprocess import additional_preprocess
except Exception:
    additional_preprocess = None

adata = sc.read_h5ad("/tmp/krasnow_sp400.h5ad")
print(f"[1] loaded adata {adata.shape}", flush=True)
model = scPrint.load_from_checkpoint("medium-v1.5.ckpt", precpt_gene_emb=None)
model = model.to(DEVICE)
print(f"[2] model on {DEVICE}", flush=True)
pp = Preprocessor(do_postp=False, force_preprocess=True,
                  **({"additional_preprocess": additional_preprocess} if additional_preprocess else {}))
adata = pp(adata)
print(f"[3] preprocessed -> {adata.shape}", flush=True)
emb = Embedder(doclass=True, precision="32", dtype=torch.float32,
               batch_size=32, num_workers=0, doplot=False, max_len=2000)
t=time.time()
res = emb(model, adata)
dt=time.time()-t
print(f"[4] embed/classify done in {dt:.1f}s, device={DEVICE}", flush=True)
ad = res[0] if isinstance(res,(tuple,list)) else res
predcols=[c for c in ad.obs.columns if 'pred' in c.lower()]
print("[5] pred columns:", predcols, flush=True)
# evaluate cell type ontology prediction
pc=[c for c in predcols if 'cell_type' in c]
if pc and 'cell_type_ontology_term_id' in ad.obs:
    pred=ad.obs[pc[0]].astype(str).values; truth=ad.obs['cell_type_ontology_term_id'].astype(str).values
    exact=(pred==truth).mean()
    print(f"[6] scprint cell-type exact CL match: {exact:.3f} over {len(truth)} cells", flush=True)
    print("    sample pred:", list(pred[:5])); print("    sample truth:", list(truth[:5]))
print("SCPRINT_DONE", flush=True)

# save predictions for ontology-aware scoring
import pandas as pd
cols = {}
for c in ad.obs.columns:
    if c in ('cell_type_ontology_term_id',) or c.endswith('pred_cell_type_ontology_term_id'):
        cols[c]=ad.obs[c].astype(str).values
pd.DataFrame(cols).to_csv(f"/tmp/scprint_pred_{DEVICE}.csv", index=False)
if 'conv_pred_cell_type_ontology_term_id' in ad.obs:
    conv=ad.obs['conv_pred_cell_type_ontology_term_id'].astype(str).values
    truth=ad.obs['cell_type_ontology_term_id'].astype(str).values
    print(f"[7] conv exact CL match: {(conv==truth).mean():.3f}", flush=True)
