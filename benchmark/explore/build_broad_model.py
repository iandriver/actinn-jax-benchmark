"""Build the shipped broad-human HierarchicalReferenceModel for actinn-jax.

Embeds the broad reference with scPRINT (full genes), restricts the trained model to a
HVG panel (small/committable), builds the coarse->fine hierarchy, and saves it into the
actinn-jax package at actinn_jax/references/broad_human_v1/. Run in .venv-scprint.
"""
import os, warnings, time; warnings.filterwarnings("ignore")
import sys, numpy as np, scanpy as sc
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
import actinn_jax as aj
from actinn_jax.embed import scprint_embed

REF = "/tmp/broad_human_ref.h5ad"
OUT = "/Users/iandriver/Downloads/actinn-jax/actinn_jax/references/broad_human_v1"
CKPT = "/Users/iandriver/Downloads/actinn-jax-benchmark/medium-v1.5.ckpt"
N_HVG, N_GROUPS = 4000, 10

ref = sc.read_h5ad(REF)
print(f"broad ref {ref.shape} | {ref.obs.cell_type.nunique()} cell types | "
      f"{ref.obs.tissue.nunique()} tissues", flush=True)

# held-out split (build on 80%, validate on 20%)
rng = np.random.default_rng(0)
test = np.zeros(ref.n_obs, dtype=bool)
labels = ref.obs["cell_type"].astype(str).to_numpy()
for c in np.unique(labels):
    idx = np.where(labels == c)[0]
    test[rng.choice(idx, max(1, int(len(idx) * 0.2)), replace=False)] = True
train, held = ref[~test].copy(), ref[test].copy()

t = time.time(); emb = scprint_embed(train, ckpt=CKPT); print(f"embedded {train.n_obs} in {time.time()-t:.0f}s", flush=True)

raw = train.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
sc.pp.highly_variable_genes(raw, n_top_genes=min(N_HVG, raw.n_vars))
hvg = raw.var["highly_variable"].values
train_hvg = train[:, hvg].copy()
print(f"HVG panel: {train_hvg.n_vars} genes", flush=True)

t = time.time()
model = aj.build_hierarchical_reference(train_hvg, "cell_type", emb, n_groups=N_GROUPS, print_cost=False)
print(f"built hierarchy ({len(model.classes)} types, "
      f"{len(set(model.type_to_group.values()))} coarse groups) in {time.time()-t:.0f}s", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model.save(OUT)

# validate on held-out (CPU, no scPRINT)
out = aj.annotate(held, model, output_label_name="pred")
acc = float((out.obs["pred"].values == out.obs["cell_type"].astype(str).values).mean())
import json
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
print(f"held-out accuracy {acc:.3f} on {held.n_obs} cells | model size {sz:.1f}MB", flush=True)
print("BROAD_MODEL_DONE", flush=True)
