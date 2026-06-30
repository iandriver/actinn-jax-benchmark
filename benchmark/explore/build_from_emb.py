"""Build the shipped broad-human HierarchicalReferenceModel from cached embeddings.
Core .venv (actinn_jax). Loads /tmp/broad_emb.npz, HVG-subsets, builds hierarchy,
saves to actinn_jax/references/broad_human_v1/, validates on a held-out split.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax as aj

REF = "/tmp/broad_human_ref.h5ad"
OUT = "/Users/iandriver/Downloads/actinn-jax/actinn_jax/references/broad_human_v1"
N_HVG, N_GROUPS = 4000, 10

ref = sc.read_h5ad(REF)
emb_all = np.load("/tmp/broad_emb.npz", allow_pickle=True)["emb"]
assert emb_all.shape[0] == ref.n_obs
labels = ref.obs["cell_type"].astype(str).to_numpy()
print(f"ref {ref.shape} | {len(set(labels))} types | emb {emb_all.shape}", flush=True)

# 80/20 stratified split
rng = np.random.default_rng(0); test = np.zeros(ref.n_obs, dtype=bool)
for c in np.unique(labels):
    idx = np.where(labels == c)[0]
    test[rng.choice(idx, max(1, int(len(idx) * 0.2)), replace=False)] = True
train, held, emb_tr = ref[~test].copy(), ref[test].copy(), emb_all[~test]

# HVG panel on train (keeps the shipped model small)
raw = train.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
sc.pp.highly_variable_genes(raw, n_top_genes=min(N_HVG, raw.n_vars))
train_hvg = train[:, raw.var["highly_variable"].values].copy()
print(f"HVG panel {train_hvg.n_vars} genes; training on {train_hvg.n_obs} cells", flush=True)

t = time.time()
model = aj.build_hierarchical_reference(train_hvg, "cell_type", emb_tr, n_groups=N_GROUPS, print_cost=False)
ng = len(set(model.type_to_group.values()))
print(f"built ({len(model.classes)} types, {ng} coarse groups) in {time.time()-t:.0f}s", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model.save(OUT)

out = aj.annotate(held, model, output_label_name="pred")
exact = float((out.obs["pred"].values == out.obs["cell_type"].astype(str).values).mean())
# coarse-group accuracy: did we at least land in the right coarse group?
t2g = model.type_to_group
true_grp = np.array([t2g.get(t, "?") for t in held.obs["cell_type"].astype(str)])
coarse_ok = float((out.obs["pred_coarse"].values == true_grp).mean())
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
print(f"held-out: exact {exact:.3f} | coarse-group {coarse_ok:.3f} | {held.n_obs} cells | {sz:.1f}MB", flush=True)
print("BUILD_DONE", flush=True)
