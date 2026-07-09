"""Side-by-side annotated UMAPs: ground truth vs actinn-jax vs actinn-jax+standardize.
    python plot_umaps.py <dataset_dir> <out_png>
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax as aj
from sklearn.metrics import accuracy_score

d = sys.argv[1]; out = sys.argv[2]; name = os.path.basename(d.rstrip("/"))

def load(p):
    a = sc.read_h5ad(p, backed="r"); X = sp.csr_matrix(a.layers["counts"][:])
    obs, var, n = a.obs.copy(), a.var.copy(), list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=X, obs=obs, var=var); b.obs_names = n; return b

print(f"{name}: loading + training", flush=True)
tr = load(f"{d}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
teC = load(f"{d}/test.h5ad")
m = tr.var["hvg"].astype(bool).to_numpy(); trh = tr[:, m].copy(); teh = teC[:, m].copy()
sol = sc.read_h5ad(f"{d}/solution.h5ad", backed="r")
true = sol.obs.loc[list(teh.obs_names), "label"].astype(str).to_numpy()

preds = {}
for tag, std in [("actinn-jax", False), ("actinn-jax + std", True)]:
    mdl = aj.train_reference(trh, train_label_name="label", standardize=std, print_cost=False)
    fr, _ = mdl.predict_frame(teh, use_raw=False)
    preds[tag] = fr["celltype"].to_numpy()

# UMAP from the framework-provided PCA of the test set
te = sc.read_h5ad(f"{d}/test.h5ad")
te.obs["Ground truth"] = true
te.obs["actinn-jax"] = preds["actinn-jax"]
te.obs["actinn-jax + std"] = preds["actinn-jax + std"]
print("neighbors + umap", flush=True)
sc.pp.neighbors(te, use_rep="X_pca", n_neighbors=15)
sc.tl.umap(te)
XY = te.obsm["X_umap"]

# shared palette over the union of label sets
cats = sorted(set(true) | set(preds["actinn-jax"]) | set(preds["actinn-jax + std"]))
pal = (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors)
cmap = {c: pal[i % len(pal)] for i, c in enumerate(cats)}

panels = [("Ground truth", true, None),
          ("actinn-jax", preds["actinn-jax"], accuracy_score(true, preds["actinn-jax"])),
          ("actinn-jax + std", preds["actinn-jax + std"], accuracy_score(true, preds["actinn-jax + std"]))]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
for ax, (title, labs, acc) in zip(axes, panels):
    cols = np.array([cmap[l] for l in labs])
    order = np.argsort([np.sum(labs == l) for l in labs])[::-1]  # big types under small
    ax.scatter(XY[:, 0], XY[:, 1], c=cols, s=3, linewidths=0, rasterized=True)
    t = title if acc is None else f"{title}  (acc {acc:.3f})"
    ax.set_title(t, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=9); ax.set_ylabel("UMAP2", fontsize=9)
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c], markersize=7, label=c) for c in cats]
ncol = 1 if len(cats) <= 20 else 2
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
           fontsize=7, ncol=ncol, frameon=False, title="cell type", title_fontsize=9)
fig.suptitle(f"{name} — test set annotation ({len(cats)} types, n={te.n_obs} cells)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED", out, flush=True)
