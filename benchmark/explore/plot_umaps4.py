"""4-panel annotated UMAP: ground truth | actinn-jax | actinn-jax+std | scanvi_scarches.
    python plot_umaps4.py <dataset_dir> <scarches_pred.csv> <out_png>
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax as aj
from sklearn.metrics import accuracy_score

d, scpred_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
name = os.path.basename(d.rstrip("/"))

def load(p):
    a = sc.read_h5ad(p, backed="r"); X = sp.csr_matrix(a.layers["counts"][:])
    obs, var, n = a.obs.copy(), a.var.copy(), list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=X, obs=obs, var=var); b.obs_names = n; return b

tr = load(f"{d}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
teC = load(f"{d}/test.h5ad")
m = tr.var["hvg"].astype(bool).to_numpy(); trh = tr[:, m].copy(); teh = teC[:, m].copy()
sol = sc.read_h5ad(f"{d}/solution.h5ad", backed="r")
true = sol.obs.loc[list(teh.obs_names), "label"].astype(str).to_numpy()

pr = {}
for tag, std in [("actinn-jax", False), ("actinn-jax + std", True)]:
    mdl = aj.train_reference(trh, train_label_name="label", standardize=std, print_cost=False)
    pr[tag] = mdl.predict_frame(teh, use_raw=False)[0]["celltype"].to_numpy()

sc_df = pd.read_csv(scpred_csv); sc_df["cell"] = sc_df["cell"].astype(str); sc_df = sc_df.set_index("cell")
te = sc.read_h5ad(f"{d}/test.h5ad")
sca = sc_df.loc[[str(x) for x in te.obs_names], "scarches_pred"].astype(str).to_numpy()

sc.pp.neighbors(te, use_rep="X_pca", n_neighbors=15); sc.tl.umap(te)
XY = te.obsm["X_umap"]

panels = [("Ground truth", true, None),
          ("actinn-jax", pr["actinn-jax"], accuracy_score(true, pr["actinn-jax"])),
          ("actinn-jax + std", pr["actinn-jax + std"], accuracy_score(true, pr["actinn-jax + std"])),
          ("scanvi_scarches", sca, accuracy_score(true, sca))]
cats = sorted(set(true) | set(pr["actinn-jax"]) | set(pr["actinn-jax + std"]) | set(sca))
pal = (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors)
cmap = {c: pal[i % len(pal)] for i, c in enumerate(cats)}

fig, axes = plt.subplots(1, 4, figsize=(21, 6))
for ax, (title, labs, acc) in zip(axes, panels):
    ax.scatter(XY[:, 0], XY[:, 1], c=[cmap[l] for l in labs], s=3, linewidths=0, rasterized=True)
    ax.set_title(title if acc is None else f"{title}  (acc {acc:.3f})", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1", fontsize=8); ax.set_ylabel("UMAP2", fontsize=8)
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c], markersize=6, label=c) for c in cats]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5), fontsize=6.5,
           ncol=1 if len(cats) <= 22 else 2, frameon=False, title="cell type", title_fontsize=8)
fig.suptitle(f"{name} — test set annotation ({len(cats)} types, n={te.n_obs} cells)", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED", out, flush=True)
