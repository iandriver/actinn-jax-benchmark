"""Generate paper figures from benchmark results. Run after the matrix + scaling.
Outputs PNGs to docs/figures/. Defensive to missing methods/datasets/columns.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/iandriver/Downloads/actinn-jax-benchmark"
FIG = f"{REPO}/docs/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# colorblind-safe (Okabe-Ito); actinn-jax gets the strong vermillion, others muted
COLOR = {"actinn-jax": "#D55E00", "svm": "#0072B2", "knn": "#56B4E9",
         "celltypist": "#009E73", "linear-anova-pca": "#E69F00", "sctop": "#F0E442",
         "protocloud": "#DC267F", "singler": "#CC79A7",
         "scmap-cluster": "#999999", "scanvi": "#5D3A9B", "scarches": "#8C6D31",
         "scprint": "#000000"}
def c(m): return COLOR.get(m, "#777777")

# Unified 11-method matrix (actinn-orig dropped; linear-anova-pca/scTOP/ProtoCloud merged
# onto the same splits — see docs/results_paper_matrix_unified.csv).
main = pd.read_csv(f"{REPO}/docs/results_paper_matrix_unified.csv")
main = main[main.get("accuracy").notna()] if "accuracy" in main else main
frames = [main]
sp = f"{REPO}/results/paper_scprint/results.csv"
if os.path.exists(sp):
    df = pd.read_csv(sp); frames.append(df[df.get("accuracy").notna()] if "accuracy" in df else df)
df = pd.concat(frames, ignore_index=True)
df["method"] = df["method"].astype(str)

METRICS = ["accuracy", "macro_f1", "ontology_concordance", "fit_s", "predict_s", "peak_mem_mb"]
agg = df.groupby(["dataset", "method"])[[m for m in METRICS if m in df]].mean().reset_index()
method_order = [m for m in COLOR if m in set(df.method)]
datasets = sorted(df.dataset.unique())


# ---- Fig 1: accuracy heatmap (methods x datasets) ----
def heatmap(metric, fname, title):
    piv = agg.pivot(index="method", columns="dataset", values=metric).reindex(method_order)
    fig, ax = plt.subplots(figsize=(1.1 * len(datasets) + 3, 0.5 * len(method_order) + 1.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=0,
                   vmax=np.nanmax(piv.values))
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.6 * np.nanmax(piv.values) else "black")
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", bbox_inches="tight"); plt.close(fig)

if "accuracy" in agg: heatmap("accuracy", "fig_accuracy_heatmap.png", "Accuracy (mean over repeats)")
if "macro_f1" in agg: heatmap("macro_f1", "fig_macrof1_heatmap.png", "Macro-F1 (mean over repeats)")


# ---- Fig 2: Pareto (accuracy vs total wall time) on a representative dataset ----
def pareto(dsname):
    d = agg[agg.dataset == dsname].copy()
    if d.empty or "fit_s" not in d: return
    d["total_s"] = d["fit_s"].fillna(0) + d["predict_s"].fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in d.iterrows():
        big = r["method"] == "actinn-jax"
        ax.scatter(r["total_s"], r["accuracy"], s=180 if big else 90, color=c(r["method"]),
                   edgecolor="black", linewidth=1.4 if big else 0.6, zorder=3)
        ax.annotate(r["method"], (r["total_s"], r["accuracy"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontweight="bold" if big else "normal", fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("total time: fit + predict (s, log scale)")
    ax.set_ylabel("accuracy"); ax.set_title(f"Accuracy vs. speed — {dsname}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_pareto_{dsname}.png", bbox_inches="tight"); plt.close(fig)

for ds in ["liver_intra", "lung_intra"]:
    if ds in datasets: pareto(ds)


# ---- Fig 3: speed + memory bars (mean across datasets) ----
def speed_mem():
    g = agg.groupby("method")[["predict_s", "fit_s", "peak_mem_mb"]].mean().reindex(method_order)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ["fit_s", "predict_s", "peak_mem_mb"],
                              ["fit time (s)", "predict time (s)", "peak memory (MB)"]):
        vals = g[col].values
        ax.barh(range(len(g)), vals, color=[c(m) for m in g.index])
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index)
        ax.set_xlabel(title); ax.invert_yaxis()
    fig.suptitle("Cost per method (mean across datasets)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_speed_memory.png", bbox_inches="tight"); plt.close(fig)
speed_mem()


# ---- Fig 4: scaling curves ----
def scaling():
    sc_cells = f"{REPO}/docs/results_scaling_cells.csv"
    sc_types = f"{REPO}/docs/results_scaling_types.csv"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if os.path.exists(sc_cells):
        sd = pd.read_csv(sc_cells)
        for m in sd.method.unique():
            s = sd[sd.method == m].sort_values("n_ref")
            axes[0].plot(s.n_ref, s.fit_s, "-o", color=c(m), label=m, lw=2 if m == "actinn-jax" else 1)
        axes[0].set_xlabel("# reference cells"); axes[0].set_ylabel("fit time (s)")
        axes[0].set_title("Training time vs. reference size"); axes[0].legend(fontsize=8)
        axes[0].set_xscale("log"); axes[0].grid(True, alpha=0.25)
    if os.path.exists(sc_types):
        td = pd.read_csv(sc_types)
        for m in td.method.unique():
            s = td[td.method == m].sort_values("n_types")
            axes[1].plot(s.n_types, s.fit_s, "-o", color=c(m), label=m, lw=2 if m == "actinn-jax" else 1)
        axes[1].set_xlabel("# cell types"); axes[1].set_ylabel("fit time (s)")
        axes[1].set_title("Training time vs. #types"); axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_scaling.png", bbox_inches="tight"); plt.close(fig)
scaling()

print("FIGURES_DONE ->", FIG)
print("generated:", sorted(os.listdir(FIG)))
