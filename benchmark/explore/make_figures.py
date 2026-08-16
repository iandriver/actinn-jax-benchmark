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
    ax.set_xscale("log")
    # Labels are placed after the axes are scaled, and each one is checked against the labels
    # already down. Methods that tie -- scANVI, scArches and CellTypist land within 0.01
    # accuracy and a few seconds of each other on liver -- otherwise print on top of one
    # another, and a legend would not help because the point is which name sits where.
    ax.margins(x=0.13, y=0.13)                 # headroom, so a label cannot reach the title
    ax.autoscale_view()
    # Markers are obstacles too, not just other labels: on liver the tied methods sit close
    # enough that a label dodging its neighbour's text lands on its dot instead. Anything
    # pushed clear of its own point gets a leader line, since a label far from its marker is
    # worse than no label -- the reader has to guess which dot it belongs to.
    pts = [ax.transData.transform((r["total_s"], r["accuracy"])) for _, r in d.iterrows()]
    placed = [(x - 9, y - 9, x + 9, y + 9) for x, y in pts]
    CAND = [(9, 0), (-9, 0), (9, 15), (-9, 15), (9, -15), (-9, -15),
            (9, 29), (-9, 29), (9, -29), (-9, -29), (9, 43), (-9, 43)]
    for _, r in d.sort_values("total_s").iterrows():
        big = r["method"] == "actinn-jax"
        px, py = ax.transData.transform((r["total_s"], r["accuracy"]))
        w, h = 6.2 * len(r["method"]) + 4, 13          # display-space label box, approximate
        dx, dy = CAND[0]
        for cx, cy in CAND:
            x0 = px + cx if cx > 0 else px + cx - w
            box = (x0, py + cy - h / 2, x0 + w, py + cy + h / 2)
            if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in placed):
                dx, dy = cx, cy
                placed.append(box)
                break
        ax.annotate(r["method"], (r["total_s"], r["accuracy"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center",
                    fontweight="bold" if big else "normal", fontsize=9,
                    arrowprops=(dict(arrowstyle="-", lw=0.6, color="0.45",
                                     shrinkA=0, shrinkB=4) if abs(dy) > 8 else None))
    ax.set_xlabel("total time: fit + predict (s, log scale)")
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
    """Fit and predict, both axes.

    This figure used to plot fit time twice and nothing else, while its caption -- and the
    paper's central claim that cached inference is flat -- described predict time. The
    evidence was in the CSVs and simply never drawn. The bottom row draws it: the predict
    axes share a y-scale with each other so the flatness is a visual fact rather than an
    artefact of two differently zoomed panels.
    """
    sc_cells = f"{REPO}/docs/results_scaling_cells.csv"
    sc_types = f"{REPO}/docs/results_scaling_types.csv"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [(sc_cells, "n_ref", "# reference cells", True),
              (sc_types, "n_types", "# cell types", False)]
    pmax = 0
    for col, (path, xcol, xlabel, logx) in enumerate(panels):
        if not os.path.exists(path):
            continue
        sd = pd.read_csv(path)
        pmax = max(pmax, sd.predict_s.max())
        for row, ycol, ylabel in ((0, "fit_s", "fit time (s)"),
                                  (1, "predict_s", "predict time (s)")):
            ax = axes[row][col]
            for m in sd.method.unique():
                s = sd[sd.method == m].sort_values(xcol)
                ax.plot(s[xcol], s[ycol], "-o", color=c(m), label=m,
                        lw=2 if m == "actinn-jax" else 1)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            if logx:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
        axes[0][col].set_title(f"Training time vs. {'reference size' if logx else '#types'}")
        axes[1][col].set_title(f"Predict time vs. {'reference size' if logx else '#types'}")
        axes[0][col].legend(fontsize=8)
    for ax in axes[1]:                       # shared scale, and zero-based: the claim is flatness
        ax.set_ylim(0, pmax * 1.15)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_scaling.png", bbox_inches="tight"); plt.close(fig)
scaling()

print("FIGURES_DONE ->", FIG)
print("generated:", sorted(os.listdir(FIG)))
