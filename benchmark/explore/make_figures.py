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
# Brain ran after the unified matrix was frozen, and its two splits carry a different label
# key (Allen Subclass / Cluster, not cell_type), so they live in their own result files rather
# than being folded into the unified CSV.
for extra in ("results_brain_subclass_panel.csv", "results_brain_cluster_panel.csv"):
    q = f"{REPO}/docs/{extra}"
    if os.path.exists(q):
        b = pd.read_csv(q); frames.append(b[b.accuracy.notna()])
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


# ---- Fig 1: every split at once, ranked, with the range each method spans ----
# The single-dataset scatter this replaces could not take more datasets: five methods sit
# inside a factor of three in time and 0.02 in accuracy, so their labels overplot no matter
# how the placement is tuned. Names on the y-axis cannot collide. Accuracy is expressed as
# distance from that split's own leader, because raw accuracy is dominated by how hard the
# split is (pbmc 0.94, lung cross-dataset 0.36) and a range over raw accuracy would measure
# the datasets rather than the methods.
NAME = {"protocloud": "ProtoCloud", "sctop": "scTOP", "svm": "SVM", "knn": "kNN",
        "celltypist": "CellTypist", "singler": "SingleR", "scanvi": "scANVI",
        "scarches": "scArches"}


def cost_accuracy_ranges(exclude=("lung_cross",)):
    g = agg[agg.method.isin(COLOR) & ~agg.dataset.isin(exclude)].copy()
    if g.empty or "fit_s" not in g:
        return
    g["t"] = g.fit_s.fillna(0) + g.predict_s.fillna(0)
    g["gap"] = g.groupby("dataset").accuracy.transform("max") - g.accuracy
    # Only methods that ran every split belong on a figure whose whole point is the range
    # across splits -- scPRINT is scored ontology-only on one dataset and would otherwise
    # appear as an empty row and inflate the method count in the title.
    n_split = g.dataset.nunique()
    full = g.groupby("method").dataset.nunique() == n_split
    g = g[g.method.isin(full[full].index)]
    order = g.groupby("method").gap.mean().sort_values(ascending=False).index.tolist()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True,
                                 gridspec_kw={"wspace": 0.04})
    for i, meth in enumerate(order):
        d_ = g[g.method == meth]
        for ax, v in ((a1, d_.gap), (a2, d_.t)):
            ax.plot([v.min(), v.max()], [i, i], color=c(meth), lw=2.6, alpha=0.5,
                    solid_capstyle="round", zorder=2)
            ax.scatter(v, [i] * len(v), s=30, color=c(meth), alpha=0.9,
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.scatter([v.mean()], [i], s=150, color=c(meth), marker="D", edgecolor="black",
                       linewidth=1.6 if meth == "actinn-jax" else 0.7, zorder=4)
        leads = int((d_.gap < 1e-9).sum())
        if leads:
            a1.annotate(f"leads {leads}", (0, i), xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color="#444444")
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels([NAME.get(m, m) for m in order], fontsize=10)
    for lbl, meth in zip(a1.get_yticklabels(), order):
        if meth == "actinn-jax":
            lbl.set_fontweight("bold"); lbl.set_color(c("actinn-jax"))
    a1.axvline(0, color="0.35", lw=1.0, ls=(0, (4, 3)), zorder=1)
    a1.set_xlim(-0.055, max(0.20, g.gap.max() * 1.05))
    a2.set_xscale("log")
    a1.set_xlabel("accuracy below the best method on the same split")
    a2.set_xlabel("total time: fit + predict (s, log scale)")
    a1.set_title("A   accuracy, as distance from that split's leader", fontsize=10.5, loc="left")
    a2.set_title("B   cost, same splits", fontsize=10.5, loc="left")
    for ax in (a1, a2):
        ax.grid(axis="x", alpha=0.25, lw=0.6); ax.set_ylim(-0.7, len(order) - 0.3)
        ax.spines["left"].set_visible(False); ax.tick_params(left=False)
    fig.suptitle(f"{n_split} splits, {len(order)} methods: accuracy overlaps, cost spans three "
                 "orders of magnitude", fontsize=12.5, y=0.985)
    fig.text(0.5, -0.02, "diamond: mean over splits   ·   dots: one split each   ·   line: "
             "worst to best split", ha="center", fontsize=8.6, color="#555555")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_cost_accuracy_ranges.png", bbox_inches="tight"); plt.close(fig)


TITLE = {"pbmc": "pbmc — 8 types", "liver_intra": "liver — 36 types",
         "liver_cross": "liver cross-study — 34", "lung_intra": "lung — 46 types",
         "lung_cross": "lung cross-dataset — 46†", "blood_gut_intra": "blood+gut — 86 types",
         "brain_intra": "brain subclass — 24", "brain_cluster_intra": "brain cluster — 151"}
PANELS = ["pbmc", "liver_intra", "lung_intra", "brain_intra", "blood_gut_intra",
          "brain_cluster_intra", "liver_cross", "lung_cross"]


def pareto_facets():
    """The same data unnormalized, one panel per split -- the supplementary detail view."""
    rep = df[df.method.isin(COLOR) & df.accuracy.notna()].copy()
    rep["t"] = rep.fit_s.fillna(0) + rep.predict_s.fillna(0)
    # Range over repeats, clamped: deterministic methods return the identical score three
    # times and mean(x, x, x) can land a float epsilon below x, which errorbar rejects.
    e = rep.groupby(["dataset", "method"]).agg(
        acc=("accuracy", "mean"), alo=("accuracy", "min"), ahi=("accuracy", "max"),
        t=("t", "mean"), tlo=("t", "min"), thi=("t", "max")).reset_index()
    clip = lambda hi, lo: float(np.clip(hi - lo, 0, None))
    panels = [p for p in PANELS if p in set(e.dataset)]
    if not panels:
        return
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), sharex=True)
    for ax, ds in zip(axes.ravel(), panels):
        for _, r in e[e.dataset == ds].iterrows():
            big = r.method == "actinn-jax"
            ax.errorbar(r.t, r.acc, xerr=[[clip(r.t, r.tlo)], [clip(r.thi, r.t)]],
                        yerr=[[clip(r.acc, r.alo)], [clip(r.ahi, r.acc)]], fmt="o",
                        ms=9 if big else 6, color=c(r.method), ecolor=c(r.method),
                        elinewidth=1.3, alpha=0.95, mec="black",
                        mew=1.5 if big else 0.5, zorder=3 if big else 2)
        ax.set_xscale("log"); ax.grid(alpha=0.22, lw=0.6)
        ax.set_title(TITLE.get(ds, ds), fontsize=10, loc="left")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    for ax in axes[1]:
        ax.set_xlabel("fit + predict (s)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c(m), mec="black",
                          mew=1.4 if m == "actinn-jax" else 0.4,
                          ms=9 if m == "actinn-jax" else 6, label=NAME.get(m, m))
               for m in COLOR if m in set(e.method)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.4,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Accuracy against cost on every split — bars are the range over three repeats",
                 fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_pareto_facets.png", bbox_inches="tight"); plt.close(fig)


cost_accuracy_ranges()
pareto_facets()


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
