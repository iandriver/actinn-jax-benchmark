"""The abstain trade-off as a curve rather than four table rows.

Table 9 reports four thresholds for two methods, and the point it makes is about *shape*:
one method's confidence traces a usable curve, the other's is saturated so every threshold
lands in the same place. That is hard to see in a table of eight numbers and immediate as a
line.

Left: what you buy by abstaining -- accuracy on the cells you keep, against the fraction
kept. Right: whether abstention actually finds the novel cells, i.e. the share of
held-out-type cells flagged as the threshold rises.

    .venv/bin/python benchmark/explore/plot_abstain.py --out docs/figures/fig_abstain.png
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito, so the six curves stay separable in grayscale and to colorblind readers
STYLE = {
    "actinn-jax":       dict(color="#0072B2", marker="o", lw=2.1, zorder=6),
    "linear-anova-pca": dict(color="#009E73", marker="s", lw=1.8, zorder=5),
    "knn":              dict(color="#56B4E9", marker="v", lw=1.8, zorder=4),
    "celltypist":       dict(color="#D55E00", marker="D", lw=1.8, zorder=3),
    "protocloud":       dict(color="#E69F00", marker="P", lw=1.8, zorder=2),
    "sctop":            dict(color="#CC79A7", marker="^", lw=1.8, zorder=1),
}
PRETTY = {"linear-anova-pca": "linear pipeline", "knn": "kNN",
          "celltypist": "CellTypist", "protocloud": "ProtoCloud", "sctop": "scTOP"}
ORDER = ["actinn-jax", "linear-anova-pca", "knn", "celltypist", "protocloud", "sctop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/results_rejection.csv")
    ap.add_argument("--out", default="docs/figures/fig_abstain.png")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    for method in [m for m in ORDER if m in set(d.method)]:
        g = d[d.method == method].sort_values("min_prob")
        st = STYLE[method]
        lab = PRETTY.get(method, method)
        ax1.plot(g["coverage"], g["indist_acc_kept"], markersize=5.2, label=lab, **st)
        ax2.plot(g["min_prob"], g["ood_flagged"], markersize=5.2, label=lab, **st)
    # Six curves: per-marker threshold labels would overprint, so mark the swept
    # thresholds once on the right panel's axis instead.
    ax2.set_xticks([0.0, 0.3, 0.5, 0.7, 0.9])

    # Say the saturation out loud: four of celltypist's five thresholds land on the same
    # coordinate, so the line looks like two points unless the reader is told why.
    # 2 dp, not 3: celltypist's thresholds sit at coverage .681/.680/.680/.678, which are
    # the same operating point to any practical reader but four distinct values at 3 dp.
    sat = d[d.method == "celltypist"].round({"coverage": 2, "indist_acc_kept": 2})
    dup = sat.groupby(["coverage", "indist_acc_kept"]).size()
    if (dup > 1).any():
        (cov, acc) = dup.idxmax()
        ax1.annotate(f"CellTypist: p≥0.3–0.9\nall land here",
                     (cov, acc), textcoords="offset points", xytext=(26, -20),
                     fontsize=7.2, color="#D55E00", ha="left",
                     arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.9))
    # scTOP's score is a projection, not a calibrated probability: it discards almost the
    # whole query by p>=0.5, which is invisible unless it is said.
    s = d[d.method == "sctop"].sort_values("min_prob")
    if len(s):
        r = s[s.min_prob == 0.5]
        if len(r):
            r = r.iloc[0]
            ax1.annotate("scTOP keeps 6%\nof cells at p≥0.5",
                         (r["coverage"], r["indist_acc_kept"]), textcoords="offset points",
                         xytext=(24, -6), fontsize=7.2, color="#CC79A7",
                         arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=0.9))

    ax1.set_xlabel("coverage — fraction of cells kept", fontsize=9)
    ax1.set_ylabel("accuracy on kept cells", fontsize=9)
    ax1.set_title("What abstaining buys", fontsize=10.5)
    ax1.invert_xaxis()
    ax1.grid(alpha=0.25, lw=0.6)

    ax2.set_xlabel("confidence threshold", fontsize=9)
    ax2.set_ylabel("fraction of held-out-type cells flagged", fontsize=9)
    ax2.set_title("Whether it finds the novel cells", fontsize=10.5)
    ax2.grid(alpha=0.25, lw=0.6)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Abstention is only useful if the threshold does something: "
                 "9 of 36 liver cell types withheld", fontsize=11.5, y=1.0)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.2, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("ABSTAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
