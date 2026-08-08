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

STYLE = {
    "actinn-jax": dict(color="#0072B2", marker="o", lw=2.0, zorder=3),
    "celltypist": dict(color="#D55E00", marker="s", lw=2.0, zorder=2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/results_rejection.csv")
    ap.add_argument("--out", default="docs/figures/fig_abstain.png")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    for method, g in d.groupby("method"):
        g = g.sort_values("min_prob")
        st = STYLE.get(method, dict(color="#666", marker="^", lw=1.8))
        ax1.plot(g["coverage"], g["indist_acc_kept"], markersize=5.5, label=method, **st)
        ax2.plot(g["min_prob"], g["ood_flagged"], markersize=5.5, label=method, **st)
        # A saturated method stacks several thresholds on one point, so labelling every
        # marker just overprints; label the ends, which is where the shape shows.
        for _, r in g.iterrows():
            if r["min_prob"] in (g["min_prob"].min(), g["min_prob"].max()):
                ax1.annotate(f"p≥{r['min_prob']:g}", (r["coverage"], r["indist_acc_kept"]),
                             textcoords="offset points", xytext=(6, -9), fontsize=7,
                             color=st["color"])

    # Say the saturation out loud: four of celltypist's five thresholds land on the same
    # coordinate, so the line looks like two points unless the reader is told why.
    # 2 dp, not 3: celltypist's thresholds sit at coverage .681/.680/.680/.678, which are
    # the same operating point to any practical reader but four distinct values at 3 dp.
    sat = d[d.method == "celltypist"].round({"coverage": 2, "indist_acc_kept": 2})
    dup = sat.groupby(["coverage", "indist_acc_kept"]).size()
    if (dup > 1).any():
        (cov, acc) = dup.idxmax()
        ax1.annotate(f"p≥0.3–0.9 all land here\n({int(dup.max())} thresholds, one operating point)",
                     (cov, acc), textcoords="offset points", xytext=(-14, 26),
                     fontsize=7.5, color="#D55E00", ha="right",
                     arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.9))

    ax1.set_xlabel("coverage — fraction of cells kept", fontsize=9)
    ax1.set_ylabel("accuracy on kept cells", fontsize=9)
    ax1.set_title("What abstaining buys", fontsize=10.5)
    ax1.invert_xaxis()
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax2.set_xlabel("confidence threshold", fontsize=9)
    ax2.set_ylabel("fraction of held-out-type cells flagged", fontsize=9)
    ax2.set_title("Whether it finds the novel cells", fontsize=10.5)
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Abstention is only useful if the threshold does something: "
                 "9 of 36 liver cell types withheld", fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("ABSTAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
