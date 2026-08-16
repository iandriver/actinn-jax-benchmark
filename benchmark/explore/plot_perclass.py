"""Per-class recall across methods: do they fail on the same cell types, or different ones?

Summary accuracy cannot answer that. Two methods at 0.83 can be identical or can be failing
on disjoint halves of the label set, and only the second case makes an ensemble or a routed
workflow worth anything.

Rows are cell types, columns are methods ordered by mean recall, each cell is recall.
Classes are sorted by how much the methods *disagree* about them (max recall minus min), and
the right-hand strip shows that spread, because disagreement is what the data actually
contains. On the 86-class blood+gut split the best and worst method differ by a median of
0.30 recall per class (max 0.73), and the mean pairwise Spearman between methods' per-class
recall is 0.58 -- correlated, but a long way from the 1.0 that "they all fail on the same
types" would need.

An earlier version sorted by abundance and was captioned "methods fail in the same place:
the rare classes". Both halves were wrong. Capping per label leaves 84 of the 86 classes with
exactly 30 test cells (the other two hold 29 and 14), so rarity is not a variable here at all.

The fix for that caption then over-corrected, comparing the ten largest classes against the
ten smallest. That comparison is empty on this split: with 84 classes tied at 30 cells, both
groups are arbitrary samples from one pool, and `value_counts()` decides which is which. The
reported gap moved from 0.038 to 0.005 between two runs on identical predictions, purely
because the tie order changed. The size summary below prints the tie count instead.

    .venv/bin/python benchmark/explore/plot_perclass.py --dataset blood_gut_intra
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from benchmark.explore.plot_workflow_umap import short_names


def load(preds_dir, dataset):
    """method -> DataFrame of per-cell truth/prediction for one dataset."""
    out = {}
    for f in sorted(glob.glob(os.path.join(preds_dir, f"{dataset}__*__rep0.parquet"))):
        method = os.path.basename(f).split("__")[1]
        d = pd.read_parquet(f)
        col = "pred_label" if "pred_label" in d else "celltype"
        out[method] = d[["truth", col]].rename(columns={col: "pred"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/percell/predictions")
    ap.add_argument("--dataset", default="blood_gut_intra")
    ap.add_argument("--top", type=int, default=30,
                    help="classes to show, chosen by how much the methods disagree")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"docs/figures/fig_perclass_{a.dataset}.png"

    per = load(a.preds, a.dataset)
    if not per:
        sys.exit(f"no predictions for {a.dataset} in {a.preds}")
    truth = next(iter(per.values()))["truth"].to_numpy()
    sizes = pd.Series(truth).value_counts()
    print(f"{a.dataset}: {len(per)} methods, {sizes.size} classes, showing {a.top}",
          flush=True)

    classes = list(sizes.index)                 # score every class, then rank by spread
    rec = pd.DataFrame(index=classes, columns=sorted(per), dtype=float)
    for m, d in per.items():
        t, p = d["truth"].to_numpy(), d["pred"].to_numpy()
        for c in classes:
            sel = t == c
            rec.loc[c, m] = float((p[sel] == c).mean()) if sel.any() else np.nan
    rec = rec[rec.mean().sort_values(ascending=False).index]      # best methods leftmost
    spread = (rec.max(axis=1) - rec.min(axis=1)).sort_values(ascending=False)
    rec = rec.loc[spread.index].head(a.top)
    spread = spread.head(a.top)
    classes = list(rec.index)

    fig, (ax, axn) = plt.subplots(
        1, 2, figsize=(1.05 * len(rec.columns) + 5.0, 0.30 * len(classes) + 2.4),
        gridspec_kw={"width_ratios": [len(rec.columns), 1.5], "wspace": 0.04})

    im = ax.imshow(rec.to_numpy(dtype=float), cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rec.columns)))
    ax.set_xticklabels(rec.columns, rotation=40, ha="right", fontsize=8)
    names = short_names(classes, 34)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([names[c] for c in classes], fontsize=7)
    for i in range(len(classes)):
        for j in range(len(rec.columns)):
            v = rec.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1",
                        ha="center", va="center", fontsize=5.8,
                        color="#222" if 0.25 < v < 0.85 else "white")
    ax.set_title(f"per-class recall — {a.dataset}", fontsize=11, pad=8)

    # Spread, not class size. The split is capped per label, so a size strip here is a
    # column of identical bars; the informative quantity is how far apart the methods are.
    axn.barh(range(len(classes)), [spread[c] for c in classes], color="#B23A48", height=0.72)
    axn.set_yticks([]); axn.set_xlim(0, 1)
    axn.set_xlabel("best − worst\nmethod recall", fontsize=8)
    axn.tick_params(labelsize=7)
    axn.invert_yaxis(); ax.invert_yaxis()
    for s in ("top", "right"):
        axn.spines[s].set_visible(False)

    cb = fig.colorbar(im, ax=axn, fraction=0.06, pad=0.22)
    cb.set_label("recall", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Methods disagree about which cell types are hard — a class one method "
                 "misses is often another's best", fontsize=12, y=1.0)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # The claim in the title is checkable, so check it rather than asserting it.
    full = pd.DataFrame(index=list(sizes.index), columns=sorted(per), dtype=float)
    for mth, dd in per.items():
        t, pp = dd["truth"].to_numpy(), dd["pred"].to_numpy()
        for c in full.index:
            sel = t == c
            full.loc[c, mth] = float((pp[sel] == c).mean()) if sel.any() else np.nan
    m = full.dropna()
    if len(m.columns) > 1:
        cc = m.corr(method="spearman").to_numpy()
        off = cc[np.triu_indices_from(cc, k=1)]
        print(f"mean pairwise Spearman of per-class recall between methods: {off.mean():.3f} "
              f"(1.0 would mean identical difficulty ranking)")
        top = sizes.value_counts().idxmax()
        print(f"class sizes in the test split: {sizes.min()}-{sizes.max()} cells, "
              f"{(sizes == top).sum()} of {len(sizes)} classes tied at {top} "
              f"-- too little spread to attribute recall differences to class size")
        sp = (m.max(axis=1) - m.min(axis=1))
        print(f"best-minus-worst method recall per class: median {sp.median():.2f}, "
              f"max {sp.max():.2f}")
    print("PERCLASS_DONE", flush=True)


if __name__ == "__main__":
    main()
