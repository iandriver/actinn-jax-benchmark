"""Figure: what the broad pass and the focused pass actually do to one dataset.

"Tier 1 / Tier 2" was jargon the paper never defined. This makes the two passes concrete on
a single query -- the withheld HLiCA liver study -- by showing the same UMAP three times:

  broad pass    the shipped census-wide reference (~800 human types). Identifies roughly
                what is present and which focused reference to load; abstains (grey) where
                confidence is below min_prob.
  focused pass  a 48-type HLiCA liver reference re-annotating the same cells.
  truth         the study's own labels.

The palettes deliberately differ between the broad panel and the other two: the broad
reference answers in the census vocabulary, the focused one in the study's. That mismatch is
the reason the hand-off exists, so hiding it would misrepresent the figure.

    .venv/bin/python benchmark/explore/plot_workflow_umap.py [--out docs/figures/fig_workflow_umap.png]
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import actinn_jax as aj
from benchmark import metrics

GREY = "#d9d9d9"

# Twelve hand-picked hues (Paul Tol bright/muted, extended). tab20 was the default and its
# paired light/dark ramps are hard to tell apart at 3-point marker size, which is exactly
# the regime this figure lives in.
PALETTE12 = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
             "#EE8866", "#117733", "#882255", "#44AA99", "#999933", "#332288"]
OTHER = "other"


# Blind truncation destroyed the distinction the figure exists to show: four endothelial
# subtypes all became "endothelial cell o…". Abbreviate the boilerplate instead, then check
# uniqueness, and only truncate what is still too long.
ABBREV = [
    ("CD8-positive, alpha-beta cytotoxic T cell", "CD8+ cytotoxic T"),
    ("CD8-positive, alpha-beta T cell", "CD8+ T"),
    ("mucosal-associated invariant T cell", "MAIT cell"),
    ("vascular associated smooth muscle cell", "vascular SMC"),
    ("endothelial cell of pericentral hepatic sinusoid", "EC pericentral sinusoid"),
    ("endothelial cell of periportal hepatic sinusoid", "EC periportal sinusoid"),
    ("endothelial cell of lymphatic vessel", "EC lymphatic"),
    ("endothelial cell of vascular tree", "EC vascular tree"),
    ("endothelial cell of sinusoid", "EC sinusoid"),
    ("endothelial cell of artery", "EC artery"),
    ("plasmacytoid dendritic cell", "pDC"),
    ("conventional dendritic cell", "cDC"),
    ("migratory dendritic cell", "migratory DC"),
    ("liver dendritic cell", "liver DC"),
    ("lipid-associated macrophage", "LAM"),
    ("large mucus secreting cholangiocyte", "mucus-secreting cholangiocyte"),
    ("centrilobular region hepatocyte", "centrilobular hepatocyte"),
    ("periportal region hepatocyte", "periportal hepatocyte"),
    ("midzonal region hepatocyte", "midzonal hepatocyte"),
    ("hepatic portal fibroblast", "portal fibroblast"),
    ("non-classical monocyte", "non-classical mono."),
    ("classical monocyte", "classical mono."),
    ("natural killer cell", "NK cell"),
    ("regulatory T cell", "Treg"),
    ("dendritic cell", "DC"),
]


def short_name(label, limit=24):
    out = label
    for long, brief in ABBREV:
        if out == long:
            out = brief
            break
    return (out[: limit - 1] + "…") if len(out) > limit else out


def short_names(labels, limit=24):
    """Shorten a set of labels, lengthening any that would otherwise collide."""
    out, seen = {}, {}
    for l in labels:
        cand = short_name(l, limit)
        if cand in seen and seen[cand] != l:      # two names collapsed onto one
            cand = short_name(l, limit + 12)
        seen[cand] = l
        out[l] = cand
    return out


def tissue_vote(labels, class_to_tissue, top=6):
    """Which tissue the broad pass implies, which is the step that picks the next reference.

    Classes the census marks pan-tissue ('*') carry no location information and are dropped;
    a class listed in several tissues splits its vote evenly rather than counting once per
    tissue, which would let a promiscuous label outvote a specific one.
    """
    import collections
    votes, pan, unmapped = collections.Counter(), 0, 0
    for l in labels:
        ts = class_to_tissue.get(l)
        if not ts:
            unmapped += 1
        elif "*" in ts:
            pan += 1
        else:
            for t in ts:
                votes[t] += 1.0 / len(ts)
    total = sum(votes.values()) or 1.0
    ranked = [(t, v / total) for t, v in votes.most_common(top)]
    return ranked, pan / max(1, len(labels)), unmapped


def route_panel(ax, ranked, pan_frac):
    names = [t for t, _ in ranked][::-1]
    fracs = [f for _, f in ranked][::-1]
    colors = ["#0072B2" if f == max(fracs) else "#c9c9c9" for f in fracs]
    ax.barh(range(len(names)), fracs, color=colors, height=0.68)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50%", "100%"], fontsize=7)
    ax.set_title("which tissue?", fontsize=10, pad=6)
    top_t, top_f = ranked[0]
    ax.text(0.5, -0.13, f"→ load the {top_t} reference\n({top_f:.0%} of tissue-specific calls;"
                        f" {pan_frac:.0%} pan-tissue, not counted)",
            transform=ax.transAxes, ha="center", fontsize=8, color="#0072B2")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)


def anchor(xy, mask, bins=14):
    """Where to write a label: the centre of its densest patch, not its median. A cell type
    split across two lobes of the embedding has its median in the empty space between them,
    which labels nothing and collides with whatever is there."""
    pts = xy[mask]
    if len(pts) < 25:
        return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
    i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    sel = ((pts[:, 0] >= xe[i]) & (pts[:, 0] <= xe[i + 1]) &
           (pts[:, 1] >= ye[j]) & (pts[:, 1] <= ye[j + 1]))
    core = pts[sel] if sel.any() else pts
    return float(np.median(core[:, 0])), float(np.median(core[:, 1]))


def declash(pos, min_dx, min_dy, iters=400):
    """Separate labels that would overlap. Text placed at cluster centres collided in six
    pairs on the first version ('hepatclymphocyte'), so nudge them apart vertically -- the
    cheap axis, since these names are wide and short."""
    p = np.array(pos, dtype=float)
    for _ in range(iters):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dx, dy = p[j, 0] - p[i, 0], p[j, 1] - p[i, 1]
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    push = (min_dy - abs(dy)) / 2 + 1e-6
                    sign = 1.0 if dy >= 0 else -1.0
                    p[i, 1] -= sign * push
                    p[j, 1] += sign * push
                    moved = True
        if not moved:
            break
    return p


def top_labels(labels, n=12):
    """The n most abundant labels, most-frequent first. Everything else is folded into a
    single grey class: past a dozen categories a qualitative palette stops being readable,
    and the tail here is a long list of one-cell types."""
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    ranked = [l for l, _ in sorted(counts.items(), key=lambda kv: -kv[1])
              if l not in ("unknown",)]
    return ranked[:n]


def collapse(labels, keep):
    keep = set(keep)
    return np.array([l if l in keep else (l if l == "unknown" else OTHER) for l in labels])


def concordance(true_cl, pred_cl, anc):
    ok = n = 0
    for t, p in zip(true_cl, pred_cl):
        if not t or t in ("unknown", "nan", ""):
            continue
        n += 1
        ok += bool(p and p not in ("unknown", "nan", "") and
                   (p == t or p in anc.get(t, ()) or t in anc.get(p, ())))
    return ok / n if n else float("nan")


def panel(ax, xy, labels, title, palette, subtitle=None, style="legend", n_show=12):
    """Draw one UMAP. `style` is 'legend' (key below the panel) or 'ondata' (names written
    on the clusters). Grey classes are drawn first so coloured cells sit on top of them."""
    order = [l for l, _ in sorted(
        ((l, int((labels == l).sum())) for l in set(labels)),
        key=lambda kv: -kv[1])]
    named = [l for l in order if l not in (OTHER, "unknown")][:n_show]
    for lab in [l for l in order if l in (OTHER, "unknown")] + named:
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=3, linewidths=0, c=palette.get(lab, GREY),
                   zorder=1 if lab in (OTHER, "unknown") else 2)
    ax.set_title(title, fontsize=11, pad=6)
    if subtitle:
        ax.text(0.5, -0.04 if style == "ondata" else -0.06, subtitle,
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#444")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if style == "ondata":
        drawn = [(lab, anchor(xy, labels == lab)) for lab in named
                 if int((labels == lab).sum()) >= 12]
        if drawn:
            xr = float(xy[:, 0].max() - xy[:, 0].min())
            yr = float(xy[:, 1].max() - xy[:, 1].min())
            pos = declash([p for _, p in drawn], min_dx=0.34 * xr, min_dy=0.052 * yr)
            names = short_names([lab for lab, _ in drawn], limit=26)
            for (lab, (x0, y0)), (x, y) in zip(drawn, pos):
                short = names[lab]
                if abs(y - y0) > 0.012 * yr:      # show where a nudged label belongs
                    ax.plot([x0, x], [y0, y], lw=0.6, color="#555", zorder=3, alpha=0.8)
                ax.text(x, y, short, fontsize=7.0, ha="center", va="center", zorder=4,
                        color="#111", fontweight="semibold",
                        path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
        return

    names = short_names(named, limit=30)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5,
                      color=palette.get(l, GREY), label=names[l])
               for l in named]
    n_other = int((labels == OTHER).sum())
    if n_other:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=5, color=GREY,
                              label=f"other ({n_other:,} cells)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.10),
              frameon=False, fontsize=7.5, ncol=2, handletextpad=0.4,
              columnspacing=0.9, labelspacing=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="/Volumes/IanSSD/hlica/liver_query_xstudy.h5ad")
    ap.add_argument("--broad", default="broad_human_v1")
    # The SHIPPED liver_hlica_v2 is trained on all HLiCA studies including the one withheld
    # here, so scoring it on this query leaks: it reports 0.936 against the 0.86 the paper
    # quotes from a leakage-free model. Train the focused reference on the 6-study reference
    # instead, which is what section 3.5 measures.
    ap.add_argument("--focused-train", default="/Volumes/IanSSD/hlica/liver_ref_xstudy.h5ad",
                    help="h5ad to train the focused reference from (leakage-free)")
    ap.add_argument("--focused-label", default="cell_type")
    ap.add_argument("--focused", default=None,
                    help="instead load a bundled reference by name (may leak; see above)")
    # No abstain by default. With min_prob set, abstained cells go grey but still count as
    # misses in the concordance below, which prints a number far under the one the text
    # quotes for the same models (0.21 vs 0.58) -- a figure that appears to contradict the
    # paper. Abstain has its own section; this figure is about what the two passes label.
    ap.add_argument("--min-prob", type=float, default=None)
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--out", default="docs/figures/fig_workflow_umap.png")
    ap.add_argument("--style", choices=("legend", "ondata"), default="legend",
                    help="legend below each panel, or names written on the clusters")
    ap.add_argument("--top", type=int, default=12,
                    help="colour only the N most abundant labels per panel; rest go grey")
    a = ap.parse_args()

    anc = metrics.load_cl_ancestors(a.obo)
    q = sc.read_h5ad(a.query)
    truth = q.obs["cell_type"].astype(str).to_numpy()
    n_truth = len(set(truth))
    true_cl = q.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
    print(f"query {q.shape} | {len(set(truth))} truth types", flush=True)

    def focused_model():
        if a.focused:
            return aj.bundled_reference(a.focused), a.focused
        ref = sc.read_h5ad(a.focused_train)
        raw = ref.copy()
        sc.pp.normalize_total(raw, target_sum=1e4)
        sc.pp.log1p(raw)
        sc.pp.highly_variable_genes(raw, n_top_genes=min(4000, raw.n_vars))
        ref = ref[:, raw.var["highly_variable"].values].copy()
        print(f"training focused reference on {ref.shape} "
              f"({ref.obs[a.focused_label].nunique()} types)", flush=True)
        m = aj.train_reference(ref, train_label_name=a.focused_label, print_cost=False)
        return m, f"trained on {os.path.basename(a.focused_train)}"

    calls, scores = {}, {}
    for tag in ("broad", "focused"):
        if tag == "broad":
            model, name = aj.bundled_reference(a.broad), a.broad
            broad_model = model
        else:
            model, name = focused_model()
        kw = {"min_prob": a.min_prob} if a.min_prob else {}
        try:
            frame = model.predict_frame(q, **kw)[0]
        except TypeError:                 # flat ReferenceModel: no min_prob, needs use_raw
            frame = model.predict_frame(q, use_raw="auto")[0]
        lab = frame["celltype"].to_numpy().astype(str)
        # Hierarchical references carry a class->CL map; a flat one trained here does not,
        # but it predicts into the query's own vocabulary, so the query supplies the mapping.
        cl_map = dict(getattr(model, "class_to_cl", None) or {})
        if not cl_map:
            cl_map = dict(zip(truth, true_cl))
        pred_cl = np.array([cl_map.get(p, "unknown") for p in lab])
        calls[tag] = lab
        scores[tag] = (concordance(true_cl, pred_cl, anc),
                       float((lab == "unknown").mean()), len(model.classes))
        print(f"{tag:<8} {name:<20} ontology={scores[tag][0]:.3f} "
              f"abstained={scores[tag][1]:.1%} classes={scores[tag][2]}", flush=True)

    # UMAP on the query alone: the figure is about how the same cells get labelled, so the
    # embedding must not depend on either reference.
    print("computing UMAP", flush=True)
    e = q.copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    sc.pp.highly_variable_genes(e, n_top_genes=2000)
    e = e[:, e.var.highly_variable].copy()
    sc.pp.scale(e, max_value=10)
    sc.tl.pca(e, n_comps=50)
    sc.pp.neighbors(e, n_neighbors=15)
    sc.tl.umap(e)
    xy = e.obsm["X_umap"]

    # Only the most abundant labels get a colour. The focused pass and the truth share the
    # study's vocabulary, so their top set is taken from the union ranked by truth abundance
    # -- that keeps a type the same colour in both panels, which is the comparison the figure
    # is for. The broad panel answers in the census vocabulary and is ranked on its own.
    truth_top = top_labels(truth, a.top)
    focused_extra = [l for l in top_labels(calls["focused"], a.top) if l not in truth_top]
    shared_named = truth_top + focused_extra[:max(0, a.top - len(truth_top))]
    pal_shared = {l: PALETTE12[i % len(PALETTE12)] for i, l in enumerate(shared_named)}
    pal_shared.update({"unknown": GREY, OTHER: GREY})

    broad_named = top_labels(calls["broad"], a.top)
    pal_broad = {l: PALETTE12[i % len(PALETTE12)] for i, l in enumerate(broad_named)}
    pal_broad.update({"unknown": GREY, OTHER: GREY})

    raw_broad = calls["broad"].copy()      # tissue vote needs the uncollapsed labels
    truth = collapse(truth, shared_named)
    calls["focused"] = collapse(calls["focused"], shared_named)
    calls["broad"] = collapse(calls["broad"], broad_named)

    # Four columns, not three: the middle one is the routing decision, which is the step
    # that connects the two passes and was the only part of the workflow the figure never
    # actually showed.
    ranked, pan_frac, _ = tissue_vote(raw_broad,
                                      dict(getattr(broad_model, "class_to_tissue", {}) or {}))
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8 if a.style == "ondata" else 5.4),
                             gridspec_kw={"width_ratios": [1, 0.52, 1, 1], "wspace": 0.16})
    route_panel(axes[1], ranked, pan_frac)
    axes = [axes[0], axes[2], axes[3]]
    ob, ab, nb = scores["broad"]
    of, af, nf = scores["focused"]
    ab_txt = f" · {ab:.0%} abstained (grey)" if a.min_prob else ""
    af_txt = f" · {af:.0%} abstained" if a.min_prob else ""
    panel(axes[0], xy, calls["broad"],
          f"broad pass — census reference ({nb} types)",
          pal_broad,
          f"ontology concordance {ob:.2f}{ab_txt}", style=a.style, n_show=a.top)
    panel(axes[1], xy, calls["focused"],
          f"focused pass — liver reference ({nf} types)",
          pal_shared,
          f"ontology concordance {of:.2f}{af_txt}", style=a.style, n_show=a.top)
    panel(axes[2], xy, truth, "study's own labels", pal_shared,
          f"{n_truth} types, {a.top} shown", style=a.style, n_show=a.top)
    fig.suptitle("Broad pass identifies the tissue · that picks the reference · "
                 "the focused reference gives the granular labels", fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("WORKFLOW_UMAP_DONE", flush=True)


if __name__ == "__main__":
    main()
