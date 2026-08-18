"""Rebuild the paper's accuracy tables from the recorded matrix runs.

Tables 3 and 5 were assembled by hand from three result directories, which makes them
tedious to re-check and easy to drift when a dataset is added. This reads whichever of
those directories exist and prints both: the per-dataset leader board behind Table 5 and
the aggregate means behind Table 3.

Repeats are averaged, which is what those tables report. Accuracy only: the cost columns
of Table 3 were reconciled separately against idle-machine reruns, so recomputing them
here would produce a second, quieter set of numbers that disagrees with the paper.

    .venv/bin/python benchmark/explore/summarize_paper_matrix.py
"""

import argparse
import os

import pandas as pd

RUNS = ["results/paper", "results/paper_baselines", "results/paper_brain"]
# Accuracy aggregate excludes lung_cross: its exact-match score is a vocabulary artefact.
NO_ACC = {"lung_cross"}
DROP = {"actinn-orig"}          # benchmarked, then cut from the paper panel
NAMES = {"actinn-jax": "actinn-jax", "linear-anova-pca": "linear-anova-pca",
         "protocloud": "ProtoCloud", "sctop": "scTOP", "svm": "SVM", "knn": "kNN",
         "celltypist": "CellTypist", "singler": "SingleR",
         "scmap-cluster": "scmap-cluster", "scanvi": "scANVI", "scarches": "scArches"}


def load(runs):
    frames = []
    for r in runs:
        p = os.path.join(r, "results.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["run"] = r
            frames.append(df[df.get("error").isna()] if "error" in df else df)
    if not frames:
        raise SystemExit("no results found")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict the aggregate to these datasets")
    a = ap.parse_args()

    df = load(a.runs)
    df = df[~df.method.isin(DROP)]
    df["method"] = df.method.map(lambda m: NAMES.get(m, m))
    g = df.groupby(["dataset", "method"])
    acc = g.accuracy.mean().unstack("method")
    ont = g.ontology_concordance.mean().unstack("method")
    f1 = g.macro_f1.mean().unstack("method")

    print("=== per dataset (accuracy, mean of repeats) ===")
    for ds in acc.index:
        row = acc.loc[ds].dropna().sort_values(ascending=False)
        aj = row.get("actinn-jax", float("nan"))
        lead, best = row.index[0], row.iloc[0]
        o = ont.loc[ds].dropna()
        extra = (f"   [ontology: actinn-jax {o.get('actinn-jax'):.3f}, "
                 f"best {o.idxmax()} {o.max():.3f}]" if len(o) else "")
        print(f"{ds:<16} n={int(row.count()):2d}  actinn-jax {aj:.3f}   "
              f"best {lead} {best:.3f}{extra}")

    keep = a.datasets or list(acc.index)
    print(f"\n=== aggregate over {len(keep)} datasets: {', '.join(keep)} ===")
    accm = acc.loc[[d for d in keep if d not in NO_ACC]].mean()
    f1m = f1.loc[keep].mean()
    ontm = ont.loc[keep].dropna(axis=1, how="all").mean()
    n_acc = len([d for d in keep if d not in NO_ACC])
    out = pd.DataFrame({f"acc({n_acc})": accm, f"macroF1({len(keep)})": f1m,
                        f"ontology({int(ont.loc[keep].notna().any(axis=1).sum())})": ontm})
    out = out.sort_values(f"acc({n_acc})", ascending=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
