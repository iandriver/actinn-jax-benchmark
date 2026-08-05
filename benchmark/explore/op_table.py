"""Assemble the completed Open Problems grid into a results CSV and the paper's table.

Also prints the actinn-jax local-vs-AWS accuracy check: the same component, same inputs, on
a laptop instead of an r7i.8xlarge. Accuracy is machine-independent, and this is what shows
it -- without it, completing the coverage locally would be an assumption rather than a
measurement.
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AWS = "/Users/iandriver/Downloads/actinn-jax-benchmark/docs/results_openproblems_samehw.csv"
DATASETS = ["dkd", "gtex_v9", "hypomap", "immune_cell_atlas",
            "mouse_pancreas_atlas", "tabula_sapiens"]

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/opres/*.json"))]
d = pd.DataFrame(rows)
print(f"{len(d)} runs, {d.method.nunique()} methods x {d.dataset.nunique()} datasets\n")

grid = d.pivot(index="method", columns="dataset", values="accuracy").reindex(columns=DATASETS)
missing = grid.isna()
if missing.any().any():
    print("INCOMPLETE:")
    for m in grid.index:
        gaps = [c for c in grid.columns if missing.loc[m, c]]
        if gaps:
            print(f"  {m}: missing {gaps}")
    print()

print("accuracy by dataset")
print(grid.round(4).to_string(), "\n")
print("means")
print(d.groupby("method")[["accuracy", "f1_macro"]].mean().round(4).to_string(), "\n")

# --- local vs AWS, same component and inputs ---
aws = pd.read_csv(AWS).set_index("method")
if "actinn_jax" in grid.index and "actinn_jax" in aws.index:
    print("actinn-jax: local vs the controlled AWS run")
    print(f"{'dataset':<24}{'local':>9}{'aws':>9}{'diff':>9}")
    diffs = []
    for ds in DATASETS:
        loc, a = grid.loc["actinn_jax", ds], aws.loc["actinn_jax", f"acc_{ds}"]
        if pd.notna(loc):
            diffs.append(abs(loc - a))
            print(f"{ds:<24}{loc:>9.4f}{a:>9.4f}{loc - a:>+9.4f}")
    if diffs:
        print(f"{'max |diff|':<24}{'':>9}{'':>9}{max(diffs):>9.4f}")
        print(f"{'mean local':<24}{grid.loc['actinn_jax'].mean():>9.4f}"
              f"{aws.loc['actinn_jax', 'mean_accuracy']:>9.4f}"
              f"{grid.loc['actinn_jax'].mean() - aws.loc['actinn_jax', 'mean_accuracy']:>+9.4f}")

out = f"{HERE}/results_openproblems_added_methods.csv"
d.sort_values(["method", "dataset"]).to_csv(out, index=False)
print(f"\nwrote {out}")

# --- markdown table for the paper: the four added methods, all six datasets ---
ADDED = ["linear_anova_pca", "svm_sgd", "celltypist", "sctop"]
NAME = {"linear_anova_pca": "linear-anova-pca", "svm_sgd": "SVM (SGD)",
        "celltypist": "CellTypist", "sctop": "scTOP"}
sub = d[d.method.isin(ADDED)]
if len(sub):
    means = sub.groupby("method")[["accuracy", "f1_macro"]].mean()
    g = sub.pivot(index="method", columns="dataset", values="accuracy").reindex(columns=DATASETS)
    print("\n--- paper table ---")
    print("| method | mean acc | macro-F1 | " + " | ".join(DATASETS) + " |")
    print("|---" * (3 + len(DATASETS)) + "|")
    for m in sorted(ADDED, key=lambda x: -means.loc[x, "accuracy"] if x in means.index else 0):
        if m not in means.index:
            continue
        cells = " | ".join(f"{g.loc[m, c]:.3f}" if pd.notna(g.loc[m, c]) else "—"
                           for c in DATASETS)
        print(f"| {NAME[m]} | {means.loc[m, 'accuracy']:.3f} | "
              f"{means.loc[m, 'f1_macro']:.3f} | {cells} |")
