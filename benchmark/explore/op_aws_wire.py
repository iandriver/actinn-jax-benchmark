"""Wire our five components into the Open Problems label_projection benchmark workflow.

Runs ON the AWS box (sent via SSM). A component alone is invisible to `run_benchmark`:
it must appear BOTH in the workflow's `dependencies:` and in the `methods = [...]` array
of main.nf, or it is silently dropped from the DAG with no error.

Also trims the method set to the CPU tier. The GPU/foundation methods (geneformer,
scgpt_*, scimilarity*, scprint, uce) need accelerators and multi-GB model downloads;
they are out of scope for this CPU box and are dropped rather than left to fail.
"""

import pathlib
import re
import sys

REPO = pathlib.Path("/home/ubuntu/task_label_projection")
OURS = ["actinn_jax", "linear_anova_pca", "sctop", "svm_sgd", "celltypist"]

# CPU-runnable OP methods + controls, in the order they should appear.
OP_KEEP = [
    "majority_vote", "random_labels", "true_labels",
    "knn", "logistic_regression", "mlp", "naive_bayes",
    "cellmapper_linear", "cellmapper_scvi",
    "scanvi", "scanvi_scarches",
    "seurat_transferdata", "singler", "xgboost",
]
CONTROLS = {"majority_vote", "random_labels", "true_labels"}

# ---- 1. dependencies ---------------------------------------------------------------
cfg_p = REPO / "src/workflows/run_benchmark/config.vsh.yaml"
cfg = cfg_p.read_text()

deps = []
for m in OP_KEEP:
    deps.append(f"  - name: {'control_methods' if m in CONTROLS else 'methods'}/{m}")
for m in OURS:
    deps.append(f"  - name: methods/{m}")

block = re.search(r"^dependencies:\n(?:.*\n)*?(?=^\S|\Z)", cfg, re.M)
if not block:
    sys.exit("could not locate dependencies: block")
old = block.group(0)
keep_head = "dependencies:\n  - name: utils/extract_uns_metadata\n    repository: openproblems\n"
new = keep_head + "\n".join(deps) + "\n  - name: metrics/accuracy\n  - name: metrics/f1\n\n"
cfg_p.write_text(cfg.replace(old, new))
print(f"dependencies: {len(deps) + 3} entries")

# ---- 2. methods array --------------------------------------------------------------
nf_p = REPO / "src/workflows/run_benchmark/main.nf"
nf = nf_p.read_text()
arr = re.search(r"methods = \[.*?\n\]", nf, re.S)
if not arr:
    sys.exit("could not locate methods = [...] array")
entries = OP_KEEP + OURS
nf_p.write_text(nf.replace(arr.group(0),
                           "methods = [\n  " + ",\n  ".join(entries) + "\n]"))
print(f"methods array: {len(entries)} entries -> {', '.join(entries)}")
