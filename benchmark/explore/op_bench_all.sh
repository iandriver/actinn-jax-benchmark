#!/bin/bash
# Download each OP label_projection dataset (log_cp10k variant) and run actinn-jax on it,
# appending to docs/results_openproblems.csv. Smallest-first; skips datasets already scored.
set -u
B="s3://openproblems-data/resources/task_label_projection/datasets/cellxgene_census"
ROOT="/Volumes/IanSSD/op_label_projection"
PY="/Users/iandriver/Downloads/celltype_predict_ACTINN/.venv/bin/python"
CSV="/Users/iandriver/Downloads/actinn-jax-benchmark/docs/results_openproblems.csv"
cd /Users/iandriver/Downloads/actinn-jax-benchmark

for ds in gtex_v9 immune_cell_atlas hypomap mouse_pancreas_atlas tabula_sapiens; do
  if grep -q "^${ds}," "$CSV" 2>/dev/null; then echo "SKIP $ds (already scored)"; continue; fi
  echo "=== $ds: downloading ==="
  mkdir -p "$ROOT/$ds"
  for f in train test solution; do
    [ -f "$ROOT/$ds/$f.h5ad" ] || aws s3 cp --no-sign-request --quiet "$B/$ds/log_cp10k/$f.h5ad" "$ROOT/$ds/$f.h5ad"
  done
  echo "=== $ds: running actinn-jax ==="
  $PY -u benchmark/explore/op_runner.py "$ROOT/$ds" "$CSV" 2>&1 | grep -iE "^${ds}:|RESULT|Error|Traceback|MemoryError" | tail -6
  echo "=== $ds DONE ==="
done
echo "OP_BENCH_ALL_DONE"
