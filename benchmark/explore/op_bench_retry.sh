#!/bin/bash
# Robust recovery pass: for any OP dataset not yet in the results CSV, (re)download its
# train/test/solution with retries + size verification, then run actinn-jax. Idempotent.
set -u
B="s3://openproblems-data/resources/task_label_projection/datasets/cellxgene_census"
ROOT="/Volumes/IanSSD/op_label_projection"
PY="/Users/iandriver/Downloads/celltype_predict_ACTINN/.venv/bin/python"
CSV="/Users/iandriver/Downloads/actinn-jax-benchmark/docs/results_openproblems.csv"
cd /Users/iandriver/Downloads/actinn-jax-benchmark

dl() {  # dl <s3key> <dest> <min_bytes> : skip if present+valid, else retry up to 4x
  local key="$1" dest="$2" minb="$3" i sz
  sz=$(stat -f%z "$dest" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$minb" ]; then echo "  have $(basename "$dest") ($sz bytes)"; return 0; fi
  for i in 1 2 3 4; do
    aws s3 cp --no-sign-request --cli-read-timeout 0 "$key" "$dest" >/dev/null 2>&1
    sz=$(stat -f%z "$dest" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$minb" ]; then return 0; fi
    echo "  retry $i for $(basename "$dest") (got $sz bytes)"; sleep 10
  done
  return 1
}

for ds in gtex_v9 immune_cell_atlas hypomap mouse_pancreas_atlas tabula_sapiens; do
  if grep -q "^${ds}," "$CSV" 2>/dev/null; then echo "SKIP $ds (scored)"; continue; fi
  echo "=== $ds: (re)downloading with retries ==="
  mkdir -p "$ROOT/$ds"
  dl "$B/$ds/log_cp10k/train.h5ad"    "$ROOT/$ds/train.h5ad"    1000000000 || { echo "FAILED dl $ds train"; continue; }
  dl "$B/$ds/log_cp10k/test.h5ad"     "$ROOT/$ds/test.h5ad"     10000000  || { echo "FAILED dl $ds test"; continue; }
  dl "$B/$ds/log_cp10k/solution.h5ad" "$ROOT/$ds/solution.h5ad" 10000000  || { echo "FAILED dl $ds solution"; continue; }
  echo "=== $ds: running actinn-jax ==="
  $PY -u benchmark/explore/op_runner.py "$ROOT/$ds" "$CSV" 2>&1 | grep -iE "^${ds}:|RESULT|Error|Traceback|MemoryError" | tail -6
  echo "=== $ds DONE ==="
done
echo "OP_RETRY_DONE"
