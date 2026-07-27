#!/usr/bin/env bash
# Provision + run the Open Problems benchmark so that losing the instance costs time,
# not work. Sent via SSM; started under systemd so it outlives the SSM command.
#
# Durability, in layers:
#   1. Nextflow `-resume` against a work/ dir that lives on a volume created with
#      DeleteOnTermination=false, so every completed task survives instance death and a
#      restart picks up where it stopped.
#   2. A publisher loop mirrors out/ (trace, scores, heartbeat) to S3 every 60 s, so
#      partial results and progress are readable even when the box is gone.
#
# The heartbeat is the monitoring fix: it carries a timestamp, so a stalled or dead run
# is visible as a stale file rather than as silence. A monitor that can only report good
# news cannot distinguish "healthy" from "vanished" -- that is what went wrong last time.
set -euxo pipefail

S3=s3://rustar-bench/actinn-op-bench
REPO=/home/ubuntu/task_label_projection
OUT=/home/ubuntu/out
export DEBIAN_FRONTEND=noninteractive
export HOME=/root

# ---------- toolchain (each checked independently) ---------------------------------
if ! command -v docker >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq docker.io awscli git unzip curl
  systemctl enable --now docker
fi
# Nextflow needs Java 17+; Ubuntu 22.04's default-jre is Java 11 and it refuses to start.
if ! java -version 2>&1 | grep -qE '"(1[7-9]|2[0-9])'; then
  apt-get install -y -qq openjdk-17-jre-headless
fi
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
command -v nextflow >/dev/null || { curl -s https://get.nextflow.io | bash; mv nextflow /usr/local/bin/; chmod +x /usr/local/bin/nextflow; }
command -v viash >/dev/null || { curl -fsSL https://github.com/viash-io/viash/releases/latest/download/viash -o /usr/local/bin/viash; chmod +x /usr/local/bin/viash; }

mkdir -p "$OUT"

# ---------- repo (git runs as root against an ubuntu-owned tree) --------------------
git config --global --add safe.directory "$REPO"
git config --global --add safe.directory "$REPO/common"
[ -d "$REPO" ] || git clone -q https://github.com/openproblems-bio/task_label_projection.git "$REPO"
cd "$REPO"
git submodule update --init --recursive          # common/ holds helper.nf; build fails without it

# ---------- our components ---------------------------------------------------------
tar xzf /home/ubuntu/components.tgz -C /tmp
for c in actinn_jax linear_anova_pca sctop svm_sgd celltypist; do
  mkdir -p "src/methods/$c" && cp /tmp/$c/* "src/methods/$c/"
done
python3 /home/ubuntu/op_aws_wire.py

# ---------- data: log_cp10k only ----------------------------------------------------
# The bucket carries three normalizations (~196 GB); the workflow filters every method to
# its preferred_normalization, so the other two are pure download cost.
aws s3 sync --no-sign-request --only-show-errors \
  --exclude "*" --include "*/log_cp10k/*" \
  s3://openproblems-data/resources/task_label_projection/datasets/cellxgene_census \
  resources/datasets/cellxgene_census
echo "DATA: $(du -sh resources/datasets | cut -f1)"

# ---------- build ------------------------------------------------------------------
viash ns build
for c in methods/actinn_jax methods/linear_anova_pca methods/sctop methods/svm_sgd \
         methods/celltypist methods/knn methods/logistic_regression methods/mlp \
         methods/naive_bayes methods/xgboost methods/cellmapper_linear \
         methods/cellmapper_scvi methods/scanvi methods/scanvi_scarches \
         metrics/accuracy metrics/f1 control_methods/true_labels \
         control_methods/majority_vote control_methods/random_labels; do
  viash build "src/$c/config.vsh.yaml" --engine docker --setup cachedbuild -o /tmp/bimg || echo "IMGFAIL $c"
done

# ---------- resource config ---------------------------------------------------------
cat > /root/big.config <<'CFG'
process {
  withLabel: lowmem  { memory = '80 GB' }
  withLabel: midmem  { memory = '100 GB' }
  withLabel: highmem { memory = '120 GB' }
  withLabel: lowcpu  { cpus = 4 }
  withLabel: midcpu  { cpus = 8 }
  withLabel: highcpu { cpus = 16 }
  errorStrategy = 'ignore'
}
executor { memory = '180 GB'; cpus = 32; queueSize = 2 }
trace  { enabled = true; overwrite = false; file = '/home/ubuntu/out/trace.txt'
         fields = 'task_id,process,tag,status,exit,realtime,duration,%cpu,peak_rss,peak_vmem' }
report { enabled = true; overwrite = true; file = '/home/ubuntu/out/report.html' }
CFG

# Only the six cellxgene_census datasets: the same-hardware table this extends is built
# from those, and OP has since added allen_brain_cell_atlas.
cat > /root/params.yaml <<'YAML'
input_states: resources/datasets/cellxgene_census/**/state.yaml
rename_keys: 'input_train:output_train;input_test:output_test;input_solution:output_solution'
output_state: "state.yaml"
settings: '{"methods_include": ["actinn_jax","linear_anova_pca","sctop","svm_sgd","celltypist","mlp","knn","logistic_regression","naive_bayes","xgboost","cellmapper_linear","cellmapper_scvi","scanvi","scanvi_scarches","true_labels","majority_vote","random_labels"]}'
publish_dir: "resources/results/aws_run"
YAML

# ---------- publisher: heartbeat + outputs -> S3 every 60 s -------------------------
cat > /home/ubuntu/publish.sh <<'PUB'
#!/bin/bash
S3=s3://rustar-bench/actinn-op-bench
while true; do
  {
    echo "utc=$(date -u +%FT%TZ)"
    echo "nextflow_active=$(systemctl is-active opbench 2>/dev/null)"
    echo "trace_rows=$(( $(wc -l < /home/ubuntu/out/trace.txt 2>/dev/null || echo 1) - 1 ))"
    echo "progress=$(grep -oE '[0-9]+ of [0-9]+' /home/ubuntu/out/nextflow.log 2>/dev/null | tail -1)"
    echo "mem_used_gb=$(free -g | awk '/Mem:/{print $3}')"
    echo "completed=$(grep -cE 'process > .*Completed' /home/ubuntu/out/nextflow.log 2>/dev/null || echo 0)"
  } > /home/ubuntu/out/heartbeat.txt
  aws s3 sync /home/ubuntu/out "$S3/out" --only-show-errors \
      --exclude "*.html" 2>/dev/null || true
  aws s3 sync /home/ubuntu/task_label_projection/resources/results "$S3/results" \
      --only-show-errors 2>/dev/null || true
  sleep 60
done
PUB
chmod +x /home/ubuntu/publish.sh
systemctl reset-failed publisher 2>/dev/null || true
systemd-run --unit=publisher --collect /bin/bash /home/ubuntu/publish.sh

# ---------- the run -----------------------------------------------------------------
cat > /home/ubuntu/run_pipeline.sh <<'INNER'
#!/bin/bash
export HOME=/root JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH JAVA_CMD=$JAVA_HOME/bin/java NXF_SYNTAX_PARSER=v1
cd /home/ubuntu/task_label_projection
exec nextflow run target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker -resume -entry auto -c /root/big.config -params-file /root/params.yaml
INNER
chmod +x /home/ubuntu/run_pipeline.sh
find "$REPO/.nextflow" -name LOCK -delete 2>/dev/null || true
systemctl reset-failed opbench 2>/dev/null || true
# systemd-run detaches from SSM's process group; a plain `nohup &` is killed when the
# SSM command returns, which silently aborted the previous attempt.
systemd-run --unit=opbench --collect \
  --property=StandardOutput=append:/home/ubuntu/out/nextflow.log \
  --property=StandardError=append:/home/ubuntu/out/nextflow.log \
  --property=TimeoutStartSec=0 /bin/bash /home/ubuntu/run_pipeline.sh
sleep 15
echo "LAUNCH_OK opbench=$(systemctl is-active opbench) publisher=$(systemctl is-active publisher)"
