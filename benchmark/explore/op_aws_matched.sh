#!/usr/bin/env bash
# Complete the OP same-hardware table for our added methods, at MATCHED concurrency.
#
# The previous extension run used queueSize 6 and produced runtimes that rank threading
# and scheduler contention as much as algorithm (%cpu spanned 49% to 2338%). This run uses
# queueSize 2, matching the low-concurrency configuration behind the table in PAPER.md
# section 3.8, so the new rows sit on the same footing as the existing ones.
#
# actinn_jax is included as a CALIBRATION CONTROL: it already has a row in that table
# (165 s/dataset). If it reproduces here, the new rows are comparable to the old ones; if
# it does not, the conditions have drifted and the table must say so.
#
# THIS RUN LOST ITS RESULTS. It published to s3://.../actinn-op-bench-matched/ while the
# instance role only granted s3://.../actinn-op-bench/* -- the trailing wildcard does not
# cover the longer prefix. Every sync was denied, the publisher swallowed the error
# (2>/dev/null || true), and the box terminated with DeleteOnTermination=true. ~$3.50 and
# 1.6h gone.
#
# Two rules before reusing this script:
#   1. The FIRST sync must be loud. Resilience (|| true) belongs on later iterations, not
#      on the one that proves the credentials work; otherwise a fatal misconfiguration is
#      indistinguishable from a healthy quiet run.
#   2. Confirm the heartbeat object actually appears in S3 within ~2 minutes of launch
#      before leaving the run unattended. The watcher printed "no-heartbeat" every five
#      minutes for ninety minutes and nobody read it until the instance was gone.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive HOME=/root
REPO=/home/ubuntu/task_label_projection
OUT=/home/ubuntu/out
mkdir -p "$OUT"

if ! command -v docker >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq docker.io awscli git unzip curl
  systemctl enable --now docker
fi
java -version 2>&1 | grep -qE '"(1[7-9]|2[0-9])' || apt-get install -y -qq openjdk-17-jre-headless
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
command -v nextflow >/dev/null || { curl -s https://get.nextflow.io | bash; mv nextflow /usr/local/bin/; chmod +x /usr/local/bin/nextflow; }
command -v viash >/dev/null || { curl -fsSL https://github.com/viash-io/viash/releases/latest/download/viash -o /usr/local/bin/viash; chmod +x /usr/local/bin/viash; }

git config --global --add safe.directory "$REPO"
git config --global --add safe.directory "$REPO/common"
[ -d "$REPO" ] || git clone -q https://github.com/openproblems-bio/task_label_projection.git "$REPO"
cd "$REPO"
git submodule update --init --recursive

tar xzf /home/ubuntu/components.tgz -C /tmp
for c in actinn_jax linear_anova_pca sctop svm_sgd celltypist; do
  mkdir -p "src/methods/$c" && cp /tmp/$c/* "src/methods/$c/"
done
python3 /home/ubuntu/op_aws_wire.py

aws s3 sync --no-sign-request --only-show-errors --exclude "*" --include "*/log_cp10k/*" \
  s3://openproblems-data/resources/task_label_projection/datasets/cellxgene_census \
  resources/datasets/cellxgene_census
echo "DATA: $(du -sh resources/datasets | cut -f1)"

viash ns build
# Only the components this run executes -- building the other dozen wastes ~10 minutes.
for c in methods/actinn_jax methods/linear_anova_pca methods/sctop methods/svm_sgd \
         methods/celltypist metrics/accuracy metrics/f1 \
         control_methods/true_labels control_methods/majority_vote control_methods/random_labels; do
  viash build "src/$c/config.vsh.yaml" --engine docker --setup cachedbuild -o /tmp/bimg || echo "IMGFAIL $c"
done

cat > /root/params.yaml <<'YAML'
input_states: resources/datasets/cellxgene_census/**/state.yaml
rename_keys: 'input_train:output_train;input_test:output_test;input_solution:output_solution'
output_state: "state.yaml"
settings: '{"methods_include": ["actinn_jax","linear_anova_pca","sctop","svm_sgd","celltypist","true_labels","majority_vote","random_labels"]}'
publish_dir: "resources/results/matched"
YAML

# queueSize 2 is the whole point of this run -- do not raise it for speed.
cat > /root/big.config <<'CFG'
process {
  errorStrategy = 'ignore'
  maxRetries = 0
  withLabel: lowmem  { memory = '25 GB' }
  withLabel: midmem  { memory = '50 GB' }
  withLabel: highmem { memory = '90 GB' }
  withLabel: lowcpu  { cpus = 4 }
  withLabel: midcpu  { cpus = 8 }
  withLabel: highcpu { cpus = 16 }
}
executor { memory = '200 GB'; cpus = 32; queueSize = 2 }
trace  { enabled = true; overwrite = true; file = '/home/ubuntu/out/trace.txt'
         fields = 'task_id,process,tag,status,exit,realtime,duration,%cpu,peak_rss,peak_vmem' }
report { enabled = true; overwrite = true; file = '/home/ubuntu/out/report.html' }
CFG

cat > /home/ubuntu/run_pipeline.sh <<'INNER'
#!/bin/bash
export HOME=/root JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH JAVA_CMD=$JAVA_HOME/bin/java NXF_SYNTAX_PARSER=v1
cd /home/ubuntu/task_label_projection
exec nextflow run target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker -resume -entry auto -c /root/big.config -params-file /root/params.yaml
INNER
chmod +x /home/ubuntu/run_pipeline.sh

cat > /home/ubuntu/publish.sh <<'PUB'
#!/bin/bash
S3=s3://rustar-bench/actinn-op-bench-matched
while true; do
  { echo "utc=$(date -u +%FT%TZ)"
    echo "nextflow_active=$(systemctl is-active opbench 2>/dev/null)"
    echo "trace_rows=$(( $(wc -l < /home/ubuntu/out/trace.txt 2>/dev/null || echo 1) - 1 ))"
    echo "progress=$(grep -oE '[0-9]+ of [0-9]+' /home/ubuntu/out/nextflow.log 2>/dev/null | tail -1)"
    echo "mem_used_gb=$(free -g | awk '/Mem:/{print $3}')"
  } > /home/ubuntu/out/heartbeat.txt
  aws s3 sync /home/ubuntu/out "$S3/out" --only-show-errors 2>/dev/null || true
  aws s3 sync /home/ubuntu/task_label_projection/resources/results "$S3/results" --only-show-errors 2>/dev/null || true
  sleep 60
done
PUB
chmod +x /home/ubuntu/publish.sh

for u in opbench publisher guardian; do systemctl reset-failed $u 2>/dev/null || true; done
systemd-run --unit=publisher --collect /bin/bash /home/ubuntu/publish.sh
systemd-run --unit=opbench --collect \
  --property=StandardOutput=append:/home/ubuntu/out/nextflow.log \
  --property=StandardError=append:/home/ubuntu/out/nextflow.log \
  --property=TimeoutStartSec=0 /bin/bash /home/ubuntu/run_pipeline.sh
systemd-run --unit=guardian --collect /usr/bin/python3 /home/ubuntu/guardian.py \
  --unit opbench --out "$OUT" --repo "$REPO" --max-hours 2.0 --stall-min 150 \
  --unknown-stop-min 15 --interval 60 --s3 s3://rustar-bench/actinn-op-bench-matched
sleep 20
echo "LAUNCH_OK opbench=$(systemctl is-active opbench) publisher=$(systemctl is-active publisher) guardian=$(systemctl is-active guardian)"
