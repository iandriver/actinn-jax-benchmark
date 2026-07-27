#!/usr/bin/env bash
# Resume the OP benchmark on an instance launched from the snapshot AMI.
# Everything (datasets, docker images, repo, nextflow work cache) is already on disk,
# so this only re-tunes, restarts the pipeline, and arms the guardian.
set -euxo pipefail
export HOME=/root JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

REPO=/home/ubuntu/task_label_projection
mkdir -p /home/ubuntu/out

# cellmapper_scvi is dropped: it errored on hypomap and aborted the whole run, and it is
# an OP-only method absent from our §3.2 panel (we carry scANVI and scArches), so losing
# it costs nothing. errorStrategy is also set per-process AND at top level, because the
# process-scope setting alone did not stop that failure from killing the pipeline.
cat > /root/params.yaml <<'YAML'
input_states: resources/datasets/cellxgene_census/**/state.yaml
rename_keys: 'input_train:output_train;input_test:output_test;input_solution:output_solution'
output_state: "state.yaml"
settings: '{"methods_include": ["actinn_jax","linear_anova_pca","sctop","svm_sgd","celltypist","mlp","knn","logistic_regression","naive_bayes","xgboost","cellmapper_linear","scanvi","scanvi_scarches","true_labels","majority_vote","random_labels"]}'
publish_dir: "resources/results/aws_run"
YAML

# Sized from the observed trace (actual peak_rss 13-50 GB), not from guesses.
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
executor { memory = '200 GB'; cpus = 32; queueSize = 6 }
trace  { enabled = true; overwrite = false; file = '/home/ubuntu/out/trace.txt'
         fields = 'task_id,process,tag,status,exit,realtime,duration,%cpu,peak_rss,peak_vmem' }
report { enabled = true; overwrite = true; file = '/home/ubuntu/out/report.html' }
CFG

cd "$REPO"
find .nextflow -name LOCK -delete 2>/dev/null || true

for u in opbench publisher guardian; do systemctl reset-failed $u 2>/dev/null || true; done
systemd-run --unit=publisher --collect /bin/bash /home/ubuntu/publish.sh
systemd-run --unit=opbench --collect \
  --property=StandardOutput=append:/home/ubuntu/out/nextflow.log \
  --property=StandardError=append:/home/ubuntu/out/nextflow.log \
  --property=TimeoutStartSec=0 /bin/bash /home/ubuntu/run_pipeline.sh
systemd-run --unit=guardian --collect --setenv=MAX_HOURS=3 --setenv=STALL_MIN=25 \
  /bin/bash /home/ubuntu/guardian.sh

sleep 20
echo "RESUME_OK opbench=$(systemctl is-active opbench) publisher=$(systemctl is-active publisher) guardian=$(systemctl is-active guardian)"
