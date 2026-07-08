# AWS run: a controlled, same-hardware Open Problems benchmark

**Goal.** Put actinn-jax through OP's *own* Nextflow pipeline on a single cloud instance so
its accuracy **and** runtime/peak-memory are measured by the same harness, on the same
hardware, as the other methods — removing the cross-hardware caveat in
[OPENPROBLEMS.md](OPENPROBLEMS.md). Run **in-region** so the S3 datasets never leave
`us-west-2` (fast, egress-free), staged to a large volume so **disk space and re-downloads
are never a factor**.

Decisions: **non-GPU method tier + actinn-jax**, **OP official pipeline**, **spot, < $30**.

## 1. Instance & storage

| | choice | why |
|---|---|---|
| region | **us-west-2** | where `s3://openproblems-data` lives → S3 reads are LAN-speed, no egress |
| instance | **r7i.8xlarge** (32 vCPU, **256 GB RAM**), spot | CPU tier + actinn-jax; 256 GB clears the full-gene atlas that OOM'd at 51 GB locally, and singler/seurat's ~49 GB peaks. Spot ≈ $0.65–0.95/hr (verify) |
| storage | **500 GB gp3 EBS** (or NVMe instance store) | 63 GB (log_cp10k) + Docker images + Nextflow `work/`; huge overhead → no space pressure, no re-runs |
| AMI | Ubuntu 22.04 LTS (x86_64) | Docker + Nextflow + Viash all supported |

Fallback: if a 256 GB spot box is scarce, `r7i.4xlarge` (128 GB) is enough for the default
**HVG** run (peak ≈ 13 GB); only *full-gene* runs need the 256 GB box.

Rough cost: build containers ~20 min + sync 63 GB in-region ~5 min + methods ~2–3 h ⇒
**≈ 3–4 instance-hours ⇒ ~$3–5 spot.** Comfortably < $30 even with reruns.

## 2. Provision (spot) — needs your AWS confirmation

Credentials for IAM user `Ian` (acct `418696582915`) are configured locally, but region
default is `us-east-1` — **launch explicitly in us-west-2**. Fill the two account-specific
blanks (`KEY_NAME`, `SG_ID`) and run:

```bash
REGION=us-west-2
AMI=$(aws ec2 describe-images --region $REGION --owners 099720109477 \
  --filters 'Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
aws ec2 run-instances --region $REGION --image-id $AMI \
  --instance-type r7i.8xlarge \
  --instance-market-options 'MarketType=spot' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
  --key-name KEY_NAME --security-group-ids SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=actinn-op-bench}]'
```

(Security group must allow inbound SSH from your IP. No IAM role needed — S3 read is
`--no-sign-request` on the public bucket.)

## 3. Set up the box

```bash
sudo apt-get update && sudo apt-get install -y docker.io default-jre awscli git
sudo usermod -aG docker ubuntu && newgrp docker
curl -s https://get.nextflow.io | bash && sudo mv nextflow /usr/local/bin/
# Viash
curl -fsSL https://github.com/viash-io/viash/releases/latest/download/viash -o viash \
  && chmod +x viash && sudo mv viash /usr/local/bin/
```

## 4. Add the actinn-jax component and build

```bash
git clone --recursive https://github.com/openproblems-bio/task_label_projection.git
cd task_label_projection
git submodule update --init --recursive    # `common/` — REQUIRED (helper.nf, base configs); a --depth 1 clone skips it
# drop in the component (from this repo's openproblems_component/actinn_jax/)
mkdir -p src/methods/actinn_jax
cp /path/to/actinn-jax-benchmark/openproblems_component/actinn_jax/* src/methods/actinn_jax/
```

**Wire the method into the benchmark workflow (two edits — a component alone is invisible to `run_benchmark`):**

1. `src/workflows/run_benchmark/config.vsh.yaml` → add under `dependencies:`
   ```yaml
     - name: methods/actinn_jax
   ```
2. `src/workflows/run_benchmark/main.nf` → add to the `methods = [ ... ]` array
   ```nextflow
     actinn_jax,
   ```

Adding *only* the dependency (or *only* the array entry) silently drops the method from
the run with no error — it just never appears in the DAG. Both are required, then:

```bash
viash ns build                              # regenerates target/nextflow/**, incl. run_benchmark/main.nf
grep -c actinn_jax target/nextflow/workflows/run_benchmark/main.nf   # must be > 0
# build the CPU-tier images (docker is an *engine* in viash 0.9, not a runner):
for c in methods/actinn_jax methods/mlp methods/knn methods/logistic_regression \
         methods/naive_bayes methods/xgboost methods/singler methods/seurat_transferdata \
         methods/cellmapper_linear metrics/accuracy metrics/f1 \
         control_methods/true_labels control_methods/majority_vote control_methods/random_labels; do
  viash build "src/$c/config.vsh.yaml" --engine docker --setup cachedbuild
done
```

## 5. Sync data (in-region → fast) and run the CPU tier

```bash
scripts/sync_datasets.sh                      # ~63 GB (log_cp10k) from S3, LAN-speed in us-west-2
# run OP's pipeline restricted to the non-GPU tier + actinn-jax + controls:
cat > /tmp/params.yaml <<YAML
input_states: resources/datasets/cellxgene_census/**/state.yaml
rename_keys: 'input_train:output_train;input_test:output_test;input_solution:output_solution'
output_state: "state.yaml"
settings: '{"methods_include": ["actinn_jax","mlp","knn","logistic_regression","naive_bayes","xgboost","singler","seurat_transferdata","cellmapper_linear","true_labels","majority_vote","random_labels"]}'
publish_dir: "resources/results/aws_run"
YAML
export JAVA_CMD=/usr/lib/jvm/java-17-openjdk-amd64/bin/java   # Nextflow 26.04 needs Java 17+
export NXF_SYNTAX_PARSER=v1                                   # 26.04's strict parser rejects the OP nextflow.config
nextflow run target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker -resume -entry auto \
  -c /root/big.config -params-file /tmp/params.yaml
```

Notes learned the hard way on the r7i.8xlarge (32 vCPU / 256 GB):

- **Don't use `common/nextflow_helpers/labels_ci.config`** — it caps *every* label's memory
  at 5 GB (a CI-only config), so `random_labels`/`singler` OOM (exit 137) loading the 20 GB
  Tabula Sapiens h5ad. Use a generous custom config (`big.config`): `lowmem 80 / midmem 100 /
  highmem 120 GB`.
- **Memory thrash is the real hazard, not per-task OOM.** The box has **no swap**, so if the
  *sum* of concurrently-scheduled tasks' actual RSS exceeds ~247 GB the kernel thrashes and
  `sshd` goes unresponsive for many minutes (the EC2 system/reachability checks still pass —
  it's userspace starvation, not a dead box; `aws ec2 reboot-instances` recovers it). Two
  Tabula-Sapiens-scale tasks overlapping under `maxForks 2` + `executor.memory 250 GB` is
  enough to trigger it. **Fix:** cap `executor { memory "180 GB" }` so two `highmem` (120 GB)
  tasks can't co-schedule (240 > 180), keeping `maxForks 2` for the light tasks. `maxForks 1`
  is bulletproof but serial and slow — only worth it if a single task also can't fit.
- **`singler` (SingleR) is a time bomb.** It ran **single-threaded at 100 % of one core for
  >2 h without finishing on the *smallest* dataset** (mouse_pancreas), blocking the whole
  serial queue; on TS/hypomap it would take many hours. `seurat_transferdata` is the next
  slowest R method. For a time-boxed same-hardware run, **exclude both** from
  `methods_include` — the controlled comparison that matters is actinn-jax vs the Python ML
  tier (`mlp`, `knn`, `logistic_regression`, `naive_bayes`, `xgboost`, `cellmapper_linear`);
  the R methods' numbers can stay sourced from OP's own CI trace.
- Invoke the **compiled** `main.nf` by path (`target/nextflow/workflows/run_benchmark/main.nf`);
  `nextflow run . -main-script …` makes 26.04 try to pull a remote repo named `.`.
- `-resume` makes the run idempotent: a spot interruption or an added method re-uses cached
  results, so **nothing re-runs unnecessarily**. If a prior run was `kill -9`'d, delete the
  stale lock first: `find .nextflow -name LOCK -delete`. Note that enabling `trace`/`report`
  after tasks have run leaves those tasks untraced — enable tracing from the **first** run,
  or restart fresh, so every task carries a `realtime`/`peak_rss` row.

## 6. Collect results, then tear down

```bash
# scores (accuracy, f1) and the trace (per-method runtime + peak_rss, SAME hardware)
aws s3 cp --recursive --no-sign-request /dev/null /dev/null 2>/dev/null # (results are local)
cp resources/results/aws_run/score_uns.yaml resources/results/aws_run/trace.txt ~/out/
# pull ~/out/ back to the laptop (scp), then:
aws ec2 terminate-instances --region us-west-2 --instance-ids <id>   # STOP THE BILL
```

Parse `score_uns.yaml` + `trace.txt` exactly as in
`benchmark/explore/` and update [OPENPROBLEMS.md](OPENPROBLEMS.md): replace the
cross-hardware runtime/memory table with the **same-hardware** trace, and confirm the
accuracy numbers reproduce (they should, within seed noise).

## What this delivers

- actinn-jax's accuracy, runtime, and peak memory measured **identically to the 12 other
  CPU-tier methods on one box** → the runtime/memory comparison becomes a controlled
  head-to-head, not tier-level indicative.
- Option to also run **full-gene** actinn-jax (`--n_hvg 0`) on the 256 GB box — its native
  mode, infeasible on the 51 GB laptop — to report both HVG and full-gene.
- The component is a clean PR to `openproblems-bio/task_label_projection`, putting
  actinn-jax on the **live public leaderboard**.

## Optional later: GPU tier

To add the GPU foundation/deep methods (scanvi, scanvi_scarches, cellmapper_scvi, scgpt,
geneformer, uce, scprint, scimilarity) on identical hardware, repeat on a **g5.12xlarge**
(4× A10G, 192 GB) and widen `methods_include`. `uce` alone needs ~129 GB / ~3 h, so this is
a separate, pricier run (~$50–150) — deferred per the current budget.
