#!/usr/bin/env bash
# Self-terminating watchdog. Runs ON the benchmark box as a systemd unit.
#
# The previous run cost ~$4 sitting idle for two hours after the pipeline died: the
# monitor noticed, but nothing acted on it. Detection has to be wired to an action that
# does not depend on a human (or an agent) checking a log. This shuts the instance down
# on every terminal condition, and on a hard wall-clock budget regardless of state.
#
# Shutdown, not self-terminate-via-API: the instance is launched with
# --instance-initiated-shutdown-behavior terminate, so `shutdown -h now` destroys it and
# stops billing without the role needing ec2:TerminateInstances.
#
# Every exit path syncs to S3 first, so results survive the shutdown.
set -uo pipefail

S3=s3://rustar-bench/actinn-op-bench
MAX_HOURS="${MAX_HOURS:-3}"          # hard budget ceiling
STALL_MIN="${STALL_MIN:-25}"         # no new trace row for this long => stuck
START=$(date +%s)
TRACE=/home/ubuntu/out/trace.txt
last_rows=-1
last_change=$START

log() { echo "[$(date -u +%FT%TZ)] guardian: $*" >> /home/ubuntu/out/guardian.log; }

finish() {
  log "SHUTTING DOWN: $1"
  echo "$1" > /home/ubuntu/out/final_status.txt
  # Final sync must complete before the box dies, so it is not backgrounded.
  aws s3 sync /home/ubuntu/out "$S3/out" --only-show-errors 2>/dev/null
  aws s3 sync /home/ubuntu/task_label_projection/resources/results "$S3/results" \
      --only-show-errors 2>/dev/null
  aws s3 cp /home/ubuntu/out/final_status.txt "$S3/out/final_status.txt" 2>/dev/null
  sleep 5
  shutdown -h now
  exit 0
}

log "started (max ${MAX_HOURS}h, stall ${STALL_MIN}m)"
sleep 120   # let the pipeline get going before judging it

while true; do
  now=$(date +%s)
  elapsed_h=$(( (now - START) / 3600 ))
  rows=$(( $(wc -l < "$TRACE" 2>/dev/null || echo 1) - 1 ))
  state=$(systemctl is-active opbench 2>/dev/null)

  [ "$rows" != "$last_rows" ] && { last_rows=$rows; last_change=$now; }
  stalled_min=$(( (now - last_change) / 60 ))

  log "state=$state rows=$rows stalled=${stalled_min}m elapsed=${elapsed_h}h"

  # 1. pipeline ended (either way) -- nothing more will happen, so stop paying
  [ "$state" = "inactive" ] || [ "$state" = "failed" ] && finish "pipeline $state after ${elapsed_h}h, $rows tasks"

  # 2. hard budget ceiling, whatever the state
  [ "$elapsed_h" -ge "$MAX_HOURS" ] && finish "budget cap ${MAX_HOURS}h reached, $rows tasks"

  # 3. alive but not progressing. scanvi_scarches legitimately runs >1h on one task, so
  #    the threshold must exceed the slowest single task or this kills a healthy run.
  [ "$stalled_min" -ge "$STALL_MIN" ] && [ "$rows" -gt 0 ] && \
    log "WARNING no new task for ${stalled_min}m (slow task, or stuck)"
  [ "$stalled_min" -ge 120 ] && finish "no progress for ${stalled_min}m, $rows tasks"

  sleep 60
done
