#!/usr/bin/env bash
# Watch the AWS benchmark run via its S3 heartbeat.
#
# Reads S3, not the instance. Two reasons: it keeps working after the instance dies
# (which is exactly when you most want to know), and a heartbeat carries a timestamp, so
# a stalled run shows up as a STALE line instead of as silence. The previous monitor
# polled the box for a value that was always empty, and two hours of nothing looked the
# same as two hours of progress.
#
#   bash op_aws_watch.sh [instance-id]
set -uo pipefail

S3=s3://rustar-bench/actinn-op-bench
IID="${1:-}"
STALE_SEC=300

while true; do
  now=$(date -u +%s)
  hb=$(aws s3 cp "$S3/out/heartbeat.txt" - 2>/dev/null)

  if [ -z "$hb" ]; then
    echo "[$(date -u +%T)] NO HEARTBEAT YET (run not started, or S3 write failing)"
  else
    ts=$(sed -n 's/^utc=//p' <<<"$hb")
    age=$(( now - $(date -u -j -f %Y-%m-%dT%H:%M:%SZ "$ts" +%s 2>/dev/null || date -u -d "$ts" +%s) ))
    rows=$(sed -n 's/^trace_rows=//p' <<<"$hb")
    prog=$(sed -n 's/^progress=//p' <<<"$hb")
    act=$(sed -n 's/^nextflow_active=//p' <<<"$hb")
    mem=$(sed -n 's/^mem_used_gb=//p' <<<"$hb")
    flag=""
    [ "$age" -gt "$STALE_SEC" ] && flag=" *** STALE ${age}s — instance may be gone ***"
    [ "$act" != "active" ] && flag="$flag *** nextflow $act ***"
    echo "[$(date -u +%T)] tasks=$rows progress='$prog' mem=${mem}GB age=${age}s$flag"
    # Terminal states: the service finished (or died) and the heartbeat confirms it.
    if [ "$act" = "inactive" ] || [ "$act" = "failed" ]; then
      echo "=== RUN ENDED (nextflow $act, $rows tasks traced) ==="; break
    fi
  fi

  # Independent check: does the instance still exist? Catches the case where the
  # heartbeat stops because the box vanished rather than because the run finished.
  if [ -n "$IID" ]; then
    st=$(aws ec2 describe-instances --region us-west-2 --instance-ids "$IID" \
         --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
    [ "$st" != "running" ] && { echo "=== INSTANCE $IID is '$st' — run cannot continue ==="; break; }
  fi
  sleep 120
done
