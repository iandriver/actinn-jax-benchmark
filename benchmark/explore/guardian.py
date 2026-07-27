"""Self-terminating watchdog for the AWS benchmark box, with testable decision logic.

Two runs were lost to monitoring bugs, and both had the same shape: the automation keyed
off a signal that nobody had proved moves while the system is healthy.

  1. A poller returned empty output for two hours. Silence read as progress.
  2. This guardian's predecessor watched `trace.txt` row count. On a *resumed* run
     Nextflow does not append to an existing trace file, so the count sat frozen at 38
     while the pipeline was actively completing tasks. The guardian declared a stall and
     killed a run that was ~75% done.

The lesson is not "pick a better signal" -- it is that any single signal can be silently
wrong, so a kill decision must never rest on one. Hence:

  * Multiple independent liveness signals; a stall requires ALL of them to be stale.
    Failure (2) becomes impossible: trace.txt frozen while nextflow.log grows means the
    log signal still moves, so no stall is declared.
  * The unconditional stop conditions are the ones that cannot false-positive in a costly
    direction: the unit is gone (nothing more will happen) or the wall-clock budget is
    spent (stopping is the point).
  * A warmup window, so a slow start is never mistaken for a dead run.

`decide()` is pure so it can be tested without AWS, a clock, or a filesystem. See
test_guardian.py -- every scenario below, including the regression for failure (2).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from pathlib import Path

STOP = "stop"
CONTINUE = "continue"


@dataclasses.dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def decide(
    *,
    unit_state: str,
    elapsed_s: float,
    since_any_signal_moved_s: float,
    max_hours: float,
    stall_s: float,
    warmup_s: float = 300.0,
    since_state_known_s: float = 0.0,
    unknown_stop_s: float = 900.0,
) -> Decision:
    """Pure policy. `since_any_signal_moved_s` must already aggregate every signal:
    it is the time since the *most recently changed* signal changed.

    Fails CLOSED. Anything that leaves the guardian unable to supervise -- notably an
    unreadable unit state -- must eventually stop the box, because the failure mode of
    continuing is an instance billing indefinitely with nobody watching.
    """
    # Budget: unconditional and deliberate. Stopping a healthy-but-slow run at the cap is
    # the intended behaviour, not a false positive.
    if elapsed_s >= max_hours * 3600:
        return Decision(STOP, f"budget cap {max_hours}h reached")

    # Terminal unit states: nothing further will happen, so paying more is pure waste.
    if unit_state in ("inactive", "failed"):
        return Decision(STOP, f"pipeline {unit_state}")

    # Un-determinable state (systemctl missing, timing out, permission denied). A brief
    # blip is tolerated; a sustained one means we cannot supervise, so we stop rather
    # than let an unwatched instance bill forever. Treating "unknown" as healthy is how a
    # cost guard silently becomes a cost leak.
    if since_state_known_s >= unknown_stop_s:
        return Decision(STOP, f"pipeline state unreadable for {since_state_known_s/60:.0f}m")

    # Never judge liveness before the pipeline has had a chance to produce anything.
    if elapsed_s < warmup_s:
        return Decision(CONTINUE, "warmup")

    # Stall: only when EVERY signal has gone quiet. One frozen signal cannot kill a run.
    if since_any_signal_moved_s >= stall_s:
        return Decision(STOP, f"no signal moved for {since_any_signal_moved_s/60:.0f}m")

    return Decision(CONTINUE, "healthy")


# --------------------------------------------------------------------------------------
# Signal collection. Each is independent; if one breaks, the others still carry liveness.
# --------------------------------------------------------------------------------------

def collect_signals(out_dir: Path, repo: Path) -> dict[str, object]:
    """Snapshot every liveness signal. Values are compared for *change*, not magnitude,
    so any monotonic or mutating quantity works."""
    sig: dict[str, object] = {}

    trace = out_dir / "trace.txt"
    sig["trace_rows"] = _line_count(trace)          # froze on the resumed run -- kept, but never alone
    sig["trace_mtime"] = _mtime(trace)

    log = out_dir / "nextflow.log"
    sig["log_size"] = _size(log)                    # grows steadily while nextflow lives
    sig["log_mtime"] = _mtime(log)

    # Nextflow rewrites the progress block continuously; the completed counts move as
    # tasks land, independent of any file it may decline to append to.
    sig["log_progress"] = _tail_progress(log)

    # Work directory grows as tasks stage and publish outputs.
    sig["work_entries"] = _count_dirs(repo / "work")

    # Container churn: the set of running containers changes as tasks start and finish.
    sig["containers"] = _docker_ps()

    return sig


def _line_count(p: Path) -> int:
    try:
        with p.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def _tail_progress(p: Path, tail_bytes: int = 20000) -> str:
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - tail_bytes))
            txt = fh.read().decode("utf8", "replace")
        # The progress block lines look like "... | 5 of 6, cached: 5"; joining them gives
        # a fingerprint that changes whenever any process advances.
        return "|".join(sorted({ln.split("|")[-1].strip()
                                for ln in txt.splitlines() if " of " in ln}))
    except OSError:
        return ""


def _count_dirs(p: Path) -> int:
    try:
        return sum(1 for _ in p.rglob("*") if _.is_dir())
    except OSError:
        return -1


def _docker_ps() -> str:
    try:
        out = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip()
    except Exception:
        return ""


def unit_state(unit: str) -> str:
    try:
        out = subprocess.run(["systemctl", "is-active", unit], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="opbench")
    ap.add_argument("--out", type=Path, default=Path("/home/ubuntu/out"))
    ap.add_argument("--repo", type=Path, default=Path("/home/ubuntu/task_label_projection"))
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--stall-min", type=float, default=150.0)
    ap.add_argument("--unknown-stop-min", type=float, default=15.0,
                    help="stop if the pipeline state stays unreadable this long")
    ap.add_argument("--s3", default="s3://rustar-bench/actinn-op-bench")
    ap.add_argument("--warmup-min", type=float, default=5.0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true", help="log decisions, never shut down")
    a = ap.parse_args()

    logf = a.out / "guardian.log"
    a.out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%FT%TZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        with logf.open("a") as fh:
            fh.write(line + "\n")

    start = time.monotonic()
    prev = collect_signals(a.out, a.repo)
    last_move = start
    last_known = start
    log(f"guardian up: max={a.max_hours}h stall={a.stall_min}m "
        f"unknown_stop={a.unknown_stop_min}m signals={sorted(prev)}")

    while True:
        time.sleep(a.interval)
        now = time.monotonic()
        cur = collect_signals(a.out, a.repo)
        moved = [k for k in cur if cur[k] != prev.get(k)]
        if moved:
            last_move = now
        prev = cur

        st = unit_state(a.unit)
        if st != "unknown":
            last_known = now
        d = decide(
            unit_state=st,
            elapsed_s=now - start,
            since_any_signal_moved_s=now - last_move,
            max_hours=a.max_hours,
            stall_s=a.stall_min * 60,
            since_state_known_s=now - last_known,
            unknown_stop_s=a.unknown_stop_min * 60,
            warmup_s=a.warmup_min * 60,
        )
        log(f"state={st} elapsed={(now-start)/60:.0f}m quiet={(now-last_move)/60:.0f}m "
            f"unknown={(now-last_known)/60:.0f}m moved={moved or 'NONE'} "
            f"-> {d.action} ({d.reason})")

        if d.action == STOP:
            log(f"FINAL: {d.reason}")
            (a.out / "final_status.txt").write_text(d.reason + "\n")
            # Sync synchronously: the box is about to die and this is the only copy.
            for src, dst in ((a.out, f"{a.s3}/out"),
                             (a.repo / "resources" / "results", f"{a.s3}/results")):
                subprocess.run(["aws", "s3", "sync", str(src), dst, "--only-show-errors"],
                               timeout=900)
            if a.dry_run:
                log("dry-run: would shut down now")
                return
            subprocess.run(["shutdown", "-h", "now"])
            return


if __name__ == "__main__":
    sys.exit(main())
