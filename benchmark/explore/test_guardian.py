"""Tests for the benchmark guardian's kill policy and its liveness signals.

These exist because two AWS runs were lost to monitoring bugs that a five-minute test
would have caught. Every scenario below maps to a real or narrowly-avoided failure.

    pytest benchmark/explore/test_guardian.py -q
"""

import subprocess
import time
from pathlib import Path

import pytest

from guardian import CONTINUE, STOP, collect_signals, decide

HOUR = 3600
MIN = 60


def d(**kw):
    base = dict(unit_state="active", elapsed_s=30 * MIN, since_any_signal_moved_s=1 * MIN,
                max_hours=3.0, stall_s=150 * MIN, warmup_s=5 * MIN,
                since_state_known_s=0.0, unknown_stop_s=15 * MIN)
    base.update(kw)
    return decide(**base)


# ---- fail-closed: found by this suite, not by an invoice -----------------------------

def test_unreadable_state_eventually_stops():
    """Caught by the end-to-end test on a machine without systemctl.

    `systemctl is-active` returning nothing (missing binary, timeout, permission denied)
    yields "unknown". The first version treated that as not-terminal and ran forever --
    a cost guard that fails OPEN. If we cannot supervise, we stop.
    """
    r = d(unit_state="unknown", since_state_known_s=16 * MIN)
    assert r.action == STOP and "unreadable" in r.reason


def test_brief_unreadable_state_is_tolerated():
    """A transient systemctl hiccup must not kill a healthy run."""
    assert d(unit_state="unknown", since_state_known_s=2 * MIN).action == CONTINUE


def test_unreadable_state_stops_even_while_signals_move():
    """Losing supervision is disqualifying on its own: an unwatchable box is exactly the
    runaway-cost case, whether or not something still looks busy."""
    r = d(unit_state="unknown", since_state_known_s=20 * MIN,
          since_any_signal_moved_s=5)
    assert r.action == STOP


# ---- the regression that motivated this file ----------------------------------------

def test_frozen_trace_file_does_not_kill_a_healthy_run():
    """THE BUG THAT KILLED RUN 2.

    Nextflow will not append to an existing trace.txt, so on a resumed run the row count
    sits frozen while the pipeline works normally. The old guardian keyed on that single
    number, saw it flat for 120 min, and terminated a run that was ~75% complete.

    The policy now consumes an aggregate over all signals, so a frozen trace with a live
    log reports a *small* since-any-signal-moved and must not stop.
    """
    assert d(since_any_signal_moved_s=2 * MIN).action == CONTINUE


def test_one_frozen_signal_is_ignored_when_another_still_moves(tmp_path):
    """Same bug at the signal layer: aggregation must key on the most recent mover."""
    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    trace = out / "trace.txt"; log = out / "nextflow.log"
    trace.write_text("header\n" + "row\n" * 38)          # frozen, as on a resumed run
    log.write_text("start\n")

    first = collect_signals(out, repo)
    log.write_text("start\nmore progress | 3 of 6\n")     # only the log advances
    second = collect_signals(out, repo)

    moved = [k for k in second if second[k] != first.get(k)]
    assert moved, "log growth must register as movement"
    assert second["trace_rows"] == first["trace_rows"], "trace deliberately frozen"


# ---- terminal states -----------------------------------------------------------------

@pytest.mark.parametrize("state", ["inactive", "failed"])
def test_terminal_unit_state_stops(state):
    """Run 1 idled ~2h after the pipeline died, costing ~$4. Stop promptly instead."""
    r = d(unit_state=state)
    assert r.action == STOP and state in r.reason


def test_terminal_state_stops_even_during_warmup():
    """A pipeline that dies 30s in must not be shielded by the warmup window."""
    assert d(unit_state="failed", elapsed_s=10).action == STOP


# ---- budget --------------------------------------------------------------------------

def test_budget_cap_stops_even_when_perfectly_healthy():
    """The cap is the cost guarantee; a busy run must not be able to outvote it."""
    r = d(elapsed_s=3 * HOUR, since_any_signal_moved_s=5, unit_state="active")
    assert r.action == STOP and "budget" in r.reason


def test_budget_checked_before_stall_so_the_reason_is_accurate():
    r = d(elapsed_s=4 * HOUR, since_any_signal_moved_s=200 * MIN)
    assert "budget" in r.reason


# ---- stalls, real and apparent -------------------------------------------------------

def test_genuine_stall_stops():
    assert d(since_any_signal_moved_s=151 * MIN).action == STOP


def test_slow_single_task_is_not_a_stall():
    """scanvi_scarches legitimately runs 80+ minutes on one task while emitting nothing
    new. A threshold below that would kill healthy runs -- the failure mode we are in."""
    assert d(since_any_signal_moved_s=85 * MIN).action == CONTINUE


def test_stall_threshold_exceeds_slowest_observed_task():
    """Guard the constant itself: 150m must stay clear of the 81m worst case measured."""
    slowest_observed_min = 81
    assert 150 > slowest_observed_min * 1.5


def test_warmup_prevents_killing_a_slow_starter():
    """Image pulls and data staging can precede the first signal movement."""
    assert d(elapsed_s=2 * MIN, since_any_signal_moved_s=2 * MIN).action == CONTINUE


# ---- signal collection is defensive ---------------------------------------------------

def test_missing_files_do_not_crash_the_guardian(tmp_path):
    """A guardian that raises stops guarding, and the box then runs unattended."""
    sig = collect_signals(tmp_path / "nope", tmp_path / "alsonope")
    assert sig["trace_rows"] == -1 and sig["log_size"] == -1


def test_signals_are_independent(tmp_path):
    """No signal may be derived from another, or one bug disables several at once."""
    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    (out / "trace.txt").write_text("a\n")
    (out / "nextflow.log").write_text("x\n")
    s = collect_signals(out, repo)
    assert {"trace_rows", "log_size", "log_progress", "work_entries"} <= set(s)


def test_progress_fingerprint_changes_as_tasks_complete(tmp_path):
    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    log = out / "nextflow.log"
    log.write_text("[ab/12] proc (tag) | 1 of 6, cached: 1\n")
    a = collect_signals(out, repo)["log_progress"]
    log.write_text("[ab/12] proc (tag) | 4 of 6, cached: 1\n")
    b = collect_signals(out, repo)["log_progress"]
    assert a != b


def test_work_dir_growth_registers(tmp_path):
    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    before = collect_signals(out, repo)["work_entries"]
    (repo / "work" / "ab").mkdir()
    assert collect_signals(out, repo)["work_entries"] > before


# ---- end to end ------------------------------------------------------------------------

def test_guardian_binary_stops_on_dead_unit_and_writes_status(tmp_path):
    """Drive the real script in --dry-run against a unit that does not exist (so
    `systemctl is-active` reports inactive/unknown) and confirm it decides to stop."""
    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    (out / "trace.txt").write_text("h\n")
    (out / "nextflow.log").write_text("l\n")

    r = subprocess.run(
        [ "python3", str(Path(__file__).with_name("guardian.py")),
          "--unit", "definitely-not-a-real-unit", "--out", str(out), "--repo", str(repo),
          "--interval", "0.2", "--dry-run", "--s3", "s3://example-bucket/none",
          # Tiny thresholds so the loop reaches a decision in seconds. On Linux the unit
          # reports "inactive"; on a machine without systemctl it reports "unknown" and
          # the unreadable-state rule fires. Both must terminate -- neither may hang.
          "--unknown-stop-min", "0.02", "--max-hours", "0.01" ],
        capture_output=True, text=True, timeout=120,
    )
    assert "would shut down now" in r.stdout, r.stdout + r.stderr
    assert (out / "final_status.txt").exists(), "status must be written before shutdown"


def test_integration_frozen_trace_with_live_log_survives_then_stops_when_truly_dead(tmp_path):
    """END-TO-END REGRESSION for the failure that killed run 2.

    Reproduces it exactly: trace.txt frozen forever, nextflow.log growing. The real loop
    must keep the box alive while the log moves, then stop once everything goes quiet.
    A guardian that fails this bills for a dead run or kills a live one.
    """
    import threading

    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    (out / "trace.txt").write_text("header\n" + "row\n" * 38)   # frozen, as on resume
    log = out / "nextflow.log"
    log.write_text("start\n")

    stop_writing = threading.Event()

    def writer():
        n = 0
        while not stop_writing.is_set():
            n += 1
            with log.open("a") as fh:
                fh.write(f"[ab/12] proc (tag) | {n} of 6, cached: 1\n")
            time.sleep(0.3)

    t = threading.Thread(target=writer, daemon=True); t.start()

    proc = subprocess.Popen(
        [ "python3", str(Path(__file__).with_name("guardian.py")),
          "--unit", "nope", "--out", str(out), "--repo", str(repo), "--interval", "0.3",
          "--dry-run", "--s3", "s3://example-bucket/none",
          "--unknown-stop-min", "999", "--max-hours", "999",
          "--warmup-min", "0", "--stall-min", "0.05" ],   # 3s of quiet == stall
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(6)                       # log is moving the whole time
    assert proc.poll() is None, "guardian stopped a run whose log was still advancing"

    stop_writing.set(); t.join(timeout=5)   # now nothing moves at all
    try:
        outp = proc.communicate(timeout=40)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("guardian failed to stop after every signal went quiet")
    assert "no signal moved" in outp, outp[-2000:]


def test_guardian_never_runs_forever_without_a_decision(tmp_path):
    """The budget cap is the last line of defence: whatever else breaks, the box stops."""
    out, repo = tmp_path / "out", tmp_path / "repo"
    out.mkdir(); (repo / "work").mkdir(parents=True)
    r = subprocess.run(
        [ "python3", str(Path(__file__).with_name("guardian.py")),
          "--unit", "nope", "--out", str(out), "--repo", str(repo), "--interval", "0.2",
          "--dry-run", "--s3", "s3://example-bucket/none",
          "--unknown-stop-min", "999", "--stall-min", "999", "--max-hours", "0.002" ],
        capture_output=True, text=True, timeout=120,
    )
    assert "budget cap" in r.stdout, r.stdout + r.stderr
