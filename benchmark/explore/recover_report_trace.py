"""Recover the per-task trace from a Nextflow execution report.

Nextflow's report.html embeds the whole trace as `window.data`. That payload is JS, not
JSON: it contains the task's .command.sh, whose shell escapes (\\' and friends) are legal
JavaScript and illegal JSON, so json.loads refuses it. Rather than trying to sanitise the
escapes, pull the handful of fields we need per task with a targeted scan.

This exists because a run was terminated before its trace file was flushed, leaving
report.html as the only surviving record of per-task runtime and peak memory.

    python recover_report_trace.py report.html out.csv
"""

import csv
import re
import sys

FIELDS = ["process", "tag", "status", "exit", "realtime", "duration", "peak_rss", "%cpu"]


def scan(path):
    html = open(path, encoding="utf8", errors="replace").read()
    # Find the payload that actually holds the trace (earlier `window.data` hits belong to
    # the bundled plotting library).
    for m in re.finditer(r'window\.data\s*=\s*', html):
        b = html.index("{", m.end() - 1)
        if '"trace"' not in html[b:b + 200]:
            continue
        depth = 0
        for j in range(b, len(html)):
            if html[j] == "{":
                depth += 1
            elif html[j] == "}":
                depth -= 1
                if depth == 0:
                    return html[b:j + 1]
    return ""


def tasks(raw):
    """Yield one dict per task object by splitting on the task_id key."""
    for chunk in re.split(r'\{"task_id":', raw)[1:]:
        row = {}
        for f in FIELDS:
            # Values are either quoted strings or bare numbers/null.
            m = re.search(r'"%s":\s*("((?:[^"\\]|\\.)*)"|[-\d.]+|null)' % re.escape(f), chunk)
            if m:
                row[f] = m.group(2) if m.group(2) is not None else m.group(1)
        if row:
            yield row


raw = scan(sys.argv[1])
rows = list(tasks(raw))
print(f"recovered {len(rows)} tasks")
if rows:
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {sys.argv[2]}")
