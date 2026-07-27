"""Summarise a Nextflow trace into per-method runtime and peak memory.

Nextflow writes trace fields as human-readable strings ("40m 52s", "12.8 GB"), not raw
numbers, so they need parsing before any arithmetic. CACHED rows carry the timings of
the run that originally produced them, which is what we want for a same-hardware
comparison -- every task in this trace ran on the same instance type.

    python summarize_op_trace.py trace.txt [out.csv]
"""

import csv
import re
import sys
from collections import defaultdict

SKIP = {"extract_uns_metadata", "extract_scores", "accuracy", "f1",
        "true_labels", "random_labels", "majority_vote"}

_TIME = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_SIZE = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1000.0}


def secs(v: str) -> float:
    """'1h 20m 41s' / '40m 52s' / '55.8s' / '964ms' -> seconds."""
    if not v or v == "-":
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*(ms|[smh])", v):
        total += float(num) * _TIME[unit]
    return total


def gb(v: str) -> float:
    """'12.8 GB' / '607.2 MB' -> gigabytes."""
    if not v or v == "-":
        return 0.0
    m = re.match(r"([\d.]+)\s*([KMGT]?B)", v.strip())
    return float(m.group(1)) * _SIZE[m.group(2)] if m else 0.0


def method(process: str) -> str:
    parts = process.split(":")
    return parts[-3] if len(parts) >= 3 else process


def dataset(tag: str) -> str:
    return tag.split("/")[1] if "/" in tag else tag


rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
by = defaultdict(dict)          # method -> dataset -> (runtime_s, peak_gb)
for r in rows:
    m = method(r["process"])
    if m in SKIP or r["status"] not in ("COMPLETED", "CACHED"):
        continue
    by[m][dataset(r["tag"])] = (secs(r["realtime"]), gb(r["peak_rss"]))

datasets = sorted({d for v in by.values() for d in v})
print(f"{'method':<20} {'n':>2} {'mean s':>8} {'max s':>8} {'max GB':>7}   datasets")
print("-" * 88)
for m in sorted(by, key=lambda k: -len(by[k])):
    v = by[m]
    rt = [t for t, _ in v.values()]
    pr = [p for _, p in v.values()]
    miss = [d for d in datasets if d not in v]
    print(f"{m:<20} {len(v):>2} {sum(rt)/len(rt):>8.1f} {max(rt):>8.1f} {max(pr):>7.1f}   "
          f"{'ALL' if not miss else 'missing: ' + ','.join(x[:12] for x in miss)}")

if len(sys.argv) > 2:
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset", "runtime_s", "peak_rss_gb"])
        for m in sorted(by):
            for d in sorted(by[m]):
                t, p = by[m][d]
                w.writerow([m, d, round(t, 1), round(p, 2)])
    print(f"\nwrote {sys.argv[2]}")
