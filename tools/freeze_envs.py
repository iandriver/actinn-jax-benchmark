"""Freeze every benchmark environment to a lockfile, and record what else results depend on.

A rerun of the benchmark reproduced its splits exactly but moved accuracy by one cell in
13,550 and ontology concordance by 0.003, because the environment underneath had drifted:
a re-downloaded Cell Ontology release, and library versions that had moved since the
recorded run. Neither drift was visible anywhere.

This writes `envs/locks/<env>.lock.txt` (exact versions, one per venv) plus
`envs/locks/manifest.json` (interpreter versions, platform, the Cell Ontology release and
its checksum). `--check` compares the current environments against the locks and reports
drift instead of rewriting them.

    python tools/freeze_envs.py            # write locks + manifest
    python tools/freeze_envs.py --check    # report drift, exit 1 if any

Note this pins the *benchmark* environments, not the actinn-jax library. A library that
pins its dependents is a library nobody can install alongside anything else; actinn-jax
deliberately declares floors instead (see its pyproject.toml).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKS = os.path.join(ROOT, "envs", "locks")
OBO = os.environ.get("ONTOLOGY_OBO", "/tmp/cl-basic.obo")

ENVS = {
    "core": os.path.expanduser("~/Downloads/actinn-jax/.venv"),
    "scprint": os.path.join(ROOT, ".venv-scprint"),
    "protocloud": os.path.join(ROOT, ".venv-protocloud"),
    "scvi": os.path.join(ROOT, ".venv-scvi"),
    "tf": os.path.join(ROOT, ".venv-tf"),
    "panhuman": os.path.join(ROOT, ".venv-panhuman"),
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


# A freeze can contain entries that will not restore on another machine: editable installs
# and wheels referenced by local path. `pip freeze` emits them happily and the lock looks
# complete. actinn-jax itself was captured as `-e file:///Users/.../actinn-jax`, so a restore
# would have silently omitted the package the benchmark is about.
LOCAL_REWRITE = {"actinn-jax": "actinn-jax==0.3.1"}   # published; pin to the release


def _rewrite_local(line):
    """Return (line, warning) with local/editable installs resolved where possible."""
    raw = line.strip()
    if raw.startswith("-e ") or " @ file://" in raw or raw.startswith("file://"):
        name = (raw.split(" @ ")[0].strip() if " @ " in raw
                else os.path.basename(raw.rstrip("/")).replace("-e ", "").strip())
        for pkg, pinned in LOCAL_REWRITE.items():
            if pkg in raw.lower():
                return pinned, f"rewrote local install of {pkg} -> {pinned}"
        return f"# UNRESTORABLE (local path): {raw}", (
            f"{name}: installed from a local path that will not exist elsewhere; "
            "entry commented out")
    return raw, None


def freeze(py):
    """pip freeze, falling back to uv for environments built without pip.

    Returns (text, warnings)."""
    r = run([py, "-m", "pip", "freeze", "--all"])
    if not (r.returncode == 0 and r.stdout.strip()):
        r = run(["uv", "pip", "freeze", "--python", py])
    if r.returncode != 0:
        return "", []
    lines, warns = [], []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        new, w = _rewrite_local(line)
        lines.append(new)
        if w:
            warns.append(w)
    return "\n".join(lines) + "\n", warns


def pyver(py):
    r = run([py, "-V"])
    return (r.stdout or r.stderr).strip().replace("Python ", "")


def obo_info():
    if not os.path.exists(OBO):
        return {"path": OBO, "present": False}
    h = hashlib.sha256()
    version = None
    with open(OBO, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if version is None and b"data-version:" in chunk:
                for line in chunk.decode("utf8", "replace").splitlines():
                    if line.startswith("data-version:"):
                        version = line.split(":", 1)[1].strip()
                        break
    return {"path": OBO, "present": True, "data_version": version,
            "sha256": h.hexdigest(), "size_bytes": os.path.getsize(OBO)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare current environments against the locks; do not write")
    a = ap.parse_args()
    os.makedirs(LOCKS, exist_ok=True)

    drift, manifest = [], {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cell_ontology": obo_info(),
        "environments": {},
    }

    for name, venv in ENVS.items():
        py = os.path.join(venv, "bin", "python")
        lock = os.path.join(LOCKS, f"{name}.lock.txt")
        if not os.path.exists(py):
            print(f"{name:<12} MISSING ({venv})")
            manifest["environments"][name] = {"present": False, "path": venv}
            continue
        v = pyver(py)
        body, warns = freeze(py)
        for w in warns:
            print(f"{name:<12} warn: {w}")
        n = len([l for l in body.splitlines() if l.strip() and not l.startswith("#")])
        manifest["environments"][name] = {"present": True, "python": v, "packages": n}

        if a.check:
            if not os.path.exists(lock):
                drift.append(f"{name}: no lockfile")
                print(f"{name:<12} py{v:<9} NO LOCKFILE")
                continue
            # Comments must be stripped from BOTH sides. Stripping only the lockfile made
            # the generated "# UNRESTORABLE (local path)" line look like a new package, so
            # a clean environment reported drift on every run.
            old = [l for l in open(lock).read().splitlines()
                   if l.strip() and not l.startswith("#")]
            new = [l for l in body.splitlines()
                   if l.strip() and not l.startswith("#")]
            added, removed = set(new) - set(old), set(old) - set(new)
            if added or removed:
                drift.append(f"{name}: {len(added)} changed/added, {len(removed)} removed")
                print(f"{name:<12} py{v:<9} DRIFT  +{len(added)} -{len(removed)}")
                for line in sorted(added)[:5]:
                    print(f"               + {line}")
            else:
                print(f"{name:<12} py{v:<9} ok ({n} packages)")
        else:
            with open(lock, "w") as fh:
                fh.write(f"# {name} environment — python {v}\n")
                fh.write(f"# frozen {manifest['frozen_utc']} on {manifest['platform']}\n")
                fh.write("# regenerate: python tools/freeze_envs.py\n")
                fh.write(body if body.endswith("\n") else body + "\n")
            print(f"{name:<12} py{v:<9} wrote {n} packages -> {os.path.relpath(lock, ROOT)}")

    cl = manifest["cell_ontology"]
    print(f"\ncell ontology: {cl.get('data_version') or 'MISSING'} "
          f"sha256={cl.get('sha256', '')[:16]}")

    if not a.check:
        with open(os.path.join(LOCKS, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"wrote {os.path.relpath(os.path.join(LOCKS, 'manifest.json'), ROOT)}")
        return 0

    print(f"\n{len(drift)} environment(s) drifted from their lockfile")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
