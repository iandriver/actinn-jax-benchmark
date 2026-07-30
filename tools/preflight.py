"""Check that every method in a config can actually run, before the matrix starts.

A full matrix takes hours. Discovering at minute 40 that `celltypist` is not installed --
after it silently vanished from an environment and got captured that way in a lockfile --
wastes the run and produces a matrix with holes that look like results.

This imports each method's adapter in the interpreter the config assigns it, and reports
what is missing. Seconds instead of hours.

    python tools/preflight.py configs/paper.yaml
"""

import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What each adapter needs importable in its own environment. The adapter module itself is
# not enough: it imports its backend lazily inside fit(), which is why a missing backend
# shows up as a mid-run failure rather than an import error at startup.
BACKEND = {
    "actinn-jax": "actinn_jax",
    "celltypist": "celltypist",
    "svm": "sklearn",
    "knn": "sklearn",
    "linear-anova-pca": "sklearn",
    "sctop": "sctop",
    "scanvi": "scvi",
    "scarches": "scvi",
    "protocloud": "protocloud",
    "scprint": "scprint",
    "actinn-orig": "tensorflow",
    "panhuman-azimuth": "panhumanpy",
    # R methods do not use rpy2: r_adapter.py shells out to `Rscript --vanilla` with
    # R_LIBS_USER pointed at the project library, so the check is Rscript + the package.
    "singler": None,
    "scmap-cluster": None,
}
R_METHODS = {"singler": "SingleR", "scmap-cluster": "scmap", "scpred": "scPred"}
R_LIB = os.environ.get("R_LIBS", os.path.join(ROOT, ".Rlib"))


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/paper.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    default_py = sys.executable
    missing = []

    print(f"{'method':<20}{'interpreter':<44}status")
    print("-" * 84)
    for m in cfg["methods"]:
        name = m["name"]
        py = m.get("python", default_py)
        if not os.path.isabs(py):
            py = os.path.join(ROOT, py)
        short = py.replace(os.path.expanduser("~"), "~")
        if not os.path.exists(py):
            print(f"{name:<20}{short[:43]:<44}NO INTERPRETER")
            missing.append(f"{name}: interpreter {py} missing")
            continue
        mod = BACKEND.get(name)
        if name in R_METHODS:
            pkg = R_METHODS[name]
            env = dict(os.environ, R_LIBS_USER=R_LIB)
            r = subprocess.run(
                ["Rscript", "--vanilla", "-e",
                 f'if (requireNamespace("{pkg}", quietly=TRUE)) cat("ok")'],
                capture_output=True, text=True, timeout=300, env=env)
            ok = r.returncode == 0 and "ok" in r.stdout
            label = f"ok (Rscript, {pkg})" if ok else f"MISSING R pkg: {pkg}"
            short = f"Rscript + {os.path.relpath(R_LIB, ROOT)}"
        elif mod is None:
            print(f"{name:<20}{short[:43]:<44}skipped (no backend recorded)")
            continue
        else:
            r = subprocess.run([py, "-c", f"import {mod}; print('ok')"],
                               capture_output=True, text=True, timeout=300)
            ok = r.returncode == 0
            label = "ok" if ok else f"MISSING: {mod}"
        print(f"{name:<20}{short[:43]:<44}{label}")
        if not ok:
            missing.append(f"{name}: cannot import {mod or 'rpy2'} in {py}")

    ds_missing = []
    for d in cfg.get("datasets", []):
        for key in ("path", "ref_path", "query_path"):
            p = d.get(key)
            if p and not os.path.exists(os.path.expanduser(p)):
                ds_missing.append(f"{d['name']}: {key} not found -> {p}")

    print()
    for x in ds_missing:
        print(f"DATA MISSING  {x}")
    for x in missing:
        print(f"FAIL          {x}")
    total = len(missing) + len(ds_missing)
    print(f"\n{len(cfg['methods'])} methods, {len(cfg.get('datasets', []))} datasets, "
          f"{total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
