#!/usr/bin/env bash
# Rebuild the shipped census-wide reference (actinn_jax/references/broad_human_v1).
#
#   benchmark/explore/update_broad_reference.sh              # all stages, resumable
#   STAGES=build benchmark/explore/update_broad_reference.sh # just re-train + ship
#
# Three stages in three environments, chained through one work directory:
#
#   1 fetch   .venv-scprint    CELLxGENE census -> census_wide_ref.h5ad   (hours, network)
#   2 embed   .venv-scprint    scPRINT -> census_wide_emb.npz             (GPU/MPS, ~1h)
#   3 build   actinn-jax .venv train + calibrate + ship + build_info.json (CPU, minutes)
#   4 verify  actinn-jax .venv score the INSTALLED reference on a held-out atlas
#
# Stages 1 and 2 are skipped when their output already exists, so a failed run resumes
# and a re-train is cheap. Delete the artifact to force a stage.
#
# Every path is derived from ACTINN_REF_WORK. That is the point: the three scripts used to
# carry independent hardcoded defaults that did not agree, so a hand-run pipeline could
# train stage 3 against a stale embedding without any error.
set -euo pipefail

WORK="${ACTINN_REF_WORK:-/tmp/actinn_ref_build}"
PKG="${ACTINN_JAX_REPO:-$HOME/Downloads/actinn-jax}"
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${REF_NAME:-broad_human_v1}"
STAGES="${STAGES:-fetch embed build verify}"

PY_SCPRINT="${PY_SCPRINT:-$BENCH/.venv-scprint/bin/python}"
PY_CORE="${PY_CORE:-$PKG/.venv/bin/python}"

export ACTINN_REF_WORK="$WORK" ACTINN_JAX_REPO="$PKG" REF_NAME="$NAME"
mkdir -p "$WORK"
echo "work=$WORK  package=$PKG  reference=$NAME  stages=$STAGES"

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has fetch; then
  if [[ -f "$WORK/census_wide_ref.h5ad" ]]; then
    echo "== fetch: skip (census_wide_ref.h5ad exists)"
  else
    echo "== fetch: pulling census (CENSUS_VERSION=${CENSUS_VERSION:-stable})"
    "$PY_SCPRINT" "$BENCH/benchmark/explore/fetch_census_wide.py"
  fi
fi

if has embed; then
  if [[ -f "$WORK/census_wide_emb.npz" ]]; then
    echo "== embed: skip (census_wide_emb.npz exists)"
  else
    echo "== embed: scPRINT (GPU/MPS; the only accelerated step)"
    "$PY_SCPRINT" "$BENCH/benchmark/explore/embed_broad.py"
  fi
fi

if has build; then
  # Keep the previous reference until the new one is verified: a rebuild that trains fine
  # but annotates worse is the failure this guards, and it is not visible until stage 4.
  DEST="$PKG/actinn_jax/references/$NAME"
  if [[ -d "$DEST" ]]; then
    BAK="$DEST.bak.$(date -u +%Y%m%dT%H%M%SZ)"
    cp -R "$DEST" "$BAK"
    echo "== build: previous reference backed up to $BAK"
  fi
  echo "== build: train + calibrate + ship"
  "$PY_CORE" "$BENCH/benchmark/explore/build_census_model.py"
fi

if has verify; then
  echo "== verify: scoring the installed reference on a held-out atlas"
  "$PY_CORE" "$BENCH/benchmark/explore/verify_reference.py" --name "$NAME"
fi

echo "UPDATE_BROAD_REFERENCE_DONE name=$NAME work=$WORK"
