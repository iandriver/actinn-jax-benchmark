"""Adapters for R-based annotation methods (SingleR, scmap, scPred).

These shell out to ``benchmark/r/run_r_method.R`` running in the system R with a
project-local library (``.Rlib``). Data is exchanged as Matrix-Market ``.mtx``
(genes x cells) plus gene/label text files — no Python/R in-process bridge, so the
R toolchain stays fully isolated from the benchmark's Python env.

Set ``R_LIBS`` in the config or environment to the project R library if it is not
the default ``<repo>/.Rlib``.
"""

import os
import subprocess
import tempfile

import scipy.io as sio

from ..prep import counts_adata
from .base import AnnotationMethod, Predictions, register

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_R_SCRIPT = os.path.join(_REPO, "benchmark", "r", "run_r_method.R")
_R_LIB = os.environ.get("R_LIBS", os.path.join(_REPO, ".Rlib"))


def _export(adata, work, prefix, label_key=None):
    """Write counts (genes x cells .mtx) + gene names (+ labels) for R to read."""
    a = counts_adata(adata)  # raw counts in .X, upper-cased gene names
    sio.mmwrite(os.path.join(work, f"{prefix}_counts.mtx"), a.X.T.tocoo())
    with open(os.path.join(work, f"{prefix}_genes.txt"), "w") as fh:
        fh.write("\n".join(map(str, a.var_names)))
    if label_key is not None:
        with open(os.path.join(work, f"{prefix}_labels.txt"), "w") as fh:
            fh.write("\n".join(a.obs[label_key].astype(str)))
    return list(adata.obs_names)


class _RMethod(AnnotationMethod):
    tier = "classical"
    device = "cpu"
    r_method = None  # set by subclasses

    def fit(self, ref, label_key):
        self._work = tempfile.mkdtemp(prefix=f"r_{self.r_method}_")
        _export(ref, self._work, "ref", label_key)

    def predict(self, query):
        cell_ids = _export(query, self._work, "query")
        out = os.path.join(self._work, "pred.csv")
        env = dict(os.environ, R_LIBS_USER=_R_LIB)
        r = subprocess.run(
            ["Rscript", "--vanilla", _R_SCRIPT, self.r_method, self._work, out],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"R method {self.r_method} failed:\n{r.stderr[-1500:]}")
        import pandas as pd
        labels = pd.read_csv(out)["pred_label"].to_numpy()
        return Predictions(cell_ids=cell_ids, labels=labels)


@register
class SingleR(_RMethod):
    name = "singler"
    r_method = "singler"


@register
class ScmapCluster(_RMethod):
    name = "scmap-cluster"
    r_method = "scmap-cluster"


@register
class ScPred(_RMethod):
    name = "scpred"
    r_method = "scpred"
