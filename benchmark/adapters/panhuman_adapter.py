"""Adapter for Pan-human Azimuth (Sarkar et al., bioRxiv 2026) via ``panhumanpy``.

Runs in the ``.venv-panhuman`` environment (Keras 3 / TensorFlow; point the method's
``python:`` there). See [docs/PAN_HUMAN_AZIMUTH.md] for what the model is and why it
matters here: it is a *pretrained* hierarchical classifier over a fixed organism-wide
typology (8 levels, 382 leaves, ~7M params, 5,055-gene panel), so it is the closest
published counterpart to actinn-jax's shipped broad reference.

**It does no training.** The labeled reference is ignored entirely -- ``fit`` only warms
the model weights (downloading them on first use), and all real work happens in
``predict``. Like scPRINT, it emits labels from *its own* vocabulary rather than the
dataset's, so exact-match accuracy against a dataset's label strings is a vocabulary
artifact and not a meaningful accuracy signal (cf. the lung_cross note in PAPER.md
section 3.1). The honest comparison is **ontology-aware**: ``panhumanpy`` ships a Cell
Ontology crosswalk for every node of its typology, so we map predictions to CL ids and
return them as ``label_cl``, which the driver scores against the truth CL ids.

Install:
    uv venv --python 3.11 .venv-panhuman
    uv pip install --python .venv-panhuman/bin/python panhumanpy scanpy
"""

import numpy as np

from .base import AnnotationMethod, Predictions, register

# Ordered candidates for a gene-symbol column. `feature_name` is the CELLxGENE
# convention and is present in the lung/liver atlases used here.
_SYMBOL_COLS = ("feature_name", "gene_symbol", "gene_symbols", "gene_name",
                "symbol", "hgnc_symbol")

# Refined label columns panhumanpy produces, coarse -> fine, plus the raw deepest call.
_LEVELS = ("azimuth_broad", "azimuth_medium", "azimuth_fine", "final_level_labels")


def _looks_ensembl(names):
    head = [str(n) for n in list(names)[:50]]
    return sum(n.startswith("ENSG") for n in head) > len(head) // 2


def _to_panhuman_input(adata):
    """Raw counts + a var frame that still carries gene symbols.

    Cannot use ``benchmark.prep.counts_adata``: it strips ``.var`` down to a bare index,
    and panhumanpy needs the symbol column for Ensembl-keyed atlases.
    """
    import anndata as ad
    import scipy.sparse as sp

    if adata.raw is not None:
        X, var = adata.raw.X, adata.raw.var.copy()
    else:
        X, var = adata.X, adata.var.copy()
    a = ad.AnnData(X=sp.csr_matrix(X).astype("float32"), obs=adata.obs.copy(), var=var)

    col = next((c for c in _SYMBOL_COLS if c in a.var.columns), None)
    if _looks_ensembl(a.var_names) and col is None:
        raise ValueError(
            "Pan-human Azimuth needs gene symbols, but var_names look like Ensembl ids "
            f"and none of {_SYMBOL_COLS} is present in .var. Add a symbol column."
        )
    return a, (col if _looks_ensembl(a.var_names) else None)


@register
class PanHumanAzimuth(AnnotationMethod):
    """Pretrained pan-human hierarchical annotator; no training, fixed typology."""

    name = "panhuman-azimuth"
    tier = "pretrained"
    device = "cpu"

    def __init__(self, level="azimuth_fine", eval_batch_size=8192, refine=True):
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}")
        self.level = level
        self.eval_batch_size = eval_batch_size
        self.refine = refine

    def fit(self, ref, label_key):
        """No training. Loads the pretrained weights, so model loading is attributed
        here rather than to prediction -- the same load-once/annotate-many accounting
        the other pretrained methods get."""
        import panhumanpy as ph
        self._base = ph.AzimuthNN_base(eval_batch_size=self.eval_batch_size)

    def predict(self, query):
        a, feature_col = _to_panhuman_input(query)

        # Drive the low-level class rather than the high-level `AzimuthNN`, which takes
        # the query in its constructor and therefore reloads the weights on every call
        # (~0.4 s). Verified to produce labels identical to the high-level path.
        az = self._base
        az.query_adata(a, feature_names_col=feature_col)
        az.process_query()
        az.run_inference_model()
        az.calibrate_predictions()
        az.process_outputs(mode="minimal")
        for lvl in (("broad", "medium", "fine") if self.refine is True
                    else tuple(self.refine or ())):
            az.refine_labels(lvl)
        az.update_cells_meta()
        meta = az.cells_meta

        # When panhumanpy cannot make the levels agree it sets `full_consistent_hierarchy`
        # False and *blanks the refined columns to the boolean False* rather than to a
        # label. Those cells still carry a deepest-level call in `final_level_labels`, so
        # we fall back to it instead of discarding them: the model did predict, it just
        # could not reconcile the prediction with its coarser levels. Dropping them would
        # silently score Pan-human Azimuth only on the cells it found easy.
        level = meta[self.level] if self.level in meta else meta["final_level_labels"]
        blank = ~level.map(lambda v: isinstance(v, str)) | level.astype(str).isin(
            ("False", "nan", "")
        )
        resolved = level.astype(str).where(~blank, meta["final_level_labels"].astype(str))
        self.n_fallback = int(blank.sum())

        # Map whatever label we actually scored -- not the blanked column -- to CL.
        meta["_scored_label"] = resolved
        az.cells_meta = meta
        az.map_to_cell_ontology("_scored_label", include_cl_id=True)
        meta = az.cells_meta

        labels = resolved.to_numpy(dtype=object)
        cl = meta["_scored_label_CL_ID"].astype(str).to_numpy()
        # 'unmapped' is panhumanpy's sentinel; align it with our own marker so ontology
        # scoring treats it as a miss rather than a bogus CL id.
        cl = np.where(cl == "unmapped", "unknown", cl)

        # 'Unassigned' is the model's trained quality-control class -- an explicit refusal,
        # unlike the hierarchy-inconsistent cells above -- so report it as a rejection.
        unassigned = (labels == "Unassigned")

        conf = None
        if "final_level_confidence" in meta:
            conf = meta["final_level_confidence"].to_numpy(dtype=np.float32)

        return Predictions(
            cell_ids=list(query.obs_names),
            labels=labels,
            probabilities=conf,
            unassigned=unassigned if unassigned.any() else None,
            label_cl=cl,
        )
