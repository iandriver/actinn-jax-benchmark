"""Accuracy metrics for annotation predictions, incl. rejection + ontology scoring."""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def compute(truth, pred, unassigned=None, ontology=None, truth_cl=None, pred_cl=None):
    """Return a dict of accuracy metrics.

    Parameters
    ----------
    truth, pred : array-like of label strings
    unassigned : bool array, optional
        True where the method rejected the cell (counted separately).
    ontology : dict {CL_id: frozenset(ancestor ids)}, optional
    truth_cl, pred_cl : array-like of CL ids, optional
        Required for ontology-aware (lineage) concordance.
    """
    truth = np.asarray(truth, dtype=object)
    pred = np.asarray(pred, dtype=object)
    out = {"n": int(len(truth))}

    if unassigned is not None:
        unassigned = np.asarray(unassigned, dtype=bool)
        out["pct_rejected"] = float(unassigned.mean())
        keep = ~unassigned
    else:
        out["pct_rejected"] = 0.0
        keep = np.ones(len(truth), dtype=bool)

    if keep.sum() > 0:
        t, p = truth[keep], pred[keep]
        out["accuracy"] = float(accuracy_score(t, p))
        out["macro_f1"] = float(f1_score(t, p, average="macro", zero_division=0))
        out["kappa"] = float(cohen_kappa_score(t, p))
    else:
        out["accuracy"] = out["macro_f1"] = out["kappa"] = 0.0

    if ontology is not None and truth_cl is not None and pred_cl is not None:
        out["ontology_concordance"] = _ontology(truth_cl, pred_cl, ontology, keep)
    return out


def _ontology(truth_cl, pred_cl, anc, keep):
    truth_cl, pred_cl = np.asarray(truth_cl)[keep], np.asarray(pred_cl)[keep]
    ok = n = 0
    for t, p in zip(truth_cl, pred_cl):
        if not isinstance(t, str) or not isinstance(p, str) or not t or not p:
            continue
        n += 1
        if t == p or p in anc.get(t, ()) or t in anc.get(p, ()):
            ok += 1
    return ok / max(n, 1)


def load_cl_ancestors(obo_path):
    """Build ``{CL_id: frozenset(ancestor ids incl. self)}`` from a Cell Ontology OBO."""
    import pronto
    ont = pronto.Ontology(obo_path)
    return {t.id: frozenset(s.id for s in t.superclasses(with_self=True))
            for t in ont.terms()}
