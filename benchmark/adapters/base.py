"""Uniform adapter interface + registry for annotation methods.

Every benchmarked method subclasses :class:`AnnotationMethod` and is registered by
name. The runner times ``fit`` and ``predict`` separately under a resource monitor,
so adapters should keep import-time and weight-loading work out of those methods
where possible (or accept that it is attributed to fit).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Predictions:
    """A method's output for a query set."""
    cell_ids: List[str]
    labels: np.ndarray                      # (n,) predicted label strings
    probabilities: Optional[np.ndarray] = None   # (n,) confidence of the call
    unassigned: Optional[np.ndarray] = None      # (n,) bool, True = rejected


class AnnotationMethod:
    """Base class. Subclasses set ``name``/``tier`` and implement fit + predict."""

    name: str = "base"
    tier: str = "classical"        # classical | deep | foundation
    device: str = "cpu"            # cpu | mps

    def fit(self, ref, label_key: str) -> None:
        """Fit / train on a labeled reference AnnData (raw counts in .X or .raw)."""
        raise NotImplementedError

    def predict(self, query) -> Predictions:
        """Annotate a query AnnData; return :class:`Predictions`."""
        raise NotImplementedError


REGISTRY = {}


def register(cls):
    """Class decorator: add a method adapter to the global registry by ``name``."""
    REGISTRY[cls.name] = cls
    return cls
