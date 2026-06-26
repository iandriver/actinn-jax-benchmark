"""Importing this package registers all available method adapters.

Tier-2 (deep) and Tier-3 (foundation) adapters are added here as they are built;
each lives in its own module and is guarded so a missing optional dependency does
not break the whole registry.
"""

from .base import REGISTRY, AnnotationMethod, Predictions, register  # noqa: F401

# Tier 1 — classical (always available with the core env).
from . import actinn_jax_adapter  # noqa: F401,E402
from . import celltypist_adapter  # noqa: F401,E402
from . import svm_adapter  # noqa: F401,E402
from . import knn_adapter  # noqa: F401,E402
from . import actinn_orig_adapter  # noqa: F401,E402  (lazy TF import; runs in .venv-tf)


def get(name, **kwargs):
    """Instantiate a registered method adapter by name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
