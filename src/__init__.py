"""Shared code for the Twin-2K-500 behaviour-model study."""
import os

# The host reports many cores, but BLAS oversubscription made linear algebra ~10x slower here.
# Cap math-library threads for every script (override with LBM_THREADS).
try:
    from threadpoolctl import threadpool_limits

    threadpool_limits(int(os.environ.get("LBM_THREADS", "4")))
except ImportError:  # threadpoolctl ships with scikit-learn
    pass
