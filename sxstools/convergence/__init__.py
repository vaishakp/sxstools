"""Convergence analysis of level-resolved (multi-resolution) time series.

Core (sxs-independent): pass any ``{lev: (t, q)}`` level series to
``analyze_level_series``. SXS adapters live in ``horizons`` (Horizons.h5
mass evolution) and the catalog harvester in ``harvest``
(``python -m sxstools.convergence.harvest``).
"""

from sxstools.convergence.core import (
    ConvergenceResult,
    analyze_level_series,
    common_grid,
    convergence_order,
    convergence_order_series,
    level_differences,
    monotonicity,
    resample,
    scalar_at,
    tail_mean,
    trusted_intervals,
)
