"""Reusable convergence plots.

All functions accept either a ``core.ConvergenceResult`` or the flattened
dict produced by ``harvest.result_to_dict`` (i.e. a loaded pickle entry
``d["results"][ah][mass_type]``). Figure styling is left to the caller
(e.g. ``vlconf.conf_matplolib()``).
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_level_overlay",
    "plot_stacked_differences",
    "plot_order_series",
]


def _get(r, key):
    if isinstance(r, dict):
        return r.get(key)
    val = getattr(r, key, None)
    if key == "mono_fraction_series" and val is None:
        mono = getattr(r, "mono", None)
        return None if mono is None else mono["fraction_series"]
    return val


def _diffs(r):
    return r["diffs"] if isinstance(r, dict) else r.diffs


def _resampled(r):
    return r["resampled"] if isinstance(r, dict) else r.resampled


def plot_level_overlay(result, ax=None, label="q", time_label="t [M]"):
    """Overlay q(t) of all levels on the common grid."""
    if ax is None:
        _, ax = plt.subplots()
    grid = _get(result, "grid")
    for lev, q in sorted(_resampled(result).items()):
        ax.plot(grid, q, label=f"Lev{lev}")
    ax.set_xlabel(time_label)
    ax.set_ylabel(label)
    ax.legend()
    return ax


def plot_stacked_differences(result, ax=None, relative=True,
                             time_label="t [M]"):
    """Semilog |Δq|(t) for each successive level pair.

    The classic stacked convergence plot: curves should stack downward with
    increasing resolution wherever the data is converging.
    """
    if ax is None:
        _, ax = plt.subplots()
    grid = _get(result, "grid")
    key = "rel" if relative else "abs"
    for (lo, hi), d in sorted(_diffs(result).items()):
        ax.semilogy(grid, d[key], label=f"Lev{hi}−Lev{lo}")
    ax.set_xlabel(time_label)
    ax.set_ylabel(("relative " if relative else "") + "|Δq|")
    ax.legend()
    return ax


def plot_order_series(result, ax=None, time_label="t [M]"):
    """Time-resolved convergence rate p(t) with the monotone fraction.

    Convergence is generally not uniform in time (early inspiral is
    typically unconverged); this shows where the data actually converges.
    Requires >= 3 levels; returns None otherwise.
    """
    series = _get(result, "order_series")
    if series is None:
        return None
    if ax is None:
        _, ax = plt.subplots()
    trust = _get(result, "trust")
    if trust is not None:
        for lo, hi in trust["intervals"]:
            ax.axvspan(lo, hi, color="C2", alpha=0.12)
    t_c, p = series
    ax.plot(t_c, p, marker="o", ms=3, label="p(t)")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    order = _get(result, "order")
    if order is not None:
        ax.axhline(order, color="C1", lw=0.8, ls=":",
                   label=f"global p = {order:.2f}")
    mono = _get(result, "mono_fraction_series")
    if mono is not None:
        grid = _get(result, "grid")
        ax2 = ax.twinx()
        ax2.plot(grid, mono, color="C2", alpha=0.5, lw=1.0)
        ax2.set_ylabel("monotone fraction", color="C2")
        ax2.set_ylim(-0.05, 1.05)
    ax.set_xlabel(time_label)
    ax.set_ylabel("convergence rate p")
    ax.legend(loc="best")
    return ax
