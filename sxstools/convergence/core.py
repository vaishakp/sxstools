"""Generic convergence analysis for level-resolved time series.

A "level series" is a dict ``{lev: (t, q)}`` mapping an integer resolution
level to a time axis ``t`` of shape ``(N,)`` and a quantity ``q`` of shape
``(N,)`` (scalar) or ``(N, d)`` (vector; handled component-wise, norms taken
over all components). Time is always axis 0.

Convergence is generally *not* uniform in time: early-inspiral data is
typically unconverged (junk radiation, unresolved initial transients) while
the late inspiral and ringdown converge cleanly. All diagnostics here are
therefore available both globally and time-resolved (windowed), and a
pointwise monotonicity check flags where successive-level differences
actually shrink with resolution.

This module is pure math: numpy + scipy only, no sxs/h5py dependencies.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    "common_grid",
    "resample",
    "level_differences",
    "monotonicity",
    "convergence_order",
    "convergence_order_series",
    "trusted_intervals",
    "scalar_at",
    "tail_mean",
    "analyze_level_series",
    "ConvergenceResult",
]


def _shifted(series, align_origin=None):
    """Return ``{lev: (t - t0(lev), q)}``; identity if align_origin is None."""
    if align_origin is None:
        return dict(series)
    return {lev: (t - align_origin[lev], q) for lev, (t, q) in series.items()}


def common_grid(series, t_junk=0.0, n_points=4000, align_origin=None):
    """Uniform time grid on the overlap of all levels.

    The window is ``[max_lev t[0] + t_junk, min_lev t[-1]]``. ``align_origin``
    (``{lev: t0}``) shifts each level's time axis to ``t - t0`` before taking
    the overlap, e.g. to align remnant series on common-horizon formation.

    Raises ValueError if the overlap is empty.
    """
    shifted = _shifted(series, align_origin)
    t_start = max(t[0] for t, _ in shifted.values()) + t_junk
    t_end = min(t[-1] for t, _ in shifted.values())
    if t_end <= t_start:
        raise ValueError(
            f"Empty overlap window: [{t_start}, {t_end}] "
            f"(t_junk={t_junk}, levs={sorted(series)})"
        )
    return np.linspace(t_start, t_end, n_points)


def resample(series, grid, align_origin=None, kind="cubic"):
    """Interpolate every level onto ``grid``. Returns ``{lev: q(grid)}``.

    Cubic interp1d, not waveformtools.get_val_at_t_ref: the latter rounds to
    5 decimals, which floors level differences at 1e-5 — the scale of
    interest here.
    """
    shifted = _shifted(series, align_origin)
    out = {}
    for lev, (t, q) in shifted.items():
        out[lev] = interp1d(t, q, kind=kind, axis=0, assume_sorted=True)(grid)
    return out


def _successive_pairs(resampled):
    levs = sorted(resampled)
    return [(levs[i], levs[i + 1]) for i in range(len(levs) - 1)]


def level_differences(resampled, rel_floor=1e-30):
    """Absolute and relative differences between successive level pairs.

    Returns ``{(lev_lo, lev_hi): {"abs": |Δq|(t), "rel": |Δq|/|q_hi|(t)}}``.
    For vector q, "abs" is the pointwise Euclidean norm of the difference and
    "rel" divides by the pointwise norm of the higher level.
    """
    diffs = {}
    for lo, hi in _successive_pairs(resampled):
        d = resampled[hi] - resampled[lo]
        if d.ndim > 1:
            d_abs = np.linalg.norm(d, axis=tuple(range(1, d.ndim)))
            q_ref = np.linalg.norm(resampled[hi], axis=tuple(range(1, d.ndim)))
        else:
            d_abs = np.abs(d)
            q_ref = np.abs(resampled[hi])
        diffs[(lo, hi)] = {"abs": d_abs, "rel": d_abs / np.maximum(q_ref, rel_floor)}
    return diffs


def monotonicity(resampled, diffs=None):
    """Pointwise check that successive-level differences shrink with resolution.

    At each time, convergence is "monotone" if |q_L2 - q_L1| > |q_L3 - q_L2| >
    ... for ascending levels. Needs >= 3 levels; returns None otherwise.

    Returns dict with:
      - "mask": bool array, True where all successive difference pairs shrink
      - "fraction": float, fraction of the grid where mask is True
      - "fraction_series": running-window fraction (same length as grid,
        window ~ len/20) — shows *where* in time convergence is monotone
        (typically poor in early inspiral)
    """
    if len(resampled) < 3:
        return None
    if diffs is None:
        diffs = level_differences(resampled)
    d_abs = [diffs[pair]["abs"] for pair in _successive_pairs(resampled)]
    mask = np.ones(d_abs[0].shape[0], dtype=bool)
    for coarser, finer in zip(d_abs[:-1], d_abs[1:]):
        mask &= finer < coarser
    n = mask.size
    w = max(n // 20, 3)
    kernel = np.ones(w) / w
    fraction_series = np.convolve(mask.astype(float), kernel, mode="same")
    return {
        "mask": mask,
        "fraction": float(mask.mean()),
        "fraction_series": fraction_series,
    }


def _norm(x, norm="l2"):
    if norm == "l2":
        return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    if norm == "max":
        return float(np.max(np.abs(x)))
    raise ValueError(f"Unknown norm {norm!r}")


def convergence_order(resampled, norm="l2", diffs=None):
    """Global exponential convergence rate from the three highest levels.

    SpEC Levs are not grid doublings; the convention is exponential
    convergence, error ∝ exp(-p * Lev). With the three highest levels
    L1 < L2 < L3:

        p = log(||q_L2 - q_L1|| / ||q_L3 - q_L2||) / (L3 - L2)

    (the (L3 - L2) division keeps p meaningful for non-unit Lev spacing).
    Returns None with fewer than 3 levels, and NaN if either difference norm
    vanishes.
    """
    if len(resampled) < 3:
        return None
    levs = sorted(resampled)[-3:]
    if diffs is None:
        diffs = level_differences({lev: resampled[lev] for lev in levs})
    d_lo = _norm(diffs[(levs[0], levs[1])]["abs"], norm)
    d_hi = _norm(diffs[(levs[1], levs[2])]["abs"], norm)
    if d_lo <= 0 or d_hi <= 0:
        return float("nan")
    return float(np.log(d_lo / d_hi) / (levs[2] - levs[1]))


def convergence_order_series(resampled, grid, n_windows=40, norm="l2"):
    """Time-resolved convergence rate p(t) over sliding windows.

    Convergence is not uniform in time (early inspiral is typically
    unconverged), so a single global order can mislead. This splits the grid
    into ``n_windows`` contiguous windows and computes the three-highest-level
    order in each.

    Returns ``(t_centers, p)`` arrays of length n_windows, or None with < 3
    levels. p entries are NaN where a window's difference norm vanishes.
    """
    if len(resampled) < 3:
        return None
    levs = sorted(resampled)[-3:]
    diffs = level_differences({lev: resampled[lev] for lev in levs})
    d_lo = diffs[(levs[0], levs[1])]["abs"]
    d_hi = diffs[(levs[1], levs[2])]["abs"]
    edges = np.linspace(0, len(grid), n_windows + 1).astype(int)
    t_centers = np.empty(n_windows)
    p = np.full(n_windows, np.nan)
    for i in range(n_windows):
        sl = slice(edges[i], max(edges[i + 1], edges[i] + 2))
        t_centers[i] = 0.5 * (grid[sl][0] + grid[sl][-1])
        nlo, nhi = _norm(d_lo[sl], norm), _norm(d_hi[sl], norm)
        if nlo > 0 and nhi > 0:
            p[i] = np.log(nlo / nhi) / (levs[2] - levs[1])
    return t_centers, p


def trusted_intervals(resampled, grid, n_windows=40, p_min=0.0,
                      mono_min=0.8, norm="l2"):
    """Windowed trust assessment: where in time is the data converged?

    A window is *trusted* when both hold there:
      - the three-highest-level convergence rate p > ``p_min`` (errors
        shrink with resolution), and
      - the fraction of points with monotone successive-level differences
        is >= ``mono_min``.

    Needs >= 3 levels; returns None otherwise. Returns a dict with:
      - "t_centers", "p", "mono_frac", "trusted": per-window arrays
      - "intervals": list of (t_start, t_end) spans of contiguous trusted
        windows
      - "t_onset": earliest time from which the data stays trusted to the
        end of the grid (start of the trailing trusted run), or None. Note
        convergence can also fail at *late* times (e.g. near merger), in
        which case "intervals" is more informative than "t_onset".
    """
    if len(resampled) < 3:
        return None
    levs = sorted(resampled)[-3:]
    top3 = {lev: resampled[lev] for lev in levs}
    diffs = level_differences(top3)
    d_lo = diffs[(levs[0], levs[1])]["abs"]
    d_hi = diffs[(levs[1], levs[2])]["abs"]
    mono_mask = monotonicity(resampled)["mask"]

    edges = np.linspace(0, len(grid), n_windows + 1).astype(int)
    t_centers = np.empty(n_windows)
    p = np.full(n_windows, np.nan)
    mono_frac = np.empty(n_windows)
    for i in range(n_windows):
        sl = slice(edges[i], max(edges[i + 1], edges[i] + 2))
        t_centers[i] = 0.5 * (grid[sl][0] + grid[sl][-1])
        nlo, nhi = _norm(d_lo[sl], norm), _norm(d_hi[sl], norm)
        if nlo > 0 and nhi > 0:
            p[i] = np.log(nlo / nhi) / (levs[2] - levs[1])
        mono_frac[i] = mono_mask[sl].mean()
    trusted = (p > p_min) & (mono_frac >= mono_min)  # NaN p -> untrusted

    intervals = []
    start = None
    for i in range(n_windows):
        if trusted[i] and start is None:
            start = grid[edges[i]]
        elif not trusted[i] and start is not None:
            intervals.append((float(start), float(grid[edges[i] - 1])))
            start = None
    if start is not None:
        intervals.append((float(start), float(grid[-1])))

    t_onset = None
    if intervals and trusted[-1]:
        t_onset = intervals[-1][0]

    return {
        "t_centers": t_centers,
        "p": p,
        "mono_frac": mono_frac,
        "trusted": trusted,
        "intervals": intervals,
        "t_onset": t_onset,
    }


def scalar_at(series, t_eval, kind="cubic"):
    """Interpolate each level's quantity at one time. ``{lev: q(t_eval)}``.

    Returns None entries for levels whose time range does not contain
    t_eval.
    """
    out = {}
    for lev, (t, q) in series.items():
        if t[0] <= t_eval <= t[-1]:
            out[lev] = interp1d(t, q, kind=kind, axis=0, assume_sorted=True)(
                t_eval
            )[()]
        else:
            out[lev] = None
    return out


def tail_mean(series, n=100):
    """Mean of the last ``n`` samples per level: ``{lev: mean(q[-n:])}``.

    Matches the remnant final-mass convention used in area_theorem
    helpers.py (mean of the last 100 points).
    """
    return {lev: np.mean(q[-n:], axis=0) for lev, (t, q) in series.items()}


@dataclass
class ConvergenceResult:
    """Bundle of convergence diagnostics for one level series."""

    grid: np.ndarray
    resampled: dict  # {lev: q(grid)}
    diffs: dict  # {(lev_lo, lev_hi): {"abs", "rel"}}, successive pairs
    top_pair: tuple  # (lev, lev) of the two finest levels
    order: float | None  # global rate; None if < 3 levs
    order_series: tuple | None  # (t_centers, p(t)); None if < 3 levs
    mono: dict | None  # monotonicity(); None if < 3 levs
    trust: dict | None  # trusted_intervals(); None if < 3 levs
    scalars: dict = field(default_factory=dict)

    @property
    def top_diff(self):
        """Diff dict {"abs", "rel"} of the two finest levels."""
        return self.diffs[self.top_pair]


def analyze_level_series(
    series,
    t_junk=0.0,
    n_points=4000,
    align_origin=None,
    eval_times=None,
    tail_n=100,
    n_windows=40,
    norm="l2",
    p_min=0.0,
    mono_min=0.8,
):
    """One-call convergence analysis of a level series.

    ``eval_times`` is an optional dict ``{name: t}``; each produces a
    ``scalars[name] = {lev: q(t)}`` entry (None where t is out of range).
    ``scalars["tail_mean"]`` is always included.
    """
    grid = common_grid(series, t_junk=t_junk, n_points=n_points,
                       align_origin=align_origin)
    res = resample(series, grid, align_origin=align_origin)
    diffs = level_differences(res)
    levs = sorted(res)
    scalars = {"tail_mean": tail_mean(series, n=tail_n)}
    for name, t_eval in (eval_times or {}).items():
        scalars[name] = scalar_at(series, t_eval)
    return ConvergenceResult(
        grid=grid,
        resampled=res,
        diffs=diffs,
        top_pair=(levs[-2], levs[-1]),
        order=convergence_order(res, norm=norm, diffs=None),
        order_series=convergence_order_series(res, grid, n_windows=n_windows,
                                              norm=norm),
        mono=monotonicity(res, diffs=diffs),
        trust=trusted_intervals(res, grid, n_windows=n_windows, p_min=p_min,
                                mono_min=mono_min, norm=norm),
        scalars=scalars,
    )
