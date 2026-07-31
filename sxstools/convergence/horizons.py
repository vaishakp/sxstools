"""SXS adapter: horizon-mass level series from Horizons.h5 files.

Reads Christodoulou (``chr``) and irreducible/areal (``irr``) masses of the
individual horizons AhA/AhB and the remnant AhC, and feeds them to the
generic convergence core.

Alignment policy:
  - AhA/AhB are compared on the raw coordinate time, skipping ``t_junk``
    (default 200 M) of junk-radiation-contaminated early inspiral. Note the
    early inspiral is generically unconverged; the time-resolved diagnostics
    (order_series, mono fraction_series) expose this.
  - AhC is compared primarily on the shifted time tau = t - t_cmn(lev),
    where t_cmn is the common-horizon formation time at that lev; this
    removes the trivial offset from merger occurring at slightly different
    coordinate times per resolution. An unaligned raw-t comparison is also
    produced as a diagnostic — its excess over the aligned one measures the
    merger-time-shift contribution.
"""

import h5py
import numpy as np

from sxstools.convergence.core import analyze_level_series, scalar_at

__all__ = [
    "MASS_DATASETS",
    "read_horizon_masses",
    "load_sim_lev",
    "horizon_level_series",
    "analyze_horizon_convergence",
]

MASS_DATASETS = {"chr": "ChristodoulouMass.dat", "irr": "ArealMass.dat"}
AHS = ("AhA", "AhB", "AhC")


def read_horizon_masses(h5path):
    """Read mass time series from one Horizons.h5.

    Returns ``{Ah: {"t": t, "chr": M, "irr": M}}`` for each of AhA/AhB/AhC
    present in the file (a horizon missing its group is simply omitted,
    e.g. runs without a common horizon).
    """
    out = {}
    with h5py.File(h5path, "r") as f:
        for ah in AHS:
            grp = f.get(f"{ah}.dir")
            if grp is None:
                continue
            entry = {}
            for key, dset in MASS_DATASETS.items():
                if dset in grp:
                    data = np.asarray(grp[dset])
                    entry.setdefault("t", data[:, 0])
                    entry[key] = data[:, 1]
            if "t" in entry:
                out[ah] = entry
    return out


def load_sim_lev(sim, lev, download=True):
    """Load one (sim, Lev)'s horizon masses via the sxs package.

    Returns ``(masses, info)`` where masses is ``read_horizon_masses``
    output and info holds the resolved location and metadata.
    """
    import os

    import sxs

    sdata = sxs.load(f"{sim}/Lev{lev}", ignore_deprecation=True,
                     download=download)
    h5path = f"{sdata.__file__}/{sdata.horizons_path}"
    if download and not os.path.exists(h5path):
        sdata.horizons  # sxs downloads Horizons.h5 lazily, on first access
    try:
        masses = read_horizon_masses(h5path)
    except OSError:
        # corrupt cached file (e.g. truncated download): re-fetch once
        if not download:
            raise
        os.remove(h5path)
        sdata.horizons
        masses = read_horizon_masses(h5path)
    info = {
        "location": sdata.__file__,
        "metadata": dict(sdata.metadata),
    }
    return masses, info


def horizon_level_series(per_lev_masses, ah, mass_type):
    """Assemble ``{lev: (t, M)}`` for one horizon and mass type.

    ``per_lev_masses`` is ``{lev: read_horizon_masses(...)}``. Levels
    missing the horizon or the mass dataset are dropped.
    """
    series = {}
    for lev, masses in per_lev_masses.items():
        entry = masses.get(ah)
        if entry is not None and mass_type in entry:
            series[lev] = (entry["t"], entry[mass_type])
    return series


def _safe_t_junk(series, t_junk, frac=0.1):
    """Clip the junk-time cut so short runs keep a usable overlap window.

    Ultra-short simulations (e.g. SXS:BBH:3873-3886) have horizons for less
    time than the default 200 M junk cut; fall back to skipping ``frac`` of
    the raw overlap instead.
    """
    t0 = max(t[0] for t, _ in series.values())
    t1 = min(t[-1] for t, _ in series.values())
    if t0 + t_junk >= t1:
        return frac * (t1 - t0)
    return t_junk


def analyze_horizon_convergence(
    per_lev_masses,
    t_junk=200.0,
    t_ref=None,
    pre_merger_offset=10.0,
    ahc_skip=5.0,
    n_points=4000,
    n_windows=40,
):
    """Full convergence analysis of the horizon masses of one simulation.

    Returns ``{Ah: {mass_type: ConvergenceResult}}`` with keys "AhA", "AhB",
    "AhC" (aligned on common-horizon formation) and "AhC_raw" (unaligned
    diagnostic), plus ``t_cmn`` (``{lev: common-horizon formation time}``).

    Scalars per AhA/AhB result: "t_ref" (if given) and "pre_merger"
    (evaluated at min_lev(t_cmn) - pre_merger_offset). The remnant final
    mass is the standard ``scalars["tail_mean"]``.
    """
    t_cmn = {}
    for lev, masses in per_lev_masses.items():
        if "AhC" in masses:
            t_cmn[lev] = float(masses["AhC"]["t"][0])

    results = {}
    for ah in ("AhA", "AhB"):
        eval_times = {}
        if t_ref is not None:
            eval_times["t_ref"] = t_ref
        if t_cmn:
            eval_times["pre_merger"] = min(t_cmn.values()) - pre_merger_offset
        results[ah] = {}
        for mt in MASS_DATASETS:
            series = horizon_level_series(per_lev_masses, ah, mt)
            if len(series) < 2:
                continue
            results[ah][mt] = analyze_level_series(
                series, t_junk=_safe_t_junk(series, t_junk),
                n_points=n_points, eval_times=eval_times,
                n_windows=n_windows,
            )

    results["AhC"] = {}
    results["AhC_raw"] = {}
    for mt in MASS_DATASETS:
        series = horizon_level_series(per_lev_masses, "AhC", mt)
        if len(series) < 2:
            continue
        align = {lev: t_cmn[lev] for lev in series}
        aligned = {lev: (t - align[lev], q) for lev, (t, q) in series.items()}
        results["AhC"][mt] = analyze_level_series(
            series, t_junk=_safe_t_junk(aligned, ahc_skip),
            align_origin=align, n_points=n_points, n_windows=n_windows,
        )
        results["AhC_raw"][mt] = analyze_level_series(
            series, n_points=n_points, n_windows=n_windows,
        )

    return results, t_cmn
