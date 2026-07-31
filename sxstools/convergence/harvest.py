"""Catalog-wide harvester for horizon-mass convergence data.

Usage:
    python -m sxstools.convergence.harvest --results-dir DIR
        [--only-cached] [--levs all|top2] [--max-sims N]
        [--sims SXS:BBH:0002 ...] [--overwrite] [--t-junk 200]
        [--n-points 4000]

Writes one pickle per simulation into the results dir (atomic writes, so
the run is resumable: existing outputs are skipped unless --overwrite) and
a failures.json with per-sim/per-lev failure reasons. Failed sims produce
no pickle, so a rerun retries them for free.

The loop is serial: the run is download-dominated and the sxs cache is not
designed for concurrent writers. For a pure-compute pass over an already
populated cache, use --only-cached (no network).
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from sxstools.convergence.horizons import (
    MASS_DATASETS,
    analyze_horizon_convergence,
    load_sim_lev,
)

META_KEYS = [
    "reference_mass_ratio",
    "reference_dimensionless_spin1",
    "reference_dimensionless_spin2",
    "reference_eccentricity",
    "reference_time",
    "remnant_mass",
    "remnant_dimensionless_spin",
    "number_of_orbits",
]


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def result_to_dict(cr):
    """Flatten a ConvergenceResult into a compact, pickle-friendly dict."""
    out = {
        "grid": _f32(cr.grid),
        "resampled": {lev: _f32(q) for lev, q in cr.resampled.items()},
        "diffs": {
            pair: {k: _f32(v) for k, v in d.items()}
            for pair, d in cr.diffs.items()
        },
        "top_pair": cr.top_pair,
        "order": cr.order,
        "scalars": cr.scalars,
    }
    if cr.order_series is not None:
        out["order_series"] = tuple(_f32(a) for a in cr.order_series)
    if cr.mono is not None:
        out["mono_fraction"] = cr.mono["fraction"]
        out["mono_fraction_series"] = _f32(cr.mono["fraction_series"])
    if cr.trust is not None:
        out["trust"] = {
            "t_centers": _f32(cr.trust["t_centers"]),
            "p": _f32(cr.trust["p"]),
            "mono_frac": _f32(cr.trust["mono_frac"]),
            "trusted": cr.trust["trusted"],
            "intervals": cr.trust["intervals"],
            "t_onset": cr.trust["t_onset"],
        }
    return out


def extract_meta(metadata):
    meta = {}
    for key in META_KEYS:
        meta[key] = metadata.get(key)
    return meta


def load_levs(sim, levs, only_cached, retries=2, retry_wait=5.0):
    """Try to load horizon masses for each lev. Returns (per_lev, info, fails)."""
    per_lev, info, fails = {}, None, []
    for lev in levs:
        err = None
        for attempt in range(retries + 1):
            try:
                masses, lev_info = load_sim_lev(
                    sim, lev, download=not only_cached
                )
                if masses:
                    per_lev[lev] = masses
                    info = lev_info  # keep highest successful lev's info
                else:
                    err = "no horizon groups in Horizons.h5"
                break
            except Exception as e:  # network, missing lev, corrupt h5, ...
                err = repr(e)
                if only_cached:
                    break  # not-cached is expected; don't retry or sleep
                if attempt < retries:
                    time.sleep(retry_wait)
        if err is not None:
            fails.append({"sim": sim, "lev": lev, "error": err})
    return per_lev, info, fails


def process_sim(sim, levs, args):
    """Harvest one simulation. Returns (result_dict | None, failures)."""
    per_lev, info, fails = load_levs(sim, levs, args.only_cached)
    if len(per_lev) < 2:
        fails.append({"sim": sim, "error": "fewer than 2 usable levs"})
        return None, fails

    metadata = info["metadata"]
    t_ref = metadata.get("reference_time")
    results, t_cmn = analyze_horizon_convergence(
        per_lev,
        t_junk=args.t_junk,
        t_ref=t_ref,
        n_points=args.n_points,
    )
    version = os.path.basename(info["location"]).split("v")[-1]
    out = {
        "sim": sim,
        "version": version,
        "levs": sorted(per_lev),
        "meta": extract_meta(metadata),
        "t_cmn": t_cmn,
        "results": {
            ah: {mt: result_to_dict(cr) for mt, cr in by_mt.items()}
            for ah, by_mt in results.items()
        },
    }
    return out, fails


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def sim_output_path(results_dir, sim):
    return results_dir / (sim.replace(":", "_") + ".pkl")


def select_sims(args):
    """Enumerate (sim, levs) pairs from the sxs catalog."""
    import sxs

    simulations = sxs.load("simulations", BBH=True)
    df = simulations.dataframe
    if not args.include_deprecated and "deprecated" in df:
        df = df[~df["deprecated"]]
    # BBH=True does not restrict the catalog index; NSNS/BHNS runs have no
    # (initial) horizons, so keep only BBH systems
    sim_names = list(args.sims) if args.sims else [
        s for s in df.index if s.startswith("SXS:BBH:")
    ]

    pairs, skipped = [], []
    for sim in sim_names:
        try:
            levs = sorted(simulations[sim].lev_numbers)
        except KeyError:
            skipped.append({"sim": sim, "error": "not in catalog"})
            continue
        if len(levs) < 2:
            skipped.append({"sim": sim, "error": f"only levs {levs} in catalog"})
            continue
        if args.levs == "top2":
            levs = levs[-2:]
        pairs.append((sim, levs))
    if args.max_sims:
        pairs = pairs[: args.max_sims]
    return pairs, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("horizon_convergence_results"))
    ap.add_argument("--only-cached", action="store_true",
                    help="no downloads; use only locally cached files")
    ap.add_argument("--levs", choices=["all", "top2"], default="all")
    ap.add_argument("--max-sims", type=int, default=None)
    ap.add_argument("--sims", nargs="*", default=None,
                    help="specific simulation IDs (default: whole catalog)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include-deprecated", action="store_true")
    ap.add_argument("--t-junk", type=float, default=200.0,
                    help="junk-radiation time to skip for AhA/AhB [M]")
    ap.add_argument("--n-points", type=int, default=4000)
    args = ap.parse_args(argv)

    from tqdm import tqdm

    args.results_dir.mkdir(parents=True, exist_ok=True)
    failures_path = args.results_dir / "failures.json"

    pairs, failures = select_sims(args)
    print(f"{len(pairs)} simulations to process "
          f"({len(failures)} skipped at catalog level)")

    n_done = n_skip = n_fail = 0
    for sim, levs in tqdm(pairs, unit="sim"):
        out_path = sim_output_path(args.results_dir, sim)
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            result, fails = process_sim(sim, levs, args)
        except Exception as e:  # never let one sim kill the sweep
            result, fails = None, [{"sim": sim, "error": repr(e)}]
        failures.extend(fails)
        if result is not None:
            atomic_save(result, out_path)
            n_done += 1
        else:
            n_fail += 1
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=1)

    print(f"done: {n_done} harvested, {n_skip} already present, "
          f"{n_fail} failed ({failures_path})")


if __name__ == "__main__":
    main()
