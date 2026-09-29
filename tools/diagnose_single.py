"""Measure float32 waveform error against float64 at identical inputs.

Run from the repository root, serially::

    PYTHONPATH=src python tools/diagnose_single.py

The report is written to ``tools/performance/single-precision.json`` unless
``--output`` names another file.

The corners of the validated range are T in {0.25, 10} years, f0 in
{0.1, 30} mHz, t0 in {0, 10} years and both signs of each frequency
derivative. At each corner and in every TDI format,
with and without a position cache, the float32 ``ucb_waveform`` runs with all
reformulations and with each one disabled through the private diagnostic
switch. The float64 inputs are the exact float64 values of the float32 ones,
with ``f0*T = q_ref + offset`` and the carrier pinned to the float32 carrier.
Both paths use the same ``BW`` from ``ucb_bw`` and the same explicit response
grid, which is the automatic count for T. The arbitrary-time ``tdi_response``
and the physical ``tdi_waveform`` are measured at the same corners.

A/E relative L2 error and mismatch (flat weighting over the band) are computed
in float64: raw strains near 1e-21 square below float32's range. A
reformulation is kept only if disabling it takes some corner beyond half the
budget, relative L2 5e-5 or mismatch 5e-9.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_onfly import provenance
from numpy.typing import ArrayLike

import binary_galactica as bg
from binary_galactica import single as bg32
from binary_galactica.glass import single as single_impl
from binary_galactica.glass.constants import YEAR
from binary_galactica.glass.waveform import _AUTO_RESPONSE_SAMPLES

# The float64 reference requires JAX double precision; importing
# binary_galactica does not enable it. The float32 path is unaffected.
jax.config.update("jax_enable_x64", True)

BUDGET = {"relative_l2": 1e-4, "mismatch": 1e-8}
HALF_BUDGET = {name: value / 2 for name, value in BUDGET.items()}
DURATIONS_YEARS = (0.25, 10.0)
FREQUENCIES_HZ = (1e-4, 3e-2)
EPOCHS_YEARS = (0.0, 10.0)
#: Fractional frequency bin at time zero, the float32 native offset.
OFFSET = 0.3719
#: Drift, in bins, of the first and second derivative terms from time zero to
#: the end of the observation, taken with each sign.
FIRST_DRIFT, SECOND_DRIFT = 4.0, 1.0
SIGNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
#: Near the ecliptic, where the Doppler delay k.x0/c is largest.
SKY = {"beta": 0.1, "lam": 4.1}
SOURCE = {"amp": 1e-22, "phi0": 0.7, "iota": 1.1, "psi": 0.4}
#: Seconds per sample of the full data grid, which only clamps ucb_bw.
CADENCE = 5.0
RESPONSE_TIMES = 257
CONFIGURATIONS = ("all", *(f"without_{name}" for name in single_impl.REFORMULATIONS))


def automatic_samples(duration: float) -> int:
    """Return the automatic sparse response count used for ``duration``."""
    boundaries = YEAR * np.asarray(_AUTO_RESPONSE_SAMPLES[:-1]) / 128
    return _AUTO_RESPONSE_SAMPLES[int(np.sum(boundaries < duration))]


def corner(
    years: float, frequency: float, epoch_years: float, signs: tuple[float, float]
) -> dict:
    """Return float32 inputs, their float64 twins and the band for one corner."""
    duration32 = np.float32(years * YEAR)
    epoch32 = np.float32(epoch_years * YEAR)
    duration, epoch = float(duration32), float(epoch32)
    q_ref = math.floor(frequency * duration)
    offset = np.float32(OFFSET)
    end = epoch + duration
    fdot = signs[0] * FIRST_DRIFT / (duration * end)
    fddot = signs[1] * 2 * SECOND_DRIFT / (duration * end**2)
    params = np.array(
        [
            offset,
            np.sin(SKY["beta"]),
            SKY["lam"],
            np.log(2 * SOURCE["amp"]),
            np.cos(SOURCE["iota"]),
            SOURCE["psi"],
            -SOURCE["phi0"],
            fdot * duration**2,
            fddot * duration**3,
        ],
        dtype=np.float32,
    )
    native = params.astype(np.float64)
    native[0] = q_ref + native[0]
    bandwidth = bg.ucb_bw(
        native[0] / duration,
        fdot,
        native[1],
        SOURCE["amp"],
        duration,
        int(duration / CADENCE),
    )
    return {
        "T": duration32,
        "t0": epoch32,
        "q_ref": q_ref,
        "params": params,
        "native": native,
        "BW": int(bandwidth),
        "response_samples": automatic_samples(duration),
        "inputs": {
            "T_years": years,
            "f0_hz": native[0] / duration,
            "t0_years": epoch_years,
            "T": duration,
            "t0": epoch,
            "q_ref": q_ref,
            "offset": float(offset),
            "fdot": float(params[7]) / duration**2,
            "fddot": float(params[8]) / duration**3,
            "derivative_signs": list(signs),
            "params_float32": params.tolist(),
        },
    }


def errors(single: ArrayLike, double: ArrayLike) -> dict[str, float]:
    """Relative L2 error and flat-weighted mismatch, in float64."""
    a = np.asarray(single, dtype=np.complex128).ravel()
    b = np.asarray(double, dtype=np.complex128).ravel()
    overlap = np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    return {
        "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "mismatch": float(1 - overlap),
    }


def channels(result) -> np.ndarray:
    """Stack A and E."""
    return np.stack([np.asarray(result["A"]), np.asarray(result["E"])])


def edge_fraction(result) -> float:
    """Fraction of A/E power in the outer eighth of the band at each end."""
    power = np.sum(np.abs(channels(result).astype(np.complex128)) ** 2, axis=0)
    edge = max(1, power.size // 8)
    return float((power[:edge].sum() + power[-edge:].sum()) / power.sum())


def check_single(result) -> None:
    """Require float32 dtypes and finite values from the float32 path."""
    for name, value in result.items():
        value = np.asarray(value)
        expected = np.int32 if name in ("qmin", "freq_bin_indices") else np.complex64
        assert value.dtype == expected, (name, value.dtype)
        assert np.all(np.isfinite(value)), name


def waveform_rows(case: dict) -> list[dict[str, Any]]:
    """Measure every format, cache choice and switch setting at one corner."""
    duration, epoch, bandwidth = case["T"], case["t0"], case["BW"]
    samples = case["response_samples"]
    cache32 = np.asarray(
        bg.orbit_cache(float(duration), float(epoch), bandwidth), dtype=np.float32
    )
    rows = []
    for fmt in bg.FORMATS:
        for cached in (False, True):
            positions32 = cache32 if cached else None
            positions64 = cache32.astype(np.float64) if cached else None
            reference = bg.ucb_waveform(
                case["native"],
                T=float(duration),
                BW=bandwidth,
                t0=float(epoch),
                fmt=fmt,
                positions=positions64,
                q=float(case["q_ref"]),
                response_samples=samples,
            )
            row: dict[str, Any] = {
                "fmt": fmt,
                "cache": cached,
                "edge_power_fraction": edge_fraction(reference),
            }
            for configuration in CONFIGURATIONS:
                names = () if configuration == "all" else (configuration[8:],)
                with single_impl._without(*names):
                    result = bg32.ucb_waveform(
                        case["params"],
                        q_ref=case["q_ref"],
                        T=duration,
                        BW=bandwidth,
                        t0=epoch,
                        fmt=fmt,
                        positions=positions32,
                        response_samples=samples,
                    )
                check_single(result)
                assert int(result["qmin"]) == int(reference["qmin"])
                row[configuration] = errors(channels(result), channels(reference))
            rows.append(row)
    return rows


def response_rows(case: dict) -> list[dict[str, Any]]:
    """Measure the arbitrary-time response across the observation."""
    duration, epoch = case["T"], case["t0"]
    grid = np.linspace(0.0, 1.0, RESPONSE_TIMES)
    times = (np.float64(epoch) + np.float64(duration) * grid).astype(np.float32)
    times64 = times.astype(np.float64)
    positions32 = np.asarray(bg.analytic_orbits(times64), dtype=np.float32)
    rows = []
    for given in (False, True):
        reference = bg.tdi_response(
            case["native"],
            times64,
            T=float(duration),
            positions=positions32.astype(np.float64) if given else None,
        )
        row: dict[str, Any] = {"positions": given}
        for configuration in ("all", "without_orbit"):
            names = () if configuration == "all" else ("orbit",)
            with single_impl._without(*names):
                result = bg32.tdi_response(
                    case["params"],
                    times,
                    q_ref=case["q_ref"],
                    T=duration,
                    positions=positions32 if given else None,
                )
            assert np.asarray(result).dtype == np.complex64
            row[configuration] = errors(result, reference)
        rows.append(row)
    return rows


def physical_row(case: dict) -> dict[str, Any]:
    """Measure the physical adapter with all reformulations, Sangria, no cache."""
    duration32, epoch, bandwidth = case["T"], case["t0"], case["BW"]
    duration = float(duration32)
    theta = np.array(
        [
            SOURCE["amp"],
            float(case["params"][0]) / duration,
            float(case["inputs"]["fdot"]),
            SOURCE["phi0"],
            SOURCE["iota"],
            SOURCE["psi"],
            SKY["lam"],
            SKY["beta"],
        ]
    )
    theta32 = theta.astype(np.float32)
    theta64 = theta32.astype(np.float64)
    theta64[1] = case["q_ref"] / duration + theta64[1]
    result = bg32.tdi_waveform(
        theta32,
        q_ref=case["q_ref"],
        T=duration32,
        BW=bandwidth,
        t0=epoch,
        response_samples=case["response_samples"],
    )
    check_single(result)
    reference = bg.ucb_waveform(
        np.asarray(bg.to_glass_vector(jnp.asarray(theta64), duration)),
        T=duration,
        BW=bandwidth,
        t0=float(epoch),
        q=float(case["q_ref"]),
        response_samples=case["response_samples"],
    )
    return errors(channels(result), channels(reference))


def summarize(corners: list[dict[str, Any]]) -> dict[str, Any]:
    """Largest errors per configuration, where they occur, and the decisions."""

    def rows(configuration):
        for entry in corners:
            where = {
                key: entry["inputs"][key]
                for key in ("T_years", "f0_hz", "t0_years", "derivative_signs")
            }
            for row in entry["waveform"]:
                yield row[configuration], {
                    **where,
                    "entry": "ucb_waveform",
                    "fmt": row["fmt"],
                    "cache": row["cache"],
                }
            if configuration in ("all", "without_orbit"):
                for row in entry["response"]:
                    yield row[configuration], {
                        **where,
                        "entry": "tdi_response",
                        "positions": row["positions"],
                    }

    def largest(configuration, metric):
        value, where = max(rows(configuration), key=lambda row: row[0][metric])
        return {"value": value[metric], "at": where}

    worst = {
        configuration: {metric: largest(configuration, metric) for metric in BUDGET}
        for configuration in CONFIGURATIONS
    }
    physical = {
        metric: max(entry["physical"][metric] for entry in corners) for metric in BUDGET
    }
    decisions = {
        name: {
            "largest_when_disabled": {
                metric: worst[f"without_{name}"][metric]["value"] for metric in BUDGET
            },
            "keep": any(
                worst[f"without_{name}"][metric]["value"] > HALF_BUDGET[metric]
                for metric in BUDGET
            ),
        }
        for name in single_impl.REFORMULATIONS
    }
    within = all(
        worst["all"][metric]["value"] <= BUDGET[metric]
        and physical[metric] <= BUDGET[metric]
        for metric in BUDGET
    )
    return {
        "largest": worst,
        "largest_physical": physical,
        "within_budget": within,
        "reformulations": decisions,
    }


def main() -> None:
    """Measure every corner and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/performance/single-precision.json"),
    )
    arguments = parser.parse_args()
    start = time.perf_counter()
    corners = []
    for years in DURATIONS_YEARS:
        for frequency in FREQUENCIES_HZ:
            for epoch_years, signs in itertools.product(EPOCHS_YEARS, SIGNS):
                case = corner(years, frequency, epoch_years, signs)
                entry = {
                    "inputs": case["inputs"],
                    "BW": case["BW"],
                    "response_samples": case["response_samples"],
                    "waveform": waveform_rows(case),
                    "response": response_rows(case),
                    "physical": physical_row(case),
                }
                corners.append(entry)
                largest = {
                    configuration: max(
                        row[configuration]["relative_l2"] for row in entry["waveform"]
                    )
                    for configuration in CONFIGURATIONS
                }
                print(
                    years,
                    frequency,
                    epoch_years,
                    signs,
                    case["BW"],
                    largest,
                    flush=True,
                )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance(),
        "purpose": (
            "Float32 on-fly waveform error against float64 at identical inputs, "
            "at the corners of the validated range, with each float32 "
            "reformulation disabled in turn."
        ),
        "method": {
            "budget": BUDGET,
            "adoption_threshold": HALF_BUDGET,
            "inputs": (
                "float64 inputs are the exact float64 values of the float32 "
                "inputs, with f0*T = q_ref + offset; the float64 carrier q is "
                "pinned to q_ref; caches are float32-rounded orbit_cache output"
            ),
            "metrics": (
                "A and E stacked, computed in float64; mismatch uses a flat "
                "weighting over the band. tdi_response rows use X/Y/Z at "
                f"{RESPONSE_TIMES} times spanning [t0, t0 + T]."
            ),
            "sources": {
                "offset": OFFSET,
                "first_derivative_drift_bins": FIRST_DRIFT,
                "second_derivative_drift_bins": SECOND_DRIFT,
                "drift_interval": "time zero to t0 + T",
                "derivative_signs": SIGNS,
                **SKY,
                **SOURCE,
            },
            "bandwidth": f"ucb_bw with N = T/{CADENCE} s, amplitude as above",
            "response_samples": "the automatic count for T, passed explicitly",
            "physical": "tdi_waveform, all reformulations, Sangria, no cache",
        },
        "seconds": time.perf_counter() - start,
        "summary": summarize(corners),
        "corners": corners,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
