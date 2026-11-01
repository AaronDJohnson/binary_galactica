"""Shared helpers for the segment validation tools (application side)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sangria  # enables x64 before the package is used

from binary_galactica.glass import FSTAR, ae_noise, analytic_snr
from binary_galactica.segments import CATALOGUE_FIELDS

PERFORMANCE = Path(__file__).resolve().parents[1] / "performance"
CHANNELS = ("A", "E", "T")
PERIODS = {2: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_report(name: str, report: dict) -> Path:
    """Write a JSON report under ``tools/performance``."""
    PERFORMANCE.mkdir(parents=True, exist_ok=True)
    path = PERFORMANCE / name
    path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    return path


def analytic_snrs(freq: np.ndarray, amp: np.ndarray, T: float) -> np.ndarray:
    """GLASS's analytic SNR estimate without confusion noise."""
    return analytic_snr(amp, ae_noise(freq), np.sin(freq / FSTAR), np.sqrt(T))


def catalogue_rows(path: Path, *, T: float, band=None, snr_min=None,
                   populations=("dgb", "igb")) -> np.ndarray:
    """Return dgb/igb rows in ``band`` (bins) above ``snr_min``, sorted by frequency."""
    dtype = [("Name", "U32")] + [(name, "f8") for name in CATALOGUE_FIELDS]
    parts = []
    for population in populations:
        cat = sangria.load_catalogue(population, ("Frequency", "Amplitude"), path)
        keep = np.ones(cat.shape[0], dtype=bool)
        q = cat["Frequency"] * T
        if band is not None:
            keep &= (q >= band[0]) & (q < band[1])
        if snr_min is not None:
            keep &= analytic_snrs(cat["Frequency"], cat["Amplitude"], T) >= snr_min
        rows = sangria.load_catalogue(population, ("Name", *CATALOGUE_FIELDS), path,
                                      index=np.nonzero(keep)[0])
        out = np.zeros(rows.shape[0], dtype=dtype)
        out["Name"] = [x.decode() if isinstance(x, bytes) else str(x) for x in rows["Name"]]
        for name in CATALOGUE_FIELDS:
            out[name] = rows[name]
        parts.append(out)
    rows = np.concatenate(parts)
    return rows[np.argsort(rows["Frequency"], kind="stable")]


def running_median_variance(spectra: dict, lo: int, hi: int, half_width: int = 500) -> dict:
    """Return each channel's coefficient variance on ``[lo, hi)``.

    The estimate is the running median of ``|r|^2`` divided by ``ln 2``, the
    median of an exponential variable with unit mean.
    """
    from scipy.ndimage import median_filter

    a, b = max(lo - half_width, 1), hi + half_width
    out = {}
    for ch, s in spectra.items():
        power = np.abs(s[a:b]) ** 2
        smooth = median_filter(power, size=2 * half_width + 1, mode="nearest")
        out[ch] = smooth[lo - a : hi - a] / np.log(2.0)
    return out


def frequency_data(spectra: dict, n: int, dt: float, channels=CHANNELS):
    """Return a frequency-domain enchilada L1Data of the given spectra."""
    import enchilada

    return enchilada.L1Data({ch: spectra[ch] for ch in channels}, sample_rate_hz=1.0 / dt,
                            tdi_generation="1.5", physical_observable="fractional_frequency",
                            channel_names=channels, num_time_samples=n,
                            data_domain="frequency")


def covariance(data, variances: dict, lo: int, hi: int):
    """Return a diagonal DataCovariance.

    It uses ``variances`` on ``[lo, hi)`` and ``(T/2) ae_noise`` elsewhere;
    DC is inactive.
    """
    import enchilada

    n_freq = data.num_time_samples // 2 + 1
    T = data.Tobs
    f = np.arange(n_freq) / T
    f[0] = f[1]
    base = 0.5 * T * ae_noise(f)
    m = np.zeros((n_freq, len(data.channel_names), len(data.channel_names)))
    for i, ch in enumerate(data.channel_names):
        diag = base.copy()
        diag[lo:hi] = variances[ch]
        m[:, i, i] = diag
    mask = np.ones(n_freq, dtype=bool)
    mask[0] = False
    return enchilada.DataCovariance(m, channel_names=data.channel_names,
                                    sample_rate_hz=data.fs, num_time_samples=data.N,
                                    data_domain="frequency", active_mask=mask)


def run_block(block, data, cov, turns: int, seed: int):
    """Run ``turns`` Gibbs turns of one block alone; return the state and seconds per turn."""
    rng = np.random.default_rng(seed)
    state = block.initial_block_state(data)
    seconds = []
    for _ in range(turns):
        start = time.perf_counter()
        state = block.sample(data, cov, state, rng=rng)
        seconds.append(time.perf_counter() - start)
    return state, seconds


def recovery(chain: dict, injections: np.ndarray, *, burn: float = 0.25, tol: float = 1.0):
    """Return, for each injection, how often a cold-chain source sits within ``tol`` bins.

    Also returns the histogram of the count after burn-in.
    """
    first = int(burn * chain["k"].size)
    samples, ks = chain["samples"][first:], chain["k"][first:]
    comps = samples.reshape(samples.shape[0], -1, 8)
    rows = np.arange(samples.shape[0])
    out = []
    for inj in np.atleast_2d(injections):
        d = np.abs(comps[..., 0] - inj[0])
        d = np.where(np.isnan(d), np.inf, d)
        j = np.argmin(d, axis=1)
        near = d[rows, j]
        hit = near < tol
        out.append({
            "q": float(inj[0]),
            "lnA": float(inj[3]),
            "recovered": float(hit.mean()) if hit.size else 0.0,
            "median_dq": float(np.median(near[hit])) if hit.any() else None,
            "median_dlnA": float(np.median(comps[rows, j, 3][hit] - inj[3])) if hit.any() else None,
        })
    return out, np.bincount(ks).tolist()
