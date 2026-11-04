"""Shared helpers for the segment validation tools (application side)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sangria  # enables x64 before the package is used

from binary_galactica.glass import FSTAR, ae_noise, analytic_snr
from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native, render_sources

PERFORMANCE = Path(__file__).resolve().parents[1] / "performance"
CHANNELS = ("A", "E", "T")
HALF_WIDTH = 500  # running_median_variance's half_width
START_SNR = 7.0  # analytic SNR floor of the start set
NOISE_REFERENCE = (
    "running median of obs-mbhb-vgb minus catalogue sources with analytic SNR >= 7"
)
MARGIN = 300  # bins either side of a band kept for the running-median window


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


def catalogue_rows(
    path: Path, *, T: float, band=None, snr_min=None, populations=("dgb", "igb")
) -> np.ndarray:
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
        rows = sangria.load_catalogue(
            population, ("Name", *CATALOGUE_FIELDS), path, index=np.nonzero(keep)[0]
        )
        out = np.zeros(rows.shape[0], dtype=dtype)
        out["Name"] = [
            x.decode() if isinstance(x, bytes) else str(x) for x in rows["Name"]
        ]
        for name in CATALOGUE_FIELDS:
            out[name] = rows[name]
        parts.append(out)
    rows = np.concatenate(parts)
    return rows[np.argsort(rows["Frequency"], kind="stable")]


def target_band(path, *, T, core, span, snr_min=START_SNR, extra=None):
    """Return ``(band, rows)`` for a band of ``span`` bins either side of ``core``.

    ``rows`` holds every dgb/igb catalogue row in ``band`` with analytic SNR >= ``snr_min``,
    plus the rows of ``extra`` (merged by Name, no duplicates), sorted by frequency.
    """
    band = (int(core[0]) - int(span), int(core[1]) + int(span))
    rows = catalogue_rows(path, T=T, band=band, snr_min=snr_min)
    if extra is not None and len(extra):
        new = extra[~np.isin(extra["Name"], rows["Name"])]
        rows = np.concatenate([rows, new.astype(rows.dtype)])
        rows = rows[np.argsort(rows["Frequency"], kind="stable")]
    return band, rows


def inner_segments(block, core):
    """Return the block's segments except the first and last.

    The edge segments' outer padding holds catalogue sources outside the band that no
    segment models, so recovery there is not judged. Raises ValueError if ``core``
    overlaps the first or last segment's interior (widen ``span``).
    """
    segs = block.table.segments
    if len(segs) < 3:
        raise ValueError("fewer than three segments: widen span")
    for edge in (segs[0], segs[-1]):
        if core[0] < edge.hi and core[1] > edge.lo:
            raise ValueError(
                f"core {tuple(core)} overlaps edge segment {edge.index}; widen span"
            )
    return list(segs[1:-1])


def running_median_variance(
    spectra: dict, lo: int, hi: int, half_width: int = HALF_WIDTH
) -> dict:
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


def residual_spectra(
    spectra: dict,
    path,
    *,
    T: float,
    lo: int,
    hi: int,
    snr_min: float | None = START_SNR,
    BW: int = 512,
) -> tuple[dict, np.ndarray]:
    """Return ``spectra`` minus every dgb/igb catalogue source with analytic SNR >= ``snr_min``
    whose bin lies in ``[lo, hi)``, plus the subtracted rows.

    Sources are rendered with ``render_sources`` (on-fly backend, ``fmt="sangria"``, every
    channel in ``spectra``) at ``ldc_to_native`` parameters, i.e. the true injected waveforms,
    in the same ``dt * rfft`` units as ``spectra``. Callers pass ``[lo, hi)`` wide enough to
    cover the running-median window (band +- MARGIN +- half_width), so the median never sees an
    unsubtracted start-rule source. ``spectra`` is not modified.
    """
    channels = tuple(spectra)
    if channels not in (("A", "E"), ("A", "E", "T")):
        raise ValueError(
            f"spectra channels must be ('A', 'E') or ('A', 'E', 'T'), got {channels}"
        )
    n_freq = len(spectra["A"])
    rows = catalogue_rows(path, T=T, band=(lo, hi), snr_min=snr_min)
    model = np.zeros((len(channels), n_freq), dtype=np.complex128)
    if len(rows):
        contrib = render_sources(
            ldc_to_native(rows, T),
            T=T,
            t0=0.0,
            BW=BW,
            n_channels=len(channels),
            n_freq=n_freq,
        )
        contrib.add_to(model)
    row = {ch: i for i, ch in enumerate(channels)}
    resid = {ch: np.array(spectra[ch]) - model[row[ch]] for ch in channels}
    return resid, rows


def frequency_data(spectra: dict, n: int, dt: float, channels=CHANNELS):
    """Return a frequency-domain enchilada L1Data of the given spectra."""
    import enchilada

    return enchilada.L1Data(
        {ch: spectra[ch] for ch in channels},
        sample_rate_hz=1.0 / dt,
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        channel_names=channels,
        num_time_samples=n,
        data_domain="frequency",
    )


def instrument_variance(f, T: float):
    """Return the instrument-noise variance of a Sangria-format ``dt * rfft`` A/E coefficient.

    ``ae_noise`` is GLASS's phase-format A/E PSD. Sangria-format templates and data carry
    ``P = 4 x sin(x) sqrt(T)`` against phase's ``2 sin(x) sqrt(T)`` (``glass/tdi.py``, the
    table of ``P`` per format), so in power the factor is ``4 x^2`` with ``x = f / FSTAR``:
    ``0.5 * T * ae_noise(f) * 4 * (f / FSTAR)**2``.
    """
    f = np.asarray(f, dtype=np.float64)
    return 0.5 * T * ae_noise(f) * 4.0 * (f / FSTAR) ** 2


def covariance(data, variances: dict, lo: int, hi: int):
    """Return a diagonal DataCovariance.

    It uses ``variances`` on ``[lo, hi)`` and the instrument-only
    :func:`instrument_variance` elsewhere; DC is inactive. Callers must pass ``lo`` and
    ``hi`` covering the block's whole padded band (``segments[0].bin_lo`` to
    ``segments[-1].bin_hi``), or the block sees instrument-only noise there.
    """
    import enchilada

    n_freq = data.num_time_samples // 2 + 1
    T = data.Tobs
    f = np.arange(n_freq) / T
    f[0] = f[1]
    base = instrument_variance(f, T)
    m = np.zeros((n_freq, len(data.channel_names), len(data.channel_names)))
    for i, ch in enumerate(data.channel_names):
        diag = base.copy()
        diag[lo:hi] = variances[ch]
        m[:, i, i] = diag
    mask = np.ones(n_freq, dtype=bool)
    mask[0] = False
    return enchilada.DataCovariance(
        m,
        channel_names=data.channel_names,
        sample_rate_hz=data.fs,
        num_time_samples=data.N,
        data_domain="frequency",
        active_mask=mask,
    )


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


def recovery(
    chain: dict, injections: np.ndarray, *, burn: float = 0.25, tol: float = 1.0
):
    """Return, for each injection, how often a cold-chain source sits within ``tol`` bins.

    Matching is one-to-one within each sample: injection-component pairs are
    taken greedily by increasing distance, so two injections within ``tol`` of
    one component cannot both claim it. Also returns the histogram of the count
    after burn-in.
    """
    first = int(burn * chain["k"].size)
    samples, ks = chain["samples"][first:], chain["k"][first:]
    comps = samples.reshape(samples.shape[0], -1, 8)
    inj = np.atleast_2d(injections)
    n_s, n_i = comps.shape[0], len(inj)
    near = np.full((n_s, n_i), np.inf)
    match = np.zeros((n_s, n_i), dtype=int)
    for s in range(n_s):
        d = np.abs(comps[s, :, 0][None, :] - inj[:, 0][:, None])  # (n_i, n_comp)
        d = np.where(np.isnan(d), np.inf, d)
        for _ in range(min(n_i, d.shape[1])):
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if not np.isfinite(d[i, j]):
                break
            near[s, i], match[s, i] = d[i, j], j
            d[i, :] = np.inf
            d[:, j] = np.inf
    out = []
    rows = np.arange(n_s)
    for i, row in enumerate(inj):
        hit = near[:, i] < tol
        out.append(
            {
                "q": float(row[0]),
                "lnA": float(row[3]),
                "recovered": float(hit.mean()) if hit.size else 0.0,
                "median_dq": float(np.median(near[hit, i])) if hit.any() else None,
                "median_dlnA": float(
                    np.median(comps[rows, match[:, i], 3][hit] - row[3])
                )
                if hit.any()
                else None,
            }
        )
    return out, np.bincount(ks).tolist()
