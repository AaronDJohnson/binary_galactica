"""Conventions gate: catalogue-driven templates against noise-free Sangria streams.

For each verification binary (``sky/vgb``) and a few bright isolated ``dgb`` and
``igb`` sources, convert the catalogue row with
:func:`binary_galactica.segments.ldc_to_native`, render A/E with both backends
on the 12-month grid, and compare with ``dt * rfft`` of the population's own
noise-free stream. Report the 1/ae_noise-weighted match and complex scale on a
``2*BW`` window around the carrier. No taper is applied.

Two generator bandwidths are measured per source:

- ``GATE_MIN_BW`` (``BW = max(512, ucb_bw)``) measures the conventions with the
  band-truncation of the template made negligible. The per-backend entries
  ``entry[backend] = {match, scale, bw, bin_lo}`` and the neighbour counts use
  this BW and window; the acceptance test reads them.
- ``entry["glass_bw"][backend]`` is the same measurement with GLASS's default
  ``ucb_bw`` (informational: how much the BW heuristic truncates).

``bright_isolated`` only considers sources below ``MAX_BRIGHT_HZ``; the catalogue
has one dgb outlier at 43 mHz that the stream does not carry. The report lists
such skipped sources under ``"skipped_above_cap"``.

    PYTHONPATH=src python tools/segments/conventions_gate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sangria  # enables x64 before the package is used

from binary_galactica.glass import (
    FSTAR,
    ae_noise,
    analytic_snr,
    template_on_window,
    ucb_bw,
)
from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native

BACKENDS = ("fourier", "onfly")
#: Floor on the generator BW of the gate measurement (half-window in bins).
GATE_MIN_BW = 512
#: Isolated dgb/igb candidates must lie below this frequency [Hz].
MAX_BRIGHT_HZ = 0.030
OUT = Path(__file__).resolve().parents[1] / "performance" / "segments-conventions.json"


def window(
    native: np.ndarray, T: float, n_freq: int, min_bw: int = GATE_MIN_BW
) -> tuple[int, int, int]:
    """Return ``(bin_lo, n_bins, BW)`` of the ``2*BW`` window around a source.

    ``BW = max(min_bw, ucb_bw)``; ``min_bw=0`` gives GLASS's own choice.
    """
    bw = ucb_bw(
        native[0] / T, native[7] / T**2, native[1], float(np.exp(native[3])), T, n_freq
    )
    bw = max(int(bw), int(min_bw))
    q = int(np.floor(native[0]))
    return q - bw, 2 * bw, bw


def match_source(native, spectra, T, n_freq, backend, min_bw=GATE_MIN_BW):
    """Return the noise-weighted match and least-squares complex scale."""
    bin_lo, n_bins, bw = window(native, T, n_freq, min_bw)
    h = np.asarray(
        template_on_window(
            native, T=T, bin_lo=bin_lo, n_bins=n_bins, BW=bw, backend=backend
        )
    )
    sl = slice(bin_lo, bin_lo + n_bins)
    d = np.stack([spectra["A"][sl], spectra["E"][sl]])
    w = 1.0 / ae_noise((bin_lo + np.arange(n_bins)) / T)
    hd = np.sum(w * np.conj(h) * d)
    hh = np.sum(w * np.abs(h) ** 2)
    dd = np.sum(w * np.abs(d) ** 2)
    scale = hd / hh
    return {
        "match": float(hd.real / np.sqrt(hh * dd)),
        "scale": [float(scale.real), float(scale.imag)],
        "bw": int(bw),
        "bin_lo": int(bin_lo),
    }


def bright_isolated(population, T, n_freq, count, path):
    """Return ``(indices, skipped)``: the brightest sources with no rival above
    5% SNR and ``f < MAX_BRIGHT_HZ``, and ``(name, f)`` of catalogue sources
    above the cap that outrank the picks (the stream carries no power there).
    """
    fields = ("Frequency", "FrequencyDerivative", "Amplitude", "EclipticLatitude")
    cat = sangria.load_catalogue(population, fields, path)
    f, a = cat["Frequency"], cat["Amplitude"]
    snr = analytic_snr(a, ae_noise(f), np.sin(f / FSTAR), np.sqrt(T))
    order = np.argsort(f)
    f_sorted, snr_sorted = f[order], snr[order]
    picked: list[int] = []
    high = np.flatnonzero(f >= MAX_BRIGHT_HZ)
    snr_ranked = np.where(f < MAX_BRIGHT_HZ, snr, -np.inf)
    for i in np.argsort(snr_ranked)[::-1][:2000]:
        bw = max(
            GATE_MIN_BW,
            ucb_bw(
                f[i],
                cat["FrequencyDerivative"][i],
                np.sin(cat["EclipticLatitude"][i]),
                a[i],
                T,
                n_freq,
            ),
        )
        lo, hi = np.searchsorted(f_sorted, [f[i] - bw / T, f[i] + bw / T])
        rival = np.sort(snr_sorted[lo:hi])[-2] if hi - lo > 1 else 0.0
        if rival < 0.05 * snr[i]:
            picked.append(int(i))
        if len(picked) == count:
            break
    skipped = []
    if high.size:
        names = sangria.load_catalogue(population, ("Name",), path, index=high)["Name"]
        skipped = [
            (int(n), float(f[i]))
            for n, i in zip(np.ravel(names), sorted(high.tolist()))
        ]
    return sorted(picked), skipped


def neighbours(freqs: np.ndarray, j: int, T: float, half_bins: int) -> int:
    """Return how many other sources of the list sit within ``half_bins``."""
    return int(np.sum(np.abs(freqs - freqs[j]) * T < half_bins) - 1)


def run(path: Path, count: int) -> dict:
    """Return the gate report for every VGB and ``count`` dgb/igb sources."""
    n, dt = sangria.grid(path)
    T, n_freq = n * dt, n // 2 + 1
    report: dict = {
        "T": T,
        "dt": dt,
        "gate_min_bw": GATE_MIN_BW,
        "max_bright_hz": MAX_BRIGHT_HZ,
        "populations": {},
        "skipped_above_cap": {},
    }
    for population in ("vgb", "dgb", "igb"):
        spectra = sangria.aet_spectra(*sangria.load_xyz(population, path))
        index = None
        if population != "vgb":
            index, skipped = bright_isolated(population, T, n_freq, count, path)
            report["skipped_above_cap"][population] = skipped
        rows = sangria.load_catalogue(
            population, ("Name", *CATALOGUE_FIELDS), path, index=index
        )
        native = ldc_to_native(rows, T)
        freqs = np.asarray(rows["Frequency"], dtype=np.float64)
        entries = []
        for j, row in enumerate(rows):
            _, _, bw = window(native[j], T, n_freq)
            name = row["Name"]
            entry = {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "frequency_hz": float(freqs[j]),
                "fdot": float(row["FrequencyDerivative"]),
                "neighbours_in_window": neighbours(freqs, j, T, bw)
                if population == "vgb"
                else 0,
            }
            for backend in BACKENDS:
                entry[backend] = match_source(native[j], spectra, T, n_freq, backend)
            entry["glass_bw"] = {
                backend: match_source(native[j], spectra, T, n_freq, backend, 0)
                for backend in BACKENDS
            }
            entries.append(entry)
        report["populations"][population] = entries
    isolated = [
        e
        for entries in report["populations"].values()
        for e in entries
        if e["neighbours_in_window"] == 0
    ]
    report["worst_match"] = {b: min(e[b]["match"] for e in isolated) for b in BACKENDS}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(args.path, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for population, entries in report["populations"].items():
        for e in entries:
            cells = "  ".join(
                f"{b}: m={e[b]['match']:.6f} "
                f"s={e[b]['scale'][0]:+.5f}{e[b]['scale'][1]:+.5f}i"
                for b in BACKENDS
            )
            g = e["glass_bw"]
            cells += "  | glass bw=%d: " % g["fourier"]["bw"] + "  ".join(
                f"{b}: m={g[b]['match']:.6f} "
                f"s={g[b]['scale'][0]:+.5f}{g[b]['scale'][1]:+.5f}i"
                for b in BACKENDS
            )
            print(
                f"{population:4s} {e['name']:>16s} {e['frequency_hz'] * 1e3:8.4f} mHz"
                f"  nb={e['neighbours_in_window']}  {cells}"
            )
    for population, skipped in report["skipped_above_cap"].items():
        if skipped:
            listing = ", ".join(f"{n} ({f * 1e3:.2f} mHz)" for n, f in skipped)
            print(
                f"note: skipped {population} sources above {MAX_BRIGHT_HZ * 1e3:.0f} mHz "
                f"(stream carries no power there): {listing}"
            )
    print("worst isolated match:", report["worst_match"])


if __name__ == "__main__":
    main()
