"""Capture float64 golden outputs of the public waveform API.

Run from the repository root::

    PYTHONPATH=src python tools/float64_golden.py

The committed baseline ``tests/data/float64_golden.npz`` was captured with the
sources of commit 2171a87, which restored the physical native amplitude
(0.5.0; every native-API output doubled), and ``tests/test_float64_golden.py``
requires later float64 outputs to match it.
Every case is a fixed input definition and nothing is random, so rerunning the
script on the same JAX build, CPU and sources reproduces the archive byte for
byte. Regenerate it only for an intended float64 change, and say so in the
commit.

The cases cover the on-fly ``ucb_waveform`` in all three TDI formats with 7-,
8- and 9-entry native vectors (signed ``fdot``, nonzero ``fddot``), nonzero
epochs, explicit and automatic response grids, a ``q`` override and a legacy
``orbit_cache`` cache; one legacy Fourier-backend spectrum; ``template_on_window``;
``tdi_response`` at arbitrary times with and without positions; the physical
``tdi_waveform`` adapter; the source-only polarization helpers; and a forward
Jacobian of ``A``/``E`` with respect to the native parameters.

Double precision is enabled here explicitly, before the package is imported,
so the script does not rely on the package enabling it on import.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import subprocess
import zipfile
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import binary_galactica as bg  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests/data/float64_golden.npz"

QUARTER = 7_884_000.0
YEAR = 31_557_600.0
FOUR_YEARS = 126_230_400.0

# Native vectors [f0*T, sin(beta), longitude, log(A), cos(iota), psi, phi0,
# fdot*T**2, fddot*T**3] for 3.1 mHz over a year, 1 mHz over a quarter year,
# 10 mHz over four years, 27 mHz over a year and 4 mHz over a year. Carrier
# fractions stay away from 0 and 1/2; chirps take both signs.
SEVEN = [97_828.61, 0.27, 2.3, math.log(3e-22), -0.58, 1.9, 0.83]
EIGHT = [7_884.3, -0.21, 1.1, math.log(8e-22), 0.37, 0.63, 0.41, -0.8]
NINE = [1_262_304.43, 0.81, 5.2, math.log(2e-22), 0.34, 0.2, 2.9, 37.5, -1.7]
FAST = [852_055.37, -0.44, 3.8, math.log(5e-23), 0.91, 2.6, 5.1, -12.0, 0.9]
JACOBIAN = [126_230.65, 0.2, 2.3, math.log(1e-22), 0.05, 0.8, 0.4, -0.2, 2.0]
# Physical GBParams order: amp, f0, fdot, phi0, iota, psi, lam, beta.
PHYSICAL = [2.1e-22, 2.3e-3, -3.1e-17, 0.7, 1.1, 0.4, 4.1, -0.3]
SOURCE = [1e-22, 1.2345e-3, 3.21e-16, 0.7, 1.1, 0.4, 4.1, -0.3]
# Irregular times before, inside and beyond a four-year observation.
TIMES = [-3.2e6, 0.0, 1.7e5, 4.4e7, 9.9e7, 1.3e8, 2.9e8]

CASES: list[dict[str, Any]] = [
    # One quarter-year source in every format, with a negative chirp, a nonzero
    # epoch and an explicit sparse response grid; then the same source with a
    # legacy position cache in place of direct analytic geometry.
    *(
        {
            "name": f"ucb_8p_{fmt}",
            "function": "ucb_waveform",
            "params": EIGHT,
            "T": QUARTER,
            "BW": 64,
            "t0": 12_345.0,
            "fmt": fmt,
            "response_samples": 48,
        }
        for fmt in ("sangria", "frequency", "phase")
    ),
    {
        "name": "ucb_8p_sangria_cache",
        "function": "ucb_waveform",
        "params": EIGHT,
        "T": QUARTER,
        "BW": 64,
        "t0": 12_345.0,
        "fmt": "sangria",
        "response_samples": 48,
        "positions": "orbit_cache",
    },
    # Automatic response grids: 128 samples for one year, 512 for four.
    {
        "name": "ucb_7p_sangria_auto",
        "function": "ucb_waveform",
        "params": SEVEN,
        "T": YEAR,
        "BW": 128,
        "t0": 0.0,
        "fmt": "sangria",
        "response_samples": None,
    },
    {
        "name": "ucb_9p_frequency_auto",
        "function": "ucb_waveform",
        "params": NINE,
        "T": FOUR_YEARS,
        "BW": 256,
        "t0": 22_090_320.0,
        "fmt": "frequency",
        "response_samples": None,
    },
    # A 27 mHz source observed from 2.1 years, with its carrier pinned one bin
    # below GLASS's default.
    {
        "name": "ucb_9p_phase_q",
        "function": "ucb_waveform",
        "params": FAST,
        "T": YEAR,
        "BW": 256,
        "t0": 66_270_960.0,
        "fmt": "phase",
        "response_samples": 256,
        "q": 852_054.0,
    },
    # The legacy single-link path shares orbit, polarization and AE kernels.
    {
        "name": "fourier_9p_phase",
        "function": "ucb_waveform",
        "params": NINE,
        "T": FOUR_YEARS,
        "BW": 256,
        "t0": 22_090_320.0,
        "fmt": "phase",
        "backend": "fourier",
    },
    # The window starts 20 bins below the carrier and runs past the block.
    {
        "name": "template_on_window",
        "function": "template_on_window",
        "params": EIGHT,
        "T": QUARTER,
        "BW": 64,
        "t0": 12_345.0,
        "fmt": "sangria",
        "response_samples": None,
        "bin_lo": 7_864,
        "n_bins": 80,
    },
    {
        "name": "tdi_response_7p",
        "function": "tdi_response",
        "params": SEVEN,
        "T": YEAR,
        "times": TIMES,
    },
    {
        "name": "tdi_response_9p",
        "function": "tdi_response",
        "params": NINE,
        "T": FOUR_YEARS,
        "times": TIMES,
    },
    # Explicit geometry distinct from the default: a rotated analytic orbit.
    {
        "name": "tdi_response_9p_positions",
        "function": "tdi_response",
        "params": NINE,
        "T": FOUR_YEARS,
        "times": TIMES,
        "positions": "analytic_orbits",
        "kappa_0": 0.3,
    },
    {
        "name": "tdi_waveform",
        "function": "tdi_waveform",
        "theta": PHYSICAL,
        "T": YEAR,
        "BW": 128,
        "t0": 9_467_280.0,
        "fmt": "sangria",
        "response_samples": None,
    },
    {
        "name": "source_polarizations",
        "function": "polarizations",
        "theta": SOURCE,
        "times": [0.0, 1.0e3, 86_400.0, 3.1e7, 1.26e8],
    },
    {
        "name": "jacobian_9p",
        "function": "jacfwd",
        "params": JACOBIAN,
        "T": YEAR,
        "BW": 64,
        "t0": 12_623_040.0,
        "fmt": "sangria",
        "response_samples": None,
    },
]

WAVEFORM_OPTIONS = ("T", "BW", "t0", "fmt", "response_samples", "q", "backend")


def _options(case: dict[str, Any]) -> dict[str, Any]:
    """Return the keyword arguments a case passes to the waveform API."""
    return {name: case[name] for name in WAVEFORM_OPTIONS if name in case}


def _ucb_waveform(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a native spectrum, first building its position cache if any."""
    options, outputs = _options(case), {}
    if case.get("positions") == "orbit_cache":
        positions = bg.orbit_cache(case["T"], case["t0"], case["BW"])
        options["positions"] = outputs["positions"] = positions
    outputs.update(bg.ucb_waveform(np.asarray(case["params"]), **options))
    return outputs


def _template_on_window(case: dict[str, Any]) -> dict[str, Any]:
    """Place continuous-Fourier A and E on a fixed analysis window."""
    window = bg.template_on_window(
        np.asarray(case["params"]),
        bin_lo=case["bin_lo"],
        n_bins=case["n_bins"],
        **_options(case),
    )
    return {"window": window}


def _tdi_response(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate demodulated XYZ at arbitrary times, optionally with geometry."""
    times, outputs = np.asarray(case["times"]), {}
    positions = None
    if case.get("positions") == "analytic_orbits":
        positions = outputs["positions"] = bg.analytic_orbits(
            times, kappa_0=case["kappa_0"]
        )
    outputs["xyz"] = bg.tdi_response(
        np.asarray(case["params"]), times, T=case["T"], positions=positions
    )
    return outputs


def _tdi_waveform(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the physical adapter and the native vector it converts to."""
    theta = np.asarray(case["theta"])
    return {
        "native": bg.to_glass_vector(theta, case["T"]),
        **bg.tdi_waveform(theta, **_options(case)),
    }


def _polarizations(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the source-only phase, frequency, amplitude and strain helpers."""
    theta, times = jnp.asarray(case["theta"]), jnp.asarray(case["times"])
    amp_plus, amp_cross = bg.amplitudes(theta)
    hp, hc = bg.polarizations(theta, times)
    return {
        "phase": bg.phase(theta, times),
        "frequency": bg.frequency(theta, times),
        "amp_plus": amp_plus,
        "amp_cross": amp_cross,
        "hp": hp,
        "hc": hc,
    }


def _jacfwd(case: dict[str, Any]) -> dict[str, Any]:
    """Differentiate stacked complex A and E, shape ``(2, BW, len(params))``."""
    options = _options(case)

    def channels(params):
        waveform = bg.ucb_waveform(params, **options)
        return jnp.stack((waveform["A"], waveform["E"]))

    return {"jacobian": jax.jacfwd(channels)(jnp.asarray(case["params"]))}


EVALUATORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "ucb_waveform": _ucb_waveform,
    "template_on_window": _template_on_window,
    "tdi_response": _tdi_response,
    "tdi_waveform": _tdi_waveform,
    "polarizations": _polarizations,
    "jacfwd": _jacfwd,
}


def evaluate(case: dict[str, Any]) -> dict[str, np.ndarray]:
    """Evaluate one case through the public API and return host arrays."""
    outputs = EVALUATORS[case["function"]](case)
    return {name: np.asarray(value) for name, value in outputs.items()}


def environment() -> dict[str, Any]:
    """Describe the build that fixes the outputs' exact bits."""
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
    }


def sources() -> dict[str, Any]:
    """Identify the evaluated package sources without volatile run details."""
    package = Path(bg.__file__).resolve().parent

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(package), *arguments],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tree = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        tree.update(path.relative_to(package).as_posix().encode())
        tree.update(path.read_bytes())
    status = git("status", "--porcelain", "--", ".")
    return {
        "package_version": bg.__version__,
        "source_tree_sha256": tree.hexdigest(),
        # The last commit touching the package, rather than HEAD, so that
        # documentation-only commits leave the archive unchanged.
        "source_commit": git("log", "-1", "--format=%H", "--", "."),
        "source_modified": None if status is None else bool(status),
    }


def save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an ``np.load``-compatible archive with fixed member timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def generate(output: Path) -> None:
    """Evaluate every case and save its outputs with the input definitions."""
    arrays, cases = {}, []
    for case in CASES:
        outputs = evaluate(case)
        described = {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in outputs.items()
        }
        cases.append({**case, "outputs": described})
        arrays.update({f"{case['name']}/{name}": v for name, v in outputs.items()})
    record = {
        "description": "Float64 public-API outputs; see tools/float64_golden.py.",
        "environment": environment(),
        "sources": sources(),
        "cases": cases,
    }
    arrays["metadata"] = np.array(json.dumps(record, indent=1, sort_keys=True))
    save(output, arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate(parser.parse_args().output)
