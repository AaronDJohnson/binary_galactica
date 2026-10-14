"""Double precision is a correctness requirement, so it is tested like one."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import binary_galactica
from binary_galactica import YEAR, phase

#: pi to 50 significant figures, for a reference phase computed outside
#: floating point entirely.
PI_50 = Decimal("3.1415926535897932384626433832795028841971693993751")

#: Every guarded float64 entry point, called with valid arguments unless the
#: name says otherwise. Each must refuse to run while x64 is disabled.
GUARDED_CALLS = {
    "ucb_waveform": "bg.ucb_waveform(native, T=T, BW=BW)",
    "ucb_waveform, fourier backend": (
        'bg.ucb_waveform(native, T=T, BW=BW, backend="fourier")'
    ),
    "ucb_waveform, invalid arguments": "bg.ucb_waveform(native, T=-T, BW=3)",
    "glass.fourier.ucb_waveform": "fourier.ucb_waveform(native, T=T, BW=BW)",
    "tdi_waveform": "bg.tdi_waveform(physical, T=T, BW=BW)",
    "tdi_response": "bg.tdi_response(native, times, T=T)",
    "template_on_window": (
        "bg.template_on_window(native, T=T, bin_lo=7870, n_bins=16, BW=BW)"
    ),
    "orbit_cache": "bg.orbit_cache(T, 0.0, BW)",
    "phase": "bg.phase(physical, times)",
    "frequency": "bg.frequency(physical, times)",
    "amplitudes": "bg.amplitudes(physical)",
    "polarizations": "bg.polarizations(physical, times)",
}

#: Prepared in the child process before the guarded calls are attempted.
GUARD_PRELUDE = """
import json

import jax
import numpy as np

import binary_galactica as bg
import binary_galactica.glass.fourier as fourier

T, BW = 7_884_000.0, 32
native = bg.params_from_native(
    f0=1e-3, fdot=1e-18, amp=7e-22, phi0=0.5, colat=1.78, lon=0.0, psi=0.7,
    cos_iota=0.3, T=T,
)
physical = np.array([1e-22, 1e-3, 1e-18, 0.5, 1.1, 0.4, 4.1, -0.3])
times = np.linspace(0.0, T, 5)
results = {}


def attempt(name, call):
    try:
        call()
    except Exception as error:  # report every outcome to the parent
        results[name] = [type(error).__name__, str(error)]
    else:
        results[name] = ["returned", ""]
"""


def run_without_x64(code: str) -> str:
    """Run ``code`` in a fresh interpreter with JAX's default x64 setting.

    The child imports the same ``binary_galactica`` as this process, whether
    that is a source checkout or an installed wheel. Its stdout is returned.
    """
    environment = dict(os.environ)
    environment.pop("JAX_ENABLE_X64", None)
    package_parent = str(Path(binary_galactica.__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (package_parent, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def exact_phase(theta, time: float) -> Decimal:
    """Phi(t) in 50-digit decimal arithmetic."""
    getcontext().prec = 50
    t = Decimal(repr(float(time)))
    phi0 = Decimal(repr(float(theta.phi0)))
    f0 = Decimal(repr(float(theta.f0)))
    fdot = Decimal(repr(float(theta.fdot)))
    return phi0 + 2 * PI_50 * f0 * t + PI_50 * fdot * t * t


def test_importing_the_package_leaves_x64_disabled():
    """No module of the package changes JAX's process-global precision."""
    code = """
import importlib
import pkgutil

import jax

assert not jax.config.read("jax_enable_x64"), "x64 enabled before import"
import binary_galactica

for module in pkgutil.walk_packages(binary_galactica.__path__, "binary_galactica."):
    importlib.import_module(module.name)
print(jax.config.read("jax_enable_x64"))
"""
    assert run_without_x64(code).strip() == "False"


def test_float64_entry_points_require_x64():
    """Each float64 entry point raises before computing anything in float32.

    Every guarded call is attempted in one fresh process with x64 disabled; a
    failure lists each call that did not refuse.
    """
    calls = "".join(
        f"attempt({name!r}, lambda: {call})\n" for name, call in GUARDED_CALLS.items()
    )
    report = 'print(json.dumps([results, jax.config.read("jax_enable_x64")]))\n'
    outcomes, x64_after = json.loads(run_without_x64(GUARD_PRELUDE + calls + report))
    assert x64_after is False
    assert set(outcomes) == set(GUARDED_CALLS)
    refused = {
        name: kind == "RuntimeError"
        and 'jax.config.update("jax_enable_x64", True)' in message
        and "binary_galactica.single" in message
        for name, (kind, message) in outcomes.items()
    }
    assert {name: outcomes[name] for name, ok in refused.items() if not ok} == {}


def test_phase_is_accurate_over_a_four_year_observation(params):
    """Absolute phase error stays far below the ~1e-3 rad that matters.

    Over four years the phase reaches ~1e6 rad.  Double precision leaves
    ~1e-9 rad of rounding; single precision would leave ~0.06 rad, which is
    enough to destroy a matched filter while remaining invisible in a plot.
    """
    for fraction in (0.25, 0.5, 1.0):
        time = 4.0 * YEAR * fraction
        computed = Decimal(repr(float(phase(params, time))))
        error = abs(computed - exact_phase(params, time))
        assert error < Decimal("1e-6"), f"phase error {error} rad at t={time}"
