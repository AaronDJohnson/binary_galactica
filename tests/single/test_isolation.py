"""The float32 path in a process without x64, and without double precision in it.

A fresh interpreter with x64 disabled runs every ``single`` function and
saves the outputs; this process, where the suite enables x64, repeats the same
calls and must compute the same bits. In this process, traced ``single``
functions must hold no float64 or complex128 values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from tests.conftest import double_precision_values
from tests.single.support import physical, source
from tests.test_precision import run_without_x64

#: One year at 10 mHz, ten years late, with both frequency derivatives.
CASE = source(1.0, 1e-2, 10.0, (1, -1))
TIMES = np.linspace(-1e6, 6.2e8, 65).astype(np.float32)

#: Every float32 function, as run in both processes. ``data`` holds the
#: float64 inputs; the splits and calls define ``outputs``. The waveforms take
#: eight entries, because a nine-entry waveform takes seconds to compile in a
#: fresh process; the nine-entry phase polynomial, whose ``fddot/6`` term uses
#: an import-time float32 pair like the orbital rate, is compiled on its own.
CALLS = """
import jax

from binary_galactica.glass import _twofloat, single as kernels

duration = float(data["duration"])
observation = {
    "T": np.float32(duration),
    "t0": np.float32(data["epoch"]),
    "BW": int(data["BW"]),
    "response_samples": int(data["samples"]),
}
q_ref, params = bg32.split_native(data["native"], duration)
q_physical, theta = bg32.split_physical(data["theta"], duration)
outputs = {
    "split_native": {"q_ref": np.int32(q_ref), "params": params},
    "split_physical": {"q_ref": np.int32(q_physical), "theta": theta},
    "ucb_waveform": bg32.ucb_waveform(params[:8], q_ref=q_ref, **observation),
    "grid_cycles": jax.jit(kernels.grid_cycles, static_argnums=4)(
        _twofloat.pair(params[0]),
        params,
        observation["T"],
        observation["t0"],
        observation["BW"],
    ),
    "template_on_window": bg32.template_on_window(
        params[:8],
        q_ref=q_ref,
        bin_lo=int(data["bin_lo"]),
        n_bins=int(data["n_bins"]),
        **observation,
    ),
    "tdi_response": bg32.tdi_response(
        params, data["times"], q_ref=q_ref, T=observation["T"]
    ),
    "tdi_waveform": bg32.tdi_waveform(theta, q_ref=q_physical, **observation),
}
"""

#: The x64-free child: run ``CALLS`` on the inputs in ``directory``, save every
#: output there and report types, dtypes and the x64 flag.
CHILD = (
    """
import json

import jax
import numpy as np

from binary_galactica import single as bg32

flags = {"after import": jax.config.read("jax_enable_x64")}
data = np.load(directory + "/inputs.npz")
"""
    + CALLS
    + """
flags["at exit"] = jax.config.read("jax_enable_x64")
report = {
    "flags": flags,
    "split": [
        type(q_ref).__name__,
        type(q_physical).__name__,
        str(params.dtype),
        str(theta.dtype),
    ],
    "outputs": {},
}
saved = {}
for name, result in outputs.items():
    values = result if isinstance(result, dict) else {"value": result}
    report["outputs"][name] = {
        key: [str(np.asarray(v).dtype), bool(np.all(np.isfinite(v)))]
        for key, v in values.items()
    }
    saved.update({name + "/" + k: np.asarray(v) for k, v in values.items()})
np.savez(directory + "/single.npz", **saved)
print(json.dumps(report))
"""
)


def write_inputs(directory: Path) -> None:
    """Write ``CASE`` as float64 native and physical inputs for ``CALLS``."""
    _, double_theta = physical(CASE)
    np.savez(
        directory / "inputs.npz",
        native=CASE.native,
        theta=double_theta,
        duration=float(CASE.duration),
        epoch=float(CASE.epoch),
        BW=CASE.bandwidth,
        samples=CASE.samples,
        bin_lo=CASE.carrier - 100,
        n_bins=200,
        times=TIMES,
    )


def test_entry_points_without_x64(tmp_path):
    """A fresh process with x64 disabled computes the same bits as this one.

    An import-time float32 pair such as the orbital rate ``1/YEAR`` would lose
    its low part if it were built with ``jnp`` while x64 is off; only a
    comparison across the two settings shows that.
    """
    write_inputs(tmp_path)
    report = json.loads(run_without_x64(f"directory = {str(tmp_path)!r}\n" + CHILD))
    assert report["flags"] == {"after import": False, "at exit": False}
    assert report["split"] == ["int", "int", "float32", "float32"]
    for name, values in report["outputs"].items():
        for key, (dtype, finite) in values.items():
            if name.startswith("split") or name == "grid_cycles":
                expected = "int32" if key == "q_ref" else "float32"
            elif key in ("qmin", "freq_bin_indices"):
                expected = "int32"
            else:
                expected = "complex64"
            assert (dtype, finite) == (expected, True), (name, key)

    namespace: dict[str, Any] = {"bg32": bg32, "np": np}
    namespace["data"] = np.load(tmp_path / "inputs.npz")
    exec(CALLS, namespace)
    saved = np.load(tmp_path / "single.npz")
    compared = set()
    for name, result in namespace["outputs"].items():
        values = result if isinstance(result, dict) else {"value": result}
        for key, value in values.items():
            child = saved[f"{name}/{key}"]
            assert child.dtype == np.asarray(value).dtype, (name, key)
            np.testing.assert_array_equal(
                child, np.asarray(value), err_msg=f"{name}/{key}"
            )
            compared.add(f"{name}/{key}")
    assert compared == set(saved.files)


CACHE = np.asarray(
    bg.orbit_cache(float(CASE.duration), float(CASE.epoch), CASE.bandwidth)
).astype(np.float32)
THETA32, _ = physical(CASE)
STATIC: dict[str, Any] = {"BW": CASE.bandwidth, "response_samples": CASE.samples}


def waveform(fmt="sangria", length=9, **options):
    """Return a traceable float32 waveform of every array argument."""

    def evaluate(params, q_ref, duration, epoch, *extra):
        keywords = dict(zip(options, extra, strict=True))
        return bg32.ucb_waveform(
            params[:length],
            q_ref=q_ref,
            T=duration,
            t0=epoch,
            fmt=fmt,
            **STATIC,
            **keywords,
        )

    return evaluate, tuple(options.values())


def automatic_grid(params, q_ref, duration, epoch):
    """Trace the automatic grid's switch, which a traced duration keeps."""
    with pytest.warns(bg.ResponseSamplesWarning):
        return bg32.ucb_waveform(
            params, q_ref=q_ref, T=duration, t0=epoch, BW=CASE.bandwidth
        )


#: Each float32 function as ``(function, extra traced arguments)``; the
#: first four arguments are always ``params, q_ref, T, t0``. Each entry takes
#: a distinct traced path: the three formats, the 7-, 8- and 9-entry phase
#: polynomials, a cache, an override, the automatic grid's switch, the window
#: scatter, the response on the analytic orbit and on given positions, and the
#: physical adapter from an array and from ``GBParams`` with a cache.
TRACED = {
    "ucb_waveform, sangria": waveform(),
    "ucb_waveform, frequency": waveform("frequency"),
    "ucb_waveform, phase": waveform("phase"),
    "ucb_waveform, 7 entries": waveform(length=7),
    "ucb_waveform, 8 entries": waveform(length=8),
    "ucb_waveform, cache": waveform(positions=CACHE),
    "ucb_waveform, override": waveform(q=np.int32(CASE.carrier + 2)),
    "ucb_waveform, automatic grid": (automatic_grid, ()),
    "template_on_window": (
        lambda p, q, duration, epoch, positions: bg32.template_on_window(
            p,
            q_ref=q,
            T=duration,
            t0=epoch,
            positions=positions,
            bin_lo=CASE.carrier - 60,
            n_bins=120,
            fmt="phase",
            **STATIC,
        ),
        (CACHE,),
    ),
    "tdi_response": (
        lambda p, q, duration, epoch, times: bg32.tdi_response(
            p, times + epoch, q_ref=q, T=duration
        ),
        (TIMES,),
    ),
    "tdi_response, positions": (
        lambda p, q, duration, epoch, positions: bg32.tdi_response(
            p, TIMES, q_ref=q, T=duration, positions=positions
        ),
        (np.asarray(bg.analytic_orbits(TIMES.astype(np.float64))).astype(np.float32),),
    ),
    "tdi_waveform": (
        lambda p, q, duration, epoch, theta: bg32.tdi_waveform(
            theta, q_ref=q, T=duration, t0=epoch, **STATIC
        ),
        (THETA32,),
    ),
    "tdi_waveform, GBParams and cache": (
        lambda p, q, duration, epoch, theta, positions: bg32.tdi_waveform(
            bg.GBParams(*theta),
            q_ref=q,
            T=duration,
            t0=epoch,
            positions=positions,
            fmt="frequency",
            **STATIC,
        ),
        (THETA32, CACHE),
    ),
}


def arguments(extra):
    """Return the traced arguments, reference bin as int32."""
    return (CASE.params, np.int32(CASE.q_ref), CASE.duration, CASE.epoch, *extra)


def test_jaxprs_hold_no_double_precision():
    """Every entry of ``TRACED``; a failure lists each offending path."""
    found = {}
    for name, (function, extra) in TRACED.items():
        jaxpr = jax.make_jaxpr(function)(*arguments(extra))
        found[name] = double_precision_values(jaxpr)
    assert {name: values for name, values in found.items() if values} == {}


def test_transformed_jaxprs_hold_no_double_precision():
    """Forward-mode AD and batching add no double-precision values either."""
    function, _ = TRACED["ucb_waveform, sangria"]

    def channels_of(params, q_ref):
        result = function(params, q_ref, CASE.duration, CASE.epoch)
        return result["A"], result["E"]

    batch = np.int32(CASE.q_ref) + np.arange(3, dtype=np.int32)
    traced = {
        "jacfwd": jax.make_jaxpr(jax.jacfwd(channels_of))(
            CASE.params, np.int32(CASE.q_ref)
        ),
        "vmap": jax.make_jaxpr(jax.vmap(channels_of))(
            np.stack([CASE.params] * 3), batch
        ),
    }
    found = {name: double_precision_values(jaxpr) for name, jaxpr in traced.items()}
    assert {name: values for name, values in found.items() if values} == {}


def test_host_double_inputs_are_rounded_before_tracing():
    """Float64 host params, duration, epoch and cache become float32 constants."""
    params = CASE.params.astype(np.float64)
    duration, epoch = float(CASE.duration), float(CASE.epoch)

    def evaluate(q_ref):
        return bg32.ucb_waveform(
            params,
            q_ref=q_ref,
            T=duration,
            t0=epoch,
            positions=CACHE.astype(np.float64),
            **STATIC,
        )

    assert double_precision_values(jax.make_jaxpr(evaluate)(np.int32(CASE.q_ref))) == []
