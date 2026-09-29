"""The float32 path in a process without x64, and without double precision in it.

A fresh interpreter with x64 disabled runs every ``single`` function on inputs
and float64 references that this process writes to ``tmp_path``. In this
process, where the suite enables x64, traced ``single`` functions must hold
no float64 or complex128 values, and their compiled programs none at all.
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
from tests.single.support import (
    MISMATCH,
    RELATIVE_L2,
    YEAR,
    errors,
    physical,
    source,
)
from tests.test_precision import run_without_x64

#: A late ten-year observation at 20 mHz, as float64 native coordinates for a
#: float64 duration that float32 does not represent.
DURATION, EPOCH = 10 * YEAR, 10 * YEAR
BW, SAMPLES = 4096, 2048
NATIVE = np.array(
    [0.02 * DURATION, np.sin(-0.4), 2.2, np.log(3e-22), 0.6, 1.3, 0.9, 2.5, -0.4]
)
THETA = np.array([1.5e-22, 0.0123456789, 2e-17, 0.9, 0.8, 1.3, 2.2, -0.4])
TIMES = np.linspace(-1e6, 6.2e8, 65).astype(np.float32)
WINDOW: dict[str, Any] = {"bin_lo": int(NATIVE[0]) - 1500, "n_bins": 3000}

#: The x64-free child: run every float32 function, compare with the float64
#: references in ``directory``, save the outputs there and try every guarded
#: float64 call. The parent defines ``directory`` first.
CHILD = """
import json

import jax
import numpy as np

flags = {"before import": jax.config.read("jax_enable_x64")}
import binary_galactica as bg
from binary_galactica import single as bg32

flags["after import"] = jax.config.read("jax_enable_x64")
data = np.load(directory + "/reference.npz")
duration = float(data["duration"])
T, t0 = np.float32(duration), np.float32(data["epoch"])
observation = {"T": T, "t0": t0, "BW": int(data["BW"]),
               "response_samples": int(data["samples"])}
window = {"bin_lo": int(data["bin_lo"]), "n_bins": int(data["n_bins"])}

q_ref, params = bg32.split_native(data["native"], duration)
q_physical, theta = bg32.split_physical(data["theta"], duration)
outputs = {
    "ucb_waveform": bg32.ucb_waveform(params, q_ref=q_ref, **observation),
    "ucb_waveform, phase and cache": bg32.ucb_waveform(
        params, q_ref=q_ref, fmt="phase", positions=data["cache"], **observation
    ),
    "ucb_waveform, jit": jax.jit(
        lambda q: bg32.ucb_waveform(params, q_ref=q, **observation)
    )(q_ref),
    "template_on_window": bg32.template_on_window(
        params, q_ref=q_ref, **window, **observation
    ),
    "tdi_response": bg32.tdi_response(params, data["times"], q_ref=q_ref, T=T),
    "tdi_response, positions": bg32.tdi_response(
        params, data["times"], q_ref=q_ref, T=T, positions=data["positions"]
    ),
    "tdi_waveform": bg32.tdi_waveform(theta, q_ref=q_physical, **observation),
}


def compared(values):
    if isinstance(values, dict):
        values = np.stack([values["A"], values["E"]])
    return np.asarray(values)


report = {
    "split": {
        "q_ref": [type(q_ref).__name__, q_ref],
        "q_physical": [type(q_physical).__name__, q_physical],
    },
    "dtypes": {},
    "errors": {},
    "guards": {},
}
saved = {"split_native": params, "split_physical": theta}
for name, result in outputs.items():
    values = result if isinstance(result, dict) else {"value": result}
    report["dtypes"][name] = {
        key: str(np.asarray(value).dtype) for key, value in values.items()
    }
    saved.update({name + "/" + k: np.asarray(v) for k, v in values.items()})
    a = compared(result).astype(np.complex128).ravel()
    b = data["reference/" + name].ravel()
    overlap = np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    report["errors"][name] = [
        float(np.linalg.norm(a - b) / np.linalg.norm(b)), float(1 - overlap)
    ]
report["dtypes"]["split"] = [str(params.dtype), str(theta.dtype)]
np.savez(directory + "/single.npz", **saved)

native = data["native"]
times = np.linspace(0.0, duration, 5)
guarded = {
    "ucb_waveform": lambda: bg.ucb_waveform(native, T=duration, BW=32),
    "tdi_waveform": lambda: bg.tdi_waveform(data["theta"], T=duration, BW=32),
    "tdi_response": lambda: bg.tdi_response(native, times, T=duration),
    "template_on_window": lambda: bg.template_on_window(
        native, T=duration, BW=32, bin_lo=int(native[0]) - 8, n_bins=16
    ),
    "orbit_cache": lambda: bg.orbit_cache(duration, 0.0, 32),
    "phase": lambda: bg.phase(data["theta"], times),
    "frequency": lambda: bg.frequency(data["theta"], times),
    "amplitudes": lambda: bg.amplitudes(data["theta"]),
    "polarizations": lambda: bg.polarizations(data["theta"], times),
}
for name, call in guarded.items():
    try:
        call()
    except Exception as error:  # report every outcome to the parent
        report["guards"][name] = [type(error).__name__, str(error)]
    else:
        report["guards"][name] = ["returned", ""]
flags["at exit"] = jax.config.read("jax_enable_x64")
report["flags"] = flags
print(json.dumps(report))
"""


DURATION32, EPOCH32 = np.float32(DURATION), np.float32(EPOCH)
OBSERVATION: dict[str, Any] = {"BW": BW, "response_samples": SAMPLES}


def float32_inputs() -> dict[str, Any]:
    """Split the float64 sources and round the geometry, as the child does."""
    q_ref, params = bg32.split_native(NATIVE, DURATION)
    q_physical, theta = bg32.split_physical(THETA, DURATION)
    cache = bg.orbit_cache(float(DURATION32), float(EPOCH32), BW)
    positions = bg.analytic_orbits(TIMES.astype(np.float64))
    return {
        "q_ref": q_ref,
        "params": params,
        "q_physical": q_physical,
        "theta": theta,
        "cache": np.asarray(cache).astype(np.float32),
        "positions": np.asarray(positions).astype(np.float32),
    }


def single_outputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the child's float32 calls in this process, where x64 is on."""
    q_ref, params = inputs["q_ref"], inputs["params"]
    observation: dict[str, Any] = {"T": DURATION32, "t0": EPOCH32, **OBSERVATION}
    return {
        "ucb_waveform": bg32.ucb_waveform(params, q_ref=q_ref, **observation),
        "ucb_waveform, phase and cache": bg32.ucb_waveform(
            params, q_ref=q_ref, fmt="phase", positions=inputs["cache"], **observation
        ),
        "template_on_window": bg32.template_on_window(
            params, q_ref=q_ref, **WINDOW, **observation
        ),
        "tdi_response": bg32.tdi_response(params, TIMES, q_ref=q_ref, T=DURATION32),
        "tdi_response, positions": bg32.tdi_response(
            params, TIMES, q_ref=q_ref, T=DURATION32, positions=inputs["positions"]
        ),
        "tdi_waveform": bg32.tdi_waveform(
            inputs["theta"], q_ref=inputs["q_physical"], **observation
        ),
    }


def write_references(directory: Path, inputs: dict[str, Any]) -> None:
    """Write the inputs and float64 references at the float32 inputs."""
    duration = float(DURATION32)
    native = inputs["params"].astype(np.float64)
    native[0] += inputs["q_ref"]
    double_theta = inputs["theta"].astype(np.float64)
    double_theta[1] += inputs["q_physical"] / duration
    cache = inputs["cache"].astype(np.float64)
    positions = inputs["positions"].astype(np.float64)
    times = TIMES.astype(np.float64)
    observation: dict[str, Any] = {"T": duration, "t0": float(EPOCH32), **OBSERVATION}
    carrier = float(inputs["q_ref"] + np.floor(inputs["params"][0]))

    def stacked(result):
        return np.stack([np.asarray(result["A"]), np.asarray(result["E"])])

    waveform = stacked(bg.ucb_waveform(native, q=carrier, **observation))
    phase = bg.ucb_waveform(
        native, q=carrier, fmt="phase", positions=cache, **observation
    )
    references = {
        "ucb_waveform": waveform,
        "ucb_waveform, jit": waveform,
        "ucb_waveform, phase and cache": stacked(phase),
        "template_on_window": bg.template_on_window(native, **WINDOW, **observation),
        "tdi_response": bg.tdi_response(native, times, T=duration),
        "tdi_response, positions": bg.tdi_response(
            native, times, T=duration, positions=positions
        ),
        "tdi_waveform": stacked(bg.tdi_waveform(double_theta, **observation)),
    }
    arrays: dict[str, Any] = {
        "native": NATIVE,
        "theta": THETA,
        "duration": DURATION,
        "epoch": EPOCH,
        "BW": BW,
        "samples": SAMPLES,
        "cache": inputs["cache"],
        "times": TIMES,
        "positions": inputs["positions"],
        **WINDOW,
        **{f"reference/{k}": np.asarray(v) for k, v in references.items()},
    }
    np.savez(directory / "reference.npz", **arrays)


def test_every_single_function_runs_without_x64(tmp_path):
    inputs = float32_inputs()
    write_references(tmp_path, inputs)
    report = json.loads(run_without_x64(f"directory = {str(tmp_path)!r}\n" + CHILD))

    assert report["flags"] == {
        "before import": False,
        "after import": False,
        "at exit": False,
    }
    assert report["split"] == {
        "q_ref": ["int", inputs["q_ref"]],
        "q_physical": ["int", inputs["q_physical"]],
    }
    assert report["dtypes"].pop("split") == ["float32", "float32"]
    for name, dtypes in report["dtypes"].items():
        for key, dtype in dtypes.items():
            expected = "int32" if key in ("qmin", "freq_bin_indices") else "complex64"
            assert dtype == expected, (name, key)
    assert set(report["errors"]) == set(report["dtypes"])
    for name, (relative_l2, mismatch) in report["errors"].items():
        assert relative_l2 <= RELATIVE_L2, (name, relative_l2)
        assert mismatch <= MISMATCH, (name, mismatch)

    # The splits are host NumPy. The float32 kernels compute the same bits
    # whether or not x64 is enabled.
    saved = np.load(tmp_path / "single.npz")
    np.testing.assert_array_equal(saved["split_native"], inputs["params"])
    np.testing.assert_array_equal(saved["split_physical"], inputs["theta"])
    for name, result in single_outputs(inputs).items():
        values = result if isinstance(result, dict) else {"value": result}
        for key, value in values.items():
            child_value = saved[f"{name}/{key}"]
            assert child_value.dtype == np.asarray(value).dtype, (name, key)
            np.testing.assert_array_equal(
                child_value, np.asarray(value), err_msg=f"{name}/{key}"
            )

    for name, (kind, message) in report["guards"].items():
        assert kind == "RuntimeError", f"{name}: {kind} {message}"
        assert 'jax.config.update("jax_enable_x64", True)' in message, name
        assert "binary_galactica.single" in message, name
    assert len(report["guards"]) == 9


CASE = source(1.0, 1e-2, 10.0, (1, -1))
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


#: Each float32 function as ``(function, extra traced arguments)``; the
#: first four arguments are always ``params, q_ref, T, t0``.
TRACED = {
    "ucb_waveform, sangria": waveform(),
    "ucb_waveform, frequency": waveform("frequency"),
    "ucb_waveform, phase": waveform("phase"),
    "ucb_waveform, 7 entries": waveform(length=7),
    "ucb_waveform, 8 entries": waveform(length=8),
    "ucb_waveform, cache": waveform(positions=CACHE),
    "ucb_waveform, override": waveform(q=np.int32(CASE.carrier + 2)),
    "ucb_waveform, automatic grid": (
        lambda p, q, duration, epoch: bg32.ucb_waveform(
            p, q_ref=q, T=duration, t0=epoch, BW=CASE.bandwidth
        ),
        (),
    ),
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


@pytest.mark.parametrize("name", list(TRACED))
def test_jaxprs_hold_no_double_precision(name):
    function, extra = TRACED[name]
    jaxpr = jax.make_jaxpr(function)(*arguments(extra))
    assert double_precision_values(jaxpr) == []


@pytest.mark.parametrize("transform", ["jacfwd", "vmap"])
def test_transformed_jaxprs_hold_no_double_precision(transform):
    function, _ = TRACED["ucb_waveform, sangria"]

    def channels_of(params, q_ref):
        result = function(params, q_ref, CASE.duration, CASE.epoch)
        return result["A"], result["E"]

    if transform == "jacfwd":
        traced = jax.make_jaxpr(jax.jacfwd(channels_of))(
            CASE.params, np.int32(CASE.q_ref)
        )
    else:
        batch = np.int32(CASE.q_ref) + np.arange(3, dtype=np.int32)
        traced = jax.make_jaxpr(jax.vmap(channels_of))(
            np.stack([CASE.params] * 3), batch
        )
    assert double_precision_values(traced) == []


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


@pytest.mark.parametrize(
    "name", ["ucb_waveform, phase", "template_on_window", "tdi_response, positions"]
)
def test_compiled_programs_hold_no_double_precision(name):
    """After XLA's optimization no double-precision value remains at all.

    The jaxprs above may pass Python scalars into ``jnp`` helpers as weakly
    typed float64 constants that are converted before use; compilation folds
    those conversions away.
    """
    function, extra = TRACED[name]
    compiled = jax.jit(function).lower(*arguments(extra)).compile().as_text()
    assert "f64" not in compiled
    assert "c128" not in compiled


def test_traced_forms_match_eager_calls():
    """The functions whose jaxprs are checked above compute the waveform."""
    function, _ = TRACED["ucb_waveform, cache"]
    traced = jax.jit(function)(*arguments((CACHE,)))
    eager = CASE.single(positions=CACHE)
    assert errors(np.asarray(traced["A"]), np.asarray(eager["A"]))[0] < 1e-5
