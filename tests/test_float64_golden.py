"""Float64 outputs stay identical to a baseline captured before float32 support.

``tools/float64_golden.py`` recorded public-API outputs with the sources of
commit 8363ec0. Bitwise equality is only meaningful on the same JAX/XLA build
and CPU. When the recorded jax, jaxlib, NumPy, machine and default backend
match the running environment, every output must keep its recorded dtype,
shape and exact bits, except outputs computed by the on-fly response kernel,
which must agree to 1e-13 of peak. Three later changes moved their last
bits: public calls now choose the automatic response grid on the host and
compile only it, not the recorded switch over every grid; the kernel now forms
its rotations from a shared cosine-sine pair and combines its eight delayed
differences in one signed sum; and ``template_on_window`` now forms A and E
from the sparse response before the transform. Bin indices, geometry and the
native vector still match bitwise. Elsewhere a different compiler or library may
legitimately move the last bits, so outputs must instead agree to within a
relative 1e-12 of each array's peak magnitude, or 1e-8 for the legacy Fourier
backend, whose sines of full phases near 1e7 rad differ by about 5e-10 of peak
between builds. Failure messages state which comparison applied. Double
precision must already be enabled, by the test configuration, before these
tests run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "data" / "float64_golden.npz"
GENERATOR = ROOT / "tools" / "float64_golden.py"
#: Allowed difference on another build, relative to each array's peak.
RELATIVE_TOLERANCE = 1e-12
#: The same for the legacy Fourier backend. It takes sines of full phases near
#: 1e7 rad, where one rounding of the argument is about 1e-9 rad; on x86_64
#: (JAX 0.4.38) its outputs moved by up to 5.4e-10 of peak with unchanged code.
FOURIER_TOLERANCE = 1e-8
#: The same on the recording build for outputs of the on-fly response kernel.
#: Automatic-grid cases were recorded through a ``lax.switch`` over every
#: grid, which host resolution no longer compiles; the kernel's rotations
#: and its signed sum over the eight delayed differences were since
#: reorganized; and the window now forms A and E before the transform. XLA
#: rounds each differently. On that build every such output moved by at most
#: 2.2e-15 of peak, while choosing a neighbouring grid moves an output by
#: about 1e-10 of its norm and a wrong sign in the sum by order one.
ONFLY_TOLERANCE = 1e-13
#: Outputs that never pass through the response kernel: exact on that build.
KERNEL_FREE_OUTPUTS = ("qmin", "freq_bin_indices", "positions", "native")
#: Environment fields that fix the exact bits of a float64 result. NumPy is
#: included because the interpolation coefficients are host NumPy constants.
BUILD = ("jax", "jaxlib", "numpy", "machine", "backend")


def load_golden():
    """Return the recorded metadata and output arrays keyed ``case/output``."""
    with np.load(GOLDEN, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return json.loads(arrays.pop("metadata").item()), arrays


METADATA, ARRAYS = load_golden()
CASES = {case["name"]: case for case in METADATA["cases"]}


@pytest.fixture(scope="module")
def generator():
    """Load the generator's evaluators without changing global configuration."""
    # The generator enables x64 on import. Requiring it to be enabled already
    # makes that a no-op here and keeps float64 inputs float64.
    assert jax.config.read("jax_enable_x64"), "enable x64 before these tests"
    spec = importlib.util.spec_from_file_location("float64_golden", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(environment):
    """Summarize the fields that decide whether exact bits are comparable."""
    return ", ".join(f"{key} {environment[key]}" for key in BUILD)


def test_recorded_inputs_match_the_generator(generator):
    """A changed case definition requires a deliberately regenerated archive."""
    recorded = [
        {key: value for key, value in case.items() if key != "outputs"}
        for case in METADATA["cases"]
    ]
    assert recorded == json.loads(json.dumps(generator.CASES))


def test_golden_cases_cover_the_float64_entry_points():
    """Keep the baseline from shrinking below the features it protects."""
    functions = {case["function"] for case in CASES.values()}
    assert functions == {
        "ucb_waveform",
        "template_on_window",
        "tdi_response",
        "tdi_waveform",
        "polarizations",
        "jacfwd",
    }
    onfly = [
        case
        for case in CASES.values()
        if case["function"] == "ucb_waveform" and "backend" not in case
    ]
    assert {case["fmt"] for case in onfly} == {"sangria", "frequency", "phase"}
    assert {len(case["params"]) for case in onfly} == {7, 8, 9}
    chirps = [case["params"][7] for case in onfly if len(case["params"]) > 7]
    assert min(chirps) < 0 < max(chirps)
    assert any(case["params"][8] != 0 for case in onfly if len(case["params"]) > 8)
    assert any(case["t0"] != 0 for case in onfly)
    samples = {case["response_samples"] for case in onfly}
    assert None in samples and len(samples) > 1
    assert any(case.get("positions") == "orbit_cache" for case in onfly)
    assert any("q" in case for case in onfly)
    responses = [case for case in CASES.values() if case["function"] == "tdi_response"]
    assert {"positions" in case for case in responses} == {False, True}
    assert GOLDEN.stat().st_size < 1_000_000


@pytest.mark.parametrize("name", list(CASES))
def test_float64_outputs_match_golden(generator, name):
    """Exact bits on the recording build outside the on-fly kernel."""
    recorded, running = METADATA["environment"], generator.environment()
    same_build = all(recorded[key] == running[key] for key in BUILD)
    case = CASES[name]
    onfly = case.get("backend") != "fourier" and case["function"] != "polarizations"
    if case.get("backend") == "fourier":
        tolerance = FOURIER_TOLERANCE
    elif same_build:
        tolerance = ONFLY_TOLERANCE
    else:
        tolerance = RELATIVE_TOLERANCE
    actual = generator.evaluate(case)
    assert sorted(actual) == sorted(case["outputs"])
    for output, value in actual.items():
        exact = same_build and not (onfly and output not in KERNEL_FREE_OUTPUTS)
        if exact:
            comparison = f"bitwise, same build: {_build(running)}"
        elif same_build:
            comparison = (
                f"relative {tolerance:g} of peak, on-fly kernel output, same "
                f"build: {_build(running)}"
            )
        else:
            comparison = (
                f"relative {tolerance:g} of peak, recorded on "
                f"{_build(recorded)}; running {_build(running)}"
            )
        expected = ARRAYS[f"{name}/{output}"]
        label = f"{name}/{output} ({comparison})"
        assert value.dtype == expected.dtype, (
            f"{label}: dtype {value.dtype}, recorded {expected.dtype}"
        )
        assert value.shape == expected.shape, (
            f"{label}: shape {value.shape}, recorded {expected.shape}"
        )
        difference = np.max(np.abs(value - expected), initial=0.0)
        if exact:
            assert (
                np.array_equal(value, expected)
                and value.tobytes() == expected.tobytes()
            ), f"{label}: float64 output changed; max |difference| {difference:.3e}"
        else:
            peak = np.max(np.abs(expected), initial=0.0)
            assert difference <= tolerance * peak, (
                f"{label}: max |difference| {difference:.3e} exceeds "
                f"{tolerance:g} of peak {peak:.3e}"
            )
