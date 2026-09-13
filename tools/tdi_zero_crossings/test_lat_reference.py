"""Optional LAT response checks; set LAT_REFERENCE_CHECKOUT to a local checkout."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import root_scalar

from .lat_reference import make_frozen_chirp_callback
from .solver import refine_candidate


@pytest.fixture(scope="module")
def lat_responses():
    """Build reference callbacks only when an explicit checkout is supplied."""
    checkout = os.environ.get("LAT_REFERENCE_CHECKOUT")
    if not checkout:
        pytest.skip("Set LAT_REFERENCE_CHECKOUT to enable local LAT response tests")
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        demodulated = make_frozen_chirp_callback(checkout)
        raw = make_frozen_chirp_callback(checkout, demodulate=False)
        yield Path(checkout).resolve(), demodulated, raw
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("time", [-100.0, 0.0, 8.106, 100.0])
def test_demodulation_preserves_amplitude_and_uses_lat_phase_sign(lat_responses, time):
    """Check the known exp(-i phase) convention independently of the loader."""
    _, demodulated, raw = lat_responses
    phase = 2.0 * np.pi * ((299792458.0 / 5e9) * time + 0.5e-7 * time**2)
    raw_value, demodulated_value = raw(time), demodulated(time)
    assert demodulated_value == pytest.approx(
        raw_value * np.exp(1j * phase), rel=1e-10, abs=1e-13
    )
    assert abs(demodulated_value) == pytest.approx(abs(raw_value), rel=1e-12)


@pytest.mark.parametrize("method", ["bounded", "brentq", "toms748"])
def test_reference_minimum_is_rejected_and_evaluation_counts_are_real(
    lat_responses, method
):
    """A physical response minimum must fail the complete complex zero check."""
    _, response, _ = lat_responses
    calls = []
    values = {}

    def observed(time):
        calls.append(time)
        values[time] = response(time)
        return values[time]

    result = refine_candidate(observed, (-100.0, 100.0), method=method)
    assert result.converged
    assert result.kind == "near_miss"
    # This nonzero residual sends either projected solver to bounded refinement.
    assert result.method == "bounded"
    assert result.evaluations == len(calls) == len(set(calls))
    assert result.evaluations > 3
    assert {-100.0, 0.0, 100.0}.issubset(calls)
    assert all(-100.0 <= time <= 100.0 for time in calls)
    assert result.value == values[result.time]
    assert result.residual == min(abs(value) for value in values.values())
    # Cancellation and the flat minimum preclude a tight physical time assertion.
    assert result.time == pytest.approx(8.106, abs=0.01)
    assert result.residual == pytest.approx(5.265961e-5, rel=1e-5)
    assert result.residual / result.scale > 0.05
    if method != "bounded":
        left, right = response(-100.0), response(100.0)
        direction = (right - left) / abs(right - left)
        projected = root_scalar(
            lambda time: (direction.conjugate() * response(time)).real,
            bracket=(-100.0, 100.0),
            method=method,
            xtol=1e-8,
        )
        projected_residual = abs(response(projected.root))
        assert projected.converged
        assert projected_residual > 1e-8 * result.scale
        assert result.residual <= projected_residual * (1.0 + 1e-10)


def test_optional_cli_records_full_results_costs_and_exact_source_hashes(
    lat_responses, tmp_path
):
    """Exercise the real CLI and verify its report against the selected files."""
    checkout, _, _ = lat_responses
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.tdi_zero_crossings.benchmark_lat",
            "--lat-checkout",
            str(checkout),
            "--output",
            str(output),
            "--repeats",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert "do not establish production speedup" in completed.stdout
    assert report["configuration"]["repeats"] == 2
    assert report["environment"]["jax"]
    assert report["environment"]["scipy"]
    assert len(report["reference"]["loaded_modules"]) == 4
    assert set(report["reference"]["loaded_modules"]).issubset(
        report["reference"]["sha256"]
    )
    for relative_path, digest in report["reference"]["sha256"].items():
        assert (
            hashlib.sha256((checkout / relative_path).read_bytes()).hexdigest()
            == digest
        )
    for method in ("bounded", "brentq", "toms748"):
        rows = report["results"][method]
        assert len(rows) == 2
        assert {row["repeat"] for row in rows} == {0, 1}
        for row in rows:
            assert row["kind"] == "near_miss"
            assert row["converged"]
            assert row["method_requested"] == method
            assert row["method"] == "bounded"
            assert row["evaluations"] == row["observed_evaluations"] > 3
            assert abs(complex(*row["value"])) == pytest.approx(row["residual"])
            assert row["relative_residual"] == pytest.approx(
                row["residual"] / row["scale"]
            )
            assert row["wall_seconds"] > 0
            assert row["cpu_seconds"] > 0
        summary = report["summary"][method]
        assert summary["methods_used"] == {"bounded": 2}
        assert summary["fallbacks"] == (0 if method == "bounded" else 2)
