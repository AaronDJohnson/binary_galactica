"""Regressions separating zero classification from accurate zero localization."""

import numpy as np
import pytest

from .solver import refine_candidate, scan_candidates


@pytest.mark.parametrize("method", ["bounded", "brentq", "toms748"])
@pytest.mark.parametrize("width", [1.0, 2e-5])
def test_localizes_nonlinear_crossing_at_requested_time_accuracy(method, width):
    """The minimizer's relative stopping floor must not dominate time_atol."""
    result = refine_candidate(
        lambda t: complex(np.expm1(t / width) - 0.5),
        (0.0, width),
        method=method,
        time_atol=width * 1e-12,
        amplitude_rtol=1e-10,
    )
    assert result.kind == "crossing"
    assert abs(result.time / width - np.log(1.5)) < 1e-12


@pytest.mark.parametrize("method", ["bounded", "brentq", "toms748"])
@pytest.mark.parametrize("order", [2, 4])
def test_localizes_tangent_zero_instead_of_accepting_only_a_small_residual(
    method, order
):
    """A flat objective can have a tiny residual and a poorly located minimum."""
    root = 0.37123456789
    result = refine_candidate(
        lambda t: (t - root) ** order * np.exp(0.2j * t),
        (0.0, 1.0),
        method=method,
        time_atol=1e-12,
        amplitude_rtol=1e-10,
    )
    assert result.kind == "touch"
    assert abs(result.time - root) < 1e-12


@pytest.mark.parametrize("method", ["brentq", "toms748"])
@pytest.mark.parametrize("order", [1, 2])
def test_rejected_projected_root_falls_back_to_the_complex_minimum(method, order):
    """Phase rotation can make a projection cross zero away from a true zero."""
    result = refine_candidate(
        lambda t: (t - 0.37) ** order * np.exp(8j * (t - 0.37)),
        (0.0, 1.0),
        method=method,
        time_atol=1e-12,
    )
    assert result.kind == ("crossing" if order == 1 else "touch")
    assert abs(result.time - 0.37) < 1e-12


@pytest.mark.parametrize("method", ["bounded", "brentq", "toms748"])
@pytest.mark.parametrize("root", [1e-10, 1.0 - 1e-10])
def test_small_endpoint_amplitude_does_not_hide_a_resolvable_interior_root(
    method, root
):
    """Amplitude acceptance tolerance must not move a root onto the boundary."""
    result = refine_candidate(
        lambda t: complex(t - root),
        (0.0, 1.0),
        method=method,
        time_atol=1e-12,
        amplitude_rtol=1e-8,
    )
    # Conservative direction checks may need better local amplitude resolution.
    assert result.kind in ("crossing", "unresolved")
    assert abs(result.time - root) < 1e-12


@pytest.mark.parametrize("method", ["bounded", "brentq", "toms748"])
def test_extra_precision_does_not_turn_a_resolved_near_miss_into_a_zero(method):
    """Fallback and polishing must keep the full complex residual check."""
    result = refine_candidate(
        lambda t: complex(t - 0.37, 3e-8),
        (0.0, 1.0),
        method=method,
        time_atol=1e-12,
        amplitude_rtol=1e-8,
    )
    assert result.kind == "near_miss"
    assert result.residual >= 3e-8


def test_tighter_tolerance_still_respects_physical_time_representation():
    """Centering a solver cannot restore bits missing from callback timestamps."""
    with pytest.raises(ValueError, match="time resolution"):
        refine_candidate(
            lambda t: complex(t - 30_000_000.3),
            (30_000_000, 30_000_001),
            time_atol=1e-12,
        )


@pytest.mark.parametrize("method", ["brentq", "toms748"])
@pytest.mark.parametrize(
    "root,order,rotation", [(0.500001, 2, 8), (0.5000000001, 1, 4)]
)
def test_small_cached_residual_does_not_mask_a_rejected_projected_root(
    method, root, order, rotation
):
    """A nearby small midpoint amplitude is not an accurately localized zero."""
    result = refine_candidate(
        lambda t: (t - root) ** order * np.exp(rotation * 1j * (t - root)),
        (0.0, 1.0),
        method=method,
        time_atol=1e-12,
    )
    assert result.kind == ("crossing" if order == 1 else "touch")
    assert abs(result.time - root) < 1e-12


def test_optimization_stages_share_the_requested_evaluation_budget():
    """A second minimizer must not silently reset the work limit."""
    calls = []

    def response(t):
        calls.append(t)
        return complex(np.expm1(t) - 0.5)

    limited = refine_candidate(response, (0.0, 1.0), time_atol=1e-12, maxiter=15)
    assert limited.kind == "unresolved"
    assert not limited.converged
    assert limited.evaluations == len(calls) <= 18  # Three initial samples extra.
    accurate = refine_candidate(response, (0.0, 1.0), time_atol=1e-12, maxiter=30)
    assert accurate.kind == "crossing"
    assert abs(accurate.time - np.log(1.5)) < 1e-12


def test_scanner_keeps_two_adjacent_exact_sampled_zeros():
    """Distinct known zeros must not collapse into one equal-minimum plateau."""
    results = scan_candidates(
        lambda t: complex((t - 0.25) * (t - 0.5)),
        [0.0, 0.25, 0.5, 0.75, 1.0],
        subdivisions=1,
        time_atol=1e-12,
    )
    roots = [r.time for r in results if r.kind == "crossing"]
    assert roots == pytest.approx([0.25, 0.5], rel=0, abs=1e-12)


@pytest.mark.parametrize("first,last", [(0.0, 0.25), (0.75, 1.0)])
def test_scanner_keeps_an_exact_endpoint_and_its_adjacent_zero(first, last):
    """An endpoint bracket must not return the neighboring sampled zero twice."""
    results = scan_candidates(
        lambda t: complex((t - first) * (t - last)),
        [0.0, 1.0],
        time_atol=1e-12,
    )
    assert [r.time for r in results] == pytest.approx([first, last], rel=0, abs=1e-12)
    assert sorted(r.kind for r in results) == ["boundary", "crossing"]
