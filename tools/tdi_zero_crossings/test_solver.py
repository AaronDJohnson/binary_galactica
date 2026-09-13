"""Analytic regressions for the experimental TDI zero finder."""

import numpy as np
import pytest

from .solver import refine_candidate, scan_candidates

METHODS = ("bounded", "brentq", "toms748")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("root", [1.5, 2.2])
def test_refines_tied_and_final_interval_crossings(method, root):
    """Catch the two missed-crossing cases of the original extractor."""
    result = refine_candidate(lambda t: complex(t - root), (0.0, 3.0), method=method)
    assert result.converged
    assert result.kind == "crossing"
    assert result.time == pytest.approx(root, rel=0.0, abs=1e-7)
    assert result.residual < 1e-8


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [1e-200, 1e-22, 1.0, 1e200])
def test_classification_is_independent_of_amplitude_scale(method, scale):
    """Squaring unnormalized amplitudes must not underflow or overflow."""
    result = refine_candidate(
        lambda t: scale * (t - 0.37) * np.exp(0.3j * t),
        (0.0, 1.0),
        method=method,
    )
    assert result.kind == "crossing"
    assert result.time == pytest.approx(0.37, rel=0.0, abs=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_near_miss_does_not_become_a_sign_flip(method):
    """A scalar projection root is insufficient evidence of a complex zero."""
    result = refine_candidate(
        lambda t: complex(t - 2.2, 0.1), (1.0, 3.0), method=method
    )
    assert result.kind == "near_miss"
    assert result.time == pytest.approx(2.2, rel=0.0, abs=1e-7)
    assert result.residual == pytest.approx(0.1)


@pytest.mark.parametrize("method", METHODS)
def test_tangential_zero_does_not_request_a_sign_flip(method):
    """An even-order zero has the same complex direction on both sides."""
    result = refine_candidate(
        lambda t: (t - 0.37) ** 2 * np.exp(0.2j * t),
        (0.0, 1.0),
        method=method,
    )
    assert result.kind == "touch"
    assert result.time == pytest.approx(0.37, rel=0.0, abs=1e-6)


@pytest.mark.parametrize("root", [0.0, 1.0])
def test_boundary_zero_requires_more_data_to_classify(root):
    """Do not infer a sign change without samples on both sides."""
    result = refine_candidate(lambda t: complex(t - root), (0.0, 1.0))
    assert result.kind == "boundary"
    assert result.time == root


def test_zero_everywhere_is_not_an_isolated_crossing():
    """A vanishing channel provides no unique zero time or phase continuation."""
    result = refine_candidate(lambda t: 0j, (0.0, 1.0))
    assert result.kind == "unresolved"


def test_failed_minimization_is_not_accepted_as_a_crossing():
    """The solver's convergence status must survive result classification."""
    result = refine_candidate(lambda t: complex(np.exp(t) - 1.5), (0, 1), maxiter=1)
    assert not result.converged
    assert result.kind == "unresolved"


def test_callback_is_evaluated_between_supplied_samples_and_calls_are_counted():
    """Refine the actual function and account for verification probes too."""
    calls = []

    def response(t):
        calls.append(t)
        return complex(np.exp(t) - 1.5)

    result = refine_candidate(response, (0.0, 1.0))
    assert result.time == pytest.approx(np.log(1.5), rel=0.0, abs=1e-7)
    assert result.kind == "crossing"
    assert result.evaluations == len(calls)
    assert any(t not in (0.0, 0.5, 1.0) for t in calls)


def test_time_normalization_preserves_large_epoch_and_short_interval():
    """Time precision is set in physical units, not by the epoch magnitude."""
    epoch = 30_000_000.0
    result = refine_candidate(
        lambda t: complex(t - (epoch + 0.37)),
        (epoch, epoch + 1.0),
        time_atol=1e-7,
        amplitude_rtol=1e-6,
    )
    assert result.kind == "crossing"
    assert result.time == pytest.approx(epoch + 0.37, rel=0.0, abs=1e-7)


@pytest.mark.parametrize("times", [[0, 1, 2, 3], [0, 0.15, 1.4, 3]])
def test_scan_finds_crossing_on_uniform_and_nonuniform_grids(times):
    """Candidate discovery cannot depend on an equally spaced stencil."""
    results = scan_candidates(lambda t: complex(t - 1.5), times)
    crossings = [result for result in results if result.kind == "crossing"]
    assert len(crossings) == 1
    assert crossings[0].time == pytest.approx(1.5, rel=0.0, abs=1e-7)


def test_scan_retains_final_interval_and_exact_sample_zero():
    """A root on a sample must be bracketed by its neighboring samples."""
    for root in (2.2, 2.0):
        results = scan_candidates(lambda t, root=root: complex(t - root), [0, 1, 2, 3])
        assert [result.kind for result in results] == ["crossing"]
        assert results[0].time == pytest.approx(root, rel=0.0, abs=1e-7)


def test_scan_refines_multiple_separated_crossings():
    """Report distinct crossings once each when the mesh resolves them."""
    results = scan_candidates(lambda t: complex((t - 0.3) * (t - 0.7)), [0, 0.5, 1])
    roots = [result.time for result in results if result.kind == "crossing"]
    assert roots == pytest.approx([0.3, 0.7], rel=0.0, abs=1e-7)


@pytest.mark.parametrize("interval", [(1, 0), (0, 0), (0, np.inf)])
def test_rejects_invalid_intervals(interval):
    """Invalid bounds must not reach a stock solver."""
    with pytest.raises(ValueError):
        refine_candidate(lambda t: 1j, interval)


@pytest.mark.parametrize("value", [complex(np.nan), complex(np.inf), [1, 2]])
def test_rejects_nonfinite_or_nonscalar_response(value):
    """Invalid response values cannot be classified as minima or zeros."""
    with pytest.raises(ValueError):
        refine_candidate(lambda t: value, (0.0, 1.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "newton"},
        {"time_atol": 0},
        {"amplitude_rtol": -1},
        {"amplitude_atol": np.nan},
        {"maxiter": 0},
    ],
)
def test_rejects_invalid_solver_controls(kwargs):
    """Validate controls before evaluating a physical response."""
    with pytest.raises(ValueError):
        refine_candidate(lambda t: complex(t), (-1, 1), **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"method": "bad"}, {"time_atol": -1}, {"maxiter": 0}]
)
def test_empty_scan_still_validates_controls_before_calling_response(kwargs):
    """An empty set of candidates must not hide malformed solver controls."""
    calls = []

    def response(t):
        calls.append(t)
        return 1j

    with pytest.raises(ValueError):
        scan_candidates(response, [0.0, 1.0], **kwargs)
    assert calls == []


def test_preserves_an_exact_midpoint_zero_in_a_nonunimodal_interval():
    """A minimizer must not discard a zero already evaluated at the midpoint."""
    result = refine_candidate(lambda t: (t - 0.5) * np.exp(6 * t), (0.0, 1.0))
    assert result.kind == "crossing"
    assert result.time == 0.5
    assert result.residual == 0.0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("interval", [(0.001, 0.01), (-0.1, 0.2)])
def test_normalized_solver_never_evaluates_outside_closed_interval(method, interval):
    """Preserve endpoints when affine floating-point arithmetic rounds outward."""
    lower, upper = interval

    def response(t):
        assert lower <= t <= upper
        return complex(t - 0.005)

    result = refine_candidate(response, interval, method=method)
    assert result.kind == "crossing"
    assert result.time == pytest.approx(0.005, rel=0.0, abs=1e-8)


def test_rejects_unrepresentable_normalized_power_with_actionable_error():
    """Finite samples can hide an interior scale too large to square safely."""
    with pytest.raises(ValueError, match="subdivide"):
        refine_candidate(
            lambda t: complex(1 + 1e160 * (t * (t - 0.5) * (t - 1)) ** 2),
            (0.0, 1.0),
        )
