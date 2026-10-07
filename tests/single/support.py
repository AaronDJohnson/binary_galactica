"""Sources, float64 references and error metrics shared by the float32 tests.

Every comparison is made at identical inputs. The float32 functions receive
float32 values; the float64 functions receive the exact float64 values of
those, with ``f0*T = q_ref + params[0]`` and the carrier pinned to the float32
carrier. Errors are computed in float64, because raw strains near 1e-21
square below float32's smallest normal number.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import lisaconstants
import numpy as np

import binary_galactica as bg
from binary_galactica import single as bg32

YEAR = lisaconstants.ASTRONOMICAL_YEAR
#: The validated range: durations and epochs in years, frequencies in hertz.
DURATIONS = (0.25, 1.0, 4.0, 10.0)
FREQUENCIES = (1e-4, 1e-3, 1e-2, 3e-2)
EPOCHS = (0.0, 10.0)
#: Signs of the first and second frequency-derivative terms.
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
#: Drift, in bins, of the first and second derivative terms from time zero to
#: the end of the observation. The band is centred on the frequency at time
#: zero, so a source must stay within a few bins of it to fit.
DRIFT = (3.0, 1.0)
#: The accuracy target of the float32 path against float64.
RELATIVE_L2 = 1e-4
MISMATCH = 1e-8
#: Near the ecliptic the Doppler delay ``k.x0/c``, and the band, are widest.
BETA = 0.1
AMPLITUDE = 1e-22
INT32_MIN = int(np.iinfo(np.int32).min)
INT32_MAX = int(np.iinfo(np.int32).max)


def automatic_samples(duration: float) -> int:
    """Return the documented automatic response count for a float64 duration.

    Powers of two from 32 to 2048, stepping up above each of 0.25, 0.5, 1, 2,
    4 and 8 years. Passing the count explicitly compiles one grid instead of
    every branch of the automatic choice.
    """
    counts = (32, 64, 128, 256, 512, 1024, 2048)
    thresholds = YEAR * np.array(counts[:-1], dtype=np.float64) / 128
    return counts[int(np.sum(thresholds < duration))]


@dataclass(frozen=True)
class Source:
    """A float32 source in offset native coordinates, with its observation."""

    q_ref: int
    params: np.ndarray
    duration: np.float32
    epoch: np.float32
    bandwidth: int
    samples: int

    @property
    def native(self) -> np.ndarray:
        """Return the exact float64 native coordinates, ``f0*T`` included."""
        native = self.params.astype(np.float64)
        native[0] += self.q_ref
        return native

    @property
    def carrier(self) -> int:
        """Return the documented default float32 carrier bin."""
        return self.q_ref + math.floor(float(self.params[0]))

    def observation(self) -> dict[str, Any]:
        """Return the float32 observation keywords."""
        return {
            "T": self.duration,
            "t0": self.epoch,
            "BW": self.bandwidth,
            "response_samples": self.samples,
        }

    def double_observation(self) -> dict[str, Any]:
        """Return the same observation as float64 keywords."""
        return {
            "T": float(self.duration),
            "t0": float(self.epoch),
            "BW": self.bandwidth,
            "response_samples": self.samples,
        }

    def single(self, **options: Any) -> Any:
        """Evaluate the float32 waveform with its default carrier."""
        return bg32.ucb_waveform(
            self.params, q_ref=self.q_ref, **{**self.observation(), **options}
        )

    def double(self, **options: Any) -> Any:
        """Evaluate the float64 waveform with the carrier pinned."""
        options = {"q": float(self.carrier), **options}
        return bg.ucb_waveform(self.native, **{**self.double_observation(), **options})


def source(
    years: float,
    frequency: float,
    epoch_years: float = 0.0,
    signs: tuple[int, int] = (1, 1),
    *,
    drift: tuple[float, float] = DRIFT,
    length: int = 9,
    bandwidth: int | None = None,
) -> Source:
    """Build a float32 source at ``frequency`` hertz and its observation.

    ``q_ref`` is ``floor(f0*T)`` for the float32 duration, and the float32
    offset is the rounded remainder. The derivative terms drift ``drift`` bins
    with the given signs by the end of the observation. ``length`` truncates
    the native vector to 7 or 8 entries. The band holds the Doppler spread and
    drift, as chosen by :func:`binary_galactica.ucb_bw`, unless ``bandwidth``
    is given.
    """
    duration = np.float32(years * YEAR)
    epoch = np.float32(epoch_years * YEAR)
    scaled = frequency * float(duration)
    q_ref = math.floor(scaled)
    offset = np.float32(scaled - q_ref)
    assert 0 <= offset < 1
    end = (float(epoch) + float(duration)) / float(duration)
    first = signs[0] * drift[0] / end
    second = signs[1] * 2 * drift[1] / end**2
    params = np.array(
        [
            offset,
            np.sin(BETA),
            4.1,
            np.log(2 * AMPLITUDE),
            np.cos(1.1),
            0.4,
            -0.7,
            first,
            second,
        ],
        dtype=np.float32,
    )[:length]
    if bandwidth is None:
        fdot = sum(drift) / float(duration) ** 2
        bandwidth = bg.ucb_bw(
            frequency, fdot, np.sin(BETA), AMPLITUDE, float(duration), 2**31
        )
    return Source(
        q_ref=q_ref,
        params=params,
        duration=duration,
        epoch=epoch,
        bandwidth=int(bandwidth),
        samples=automatic_samples(float(duration)),
    )


def chirping(
    years: float, frequency: float, chirp_mass: float, epoch_years: float = 0.0
) -> Source:
    """Build a source whose frequency derivatives come from gravitational radiation.

    At time zero ``fdot`` is :func:`binary_galactica.ucb_fdot` for
    ``chirp_mass`` solar masses at ``frequency`` hertz, and
    ``fddot = 11/3*fdot**2/f0``. Chirp masses up to one solar mass lie in the
    package's own prior; at tens of millihertz the phase terms reach thousands
    of cycles. The band from :func:`binary_galactica.ucb_bw` holds the drift
    about the frequency at the middle of the observation.
    """
    case = source(years, frequency, epoch_years)
    duration = float(case.duration)
    fdot = bg.ucb_fdot(chirp_mass, frequency)
    fddot = 11 / 3 * fdot**2 / frequency
    params = case.params.copy()
    params[7], params[8] = fdot * duration**2, fddot * duration**3
    middle = float(case.epoch) + duration / 2
    bandwidth = bg.ucb_bw(
        frequency + fdot * middle + fddot * middle**2 / 2,
        fdot + fddot * middle,
        np.sin(BETA),
        AMPLITUDE,
        duration,
        2**31,
    )
    return dataclasses.replace(case, params=params, bandwidth=int(bandwidth))


def middle_bin(case: Source) -> int:
    """Return the bin holding the source's frequency mid-observation.

    A carrier pinned there centres the band on a source that has drifted far
    from its frequency at time zero.
    """
    native = case.native
    u = (float(case.epoch) + float(case.duration) / 2) / float(case.duration)
    return math.floor(native[0] + native[7] * u + native[8] * u**2 / 2)


def physical(case: Source) -> tuple[np.ndarray, np.ndarray]:
    """Return ``case``'s source as float32 physical offset parameters, and float64.

    ``(amp, f0 - q_ref/T, fdot, phi0, iota, psi, lam, beta)`` for the float32
    duration; physical parameters carry no second derivative. The float64
    parameters are the exact float32 values with ``q_ref/T`` added to the
    offset.
    """
    duration = float(case.duration)
    offset = float(case.params[0]) / duration
    fdot = float(case.params[7]) / duration**2
    theta = np.array(
        [AMPLITUDE, offset, fdot, 0.7, 1.1, 0.4, 4.1, BETA], dtype=np.float32
    )
    double = theta.astype(np.float64)
    double[1] += case.q_ref / duration
    return theta, double


def channels(result: Any) -> np.ndarray:
    """Stack the A and E channels of a waveform result."""
    return np.stack([np.asarray(result["A"]), np.asarray(result["E"])])


def errors(single: Any, double: Any) -> tuple[float, float]:
    """Return the relative L2 error and flat-weighted mismatch, in float64."""
    a = np.asarray(single, dtype=np.complex128).ravel()
    b = np.asarray(double, dtype=np.complex128).ravel()
    overlap = np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)), float(1 - overlap)


def assert_accurate(single: Any, double: Any, label: str = "") -> None:
    """Require the float32 target: relative L2 and mismatch against float64."""
    relative_l2, mismatch = errors(single, double)
    assert relative_l2 <= RELATIVE_L2, f"{label}: relative L2 {relative_l2:.3g}"
    assert mismatch <= MISMATCH, f"{label}: mismatch {mismatch:.3g}"
