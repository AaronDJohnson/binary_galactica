"""Structured return type for the native and physical TDI waveform APIs."""

from typing import TypedDict

from jax import Array


class WaveformResult(TypedDict):
    """Raw GLASS channels and their absolute Fourier-bin coordinates.

    ``qmin`` is a float64 scalar; ``freq_bin_indices`` is a float64 array
    of length BW with integral values for valid inputs. ``X/Y/Z/A/E`` are
    complex128 arrays of length BW, all with GLASS normalization. The float32
    functions of :mod:`binary_galactica.single` return int32 bin coordinates
    and complex64 channels under the same keys.
    """

    qmin: Array
    freq_bin_indices: Array
    X: Array
    Y: Array
    Z: Array
    A: Array
    E: Array
