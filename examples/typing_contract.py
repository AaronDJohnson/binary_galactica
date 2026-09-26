"""Statically checked consumers of the public waveform conversion API."""

from collections.abc import Mapping

from jax.typing import ArrayLike

from binary_galactica import (
    GBParams,
    WaveformResult,
    tdi_response,
    tdi_waveform,
    to_continuous_ft,
    ucb_waveform,
)


def native_continuous(params: ArrayLike, duration: ArrayLike) -> WaveformResult:
    """Preserve named output fields through the native waveform conversion."""
    raw = ucb_waveform(params, T=duration, BW=32)
    return to_continuous_ft(raw, duration)


def physical_continuous(params: GBParams, duration: ArrayLike) -> WaveformResult:
    """Preserve named output fields through the physical waveform conversion."""
    raw = tdi_waveform(params, T=duration, BW=32)
    return to_continuous_ft(raw, duration)


def native_response(
    params: ArrayLike, times: ArrayLike, duration: ArrayLike
) -> ArrayLike:
    """Use the public demodulated XYZ response at arbitrary reception times."""
    return tdi_response(params, times, T=duration)


def partial_continuous(
    amplitude: ArrayLike, duration: ArrayLike
) -> dict[str, ArrayLike]:
    """Convert a partial dictionary while retaining scalar bin metadata."""
    waveform: dict[str, ArrayLike] = {"qmin": 7.0, "A": amplitude}
    return to_continuous_ft(waveform, duration)


def mapping_continuous(
    waveform: Mapping[str, ArrayLike], duration: ArrayLike
) -> dict[str, ArrayLike]:
    """Convert partial channel dictionaries and other read-only mappings."""
    return to_continuous_ft(waveform, duration)
