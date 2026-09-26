"""Generate default on-fly TDI channels, a fixed window, and a Jacobian.

``bandwidth`` controls returned FFT bins; ``response_samples`` independently
controls the sparse response grid. The analytic orbit is used by default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import binary_galactica as bg

duration, bandwidth, response_samples = 7_884_000.0, 512, None
native = bg.params_from_native(
    f0=1e-3,
    fdot=1e-18,
    amp=7.16e-22,
    phi0=0.5,
    colat=1.78,
    lon=0.0,
    psi=0.7,
    cos_iota=0.3,
    T=duration,
)
wave = bg.ucb_waveform(
    native, T=duration, BW=bandwidth, response_samples=response_samples
)
continuous = bg.to_continuous_ft(wave, duration)
bins = wave["freq_bin_indices"]
print(f"{bandwidth} bins from {int(bins[0])} to {int(bins[-1])}")
print(
    f"Frequency band: {float(bins[0] / duration):.7f}–"
    f"{float(bins[-1] / duration):.7f} Hz"
)
print(f"Peak continuous |A|: {float(jnp.max(jnp.abs(continuous['A']))):.6e}")

window = bg.template_on_window(
    native,
    T=duration,
    BW=bandwidth,
    response_samples=response_samples,
    bin_lo=int(bins[0]),
    n_bins=bandwidth,
)
print(f"Fixed A/E analysis window: {window.shape}")


def channels(params, observation_time):
    """Keep duration traced in JIT and differentiation wrappers."""
    result = bg.ucb_waveform(
        params,
        T=observation_time,
        BW=bandwidth,
        response_samples=response_samples,
    )
    return jnp.stack([result["A"], result["E"]])


jacobian = jax.jacfwd(channels, argnums=0)(jnp.asarray(native), duration)
print(f"Native-parameter Jacobian: {jacobian.shape} (channel, bin, parameter)")
catalogue = jnp.stack([native, native])
batch = jax.jit(jax.vmap(channels, in_axes=(0, None)))(catalogue, duration)
print(f"Batched channels: {batch.shape}")
physical = bg.from_glass_vector(native, duration)
physical_wave = bg.tdi_waveform(
    physical, T=duration, BW=bandwidth, response_samples=response_samples
)
relative = jnp.linalg.norm(physical_wave["A"] - wave["A"]) / jnp.linalg.norm(wave["A"])
print(f"Physical/native adapter relative difference: {float(relative):.3e}")
