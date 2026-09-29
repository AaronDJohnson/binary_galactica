"""Float32 A/E channels and inner products with JAX double precision disabled.

``binary_galactica.single`` serves processes that keep ``jax_enable_x64`` off.
Host NumPy code such as priors, proposals and saved catalogues stays in
float64; ``split_native`` turns each source into an integer reference bin
``q_ref`` and float32 coordinates whose first entry is the offset
``f0*T - q_ref``. Raw strains square below float32's range, so inner products
use whitened channels, or templates rescaled through ``params[3]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import binary_galactica as bg
from binary_galactica import single as bg32

if jax.config.read("jax_enable_x64"):
    raise SystemExit("Run this example with jax_enable_x64 disabled (the default).")

duration, bandwidth = 125_829_120.0, 512  # four GLASS binary years, in seconds
# A float64 NumPy vector, as a prior draw or catalogue entry would be.
native = bg.params_from_native(
    f0=4.1e-3,
    fdot=2e-17,
    amp=2e-22,
    phi0=0.5,
    colat=1.78,
    lon=0.3,
    psi=0.7,
    cos_iota=0.3,
    T=duration,
)
try:
    bg.ucb_waveform(native, T=duration, BW=bandwidth)
except RuntimeError:
    print("The float64 bg.ucb_waveform raises RuntimeError without x64")

# Rounding f0*T to float32 would shift the phase by 2*pi*drift at the end.
drift = float(np.float32(native[0])) - native[0]
print(f"f0*T = {native[0]:.4f} cycles; in float32 it moves {drift:+.4f} cycles")
q_ref, params = bg32.split_native(native, duration)
print(f"q_ref = {q_ref}, offset = {params[0]:.6f} cycles, parameters {params.dtype}")

wave = bg32.ucb_waveform(params, q_ref=q_ref, T=duration, BW=bandwidth)
bins = wave["freq_bin_indices"]  # int32, exact
print(f"A/E: {wave['A'].dtype}, bins {int(bins[0])}-{int(bins[-1])} ({bins.dtype})")

# Raw channels near 1e-20 are representable, but their squares are not.
raw = float(jnp.sum(jnp.abs(wave["A"]) ** 2))
# The waveform is linear in amplitude: lowering params[3] by log(scale)
# divides every channel by scale.
scale = 1e-21
rescaled = params.copy()
rescaled[3] -= np.log(scale)
scaled = bg32.ucb_waveform(rescaled, q_ref=q_ref, T=duration, BW=bandwidth)
norm = scale * float(jnp.linalg.norm(scaled["A"]))
host = float(np.linalg.norm(np.asarray(wave["A"], dtype=np.complex128)))
print(f"float32 sum |A|^2 of raw values: {raw:.3e} (underflow)")
print(f"|A| from the rescaled template: {norm:.6e}; host float64: {host:.6e}")

# Noise weighting over a fixed window. The one-sided A/E PSD near 5e-42 is
# itself below float32's range, so form sqrt(T * PSD) on the host in float64.
# GLASS's band-sizing model bg.ae_noise stands in for an analysis PSD.
bin_lo, n_bins = int(bins[0]), bandwidth
psd = bg.ae_noise((bin_lo + np.arange(n_bins)) / duration)
whitening = np.sqrt(duration * psd).astype(np.float32)


@jax.jit
def whitened(p):
    """Return continuous-unit A/E on the window, divided by sqrt(T * PSD)."""
    template = bg32.template_on_window(
        p, q_ref=q_ref, T=duration, bin_lo=bin_lo, n_bins=n_bins, BW=bandwidth
    )
    return template / whitening


def inner(a, b):
    """Return the noise-weighted inner product ``(a|b)`` of whitened A/E."""
    return 4 * jnp.vdot(a, b).real


data = whitened(params)
channels = np.stack([wave["A"], wave["E"]]).astype(np.complex128)
reference = 4 * np.sum(np.abs(channels) ** 2 / psd)
print(f"Whitened float32 (h|h) = {float(inner(data, data)):.6e}")
print(f"Host float64 (h|h) from the raw channels = {reference:.6e}")


def shifted(offset):
    """Return the whitened template with the frequency offset replaced."""
    return whitened(jnp.asarray(params).at[0].set(offset))


# Match against templates up to 0.02 bins away, in one compiled float32 call.
offsets = params[0] + np.linspace(-0.02, 0.02, 5, dtype=np.float32)
templates = jax.jit(jax.vmap(shifted))(offsets)
overlaps = jax.vmap(inner, in_axes=(None, 0))(data, templates)
norms = jax.vmap(inner)(templates, templates)
matches = overlaps / jnp.sqrt(inner(data, data) * norms)
for offset, match in zip(offsets - params[0], matches, strict=True):
    print(f"  offset {offset:+.3f} bins: match {float(match):.6f}")
