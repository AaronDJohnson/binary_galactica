"""Generate a galactic-binary waveform and its exact parameter Jacobian.

Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from binary_galactica import (
    DAY,
    PARAMETER_NAMES,
    YEAR,
    GBParams,
    frequency,
    polarizations,
)

params = GBParams(
    amp=1e-22,
    f0=1.2345e-3,
    fdot=3.21e-16,
    phi0=0.7,
    iota=1.1,
    psi=0.4,
    lam=4.1,
    beta=-0.3,
)

# --- a waveform -----------------------------------------------------------
t = jnp.arange(0.0, 4.0 * YEAR, 15.0)
hp, hc = jax.jit(polarizations)(params, t)

print(f"{len(t):,} samples over {t[-1] / YEAR:.2f} yr at 15 s cadence")
print(f"h+ peak-to-peak : {hp.max() - hp.min():.3e}")
print(f"hx peak-to-peak : {hc.max() - hc.min():.3e}")
print(f"frequency drift : {1e9 * (frequency(params, t[-1]) - params.f0):.4f} nHz")

# --- exact derivatives ----------------------------------------------------
# jacfwd differentiates the model analytically, so these columns carry no
# finite-difference error -- which is what a Fisher matrix needs.
short = jnp.arange(0.0, 30.0 * DAY, 15.0)
jac_hp, _ = jax.jacfwd(polarizations)(params.to_array(), short)

print(f"\nJacobian of h+ : {jac_hp.shape}  (samples, parameters)")
for name, column in zip(PARAMETER_NAMES, jac_hp.T, strict=True):
    print(f"  d h+ / d {name:<5} max |.| = {jnp.abs(column).max():.3e}")
print("\nlam and beta are exactly zero: sky position lives in the response.")

# --- a catalogue ----------------------------------------------------------
# vmap maps a (K, 8) array of sources onto K waveforms.
catalogue = jnp.stack(
    [params._replace(f0=params.f0 + shift).to_array() for shift in (0.0, 1e-5, 2e-5)]
)
batch_hp, batch_hc = jax.vmap(polarizations, in_axes=(0, None))(catalogue, short)
print(f"\nCatalogue of {catalogue.shape[0]} sources -> h+ shape {batch_hp.shape}")
