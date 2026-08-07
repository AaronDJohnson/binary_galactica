# Galactic-binary waveform model — design

**Date:** 2026-08-07
**Package:** `binary-galactica`
**Status:** implemented

## Purpose

A publication-quality implementation of the standard galactic-binary
gravitational waveform model over the 8D parameter space
`(amp, f0, fdot, phi0, iota, psi, lam, beta)`. Simple, readable, and correct
enough to underpin published results.

## Scope

**In:** the source model — the polarizations $h_+$ and $h_\times$ at the
solar-system barycentre.

**Out:** everything instrumental. LISA orbits, arm projection, light-travel
delays, TDI, and the sky-position geometry (propagation direction and
polarization basis) all belong to the response package downstream.

**Out, deliberately:** $\ddot{f}$. The parameter space is the standard 8D one.
More elaborate source models (eccentricity, third bodies, tides) are expected
later and are the reason the package is organised around a single
`polarizations` entry point rather than a monolithic generator.

### Consequence: `lam` and `beta` are inert here

Sky position does not enter $h_+$ or $h_\times$; it enters the response.
`GBParams` still carries `lam` and `beta` because they are part of the sampling
space and the response needs them, but the corresponding Jacobian columns are
identically zero. This is stated in the docstrings and asserted in the tests
rather than left as a surprise.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | polarizations only | Waveform model; instrument is a separate package |
| Backend | JAX | Exact gradients for Fisher/HMC, `jit`, `vmap` over catalogues |
| Precision | x64 enabled at import | float32 gives ~0.06 rad phase error over 4 yr — invisible and fatal |
| Batching | `vmap`, not built-in broadcasting | One code path; idiomatic JAX |
| Reference epoch | `t = 0` of the supplied array | No hidden state |
| Conventions | match `ldasoft` / `GBGPU` | Output must be safe to hand to existing response codes |

### On enabling x64 at import

Over four years the phase of a millihertz binary reaches $\sim10^6$ rad. In
float32 that carries $\sim0.06$ rad of error — enough to destroy a matched
filter, far too small to see in a plot. A model that is silently wrong by
default is a worse outcome than a documented process-global setting, so
`binary_galactica/__init__.py` sets `jax_enable_x64=True` before importing
anything else. Documented prominently in the README.

In float64 the residual phase error is $\sim10^{-9}$ rad, so no compensated
summation or phase-reduction tricks are needed.

## Architecture

```
src/binary_galactica/
├── __init__.py       public API; enables x64
├── constants.py      YEAR (sidereal), DAY
├── parameters.py     GBParams  <->  (8,) array
└── waveform.py       phase, frequency, amplitudes, polarizations
```

Four modules, each with one job. `waveform.py` holds the entire physics in
about forty lines of code.

`GBParams` is a `NamedTuple`, hence automatically a JAX pytree, so it passes
through `jit`/`vmap`/`grad` exactly like the bare array. `from_array` accepts
either representation, so callers never have to choose.

## The model

$$\Phi(t) = \phi_0 + 2\pi f_0 t + \pi \dot{f} t^2, \qquad f(t) = f_0 + \dot{f} t$$

$$\mathcal{A}_+ = \mathcal{A}(1+\cos^2\iota), \qquad \mathcal{A}_\times = -2\mathcal{A}\cos\iota$$

$$h_+^{\rm src} = \mathcal{A}_+\cos\Phi, \qquad h_\times^{\rm src} = \mathcal{A}_\times\sin\Phi$$

$$h_+ = h_+^{\rm src}\cos 2\psi + h_\times^{\rm src}\sin 2\psi, \qquad
  h_\times = -h_+^{\rm src}\sin 2\psi + h_\times^{\rm src}\cos 2\psi$$

The $\psi$-rotation sign was taken from the source rather than from memory:
`ldasoft`'s `galactic.c` and `GBGPU`'s `run_wave` both build the response from

```
DPr =  A+ cos2psi        DPi = -Ax sin2psi
DCr = -A+ sin2psi        DCi = -Ax cos2psi
```

against a wave written $e^{-i\Phi}$, giving the rotation above — the transpose
of the naive guess.

## Testing

Analytic identities rather than stored waveforms, so the suite documents the
model as well as checking it.

- **Cross-check:** an independently written NumPy implementation in
  `conftest.py`, using a complex exponential and an explicit rotation matrix.
- **Conventions** (`test_conventions.py`): reproduces `ldasoft`'s
  `DPr/DPi/DCr/DCi` coefficients; pins the sign of $\mathcal{A}_\times$ and the
  direction of the $2\psi$ rotation; pins the $\pi\dot f t^2$ factor. These are
  the tests that make the output safe for a response code — every other
  property would be satisfied equally well by $\psi \to -\psi$.
- **Physics** (`test_waveform.py`): monochromatic limit; chirp linearity;
  face-on circular and edge-on linear polarization; amplitude linearity;
  $\psi$-rotation orthogonality and period $\pi$; phase offset ≡ time shift.
- **Transforms** (`test_transforms.py`): `jit` matches eager; `vmap` matches a
  loop; `jacfwd` matches central differences on all eight parameters; sky
  columns exactly zero; `grad` through a scalar functional, checked against
  Euler's theorem in `amp`.
- **Precision** (`test_precision.py`): x64 active; outputs float64; phase
  accurate to $<10^{-6}$ rad over four years against 50-digit decimal
  arithmetic.

## Future work

Additional source models slot in beside `polarizations` with the same
signature: $\ddot{f}$, eccentric binaries, third-body Doppler modulation.
