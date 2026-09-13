# binary-galactica

GLASS fast-slow galactic-binary **LISA TDI waveforms in JAX**. This package
adopts `gbonfly` from `wdm_test`, correcting its raw normalization against
[GLASS commit 079c6a3](https://github.com/nsgs-analysis-center/glass/tree/079c6a3a6fcf380a7c107f39640fff7fc720d0f1).

The pipeline evaluates analytic LISA orbits, sky and arm projections,
light-travel response, frequency and amplitude evolution, a reduced complex
FFT, and TDI synthesis. It returns a narrow frequency band in channels
`X`, `Y`, `Z`, `A`, and `E`. JAX provides compilation, batching, and derivatives
at fixed bandwidth and carrier-bin choices.

## Install

```sh
pip install -e ".[dev]"
```

Python >=3.10, JAX >=0.4.30, and NumPy >=1.24 are required. Importing the package
enables `jax_enable_x64=True` process-wide. Long-duration millihertz phases
require double precision.

## Generate a GLASS waveform

```python
import binary_galactica as bg

T, BW = 7_884_000.0, 512
p = bg.params_from_native(
    f0=1e-3,
    fdot=1e-18,
    amp=7.16e-22,
    phi0=0.5,
    colat=1.78,
    lon=0.0,
    psi=0.7,
    cos_iota=0.3,
    T=T,
)
positions = bg.orbit_cache(T, 0.0, BW)
wave = bg.ucb_waveform(p, T=T, BW=BW, positions=positions)
frequencies = wave["freq_bin_indices"] / T
A, E = wave["A"], wave["E"]  # complex arrays of length BW
continuous = bg.to_continuous_ft(wave, T)
```

`BW` is both the slow-time sample count and the returned frequency-bin count.
It must be a positive even integer. A larger value refines the response and
widens the band. `ucb_alignment(p, T, N)` and `ucb_bw(...)` reproduce the GLASS
bandwidth heuristic, but **choose bandwidth by convergence for your analysis**.
The heuristic is not an accuracy guarantee; `512` is an example, not a universal
setting.

Cached positions depend only on `(T, t0, BW)` and can be shared across sources.
Omit `positions` to evaluate the orbit in the waveform call. Supplied positions
must have shape `(3, BW, 3)` on that exact time grid. All source parameters refer
to time zero; `t0` moves the observing interval, not the parameter reference epoch.

### Native parameters

The waveform entry points use the **GLASS-native** vector:

```text
[f0*T, cos(colat), longitude, ln(A_G), cos(iota), psi, phi0_G,
 fdot*T**2, fddot*T**3]
```

`ucb_waveform` and `template_on_window` accept 7, 8, or 9 entries; the
eighth and ninth chirp terms are optional. `ucb_alignment` requires at least
eight entries because it reads `fdot`. `UCBPrior` and the standard
single-source proposal mix are eight-dimensional.

`params_from_native` is a NumPy host packing helper that returns eight
entries unless `d2fdt2` is supplied. Use JAX operations when constructing
vectors inside a differentiated function.

Supported TDI formats are `"sangria"` (default), `"frequency"`, and `"phase"`.
They reproduce GLASS's conventions, including conjugation of `A/E` in the phase
format. Nominal arm length and transfer frequency are fixed to the pinned
constants. Numerical-orbit interpolation and GLASS's direct wavelet generator
are not included.

### Normalization

`ucb_waveform` returns **raw GLASS Fourier coefficients**:

```text
h_continuous = sqrt(T) * h_GLASS
```

With a compatible one-sided PSD, continuous coefficients use the inner product
`4 * (1/T) * sum(conj(a) * b / PSD)`, taking the real part as appropriate.
For a bare NumPy/JAX `rfft` convention, also divide continuous coefficients by
the time-domain sample interval.

For the **same native vector**, raw output is half the donor `gbonfly` output:
this package keeps GLASS's half inclination factors and uses `1/(2*BW)` for
the FFT. `to_continuous_ft` remains `sqrt(T)`. No conversion factor or SNR target
compensates for that correction. Reusing saved donor templates therefore
requires an explicit convention decision.

## Physical parameters and source polarizations

`GBParams` keeps the physical ordering
`(amp, f0, fdot, phi0, iota, psi, lam, beta)`. Pass it to `tdi_waveform`:

```python
params = bg.GBParams(
    amp=1e-22,
    f0=1e-3,
    fdot=1e-18,
    phi0=0.5,
    iota=1.1,
    psi=0.7,
    lam=0.0,
    beta=-0.2,
)
physical_wave = bg.tdi_waveform(params, T=T, BW=BW)
native = bg.to_glass_vector(params, T)
restored = bg.from_glass_vector(native, T)
```

The physical and native eight-element arrays cannot be distinguished by shape.
Use `tdi_waveform` for physical parameters and `ucb_waveform` for native ones.
The adapter makes their differences explicit:

| Quantity | Physical `GBParams` | GLASS-native |
|---|---|---|
| Amplitude | Full inclination factors times `amp` | `A_G = 2*amp`, with half factors |
| Initial phase | `+phi0` in the phase function | `phi0_G = -phi0` |
| Sky latitude | `beta` | `cos(colat) = sin(beta)` |
| Inclination | `iota` | `cos(iota)` |
| Frequency and chirp | Hz, Hz/s | `f0*T`, `fdot*T**2` |

The inverse accepts only eight parameters and returns canonical inclination
`[0, pi]` and latitude `[-pi/2, pi/2]`. It never silently discards a ninth,
second-derivative parameter.

The original `phase`, `frequency`, `amplitudes`, and `polarizations` helpers
remain available, including imports from `binary_galactica.waveform`. They
generate constant-amplitude barycentric source polarizations. The detector
model additionally applies GLASS's first-order amplitude evolution,
`1 + (2/3)*(fdot/f0)*t`, evaluated at retarded times. Source-only polarizations
are not substitutes for detector channels.

## JAX and fixed analysis windows

```python
import jax
import jax.numpy as jnp

window = bg.template_on_window(
    p,
    T=T,
    BW=BW,
    bin_lo=int(wave["qmin"]),
    n_bins=BW,
)  # (2, BW): A/E in continuous Fourier units


def channels(native, duration):
    result = bg.ucb_waveform(native, T=duration, BW=BW)
    return jnp.stack([result["A"], result["E"]])


jacobian = jax.jacfwd(channels, argnums=0)(jnp.asarray(p), T)
catalogue = jnp.stack([jnp.asarray(p), jnp.asarray(p)])
batch = jax.jit(jax.vmap(channels, in_axes=(0, None)))(catalogue, T)
```

The fixed-window helper zero-fills missing bins and drops bins outside the
requested window. Choose the window wide enough for the analysis.

Keep `T` a **traced argument**, including in outer JIT wrappers. Closing over
a constant can change division into reciprocal multiplication and move a
carrier bin at a floating-point boundary. `BW`, `fmt`, `bin_lo`, and `n_bins`
are static. `ucb_waveform(..., q=...)` can explicitly pin the carrier.

Ordinary `ucb_waveform` and `tdi_waveform` calls validate inputs and raise
`ValueError` for invalid durations, frequencies, angles, carrier overrides,
or orbit arrays. A carrier override must be a finite integer-valued scalar.
Physical `tdi_waveform` inputs use inclination in `[0, pi]` and latitude in
`[-pi/2, pi/2]`; native sky and inclination cosines must lie in `[-1, 1]`.

Inside `jit` or `vmap`, invalid traced values produce NaN outputs, including
bin metadata, because Python exceptions cannot inspect those values during
tracing. Validate catalogue values before a compiled bulk calculation and
check results for finiteness. Static shape and format errors still raise
during tracing. Results retain the existing dictionary keys and are described
by the exported `WaveformResult` type for editor and type-checker support.

Derivatives do not differentiate bandwidth selection or carrier jumps. The
removable sinc singularity is guarded. At sky-coordinate poles, the backend
regularizes singular coordinate derivatives to zero. These intentionally
differ from raw C at singular inputs.

See the [performance guide](docs/performance.md) for repeated-call and catalogue
patterns and a reproducible benchmark of waveform, fixed-window, batch, and
Jacobian workloads. The benchmark separates first-call compilation from warmed
execution and waits for all returned arrays.

## Supporting modules

The donor modules live under `binary_galactica.glass`: constants, orbit,
response, waveform, TDI, bandwidth, alignment, Fourier conversion, fixed-window
placement, priors, and proposals. Priors/proposals use NumPy on the host and
native coordinates. They are building blocks rather than an inference engine:
likelihoods, fitted noise, trans-dimensional searches, and WDM integration
remain outside this package.

The Fisher helper preserves the donor's one-eigendirection draw, but uses two
safer guards: non-finite draws and correlated jumps reaching either cosine
boundary become whole-vector self-proposals. This intentionally differs from
GLASS's deterministic one-percent fallback and component-only boundary revert;
it does not affect waveform parity. Its zero proposal correction is valid for
a Fisher decomposition fixed independently of chain state. State-dependent
cache refresh remains adaptation; freeze covariance for production sampling
or separately justify the adaptive scheme or reverse-density correction.

For a fixed covariance, a proposal loop can use the root-level helpers:

```python
import numpy as np

prior = bg.ucb_prior(T=T, fmin=0.0009, fmax=0.0011)
rng = np.random.default_rng(12)
state = prior.sample(rng)
# Illustrative fixed scales: use a measured Fisher decomposition in an analysis.
fixed_eigenvalues = 1 / (0.01 * (prior.hi - prior.lo)) ** 2
fixed_eigenvectors = np.eye(8)
mix = bg.single_source_mix(
    prior=prior,
    fisher=lambda _: (fixed_eigenvalues, fixed_eigenvectors),
    T=T,
)
candidate, log_q_ratio, proposal_name = mix(state, rng)
candidate = prior.wrap(candidate)
log_prior = prior.log_prior(candidate)
# Use log_prior and log_q_ratio with your likelihood and acceptance rule.
```

`FisherCache` counts **Fisher calls**, not total mixture steps. With the
default mixture, only about 70% of steps use Fisher proposals, so refreshing
every 80 calls does not reproduce GLASS's refresh every 80 total sampler
steps. The example's fixed decomposition avoids state-dependent adaptation.

## Validation

```sh
pytest
ruff check .
python examples/quickstart.py
```

Mandatory C reference tests cover varied sky positions, inclinations,
amplitudes, phases, 7/8/9 parameters, all three formats, signed chirps,
quarter-year through four-year observations, and carrier boundaries.
Each of the five channels must agree to relative L2 error `1e-9`. Normal tests
need no C compiler, network, or sibling repository, and missing references fail.

[Provenance and offline regeneration](tools/glass_reference/README.md) document
the pinned, unchanged upstream routines and vendored FFT source. Additional
tests cover conversions, normalized derivatives, batching, singular guards,
window placement, and donor helper behavior. These checks establish agreement
with the pinned finite-band analytic-orbit calculation, not convergence for
every source or a full GLASS pipeline.

[CI](.github/workflows/ci.yml) runs the suite on Python 3.10 with the
[minimum runtime dependencies](requirements/minimum.txt), Python 3.11–3.14
with current compatible dependencies, and macOS with Python 3.12. It also
checks lint, formatting, distribution metadata, and the installed wheel from
outside the checkout. The workflow runs on pushes and pull requests.

To reproduce the minimum-dependency check in a clean Python 3.10 environment:

```sh
python -m pip install -c requirements/minimum.txt -e ".[dev]"
python -m pytest
```

The [quality review follow-up](docs/quality-review.md) records executed
compatibility checks, resolved findings, and remaining release verification.

The [TDI zero-crossing prototype](tools/tdi_zero_crossings/README.md) compares
bounded Brent minimization with projected root finders for a callable complex
response. It includes separate tests and reproducible benchmarks; it is an
experiment for TDI-on-the-fly integration and does not alter this GLASS backend.

## Attribution

The fast-slow construction follows Cornish & Littenberg,
[Phys. Rev. D 76, 083006 (2007)](https://doi.org/10.1103/PhysRevD.76.083006).
GLASS is by Tyson B. Littenberg, Neil J. Cornish, and contributors.
See [NOTICE](NOTICE) and the included MIT, Apache-2.0, and vendored KissFFT
license notices.
