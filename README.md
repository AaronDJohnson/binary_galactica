# binary-galactica

**LISA TDI waveforms for galactic binaries, in JAX.** The default `onfly`
backend evaluates demodulated TDI `X/Y/Z` directly at sparse reception times,
interpolates the response onto the FFT times (or evaluates it there when the
sparse grid is at least as fine), restores the analytic source phase, and
transforms the channels. It follows GLASS's `LISA_spline_response` /
`LISA_TDI_spline` conventions and returns a narrow band in `X`, `Y`, `Z`, `A`,
and `E`. JAX provides compilation, batching, and derivatives at fixed discrete
settings.

The response uses analytic LISA positions by default, with instantaneous arm
lengths and geometry frozen at each reception time. This is the first-generation
frozen-geometry response used by that GLASS path; it does not implement TDI-2
or moving-spacecraft sequential delays. The prior link-FFT implementation is
available as the explicit legacy `backend="fourier"` option for reference
comparisons.

The on-fly spline response and legacy fast-slow link response use reverse
directed-link placement differently. Their finite-frequency spectra therefore
are not expected to agree, even for static equal arms; they agree in the
low-frequency limit. Independent pinned-spline C fixtures validate the new
response, while legacy reference tests remain under explicit
`backend="fourier"`. See the [on-fly reference provenance and regeneration
guide](tools/onfly_reference/README.md).

**Credit.** binary-galactica is a JAX translation of the galactic-binary
waveform and TDI response code in [GLASS](https://github.com/nsgs-analysis-center/glass),
originally written by Neil J. Cornish and Tyson B. Littenberg. The default
`onfly` backend implements their method from N. J. Cornish and T. B. Littenberg,
["TDI on the fly," arXiv:2506.08093 (2025)](https://arxiv.org/abs/2506.08093);
the legacy `fourier` backend implements their fast–slow decomposition,
[Phys. Rev. D 76, 083006 (2007)](https://doi.org/10.1103/PhysRevD.76.083006).
If you use this package, please cite their paper and GLASS as well (see
[Citing](#citing)).

## Install

```sh
python -m pip install binary-galactica
```

Or from a source checkout:

```sh
git clone https://github.com/AaronDJohnson/binary_galactica.git
cd binary_galactica
python -m pip install .
```

For development, use an editable installation with the test and lint tools:

```sh
python -m pip install -e ".[dev]"
```

Python >=3.10, JAX >=0.6.0, and NumPy >=1.25 are required.

**Migrating to 0.6.0:** JAX >=0.6.0 and NumPy >=1.25 are now required. The
unused `binary_galactica.glass.interpolation` module (a Hermite comparison
utility) is removed. Float32 waveforms whose parameters include `fddot` now
compile in about 0.3 s instead of 4–9 s; outputs are unchanged bit for bit.

**Migrating to 0.5.0:** the GLASS-native amplitude `exp(params[3])` is now the
physical strain amplitude, the LDC/Sangria catalogue `Amplitude`. Through 0.4.0
every native output was half the physical signal: `ucb_waveform`,
`template_on_window`, and `tdi_response`, on both backends and in
`binary_galactica.single`. For the same vector they now return exactly twice
their 0.4.0 values and match Sangria data. A 0.4.0 native vector `v` gives the
same signal in 0.5.0 with `v[3] - ln(2)`. The physical `GBParams` API is
unchanged: `tdi_waveform` returns the same signal, while `to_glass_vector` now
writes `ln(amp)` rather than `ln(2*amp)` and `from_glass_vector` reads
`amp = exp(v[3])`. Native callers change meaning by that factor of two, for
example `petra_catalogs` (`petra/binary_galactica.py`) with
`parameterization="glass"`. Saved native templates, SNRs, and amplitude
posteriors need an explicit decision. See the
[physical-amplitude spec](docs/superpowers/specs/2026-10-11-physical-amplitude.md).

**Migrating to 0.4.0:** importing the package no longer enables
`jax_enable_x64`. The float64 functions need it, because long-duration
millihertz phases require double precision. Enable it yourself at startup,
before creating JAX arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

While it is disabled, `ucb_waveform`, `tdi_waveform`, `tdi_response`,
`template_on_window`, `orbit_cache`, `phase`, `frequency`, `amplitudes`, and
`polarizations` raise `RuntimeError` instead of computing in float32.
Lower-level building blocks such as `analytic_orbits`, `slow_time_grid`,
`to_glass_vector`, and `from_glass_vector` now follow their input dtype:
float32 inputs give float32 results even with x64 enabled. Processes that keep
x64 disabled can use the float32 functions in
[`binary_galactica.single`](#single-precision).

## Generate a GLASS waveform

```python
import jax

import binary_galactica as bg

jax.config.update("jax_enable_x64", True)

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
wave = bg.ucb_waveform(p, T=T, BW=BW)  # backend="onfly", automatic response grid
frequencies = wave["freq_bin_indices"] / T
A, E = wave["A"], wave["E"]  # complex arrays of length BW
continuous = bg.to_continuous_ft(wave, T)
```

`BW` is the positive, even number of returned frequency bins. It controls the
FFT band, while `response_samples` independently controls the sparse response.
The default `None` selects a power of two from 32 through 2048, targeting at
least 128 samples per observation year: 32 for a quarter year, 128 for a year,
and 512 for four years. Set an integer of at least four to fix the grid.
A count below `BW` is interpolated onto the `BW` FFT times. A count at or
above `BW` would only be interpolated onto fewer points, so the response is
evaluated directly at the `BW` FFT times instead: every such count gives the
same interpolation-free result, at lower cost. A four-year observation with
`BW=128`, for example, evaluates the response at its 128 FFT times rather than
interpolating 512 samples onto them.
With a concrete `T`, public calls choose the automatic count on the host
before compiling; see [JAX and fixed analysis windows](#jax-and-fixed-analysis-windows)
for a traced `T`.
The automatic policy is a starting point, not an error bound; observations
beyond 16 years particularly need an explicit convergence check.
Choose both grids by convergence for the
analysis: increasing `BW` widens the band and does not refine the on-fly
response; increasing `response_samples` refines that interpolation, up to
`BW`, and does not change the number of returned bins.
`ucb_alignment(p, T, N)` and `ucb_bw(...)` retain the GLASS bandwidth
heuristic, but it is not an accuracy guarantee.

The sparse response removes the phase at spacecraft 0's retarded source time,
including the common orbital Doppler phase. Local polynomials through up to
eight complex samples interpolate the remaining smooth response. This avoids
time-derivative evaluations and singular amplitude/phase coordinates at zeros.
The reference phase is restored at the FFT times.

For correctness, omit `positions` and use the analytic orbit. Legacy position
caches with shape `(3, BW, 3)` remain accepted. With fewer response samples
than `BW` they are interpolated cubically to the response times, and a coarse
cache limits orbit accuracy; otherwise the cache is used at its own times,
and an `orbit_cache` reproduces the analytic orbit. All source parameters
refer to time zero; `t0` moves the observing interval, not the parameter
reference epoch.

### Native parameters

The waveform entry points use the **GLASS-native** vector:

```text
[f0*T, cos(colat), longitude, ln(A_G), cos(iota), psi, phi0_G,
 fdot*T**2, fddot*T**3]
```

`A_G` is the physical strain amplitude: the LDC/Sangria catalogue
`Amplitude`, and the value GLASS's `amplitude()` returns. On both backends the
polarizations are `h+ = A_G*(1 + cos(iota)**2)` and `hx = 2*A_G*cos(iota)`.

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
constants. `backend="fourier"` retains the old finite-band link-FFT response;
it is useful for exact legacy-reference comparisons.

### Arbitrary-time response

`tdi_response` exposes the same arbitrary-time native response in its documented
source-phase-demodulated convention:

```python
import jax.numpy as jnp

times = T * jnp.array([0.0, 0.25, 0.5, 0.75])
xyz = bg.tdi_response(p, times, T=T)  # complex array, shape (3, 4)
```

It is in Sangria analytic-response units: it is the raw internal GLASS spline
response, before FFT normalization. After the source phase is restored, its
real part is the Sangria fractional-frequency TDI signal. The waveform restores
the full analytic source phase before transforming, and its raw spectrum retains
the usual `sqrt(T)` conversion described below. `times` is one-dimensional.
An optional position array has shape `(3, n_times, 3)`; use analytic positions
unless an existing cache must be reused.

### Normalization

`ucb_waveform` returns **raw GLASS Fourier coefficients**:

```text
h_continuous = sqrt(T) * h_GLASS
```

With a compatible one-sided PSD, continuous coefficients use the inner product
`4 * (1/T) * sum(conj(a) * b / PSD)`, taking the real part as appropriate.
For a bare NumPy/JAX `rfft` convention, also divide continuous coefficients by
the time-domain sample interval.

The on-fly Fourier prefactor is `sqrt(T) / (2*BW)` for the raw GLASS spline
response, which the public `tdi_response` returns unscaled. The factor of one
half selects the positive-frequency contribution of the real time signal.
`to_continuous_ft` remains `sqrt(T)`. With these conventions, continuous
coefficients divided by the sample interval match the in-band `rfft` of
Sangria/LDC time-domain TDI generated from the same catalogue `Amplitude`. No
conversion factor or SNR target compensates for a convention change; reusing
saved templates requires an explicit convention decision.

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
physical_wave = bg.tdi_waveform(params, T=T, BW=BW, response_samples=512)
native = bg.to_glass_vector(params, T)
restored = bg.from_glass_vector(native, T)
```

The physical and native eight-element arrays cannot be distinguished by shape.
Use `tdi_waveform` for physical parameters and `ucb_waveform` for native ones.
The adapter makes their differences explicit:

| Quantity | Physical `GBParams` | GLASS-native |
|---|---|---|
| Amplitude | Full inclination factors times `amp` | `A_G = amp` (LDC `Amplitude`), same full factors |
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
batch = jax.jit(jax.vmap(lambda native: channels(native, T)))(catalogue)
```

The fixed-window helper zero-fills missing bins and drops bins outside the
requested window. Choose the window wide enough for the analysis. `bin_lo`
must be an integer and `n_bins` a non-negative integer; fractional values,
booleans, and nonfinite values raise `ValueError`. A zero-length window returns
an empty `(2, 0)` array.

`T` may be a traced argument or a constant captured by an outer JIT wrapper.
The carrier calculation preserves GLASS's float64 rounding in either case.
Pass `T` as an argument when varying or differentiating the duration without
recompiling. `BW`, `response_samples`, `backend`, `fmt`, `bin_lo`, and `n_bins`
are static.
`ucb_waveform(..., q=...)` can explicitly pin the carrier.
Automatic response counts change discretely with duration. A captured `T`, a
Python or NumPy scalar or a concrete JAX array, lets each public call choose
the count on the host and compile one grid. A traced `T` cannot: every
automatic count below `BW` (of 32 through 2048), and one direct evaluation for
the rest, is compiled behind `jax.lax.switch`, reverse-mode derivatives keep
the intermediate values of every one, and the call emits
`ResponseSamplesWarning`. With `BW` of at most 32 every automatic count is
direct, so a traced `T` compiles a single grid and does not warn. Pass an
explicit `response_samples` whenever `T` itself is a `jit`, `grad` or `vmap`
argument. AD follows the selected grid;
use an explicit `response_samples` when differentiating across a duration
threshold. For batches with different durations, grouping by duration or using
an explicit count avoids evaluating multiple automatic grid branches.

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

## Single precision

`binary_galactica.single` is a float32 implementation of the on-fly waveform
for processes that keep `jax_enable_x64` disabled, such as float32 analysis
pipelines or float32-only accelerators. Its `ucb_waveform`,
`template_on_window`, `tdi_response`, and `tdi_waveform` mirror the float64
functions, use only the on-fly response, and compute and return only float32,
complex64, and int32 values, with x64 enabled or not. The float64 functions
remain the reference implementation.

Float32 cannot hold `f0*T`, which reaches millions of cycles. Each source is
therefore an integer reference bin `q_ref` and float32 coordinates whose
frequency entry is the offset from that bin: native `params[0]` is
`f0*T - q_ref`, and the physical `tdi_waveform` frequency field is
`f0 - q_ref/T` in hertz. The host NumPy helpers `split_native` and
`split_physical` convert float64 priors, proposals, and catalogues, choosing
`q_ref = floor(f0*T)`:

```python
import binary_galactica as bg
from binary_galactica import single as bg32

T, BW = 7_884_000.0, 512
native = bg.params_from_native(  # float64 NumPy, as from a prior or catalogue
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
q_ref, params = bg32.split_native(native, T)  # Python int, float32 array
wave32 = bg32.ucb_waveform(params, q_ref=q_ref, T=T, BW=BW)
A32, E32 = wave32["A"], wave32["E"]  # complex64, raw GLASS units
bins32 = wave32["freq_bin_indices"]  # int32
```

`q_ref` is a Python integer or an int32 scalar. It may be traced, or batched
with `jax.vmap` alongside the parameters; inside `jit`, pass `numpy.int32`
values, because with x64 enabled a traced Python integer is int64 and is
rejected rather than wrapped. The carrier defaults to
`q_ref + floor(params[0])`. The float64 carrier reproduces GLASS rounding and
can differ by one bin at an exact boundary, so pin it with `q=` when comparing
the two. `qmin` and `freq_bin_indices` are exact int32; the other keys and the
format conventions match the float64 results. Eager invalid inputs raise
`ValueError`; traced invalid values give NaN channels and `INT32_MIN` bins.

Against float64 at identical inputs, the
[recorded diagnostic](tools/performance/single-precision.json) measures A/E
relative L2 errors of at most 1.2e-5 and mismatches of at most 7.7e-11, for
targets of 1e-4 and 1e-8. It covers the corners of the validated range: T of
0.25 and 10 years, f0 of 0.1 and 30 mHz, and t0 of 0 and 10 years, with
sources drifting four bins through `fdot` and one through `fddot` in every
sign combination, all three formats, and runs with and without position
caches. Stress cases add chirps driven by gravitational radiation, with chirp
masses of 0.3 and 1 solar mass and `fdot*T**2` of 94 to 20,000 bins,
carriers pinned mid-observation up to 30,525 bins from the default, and a
carrier 1,000 bins from a steady source. The physical adapter reaches 1.1e-5
and the arbitrary-time response 9.2e-6. Rounding `f0*T` itself to float32
instead gives a relative L2 error of 1.9 at ten years and 30 mHz in the same
record. These are measurements, not error bounds; `tools/diagnose_single.py`
reproduces them.

Only the frequency is split. The other float32 entries are inputs like any
other, so converting a float64 source rounds them once, and for a strong chirp
that input rounding can exceed the arithmetic error. Rounding `fdot*T**2`
moves the phase by up to about 6e-8 of its value times `(t/T)**2/2` cycles,
so the effect grows with the drift and at late epochs. The same record splits
its stress chirps from unrounded float64 values and compares them with
float64 at those values: `split_native` gave relative L2 errors of 4.7e-5 and
6.2e-5 at `t0` = 0, and 2.9e-3 and 5.2e-3 at `t0` = 10 years;
`split_physical`, run at `t0` = 0 only, gave up to 6.6e-5. Parameters
referred to a time near the observation, with a small `t0`, avoid the
late-epoch growth.

Raw channels near 1e-21 are representable in float32, but their squares, near
1e-42, are below its smallest normal number, about 1.2e-38. So is a
millihertz A/E noise PSD: `ae_noise` gives 5e-42 at 4 mHz. Form inner products
from whitened channels, dividing by an amplitude spectral density computed on
the host, or rescale: the waveform is linear in amplitude, so subtracting
`log(s)` from native `params[3]`, or dividing physical `amp` by `s`, scales
every channel by `1/s`. `tdi_response` values near 1e-24 need the same care.
[`examples/single_precision.py`](examples/single_precision.py) shows both
approaches with x64 disabled.

Position caches keep the `(3, BW, 3)` layout on the `(T, t0, BW)` grid. Host
float64 or float32 caches are rounded to float32, whose step near 1.5e11 m is
about 5e-5 s of light time. `orbit_cache` is a float64 function and raises
in an x64-disabled process, which can instead load a saved cache. Omitting
`positions` evaluates the analytic orbit, with its phase reduced in paired
float32 arithmetic.

On the recorded CPU benchmark, float32 calls took 0.62 to 1.23 times as long
as float64 ones: similar cost, not a large speedup. See the
[performance guide](docs/performance.md#single-precision-timing--september-29-2026).
Accelerator timings have not been measured.

## Supporting modules

The modules under `binary_galactica.glass` provide constants, orbit, response,
on-fly interpolation, waveform generation, TDI conversion, bandwidth,
alignment, Fourier conversion, fixed-window placement, priors, and proposals.
Priors/proposals use NumPy on the host and native coordinates. They are building
blocks rather than an inference engine:
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
pytest -m "not slow" -n auto   # fast tier, run on every push (~40 s)
pytest -n auto                 # full suite, adds the slow validation tier
ruff check .
python examples/quickstart.py
python examples/single_precision.py
```

The test configuration enables x64 for the session. A float64 golden archive,
recorded after the 0.5.0 physical-amplitude fix, must reproduce to 1e-12 of
each output's peak on other builds. On its recording JAX, NumPy, and CPU build,
outputs of the on-fly response kernel must agree to 1e-13 of peak, because
host-resolved automatic grids, a reorganized kernel and A/E formed before the
transform in `template_on_window` have since moved their last bits, and every
other output must reproduce bitwise. Two cases whose automatic response grid
is finer than their band now evaluate the response directly at the FFT times,
so on any build they differ from the recording by its interpolation error and
must agree to 5e-10 of peak. The
float32 tests compare the `single` waveform, window, arbitrary-time response,
and physical adapter with float64 at identical inputs, check that no float64
value enters their compiled programs, and require a subprocess with x64
disabled to reproduce the x64-enabled float32 outputs bit for bit. The slow
tier checks float32 accuracy at the corners of the supported range (0.25–10
years, 0.1–30 mHz, epochs 0 and 10 years), float32 Jacobians, the automatic
response grid, and a chirp drifting thousands of bins.

Legacy Fourier C reference tests cover varied sky positions, inclinations,
amplitudes, phases, 7/8/9 parameters, all three formats, signed chirps,
quarter-year through four-year observations, and carrier boundaries.
Each of the five channels must agree to relative L2 error `1e-9` with twice the
stored spectra, whose pinned polarization factors are halved. Independently of
those fixtures, `tests/test_physical_amplitude.py` pins the physical amplitude
through the physical API and a long-wavelength TDI limit. Normal tests
need no C compiler, network, or sibling repository, and missing references fail.

Separate pinned-spline fixtures cover the arbitrary-time on-fly XYZ response;
their [offline provenance and regeneration instructions](tools/onfly_reference/README.md)
keep that validation independent of the Python implementation. Legacy
finite-band reference checks select `backend="fourier"` explicitly.

[Provenance and offline regeneration](tools/glass_reference/README.md) document
the pinned, unchanged upstream routines and vendored FFT source. Additional
tests cover conversions, normalized derivatives, batching, singular guards,
window placement, and donor helper behavior. These checks establish agreement
with the pinned finite-band analytic-orbit calculation, not convergence for
every source or a full GLASS pipeline.

[CI](.github/workflows/ci.yml) runs the fast tier on every pull request and
push to `main`, on Python 3.10 with the
[minimum runtime dependencies](requirements/minimum.txt), Python 3.11–3.14
with current compatible dependencies, and macOS with Python 3.12. It also
checks lint, formatting, distribution metadata, and the installed wheel from
outside the checkout. The [full suite](.github/workflows/full.yml) runs on
`main`, nightly, and before every PyPI upload.

To reproduce the minimum-dependency check in a clean Python 3.10 environment:

```sh
python -m pip install -c requirements/minimum.txt -e ".[dev]"
python -m pytest
```

The [quality review follow-up](docs/quality-review.md) records executed
compatibility checks, resolved findings, and remaining release verification.

The [TDI zero-crossing prototype](tools/tdi_zero_crossings/README.md) compares
bounded Brent minimization with projected root finders for a callable complex
response. Its findings informed the response interpolation; it remains a
separate reproducible experiment.

## Attribution

The original C implementation of both waveforms is GLASS, by Tyson B.
Littenberg, Neil J. Cornish, and contributors; this package translates it to
JAX. The mainline method follows Cornish & Littenberg,
[TDI on the fly (2025)](https://arxiv.org/abs/2506.08093), with this package's
JAX implementation and local complex-polynomial interpolation. The legacy Fourier
calculation follows Cornish & Littenberg,
[Phys. Rev. D 76, 083006 (2007)](https://doi.org/10.1103/PhysRevD.76.083006).

## Citing

If you use binary-galactica, please cite it with the metadata in
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button exports
BibTeX and APA), and also cite GLASS and the papers its methods come from:

```bibtex
@misc{glass,
  author       = "Tyson B. Littenberg, Neil J. Cornish",
  title        = "GLASS",
  howpublished = "free software (Apache 2.0)",
  doi          = "10.5281/zenodo.2026177",
  year         = "2025"
}

@article{LittenbergCornish2023,
  author  = "Littenberg, Tyson B. and Cornish, Neil J.",
  title   = "Prototype global analysis of LISA data with multiple source types",
  journal = "Phys. Rev. D",
  volume  = "107",
  pages   = "063004",
  year    = "2023",
  doi     = "10.1103/PhysRevD.107.063004",
  eprint  = "2301.03673",
  archivePrefix = "arXiv"
}

@article{CornishLittenberg2025,
  author  = "Cornish, Neil J. and Littenberg, Tyson B.",
  title   = "TDI on the fly",
  year    = "2025",
  eprint  = "2506.08093",
  archivePrefix = "arXiv"
}

@article{CornishLittenberg2007,
  author  = "Cornish, Neil J. and Littenberg, Tyson B.",
  title   = "Tests of Bayesian model selection techniques for gravitational
             wave astronomy",
  journal = "Phys. Rev. D",
  volume  = "76",
  pages   = "083006",
  year    = "2007",
  doi     = "10.1103/PhysRevD.76.083006",
  eprint  = "0704.1808",
  archivePrefix = "arXiv"
}
```

## License

binary-galactica is licensed under the [Apache License, Version 2.0](LICENSE),
Copyright 2026 Aaron D. Johnson, Tyson B. Littenberg, Neil J. Cornish, and
Robert Rosati. Portions adapted from GLASS keep their upstream Apache-2.0
copyright notices; see [NOTICE](NOTICE). The optional C reference tool under
`tools/glass_reference` vendors KissFFT (BSD-3-Clause), which is not a runtime
dependency.
