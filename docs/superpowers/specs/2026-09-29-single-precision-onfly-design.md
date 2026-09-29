# Single-precision TDI on the fly

Add a float32 implementation of the on-fly waveform that runs in a JAX process
with `jax_enable_x64` disabled (float32-only accelerators, float32 analysis
pipelines). Work on branch `onfly-float32`, based on `8363ec0`. The float64
on-fly and legacy Fourier backends remain the reference implementations.

## Motivation and measured starting point

Running the current on-fly path with x64 disabled, against float64 at the same
float32-representable inputs, gives A/E relative L2 errors of 3e-6 (quarter
year, 1.2 mHz) to 5e-5 (ten years, 14 mHz), with mismatch at most 1e-9. The
arithmetic already works with sub-bin residual cycles and spacecraft-relative
delays. The dominant failure is input representation: a float32 `f0*T` near
1e6-4e6 resolves the frequency only to 0.06-0.25 cycles, giving relative L2
errors of 0.17 at four years and 1.5 at ten years. A float32 contract must
therefore split the frequency, and it must keep squared raw strains
(about 1e-42) out of float32 arithmetic.

## Double precision is no longer forced on import

This is a breaking change; the version becomes 0.4.0. Remove every
`jax.config.update("jax_enable_x64", True)` from the package (`__init__`,
`orbit`, `compat`, `tdi`, `response`, `fourier`). Every float64 JAX entry point
calls `_validation.require_x64()` first: `ucb_waveform`, `tdi_waveform`,
`tdi_response`, `template_on_window`, `orbit_cache`, and the source-only
`phase`, `frequency`, `amplitudes` and `polarizations`. With x64 disabled
these raise `RuntimeError` naming `jax.config.update("jax_enable_x64", True)`
and `binary_galactica.single`; nothing downcasts silently. Lower-level kernels
become dtype-generic and are unguarded. NumPy host helpers (priors, proposals,
`params_from_native`) are unaffected. The test suite enables x64 in
`tests/conftest.py` before importing the package.

## Public contract of `binary_galactica.single`

The namespace mirrors the float64 names, so `from binary_galactica import
single as bg32` reads naturally. Every function works with x64 on or off and
computes and returns only float32, complex64 and int32 values. All are on-fly
only; there is no `backend` argument.

- `ucb_waveform(params, *, q_ref, T, BW, t0=0.0, fmt="sangria",
  positions=None, q=None, response_samples=None)`.
- `template_on_window(params, *, q_ref, T, bin_lo, n_bins, BW, t0=0.0,
  fmt="sangria", positions=None, response_samples=None)`: `(2, n_bins)`
  complex64 in continuous-Fourier units.
- `tdi_response(params, times, *, q_ref, T, positions=None)`: `(3, n)`
  complex64, in the float64 function's Sangria analytic-response units.
- `tdi_waveform(theta, *, q_ref, T, BW, t0=0.0, fmt="sangria",
  positions=None, response_samples=None)`: physical parameters whose
  frequency field is the offset `f0 - q_ref/T` in hertz.
- `split_native(params, T)` and `split_physical(theta, T)`: host float64 to
  `(q_ref, float32 array)`. They bridge priors, proposals and saved float64
  catalogues. `q_ref` is `floor(f0*T)`, so the offset lies in `[0, 1)`.

Native vectors keep the 7/8/9-entry layout, except that `params[0]` is
`f0*T - q_ref`. `q_ref` is a Python integer or an int32 scalar, possibly traced
or batched. The carrier defaults to `q = q_ref + floor(params[0])`, so the
residual `f0*T - q` lies in `[0, 1)` exactly. An explicit integral `q`
override remains available. The float64 carrier `trunc((f0T/T)*T)` reproduces
GLASS rounding and may differ by one bin at an exact boundary; parity tests pin
`q`. `qmin` and `freq_bin_indices` are int32 and exact to 2**31. Other keys and
format conventions match the float64 `WaveformResult`.

Position caches keep the `(3, BW, 3)` layout on the `(T, t0, BW)` grid. Host
float64 or float32 arrays are accepted and cast to float32; the float32 step
near 1.5e11 m is about 5e-5 s of light time.

Outputs near 1e-21 are representable, but their squares underflow. Float32
inner products must use whitened or rescaled quantities. Because the waveform is
linear in amplitude, subtracting `ln(s)` from `params[3]` scales every channel
by `1/s`. This is documented rather than given a new option.

## Numerics

Kernels take their working dtype from the parameter array. A small
`glass/_precision.py` supplies the real and complex dtypes and the contraction
precision. `onfly.py`, `sparse.py` (including its host coefficient matrix),
`orbit.py`, `response.polarization_tensor`, `tdi.xyz_to_ae`,
`parameters.to_glass_vector` and the waveform cache/envelope helpers replace
hard-coded float64/complex128 casts, untyped constants and NumPy float64
arrays with dtype-typed values. Float64 outputs must remain bitwise identical.

Three float32-specific reformulations live in `glass/single.py` and a two-float
helper module:

1. **Carrier.** The residual `r = offset - floor(offset)` is exact in float32.
   No product as large as `f0*T` is ever formed.
2. **Epoch re-expansion.** With `u0 = t0/T` and `v = n/BW`, the FFT-grid phase
   is `2*pi*(c0 + c1*v + c2*v**2 + c3*v**3) - phi0 + shift`, where
   `c0 = r*u0 + F*u0**2/2 + G*u0**3/6`, `c1 = r + F*u0 + G*u0**2/2`,
   `c2 = (F + G*u0)/2`, `c3 = G/6`, `F = params[7]` and `G = params[8]`.
   This is algebraically the float64 formula, including its convention for
   `t0`, but keeps per-sample terms to a few cycles.
3. **Orbital phase.** The full Doppler delay `k.x0/c` (about 500 s) needs the
   orbital phase to about 1e-7 cycles; plain float32 at times near 6e8 s costs
   about 6e-4 rad of source phase at 30 mHz. Split `analytic_orbits` into
   `orbital_phase(t)` and a positions-from-phase function. Float64 keeps
   `PI2*t/YEAR + kappa_0` exactly. Float32 computes `frac(t/YEAR)` with two-float
   arithmetic: on the FFT grid from `(t0, T, n/BW)` without first rounding
   `t_n`, and at arbitrary response times from the float32 time value. Sparse
   response knots use plain float32: with the reference phase removed there, a
   knot-time rounding only resamples the smooth envelope (estimated at most 2e-5).

Every einsum and the interpolation-window convolution use
`jax.lax.Precision.HIGHEST`; default float32 contractions use TF32 on NVIDIA
GPUs and reduced-precision passes on TPUs. CPU float64 ignores the setting,
which the bitwise test confirms. Two-float transforms must remain correct
whether or not the compiler contracts `a*b + c` into a fused multiply-add; a
test checks the reduction accuracy on the executing backend.

The mitigations are adopted on evidence. `tools/diagnose_single.py` measures
relative L2 against float64 with each mitigation on and off at the corners of
the validated range (ten years, 30 mHz, `t0` of ten years, all formats, with
and without caches). A mitigation that the 1e-4 target does not need is
dropped. Results are recorded in `tools/performance/single-precision.json`.

Eager invalid inputs raise `ValueError` as in float64. `q_ref` must be an
integer with `q_ref - BW` and `q_ref + BW` inside int32 range. Traced invalid
values return NaN in floating outputs and `INT32_MIN` in int32 bin metadata.
Discrete carrier and automatic grid choices remain fixed under AD.

## Verification

Float32 accuracy: for T in {0.25, 1, 4, 10} years, f0 in {0.1, 1, 10, 30} mHz,
`t0` in {0, 10 years}, signed first and second frequency derivatives and all
three formats, A/E relative L2 error is at most 1e-4 and mismatch at most 1e-8
against the float64 path at identical inputs, with `q` pinned. Cover each entry
point: window placement, arbitrary-time response, the physical adapter and
position caches. Forward Jacobians agree with float64 to within 1e-3 relative
L2 error in each parameter column. Test `jit`, `vmap` over `params` and `q_ref`, split round trips within
one float32 ulp of the offset, two-float reduction accuracy, eager errors,
traced NaN/`INT32_MIN` behavior, and amplitude rescaling through `params[3]`.

A subprocess with x64 disabled imports the package, confirms the flag is still
false, runs every `single` function, checks float32/complex64/int32 outputs, and
compares them with float64 references written by the parent process. It also
checks that float64 entry points raise the x64 error. With x64 enabled, the
jaxprs of `single` functions contain no f64 or c128 values.

Float64 regression: golden outputs captured at `8363ec0` (all formats, window,
arbitrary-time response, physical adapter, caches and a Jacobian) must be
bitwise identical after the refactor. Update `tests/test_precision.py` for the
new import behavior, test the x64 guard, and run the full suite, zero-crossing
prototype tests, ruff, mypy, the quickstart and wheel verification.

## Documentation and performance

Add a README section on single precision: purpose, reference-bin input, split
helpers, measured accuracy, validated range, the underflow caveat and caches.
Add a 0.4.0 migration note to installation: enable x64 before using float64
functions. Add `examples/single_precision.py`, run with x64 disabled. Record
warmed, precompiled float32 and float64 CPU timings at BW 128, 512 and 2048 and
for a batch in `docs/performance.md`, reporting measured results without
claiming a speedup the measurements do not show. Set the version to 0.4.0.
