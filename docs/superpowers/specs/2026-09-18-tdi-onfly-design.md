# Mainline TDI on the fly

Implement the user's request to build TDI on the fly and make it the mainline
waveform backend. Work in the existing checkout, preserving its uncommitted
changes. This supersedes the September 13 choice of a mainline link-FFT backend.

## Response and interpolation

Compute the demodulated complex XYZ response directly at arbitrary reception
times, using the pinned GLASS `LISA_spline_response` / `LISA_TDI_spline`
conventions. Use analytic spacecraft positions by default, actual instantaneous
arm lengths and unit directions, and delayed source amplitude and polynomial
phase. Geometry is frozen at each reception time, as in that GLASS path;
this is not a moving-spacecraft, sequential-delay TDI-2 implementation.
Pair delayed differences analytically to regularize aligned-arm limits.

Interpolate the already-combined response on a sparse grid independent of FFT
bandwidth. Following performance validation, the mainline removes the phase
at spacecraft 0's retarded source time before evaluating exponentials. Local
Cartesian polynomials through up to eight samples reconstruct the smooth
envelope without time derivatives or iterative zero refinement. Sparse-window
coefficients are computed before gathering onto the output grid; fixed
normalized coordinates make stencil selection independent of source, duration
and epoch. Complex zeros, tangent zeros and near misses have no singular polar
representation or artificial sign changes. The original derivative-based
Hermite/refinement utility remains tested as an internal reference.

**Normalization superseded on 2026-10-11** by the
[physical-amplitude spec](2026-10-11-physical-amplitude.md): the raw spline
response is the physical Sangria response, `tdi_response` returns it unscaled,
and the Fourier prefactor is `sqrt(T)/(2*BW)`.

Restore the retarded reference phase minus the integer carrier at the FFT sample
times, then transform the three channel envelopes. The public spectrum keeps
the existing carrier-bin layout, raw-GLASS units and format conventions. The
raw GLASS spline response is twice the package's public analytic Sangria
response. The Fourier prefactor is `sqrt(T)/(4*BW)` for the internal raw
response, or `sqrt(T)/(2*BW)` for public `tdi_response`. Verify this factor
independently on a static equal-arm monochromatic constellation with directed
links reversed in the old synthesis. The pinned spline and link-FFT C paths
place directed links differently; finite-frequency spectral differences are
expected even before accounting for actual arm lengths. Convert
Sangria to frequency units with `-1j`; phase units use `1/(2j*f/FSTAR)` and
conjugate A/E as before. Phase DC is defined to be zero.

## Public contract

`ucb_waveform`, physical `tdi_waveform`, and `template_on_window` default to
`backend="onfly"`. An explicit `backend="fourier"` retains the old backend
and its exact C-reference tests. `response_samples` is a static positive integer
of at least four, or `None` for automatic selection, and is independent of `BW`.
The automatic default selects the next power of two above 128 samples per
observation year, with a floor of 32 and cap of 2048. Explicit counts retain
caller control. This is a convergence policy, not a universal accuracy bound;
longer-than-16-year observations especially require explicit convergence checks.
`BW` remains the positive even FFT length and number of output bins. Dynamic
`T`, `t0`, native 7/8/9-parameter inputs, formats, carrier overrides and invalid
input behavior remain supported. Native vectors and physical amplitude/phase
conversion do not change.

Existing `(3,BW,3)` position caches remain accepted: for on-fly evaluation,
interpolate their geometry in time and document that coarse caches constrain
orbit accuracy. New optional sparse-grid cache support may be added only if
needed for correctness or a clear performance benefit. The response interval
includes every requested FFT time without extrapolating the old cache.

Expose a validated arbitrary-time native response interface, in documented
Sangria analytic-response units, for independent testing and downstream use.
Retain Python >=3.10, JAX >=0.4.30 and NumPy >=1.24; no new runtime dependency.
Preserve float64, JIT, vmap, forward/reverse derivatives and finite derivatives
at regularized sky poles and response zeros. AD holds the carrier and automatic
grid selection fixed. Grid switches introduce small interpolation jumps; an
explicit count is recommended when differentiating across duration thresholds.
No global differentiability claim applies across those switching boundaries.

## Verification

New response tests must use extracted pinned C routines, shipped fixtures,
provenance hashes and an offline regeneration command. Fixtures cover varied
sky, frequency, inclination, epoch and chirp; document C limitations for the
optional second frequency derivative. Interpolation tests cover smooth
responses, simple crossings, exact sampled zeros, near misses, tangencies,
zero channels, small strain scales, convergence and derivatives.

End-to-end tests compare interpolated spectra to dense arbitrary-time response
evaluation, establish the new default independently of the legacy backend,
check native/physical/window adapters and numerical Jacobians, and retain all
old reference checks under explicit Fourier selection. Run full tests,
prototype tests, lint/format/type checks, example and wheel verification.
Measure response accuracy and warmed performance; report measured limits.
