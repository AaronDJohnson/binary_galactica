# TDI on-the-fly performance follow-up

The user authorized continuing after the 1.63 ms on-fly versus 0.378 ms
Fourier measurement. Preserve the working checkout and the mainline on-fly
backend while addressing the measured interpolation and response costs.

## Design and constraints

The reference phase will be `phi(t - k.x0/c)`, matching the pinned GLASS
construction. Recenter source times and delays inside the existing response
kernel before exponentiation. Restore the reference phase, including the
carrier residual, only at the output samples. The public `tdi_response` retains
its documented `phi(t)` demodulation and normalization.

Compare sampled-value natural/polar splines with local Cartesian polynomials
after this phase removal. Prefer the simplest method that meets the dense
response and derivative checks. A local polynomial is linear in complex
samples and has no singular representation at zeros; this can eliminate exact
time JVPs and iterative zero refinement from the waveform hot path. Construct
coefficients on sparse windows before gathering onto the output grid.

Keep native/physical/window APIs, 7/8/9 parameters, dynamic duration and epoch,
carrier-bin rounding, cached positions, output units/formats, float64, JIT,
vmap and forward/reverse derivatives. Keep Python >=3.10, JAX >=0.4.30 and
NumPy >=1.24 without new runtime dependencies. Initially retain the 512-sample
default; any reduction must follow convergence evidence. Preserve the existing
Hermite helper's tests while evaluating the replacement waveform path.

## Work

- [x] Reproduce sparse high-frequency failures against the independently
  assembled dense-response spectrum: one/four-year 30 mHz cases fail at
  64/256 samples before phase removal. Require per-XYZ relative L2 <2e-6.
- [x] Prototype stable phase recentering and verify reconstruction for 7/8/9
  parameters, nonzero epochs, cached geometry, time/source/duration derivatives
  and Hessians.
- [x] Compare sampled-value interpolation candidates on synthetic zeros,
  nonvanishing rotating signals and all four benchmark sources. Reject a
  candidate that obtains speed by losing required convergence or AD behavior.
- [x] Integrate the chosen interpolation and shared phase kernel; run the new
  sparse-response regressions and existing numerical/reference tests.
- [x] Check source, duration and epoch derivatives against dense calculations,
  including regularized poles and cached geometry. Validate minimum JAX.
- [x] Measure synchronized public and compiled calls, fixed-window templates,
  catalogues and Jacobians against the Fourier backend with no competing tests.
- [x] Refresh documentation, benchmark provenance and artifacts; obtain an
  independent numerical review and complete full verification.

## Decisions and verification

- Selected degree-seven local Cartesian polynomials after spacecraft-reference
  phase removal. The interpolation is linear in complex samples, retains zero
  residuals, and needs neither time JVPs nor root searches in waveform calls.
- Broad production validation justified an automatic duration policy: `None`
  selects the next power of two at density 128 per GLASS orbital year, between
  32 and 2048. Explicit integer counts remain supported. The 90 tested cases
  span 0.1–100 mHz, quarter-year through four-year observations, poles, edge-on
  sources, 7/8/9 parameters and varied epochs. Maximum channel-relative L2 was
  2.06e-7 overall and 1.36e-8 through 30 mHz. These are measured limits.
- Automatic switches cause small interpolation jumps at duration thresholds;
  use a fixed count for duration derivatives crossing them. Batched differing
  durations can evaluate multiple branches. Documented both limitations and
  the 2048 cap beyond 16 years.
- All 503 current-JAX tests passed in 189.00 s. All 80 focused tests passed on
  minimum JAX 0.4.30 / NumPy 1.24.0. Independent numerical review found no
  production blocker; source/observation derivatives and regularized poles were
  checked against dense responses and finite differences. Corrected reporting
  helpers to use the production sidereal-year constant at grid boundaries.
- Quickstart and mypy passed; physical/native example difference was 4.13e-16.
  Ruff lint/format and whitespace checks passed.
- Built the 0.3.0 sdist and wheel and passed strict Twine metadata checks.
  All 77 selected installed-wheel tests passed outside the source checkout,
  including independent GLASS fixtures and the public on-fly API. The installed
  quickstart passed; all 23 packaged Python modules matched the source tree.
- The original fiducial benchmark now measures 0.323 ms for the automatic
  on-fly public call versus 0.371 ms for Fourier, compared with 2.28 ms initial
  and 1.63 ms after the first coefficient optimization. Explicit 512 now takes
  0.771 ms, so the report separates algorithm and sampling improvements.
- Matched 64-source workloads show gains at BW2048 for the analytic-orbit
  native/window/batch/Jacobian paths. Fourier remains faster at BW128 and for
  BW512 batches/Jacobians and several cached workloads. The documentation and
  recorded JSON retain these limits rather than claiming a universal speedup.
