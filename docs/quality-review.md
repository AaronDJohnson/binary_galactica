# Quality review follow-up — 2026-09-13

All concrete findings from the quality review and its follow-up review are
resolved. Changes remain uncommitted in the existing `subsume-gbonfly` checkout.

## Changes

- Sinc uses a tenth-degree Taylor expansion near zero, with both inactive
  branches guarded. Independent 70-digit Decimal references check values,
  gradients, and Hessians around the old and new transitions and at large
  arguments.
- Native and physical waveform entry points validate concrete source and
  observation inputs. Invalid traced domains produce NaNs in all returned
  fields; static shape errors still raise while tracing. Fractional carrier
  overrides and out-of-domain cosines no longer produce plausible spectra.
- Priors validate finite positive durations, eight-dimensional finite ordered
  bounds, and periodic intervals. Mixture weights must be finite with a
  positive finite total. Both helpers snapshot caller-owned configuration.
- Proposal documentation distinguishes Fisher calls from total sampler steps,
  fixed covariance from adaptation, and discrete phase/polarization symmetry
  from continuous correlation moves.
- The public API exports the Fisher helpers and `WaveformResult`. The package
  includes `py.typed`, and converter overloads preserve the structured result
  while supporting partial dictionaries and read-only mappings. Static
  consumer examples catch the typing regression found during re-review.
- The independent C corpus grew from 18 to 36 cases, with varied source
  orientations, inclinations, amplitudes, phases, signed chirps, durations,
  bandwidths, and formats. All original fixtures remain unchanged. Independent
  regeneration reproduced all 216 arrays exactly on this machine; 180 complex
  channel spectra satisfy the per-channel relative L2 tolerance of `1e-9`.
- Normalized Jacobian tests cover 7/8/9 parameters and all three TDI formats.
  A phase-sensitive complex-quadrature regression replaces the ineffective
  phase-independent power comparison. Stop-gradient mutations verified that
  the phase and ninth-parameter regressions detect missing derivatives.
- CI includes minimum runtime dependencies, current compatible dependencies
  on Python 3.11–3.14, macOS, lint, formatting, public typing, package metadata,
  and tests of the installed wheel outside the checkout. Making `tests` a
  package fixed an import failure exposed by that installed-wheel check.

## Verification

All tests below ran on macOS arm64 with CPU JAX:

| Environment | Result |
|---|---|
| Python 3.10.17, JAX/JAXlib 0.4.30, NumPy 1.24.0 | 321 passed in 24.40 s |
| Python 3.12.12, JAX/JAXlib 0.11.1, NumPy 2.5.3 | 321 passed in 23.87 s |
| Python 3.14.2, JAX 0.7.2, NumPy 2.3.5 | 321 passed in 26.11 s |
| Installed wheel, Python 3.12.12, outside the checkout | 318 passed in 22.53 s |

The 321-test suite includes three additional source-module doctests; the
installed-wheel command selects the 318 tests under `tests/`.

Ruff lint and formatting, whitespace checks, actionlint, and mypy passed.
The quickstart and all four README Python examples executed successfully.
Both distributions built and passed `twine check --strict`. All 19 Python
modules in the installed wheel matched the current sources byte-for-byte;
the typing marker and license notices were present.

The initial local Python 3.10.13 interpreter crashed on its own standard-library
`readline` import, before package tests ran. A fresh temporary environment using
the working Python 3.10.17 interpreter and the same pinned dependency files
completed the full suite normally, including doctests. No test skips or
interpreter workarounds are required by the repository or CI workflow.

Independent review reproduced the original edge cases and confirmed their
resolution. Its one additional finding, the converter's TypedDict compatibility,
was corrected and independently re-reviewed with static and eager/JIT checks.
No actionable findings remain within the reviewed scope.

Hosted GitHub Actions has not run for these unpushed changes. Its workflow was
validated locally, and the referenced action versions were verified. Linux and
the remaining Python versions are configured in CI but were not executed here.
Reference agreement covers the documented pinned analytic-orbit calculation;
it does not establish bandwidth convergence for every possible source.

## Revised metrics

| Metric | Before | After |
|---|---:|---:|
| Correctness | 8.5 | 9.0 |
| Ease of use | 7.0 | 8.5 |
| Readability | 8.0 | 8.5 |
| Maintainability | 7.0 | 8.5 |
| Test quality | 8.0 | 9.0 |
| Release readiness | 6.0 | 8.5 |
| Overall | 7.5 | 8.7 |

The remaining release-readiness limit is verification of the hosted platform
matrix before publishing. These ratings reflect this scoped review and the
executed checks, rather than a claim of exhaustive correctness.
