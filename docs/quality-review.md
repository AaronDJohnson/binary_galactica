# Quality review follow-up — 2026-09-14

This is the historical Fourier-backend assessment. The September 18 migration
now makes TDI on the fly the default; see its [implementation and verification
record](superpowers/plans/2026-09-18-tdi-onfly.md) and the updated
[performance guide](performance.md). The older migration-status statements
below describe the state at the time of this review.

This assessment covers the current working-tree fixes following the September
14 review. It supersedes the earlier report's test counts and unverified CI
status. The changes have not been pushed or published.

**Scope correction after the user's clarification:** these scores and tests
assess the existing GLASS Fourier backend. They do not establish readiness for
the intended exclusive TDI-on-the-fly backend with integrated zero-crossing
handling. That migration remains incomplete. The earlier on-the-fly headline
conflated evaluating a waveform on demand with the distinct sparse TDI method;
the README now states the actual implementation status.

## Fixes

- Fixed-window templates validate `bin_lo` and `n_bins` before compilation.
  Fractional values, booleans, and nonfinite values raise `ValueError`;
  negative lengths are rejected. Python and NumPy integers are accepted,
  and a zero-length window retains its `(2, 0)` output.
- Alignment validates parameter rank and integer segment indices. Bandwidth
  requires a positive integer grid length, and Fisher cache refresh intervals
  require positive integers. Converting accepted NumPy integers to Python
  integers also fixes unsigned underflow in negative segment offsets.
  GLASS's short-grid bandwidth clamp remains unchanged.
- Carrier selection preserves the separate float64 division and multiplication
  used by the pinned GLASS code, including when an outer JIT captures duration
  or all source inputs as constants. Optimization barriers apply only to the
  discrete carrier; continuous source and duration derivatives remain intact.
  A batching rule handles independent and nested source/duration axes.
- README and package descriptions identify the implemented Fourier backend
  and the outstanding TDI-on-the-fly migration. The README describes the
  link-response envelope FFT and subsequent TDI synthesis, normal source
  installation, and updated
  JIT usage; package metadata now includes source, documentation, and issue URLs.
  Physical parameter documentation no longer describes native arrays as having
  the physical ordering.
- Corrected the import ordering in `tests/test_waveform.py` that failed hosted
  CI. An uncached local Ruff run reproduced the failure and passed after the
  fix. The earlier cached lint result had missed it.

## Verification

Fresh local checks ran on macOS arm64 with CPU JAX:

| Check | Result |
|---|---|
| Python 3.10.17, JAX/JAXlib 0.4.30, NumPy 1.24.0 | 423 passed |
| Python 3.12.12, JAX/JAXlib 0.11.1, NumPy 2.5.3 | 423 passed |
| Fresh installed wheel, Python 3.12.12, outside checkout | 420 passed |
| TDI zero-crossing solver and synthetic benchmark tests | 102 passed |
| README Python examples | All four executed successfully |
| Installed-wheel quickstart | Passed, including batching and derivatives |
| Ruff lint with `--no-cache`, formatting, whitespace | Passed |
| Public conversion-consumer mypy example | Passed |
| Sdist build, wheel built from sdist, strict Twine checks | Passed |

The source suite includes three module doctests in addition to the 420 tests
under `tests/`. Regression tests were observed failing before the fixes.
Carrier tests cover integer-boundary neighbors, constant and dynamic durations,
all TDI formats, 7/8/9 parameters, physical and fixed-window adapters, nested
batching, forward/reverse derivatives, and duration finite differences.
Independent review also checked all 36 pinned C spectrum cases through
constant-duration outer JIT on both JAX versions, retaining exact bins and
the existing `1e-9` per-channel relative L2 threshold.

All 19 Python modules in the fresh wheel matched the source checkout. The
typing marker, license files, and project URLs were present. The sdist included
the new carrier tests and the unchanged C reference fixtures. The wheel was
installed into a fresh temporary target using the existing Python 3.12
dependency environment; imports were confirmed to resolve to that target.
Independent review found no blocking issues in the changes.

## Hosted CI

[Run 34787515134](https://github.com/AaronDJohnson/binary_galactica/actions/runs/34787515134)
tested commit `9e6a9ce23897671954cdd452cedebe2d5cf12246`. All six runtime jobs
passed: Linux Python 3.10 with minimum dependencies, Linux Python 3.11–3.14
with current dependencies, and macOS Python 3.12. The distribution job failed
on the import-order violation fixed above, before its build and wheel checks.

These hosted results apply to the preceding commit. The current fixes have
passed local checks and still need a hosted run after pushing.

## Revised metrics

| Metric | Before these fixes | After |
|---|---:|---:|
| Correctness | 9.0 | 9.0 |
| Code readability | 8.5 | 8.5 |
| Ease of use | 8.0 | 9.0 |
| Release readiness | 8.5 | 9.0 |
| Overall, equal weighting | 8.5 | 8.9 |

The fixes remove concrete API hazards and improve the evidence for correctness.
The numerical code still uses dense domain notation and some untyped helper
interfaces. For minimum JAX compatibility, carrier stabilization uses an
isolated fallback to JAX 0.4.30's private optimization-barrier implementation;
the public API is used where available. Both paths were exercised above.

Reference agreement establishes fidelity to the pinned finite-band analytic
orbit calculation. Bandwidth convergence remains analysis-dependent, and
sky-pole derivatives use the documented coordinate regularization. GPU behavior
was not tested in this review. The arbitrary-time LAT TDI callback remains an
experiment under `tools/`; this revision retains the public backend's
carrier-removal and envelope-FFT calculation.
