# TDI zero-crossing prototype

The user approved the proposed bounded-Brent prototype and comparison with
projected Brent/TOMS 748. This is a CPU experiment under
`tools/tdi_zero_crossings`, separate from the GLASS waveform backend and the
sibling LAT checkout. It accepts a callable complex response, so refinement
evaluates the response between the original samples.

- [x] Write regression tests for midpoint ties, final intervals, near misses,
  tangencies, amplitude scaling, nonuniform times, and failed refinement.
- [x] Implement bounded minimization of normalized squared magnitude, full
  complex residual checks, and conservative crossing/touch classification.
- [x] Implement projected Brent and TOMS 748 alternatives and report fallback,
  convergence, and actual response-evaluation counts.
- [x] Scan a subdivided time grid for candidate minima, retaining ties and
  boundary candidates. Document that finite sampling cannot guarantee discovery
  of arbitrarily narrow or multiple zeros.
- [x] Benchmark the three methods against analytic truth and the original
  three-point heuristic. Keep candidate discovery and refinement costs distinct.
- [x] Exercise an unchanged LAT complex response kernel if locally accessible;
  identify any orbit/source simplifications explicitly.
- [x] Run numerical tests, repository regression tests, formatting/lint, and an
  independent review. Save reproducible results and a recommendation.

`refine_candidate(response, interval, ...)` returns the candidate time, residual,
local scale, classification, actual solver, evaluation count, and convergence.
`scan_candidates(response, times, ...)` supplies candidate intervals using a
configurable subdivision count. Neither function changes waveform samples or
assigns phase at an exact zero. Boundary and unresolved cases require more data
before changing amplitude signs. Solver time tolerance and amplitude tolerance
are distinct and explicit.

Validation uses analytic complex curves with known zeros and near misses. The
benchmark measures classification, root localization, function calls, and CPU
wall time; it does not establish production TDI mismatch or GPU speedup.

Verified: 395 combined repository/prototype/LAT tests passed on Python 3.14;
74 prototype/LAT tests passed on Python 3.10 with minimum dependencies.
The 66 portable prototype tests also passed on Python 3.12 with current
dependencies. Ruff, formatting, actionlint, and whitespace checks passed.
Independent review found no remaining blocking issues. Reports and source
fingerprints are saved under `tools/tdi_zero_crossings/results/`.

## Higher-accuracy follow-up

Added local recentering inside sampled brackets, full-residual fallback for
projected roots, a shared objective budget, and preservation of adjacent exact
sampled zeros. Endpoint amplitude acceptance no longer bypasses localization.

Matched 104-case measurements at fractional time tolerance 1e-12 reduced
maximum normalized zero-time error from 6.253e-9 to 2.880e-13. An expanded
536-case run also achieved 100% classification and localization accuracy,
with worst normalized time error 3.195e-13. Callback costs increased; full
controls, source hashes, and before/after results are saved in results/.

Verification: 431 combined tests passed on Python 3.14. The 110 prototype/LAT
tests passed on both Python 3.10/minimum dependencies and Python 3.12/current
dependencies. Independent review findings were fixed and rechecked.
