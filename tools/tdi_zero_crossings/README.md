# Experimental TDI zero refinement

This CPU prototype tests a replacement for the strict three-point
magnitude-minimum and curvature heuristic used in LAT's amplitude/phase
extractor. It refines an isolated candidate with bounded scalar minimization,
checks the full complex residual, and then checks the complex directions on
both sides to distinguish a crossing from a tangent zero. Projected Brentq and
TOMS748 provide comparison methods. The code reports evidence about candidates;
it does not change amplitude signs or construct phase jumps.

The prototype stays under `tools/` because the public waveform API does not
currently expose the arbitrary-time, demodulated complex TDI callback needed
for this experiment. Integration would also require a caller to define its
candidate-resolution, sign-continuation, and interpolation policy. The
synthetic comparison does not validate those production choices.

From the repository root, use Python 3.10 or newer with NumPy, SciPy, and pytest:

```sh
python -m pip install 'numpy>=1.24' 'scipy>=1.10' 'pytest>=7'
python -m pytest -q tools/tdi_zero_crossings/test_solver.py tools/tdi_zero_crossings/test_accuracy.py tools/tdi_zero_crossings/test_benchmark.py
python -m tools.tdi_zero_crossings.benchmark --output /tmp/tdi_zero_crossings.json --repeats 3
```

CI runs the solver and synthetic benchmark regression tests alongside the
existing Python/dependency matrix. Optional LAT checks require an explicit
local checkout and run separately.

The benchmark needs no LAT checkout, compiled extension, JAX, or GPU. It writes
the environment and package versions, complete case definitions, per-run
classifications and callback counts, and aggregate CPU timings to JSON.
`--seed`, `--random-cases`, `--amplitude-rtol`, and `--time-fraction-atol` control
the synthetic experiment. Its default time tolerance is the candidate interval
width times `1e-12`; absolute amplitude tolerance is zero. Override the time
setting with `--time-fraction-atol` when comparing another accuracy target.

A response callback accepts one scalar time and returns one finite complex
value. Supply the actual smooth, preferably demodulated TDI response at every
requested time. The solvers evaluate between the supplied samples. Passing an
interpolated amplitude or response instead locates properties of that
interpolant and cannot establish the original response's zero times.

```python
import numpy as np

from tools.tdi_zero_crossings.solver import refine_candidate, scan_candidates


def response(t):
    # Analytic example only; replace with an arbitrary-time complex TDI callback.
    return 1e-22 * (t - 0.37) * np.exp(0.3j * t)


candidate = refine_candidate(
    response,
    (0.0, 1.0),
    method="bounded",
    amplitude_atol=0.0,
    amplitude_rtol=1e-8,
    time_atol=1e-12,
)
print(candidate.kind, candidate.time, candidate.residual, candidate.evaluations)

# Scan a nonuniform grid before refining the retained minima.
candidates = scan_candidates(
    response,
    [0.0, 0.08, 0.31, 0.68, 1.0],
    subdivisions=4,
    time_atol=1e-12,
)
crossings = [item for item in candidates if item.kind == "crossing"]
```

`refine_candidate` expects an isolated candidate on a finite increasing
interval. `bounded` minimizes `abs(response / scale)**2` on normalized time,
using SciPy's [bounded scalar minimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html).
The normalization avoids squaring raw amplitudes near `1e-22` or much smaller
values. After locating a minimum, a second bounded solve uses small time
displacements from the best evaluated sample, between its nearest evaluated
neighbors. This reduces the first solve's relative stopping floor without
assuming a fixed radius around the minimum. `brentq` and `toms748` solve a real
projection onto the endpoint complex
secant with SciPy's [Brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
and [TOMS748](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.toms748.html)
methods. They fall back to bounded minimization when that projection has no
strict sign bracket or the projected root fails the full complex residual
check. A projection can have a nonzero-response root even when the interval
contains a true complex zero; the fallback now searches the magnitude minimum
before deciding the classification. The result's actual `method` is `bounded`
when this fallback is used.

The result records `time`, complex `value`, amplitude `residual`, local `scale`,
`kind`, actual `method`, callback `evaluations`, and solver `converged` status.
`method` can be `samples` when endpoint, exact-midpoint, or identically-zero
checks decide the result before numerical refinement. The best evaluated
candidate is retained if the final solver iterate has a larger residual. Only
exactly zero endpoint samples short-circuit refinement: a small nonzero
endpoint residual can conceal a nearby interior crossing.

| Kind | Meaning at the requested tolerances |
| --- | --- |
| `crossing` | Small complex residual and consistently opposite directions on both sides. |
| `touch` | Small complex residual and consistently matching directions on both sides. |
| `near_miss` | Converged candidate with a residual above the acceptance threshold. |
| `boundary` | Accepted candidate at or too close to an interval endpoint for an interior decision. |
| `unresolved` | Insufficient direction evidence, nonconvergence, or no isolated candidate in an all-zero response. |

Acceptance uses `amplitude_atol + amplitude_rtol * scale`, with `scale` equal
to the largest magnitude at the interval endpoints and midpoint. An absolute
tolerance must match the response units: a tolerance appropriate for a
unit-amplitude signal can accept everything in a `1e-22` signal. A successful
scalar solve alone does not imply an accepted crossing. These classifications
are numerical diagnostics, not proofs of exact zeros; `near_miss` does not
prove that the interval contains no other zero. A near miss below the amplitude
tolerance can be indistinguishable from a true zero; reducing the time
tolerance alone does not resolve that ambiguity.

`time_atol` uses the callback's time units, must be positive and smaller than
the interval width, and cannot be below floating-point spacing at the interval
endpoints. Internal time normalization helps with short intervals, but it
cannot recover times finer than the floating-point representation at a large
epoch. Recentering reduces SciPy's bounded minimizer's relative stopping floor,
but cannot recover digits already lost to cancellation inside the callback.
Flat nonzero minima can also limit localization. A tight time tolerance does
not guarantee an equally small complex residual or an accurate physical zero.

Direction classification uses successively smaller probe pairs and requires
two consistent decisions. Rapid phase rotation, weak probe amplitudes, or
inadequate time resolution can leave a candidate unresolved. A demodulation
consistent with the response's phase convention reduces carrier rotation
while preserving its zeros. The caller must supply or construct that physical
response path; sampled magnitudes alone cannot recover the complex direction.

`scan_candidates` subdivides each original interval, including nonuniform
intervals, retains tied minima and global endpoint minima, and refines the
resulting candidate intervals. Exact sampled zeros receive separate local
brackets, including inward brackets at the global endpoints, so adjacent known
zeros are not merged into one plateau. `subdivisions` controls discovery resolution.
It does not guarantee completeness: nearby zeros, a rapid oscillation, or
multiple minima within one candidate require a denser scan or a separately
justified interval. Compare results under grid refinement for the response
being studied. Boundary candidates alone cannot justify a sign flip.

Each refinement counts actual callback evaluations, including endpoint,
residual, and direction checks, with repeated times cached locally. A scan's
individual result counts exclude its initial grid sampling. Wrap the callback
with an external counter to measure total scan-plus-refinement cost; summing
the result counts alone understates that cost. The synthetic benchmark wraps
each response independently and verifies reported counts against observed
calls. `maxiter` is a shared budget for objective calls across projection,
fallback, and both bounded solves, including objective calls served by the
cache. Initial samples, final residual verification, and direction probes are
additional work, so
`maxiter` is not an upper bound on the reported callback count.

The default seeded suite has 104 candidates: fixed cases and 24 random shapes
paired at amplitudes `1` and `1e-22`. It covers simple crossings, tied minima,
near-boundary and endpoint zeros, tangent zeros, near misses, rotating
responses, short local intervals, and nonuniform sampling. All refinement
methods receive identical known candidate intervals, so these measurements
exclude discovery. The legacy baseline mirrors strict minima, the curvature
tests, and suppression of a flip assigned to the final sample. It only
detects flips; its full-kind and root-time accuracy fields are intentionally
`null`. Its five sampled callback evaluations are measured for transparency,
although the production heuristic reads amplitudes already computed upstream.

The accuracy revision was compared with the preceding implementation on the
same 104 cases, seed, and three repeats, with both runs requesting a time
tolerance of `1e-12` times the interval width. Classification and localization
are scored separately. "Within time tolerance" counts every estimate of a
known true zero whose absolute time error divided by interval width is at
most `1e-12`; it does not filter on classification or convergence. There are
80 true-zero cases, including tangent and endpoint zeros, giving 240 estimates
per method. Near misses have no reference zero time.

| Method | Kind accuracy, before → after | Within time tolerance, before → after | Maximum relative time error, before → after | Mean callbacks, before → after |
| --- | ---: | ---: | ---: | ---: |
| Bounded | 100% → 100% | 50% → 100% | `6.25e-9` → `2.88e-13` | 18.0 → 26.7 |
| Projected Brentq | 92.3% → 100% | 65% → 100% | `0.130` → `2.88e-13` | 13.1 → 25.6 |
| Projected TOMS748 | 92.3% → 100% | 65% → 100% | `0.125` → `2.88e-13` | 13.6 → 26.1 |

All three revised methods also reached 100% crossing accuracy in this run,
with no false crossings. The legacy heuristic remains at 67.3% crossing
accuracy and five callback evaluations; it has no root-time accuracy score.
The additional evaluations pay for more accurate local refinement and, for
the projected methods, recovery from nonzero projection roots. In particular,
the fallback recovered the fast-rotation crossings and seeded rotating-tangent
cases missed by the preceding projected solvers.

The [matched comparison](results/accuracy_comparison.json) preserves before
and after measurements at the same tolerances, including the historical solver
hash. The [current synthetic report](results/synthetic.json) records the updated
source hashes and controls. The generated JSON adds per-estimate
`time_within_tolerance` and aggregate `time_accuracy_fraction`.
Reported time errors include every known true zero,
even an estimate that was misclassified or did not converge. This finite
synthetic suite does not establish general correctness or a production
speedup. CPU timings include each complete refinement and classification
after warmup and exclude JSON reporting; callback costs for a physical TDI
calculation may be very different.

An [expanded run](results/synthetic_expanded.json) used 536 cases, including
240 seeded random shapes at both amplitude scales, with three repeats.
All three methods reached 100% classification, crossing, and localization
accuracy at the same `1e-12` fractional time tolerance. The largest normalized
zero-time error was `3.20e-13`; mean callback counts were 32.5, 32.8, and 33.7
for bounded, Brentq, and TOMS748 respectively. Reproduce this run with
`--random-cases 240 --repeats 3`.

The optional `lat_reference.py` adapter provides a separate experiment using
an explicitly supplied, trusted LAT source checkout and controlled orbit and
source fixtures. It requires JAX with float64 enabled and a warmed callback.
The CLI enables float64 and warms compilation before timing:

```sh
python -m tools.tdi_zero_crossings.benchmark_lat \
  --lat-checkout /path/to/LISAanalysistools \
  --repeats 5 --output /tmp/tdi_zero_crossings_lat.json

LAT_REFERENCE_CHECKOUT=/path/to/LISAanalysistools \
  python -m pytest -q tools/tdi_zero_crossings/test_lat_reference.py
```

The [saved LAT report](results/lat_reference.json) records the exact loaded
source hashes, prototype hashes, fixture parameters, environment, and every
solve. This uses the unchanged local JAX TDI projection with a frozen 2.5 Gm
equilateral constellation and a constant-amplitude chirping source. It is a
controlled test near an equal-arm transfer null, with the source phase removed;
it is not a realistic orbital waveform or compiled C++ parity test.

In the updated saved run, all 15 solves classified this example as a near
miss. Both projected methods used bounded refinement after their projected
roots failed the full complex residual check. The smallest response
magnitude was approximately `5.26596e-5`, near `8.106` seconds, or `0.0543` of
the local amplitude scale. Median callback counts, including endpoint and
residual checks and fallbacks, were 28 for bounded minimization, 32 for
projected Brentq, and 34 for projected TOMS748. The corresponding median
wall times on the recorded machine were approximately 1.17, 1.33, and 1.67
milliseconds. Agreement between methods on this flat, cancellation-sensitive
minimum does not certify its physical time accuracy.
These timings include CPU callback execution after JIT warmup and exclude
candidate discovery; they do not establish a production speedup.

Bounded minimization remains the CPU reference. Projected methods now use its
refinement as a fallback, and their total callback costs must be measured on
the intended response before choosing between them. Production integration
still needs a response callback at new times, grid-convergence and
waveform-interpolation checks, a sign/phase-continuation policy, and
measurement on the intended CPU or GPU backend.
