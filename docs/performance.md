# Waveform performance

The default waveform backend is sparse TDI on the fly. Its runtime depends on
the automatic or explicitly fixed response grid as well as output bandwidth; choose
those settings for response convergence before comparing workloads. The
benchmark exercises native and physical APIs, fixed-window templates, catalogue
batches, and waveform Jacobians. It does not measure a sampler or likelihood.

## Single-precision timing — September 29, 2026

The [recorded run](../tools/performance/single-precision-timing.json) times
the float32 `binary_galactica.single.ucb_waveform` in a process with
`jax_enable_x64` disabled and the float64 on-fly `ucb_waveform` in a process
with it enabled, on a macOS arm64 CPU (Python 3.11.8, JAX/jaxlib 0.7.2,
NumPy 2.3.5). Both evaluate the same 64-source catalogue
near 3 mHz; float64 receives the exact values of the float32 inputs. Calls use
the automatic response grid, analytic orbits and `t0` = 12,345 s, and return
all seven outputs, which are synchronized before each timer stops. Outer-JIT
functions are compiled ahead of time with `lower().compile()` and take the
parameters, `q_ref`, `T` and `t0` as device arrays. The two processes
alternated for three rounds; each entry is the median over rounds of a round's
median of 200 warmed calls. Before timing, the first source's float32 A/E
agreed with float64 to a relative L2 error of at most 3.7e-6.

Milliseconds, **float64 / float32**:

| Duration (response samples) | Workload | BW=128 | BW=512 | BW=2048 |
| --- | --- | ---: | ---: | ---: |
| 7,884,000 s (32) | Public call | 0.102 / 0.115 | 0.129 / 0.154 | 0.260 / 0.249 |
|  | Outer JIT | 0.076 / 0.087 | 0.100 / 0.123 | 0.224 / 0.222 |
|  | Batch of 32, outer JIT | 0.712 / 0.546 | 0.992 / 0.743 | 2.618 / 1.613 |
| 126,000,000 s (512) | Public call | 0.343 / 0.307 | 0.369 / 0.320 | 0.434 / 0.402 |
|  | Outer JIT | 0.305 / 0.261 | 0.331 / 0.300 | 0.448 / 0.306 |
|  | Batch of 32, outer JIT | 4.495 / 3.561 | 4.935 / 3.940 | 6.177 / 4.864 |

Float32 took between 0.62 and 1.23 times the float64 time. For single
quarter-year sources, the smallest calls at about 0.1 ms, it was 13–23%
slower at BW=128 and BW=512 and within 4% at BW=2048. It was 7–32% faster for
four-year single sources and 20–38% faster for batches of 32. The float32 path
also does work that float64 does not: at every FFT sample it sums the orbital
phase and the restored source phase in paired float32 arithmetic, which keeps
strong chirps and distant carriers within the accuracy target. On this CPU,
float32 makes it possible to run without x64 at a similar cost; it is not a
large speedup. Ahead-of-time compilation took 0.63–1.12 s per function in
float64 and 0.70–1.26 s in float32.

Other work shared the machine during the run: the load average recorded by
each worker process ranged from 3.3 to 6.2 on 16 logical CPUs. For the same
workload, a round's median differed from another round's by up to 34%, and
the per-round float32/float64 ratio varied across rounds by up to 0.17. The
quarter-year BW=128 and BW=512 single-source rows were slower in float32 in
every round. Every other row was faster in float32 in every round, except
the quarter-year BW=2048 outer-JIT row, which stayed within 1% of equal. An
earlier record, before the per-sample phase sum used pairs, measured 0.56 to
1.09 under heavier load. The pairs add little work: XLA's cost estimate for
the compiled float32 quarter-year call grew by 0.5% at BW=128, 2% at BW=512
and 7% at BW=2048, so the difference between the records is mostly within
their round-to-round spread. GPU and TPU timings have not been measured, and
neither Jacobians nor `template_on_window` were timed.

Reproduce, serially and preferably on an idle machine, with:

```sh
PYTHONPATH=src python tools/benchmark_single.py --output /tmp/single-timing.json
```

## On-fly accuracy and timing — September 18, 2026

The [recorded run](../tools/performance/onfly.json) uses macOS arm64, Python
3.11.8, JAX/jaxlib 0.7.2, NumPy 2.3.5 and CPU execution. It compares sparse
interpolation against the same direct arbitrary-time response evaluated at all
2048 FFT samples. Errors are the largest complex relative L2 error among
X/Y/Z/A/E. This checks interpolation, not physical validity or FFT bandwidth
convergence; the independent C fixtures test the response separately.

The default `response_samples=None` selects a power-of-two count from 32 to
2048, targeting at least 128 samples per GLASS orbital year
(31,558,149.7632 seconds), until the cap. Explicit integers still choose the
count directly. The response removes the reference-spacecraft phase before
interpolation, following GLASS's phase choice. Local polynomials through up to
eight complex values replace exact response time derivatives and iterative
zero refinement in the waveform path. The interpolation remains linear in
complex values through zeros. (The Hermite comparison helper has been removed.)

| Source | Automatic samples | Automatic relative L2 error | Explicit 512 error |
| --- | ---: | ---: | ---: |
| 1 mHz, quarter year, nearly edge-on | 32 | 5.27e-10 | 1.19e-14 |
| 5 mHz, one year | 128 | 2.49e-10 | 9.45e-15 |
| 5 mHz, four years, nonzero fddot | 512 | 2.01e-10 | 2.01e-10 |
| 30 mHz, one year | 128 | 8.81e-10 | 1.84e-14 |

A separate [90-case production validation](../tools/performance/onfly-validation.json)
covers 0.1–100 mHz, quarter-year through four-year observations, native 7/8/9
parameters, poles, edge-on sources, and varied nonzero epochs. Its maximum
channel-relative L2 error with the automatic grid was **2.06e-7**; for
frequencies through 30 mHz it was **1.36e-8**. All outputs were finite. These
are measured cases, not universal error bounds. Coarse grids still miss
structure: the four-year 5 mHz case has error 1.59e-5 at 128 samples. Increase
`response_samples` independently of `BW` to check the intended sources,
particularly for observations beyond 16 years where the automatic count caps.

For the quarter-year source at `BW=2048`, 100 synchronized warmed calls gave:

| Path | Native public call | Outer JIT |
| --- | ---: | ---: |
| On-fly, automatic (32 samples) | **0.323 ms** | **0.192 ms** |
| On-fly, explicit 512 samples | 0.771 ms | 0.742 ms |
| Fourier | **0.371 ms** | See workload comparison |

The default native call is about **13% faster than Fourier on this case**.
The [initial on-fly implementation](../tools/performance/onfly-initial.json)
measured 2.28 ms; the [Hermite coefficient follow-up](../tools/performance/onfly-hermite.json)
measured 1.63 ms. The current default is about seven times faster than the
initial implementation, with a smaller dense-response interpolation error on
this source (5.27e-10 versus 2.88e-9). The explicit-512 row separates the
algorithm change from the automatic sampling change.

Timings exclude initial compilation. The record's `first_call_ms` is only the
first call in each timing section; earlier accuracy checks may have compiled
the same kernel. Diagnostic response-only and response-plus-time-derivative
measurements in the report are separate compiled functions, not additive
components of waveform latency. The mainline no longer needs those time JVPs.
GPU performance has not been measured.

The [September 19 GLASS comparison and compiler diagnosis](onfly-glass-comparison.md)
investigates the remaining small-band slowdown, including response recomputation
inside overlapping interpolation windows and differences in GLASS's benchmark.

### Matched inference workloads

The [on-fly workload run](../tools/performance/onfly-workloads.json) and
[Fourier workload run](../tools/performance/fourier-workloads.json) use the same
64-source catalogue near 3 mHz, quarter-year duration, dynamic `T`/epoch inputs,
environment, source tree and 100 warmed repeats. The on-fly default selects
32 response samples. The table reports **on-fly / Fourier**, in milliseconds,
with analytic orbits evaluated inside each call:

| Workload | BW=128 | BW=512 | BW=2048 |
| --- | ---: | ---: | ---: |
| Native ordinary call | 0.136 / 0.081 | 0.155 / 0.167 | 0.295 / 0.360 |
| Native outer JIT | 0.113 / 0.060 | 0.132 / 0.145 | 0.259 / 0.334 |
| Fixed-window A/E outer JIT | 0.108 / 0.054 | 0.127 / 0.133 | 0.277 / 0.310 |
| Batch of 32, outer JIT | 0.963 / 0.360 | 1.289 / 1.069 | 2.867 / 3.383 |
| A/E source Jacobian, outer JIT | 0.419 / 0.168 | 0.643 / 0.406 | 0.887 / 1.324 |

At BW=2048 the on-fly path is faster across these workloads. At BW=128,
Fourier remains faster; at BW=512 the on-fly single-source advantage does not
extend to batches or Jacobians. With supplied orbit caches, Fourier also remains
faster for the measured single-source calls. The reports include all cached
and physical-adapter timings. These backends have the documented physical
response differences; this comparison does not make their spectra equivalent.

Automatic duration branches also cost compilation time. These cache-cleared
first calls were roughly 1 second for native on-fly calls and 3 seconds for
Jacobians, versus roughly 0.1–0.3 seconds for Fourier. Reuse compiled functions
for repeated inference. The original four-source table uses a different
fiducial source and captures duration as a constant in its outer-JIT case.
The workload runs passed duration as a traced argument, so these first calls
compiled every automatic grid. Public calls now choose the automatic count on
the host whenever the duration is concrete, and compile only that grid.

Automatic grid changes introduce small interpolation jumps at duration
thresholds. AD follows the selected branch. Use an explicit sample count when
differentiating across a threshold. A traced duration with the automatic grid
compiles all seven grids behind a switch, and reverse-mode derivatives keep the
intermediate values of every one, so memory and time grow with the batch; such
calls emit `ResponseSamplesWarning`. Batching different durations also
evaluates multiple branches. Close over a fixed duration, group by duration,
or fix the count for that workload.

Reproduce convergence, timing and the broader accuracy validation with:

```sh
PYTHONPATH=src python tools/benchmark_onfly.py --output /tmp/onfly.json --repeats 100
PYTHONPATH=src python tools/validate_onfly.py --output /tmp/onfly-validation.json
```

The historical measurements below describe only the explicit legacy Fourier
path and use a different Python environment and bandwidth.

## Matched package-versus-GLASS measurements — September 19, 2026

The [matched comparison](../tools/performance/glass-comparison.json) uses
identical native source vectors near 5 mHz, BW512, epochs of 12,345 seconds,
and quarter/one/four binary years of 31,457,280 seconds. Native GLASS and JAX
run serially on this arm64 CPU. Each entry is the median of three 200-call
block means. JAX calls synchronize every result; compilation and native
orbit/basis initialization are outside the timing loops.

The complete on-fly calls have these costs, with their distinct outputs:

| Duration | Package default, outer JIT, Fourier XYZA/E | Package default, ordinary call | GLASS on-fly, WDM XYZ |
| --- | ---: | ---: | ---: |
| Quarter year | 0.136 ms | 0.157 ms | 0.571 ms |
| One year | 0.221 ms | 0.245 ms | 2.95 ms |
| Four years | 0.543 ms | 0.570 ms | 15.2 ms |

The package uses 32/128/512 response samples by default; GLASS uses
49/199/797. Both interpolate their sparse response and apply a transform, but
the package returns 512 complex Fourier bins per channel while GLASS returns
2,048/8,192/32,768 active real WDM pixels per channel. Consequently, these
costs do not establish a speedup for equivalent work. Explicitly requesting
GLASS's response counts in the package yields outer-JIT times of
0.167/0.293/0.700 ms; that still does not make the output transforms equivalent.

For a closer comparison of sparse response evaluation, use the same endpoint
grid and the same number of samples in both implementations:

| Duration | Samples | Package, analytic geometry | Package, supplied GLASS geometry | GLASS response |
| --- | ---: | ---: | ---: | ---: |
| Quarter year | 49 | 0.0356 ms | 0.0393 ms | 0.0298 ms |
| One year | 199 | 0.0904 ms | 0.0930 ms | 0.144 ms |
| Four years | 797 | 0.231 ms | 0.231 ms | 0.656 ms |

The GLASS response includes orbit-spline lookups and signed amplitude/phase
extraction. The package returns complex XYZ directly; its supplied-geometry
variant excludes those lookups. Source splines and reference phases are
prepared before the native response timer, while the package evaluates the
analytic source from its parameters. Thus equal sampling gives a useful cost
comparison, but these are not identical instruction sequences. All JAX arrays,
times and durations are passed dynamically so analytic orbits cannot be
constant-folded out of the timed kernel.

The same-output comparison is the package's explicit legacy Fourier backend
against native GLASS `ucb_waveform`, both returning XYZA/E at BW512:

| Duration | Package outer JIT | Package ordinary call | Native GLASS |
| --- | ---: | ---: | ---: |
| Quarter year | 0.142 ms | 0.170 ms | 0.104 ms |
| One year | 0.145 ms | 0.173 ms | 0.106 ms |
| Four years | 0.148 ms | 0.176 ms | 0.110 ms |

GLASS wins this comparison: JAX's compiled-call latency is about 34–37% higher.
The JAX measurements include Python dispatch and synchronization; native calls
are timed inside C. Ordinary package calls use NumPy inputs and include input
validation. This measures single-call latency, not batched/GPU throughput.

Before timing, every Fourier channel and the carrier bin were compared with
fresh native dumps. Maximum channel-relative L2 error was 1.92e-10, with exact
carrier agreement. Sparse complex responses using identical GLASS geometry
agreed within 3.53e-9 relative L2; using analytic geometry gave differences up
to 1.87e-6, which include GLASS's orbit-spline approximation. Signed-polar
outputs were reconstructed into complex values outside the timed native loop.

Reproduce using the [native build instructions](../tools/native_glass_timing/README.md)
and then:

```sh
PYTHONPATH=src python tools/compare_glass_speed.py --glass-executable /private/tmp/glass-ucb-build/native_ucb_timing --output /tmp/glass-comparison.json
```

## Native GLASS on-fly measurements — September 19, 2026

The [native C measurements](../tools/performance/glass-native.json) use GLASS
commit `079c6a3a6fcf380a7c107f39640fff7fc720d0f1`, Apple clang 15, and
`-O3 -march=native -ffp-contract=off` on this arm64 Mac. The timed upstream
function bodies and KissFFT sources are unchanged. The
[reproduction harness](../tools/native_glass_timing/README.md) builds the
waveform code without unrelated application dependencies.

For the 5 mHz source, median block-mean times were:

| Observation duration | Sparse response samples | XYZ TDI response only | Complete WDM XYZ waveform |
| --- | ---: | ---: | ---: |
| Quarter binary year | 49 | 0.0284 ms | 0.538 ms |
| One binary year | 199 | 0.138 ms | 2.81 ms |
| Four binary years | 797 | 0.644 ms | 14.9 ms |

A GLASS binary year here is 31,457,280 seconds, giving exact WDM grids.
Each case uses three independent processes, each averaging 200 warmed calls;
the table reports the median of those three means. Orbit/basis initialization
and caller output allocation are excluded. Allocations and spline/transform
work inside each waveform call are included. `LISA_spline_response` alone is
timed separately with source splines and reference phases already prepared.

The complete waveform is `ucb_waveform_wavelet`, including interpolation,
heterodyning and the wavelet transform. It returns 2,048, 8,192 or 32,768 active
real XYZ wavelet pixels per channel for these cases. These are different
outputs from this package's complex Fourier bins, so their absolute runtimes
are not an equivalent-output comparison. The record also includes native
Fourier timings at BW512, for cost context only.

Across the tested 1, 5 and 30 mHz sources, complete WDM times were
0.522–0.538 ms for a quarter year, 2.73–2.83 ms for one year, and
14.3–15.1 ms for four years. Those sources share a sky position, orientation,
amplitude and dimensionless chirp; these measurements are not universal bounds.
All measured response amplitudes/phases, active WDM outputs, and Fourier
outputs were checked for finiteness outside the timed loops.

## Historical Fourier measurements

The September 13, 2026 change kept host inputs in NumPy until the compiled
boundary, evaluated the six links together, and expanded the three-component
detector contractions into expressions that the compiler could combine with the
response. These figures apply only to `backend="fourier"`; public default
behavior has since moved to `backend="onfly"`.

These are matched CPU measurements on macOS arm64 with Python 3.14.2,
JAX/jaxlib 0.7.2, and NumPy 2.3.5. Each workload used 100 synchronized calls
after warmup, the same seeded sources, observation grid, and benchmark code.
At bandwidth 512:

| Workload | Before | After | Time reduction |
| --- | ---: | ---: | ---: |
| Native parameters, ordinary call | 306 µs | 160 µs | 48% |
| Physical `GBParams`, ordinary call | 622 µs | 169 µs | 73% |
| Native parameters, outer JIT | 149 µs | 116 µs | 22% |
| Fixed-window A/E template, outer JIT | 140 µs | 104 µs | 26% |
| Batch of 32 sources, outer JIT | 1.47 ms | 1.17 ms | 20% |
| Full A/E Jacobian, outer JIT | 510 µs | 370 µs | 27% |

The fixed window has `BW + 64` bins. Across bandwidths 128, 512, and 2048,
its latency fell by 18–26%. This is the most relevant path for repeated
single-source templates in inference. The measurements exclude subsequent
likelihood arithmetic and do not establish an equal reduction in total sampler
runtime. GPU performance has not been measured.

The [comparison](../tools/performance/comparison.json) links the complete
[before](../tools/performance/before.json) and
[after](../tools/performance/after.json) reports, including raw timings,
first-call costs, source hashes, and inputs. The full package suite passes
362 tests on JAX 0.4.30, 0.7.2, and 0.11.1, including the 36 independent GLASS
reference cases and derivative checks. There are no relaxed accuracy tolerances.

## Measuring your workload

From an installed checkout:

```sh
python tools/benchmark_waveform.py --backend onfly --output /tmp/onfly-workloads.json
python tools/benchmark_waveform.py --backend fourier --output /tmp/fourier-workloads.json
```

The report records the resolved package's source hashes, input catalogue,
environment, and individual timings. First-call time is reported separately
from repeated calls after warmup. Every timed call waits for all returned
arrays, following JAX's [benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
Compare runs with the same bandwidths, catalogue, JAX version, and hardware.
Compilation time, single-source latency, and throughput per source in a batch
answer different performance questions.

## Repeated on-fly calculations

For repeated single-source inference, use `template_on_window` with a fixed
analysis window. A concrete duration uses one automatic response count, chosen
on the host before compiling; an explicit `response_samples` also keeps the
grid fixed when the duration is traced or varies. It compiles waveform
generation and placement together. It forms A and E from the sparse response
before interpolation, so it interpolates and transforms two channels where
`ucb_waveform` transforms X, Y and Z. The default analytic orbit is the most
accurate option:

```python
import jax

import binary_galactica as bg

# Since 0.4.0 the float64 functions raise unless double precision is enabled.
jax.config.update("jax_enable_x64", True)

duration, bandwidth = 7_884_000.0, 512

# `native` is one proposed GLASS vector; this example window covers 1 mHz.
template = bg.template_on_window(
    native,
    T=duration,
    BW=bandwidth,
    bin_lo=7_500,
    n_bins=1_024,
)
```

When integrating with a likelihood, compile the surrounding array calculation
too. Close over a fixed duration, so that the automatic response count is
chosen on the host. Pass duration as an argument only when varying it between
calls, and then pass an explicit `response_samples` as well: with a traced
duration and the automatic grid, every grid is compiled and differentiated
(see above), and the call emits `ResponseSamplesWarning`. Keep the data and
noise arrays on the device across proposals. A legacy `(3, BW, 3)` orbit
cache is accepted and cubically interpolated to response times, but its coarse
grid constrains orbit accuracy; prefer the analytic orbit unless reuse is
required. The benchmark does not include likelihood integration.

- Keep an outer `jax.jit` function alive across calls. Place waveform generation
  and subsequent array operations inside it so intermediate arrays can remain
  on the device. `template_on_window` already has a compiled outer boundary.
- For catalogues, use `jax.jit(jax.vmap(...))` with sources grouped by bandwidth.
  Group different durations too, or use an explicit response sample count.
  Choose a batch size that fits the device's memory; a larger batch also makes
  derivative arrays larger.
- `T` may be dynamic or captured as a constant: the carrier calculation
  preserves GLASS's rounding in both cases. A captured `T`, a Python or NumPy
  scalar or a concrete JAX array, lets public calls choose the automatic
  response count on the host. A dynamic `T` needs an explicit
  `response_samples` to avoid compiling every automatic grid. Bandwidth,
  response sample count, backend, and format are static and changing them can
  require compilation. Derivatives hold the discrete carrier and interpolation
  choices fixed.
- Keep NumPy inputs on the host for ordinary validated calls; use device arrays
  when composing a compiled calculation. Packing each host scalar with JAX
  before every call adds dispatch work.
- Choose bandwidth by waveform convergence. Reducing bandwidth or precision
  changes the accuracy question and is not part of these optimizations.

For example, compile a reusable A/E catalogue function once:

```python
from functools import partial

import jax
import jax.numpy as jnp
import binary_galactica as bg


@partial(jax.jit, static_argnames=("bandwidth", "samples"))
def catalogue_channels(sources, duration, epoch, *, bandwidth, samples):
    def one(source):
        wave = bg.ucb_waveform(
            source, T=duration, t0=epoch, BW=bandwidth, response_samples=samples
        )
        return jnp.stack((wave["A"], wave["E"]))

    return jax.vmap(one)(sources)
```

Here `sources` holds native vectors and the result has shape
`(number_of_sources, 2, bandwidth)` in raw GLASS units. `duration` is a traced
argument, so the static `samples` fixes the response grid: the automatic policy
would choose 32 for a quarter year, 128 for a year and 512 for four years, but
check convergence for the intended sources. Without it, every automatic grid
would be compiled. If the duration never changes, close over it instead and
omit `response_samples`; the count is then chosen on the host. An A/E-only
calculation can discard unused outputs during compilation. Ordinary public calls still
perform eager input validation; traced invalid values retain the documented
NaN behavior.
