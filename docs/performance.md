# Waveform performance

The package's main computation is the GLASS fast-slow waveform. The performance
benchmark exercises its native and physical APIs, fixed-window templates,
catalogue batches, and waveform Jacobians. It does not measure a sampler or a
likelihood implementation, neither of which is part of this package. The
experimental zero finder under `tools/tdi_zero_crossings` is separate from these
waveform paths.

## Measured optimization

The September 13, 2026 change keeps host inputs in NumPy until the compiled
boundary, evaluates the six links together, and expands the three-component
detector contractions into expressions that the compiler can combine with
the response. Public APIs, float64 precision, bandwidth, and GLASS conventions
are preserved.

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
python tools/benchmark_waveform.py --output /tmp/waveform-performance.json
```

The report records the resolved package's source hashes, input catalogue,
environment, and individual timings. First-call time is reported separately
from repeated calls after warmup. Every timed call waits for all returned
arrays, following JAX's [benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
Compare runs with the same bandwidths, catalogue, JAX version, and hardware.
Compilation time, single-source latency, and throughput per source in a batch
answer different performance questions.

## Repeated calculations

For repeated single-source inference, use `template_on_window` with cached
positions and a fixed analysis window. It already compiles waveform generation
and placement together. For example, create the orbit cache once, then evaluate
each proposed native vector with the same static arguments:

```python
import binary_galactica as bg

duration, bandwidth = 7_884_000.0, 512
positions = bg.orbit_cache(duration, 0.0, bandwidth)

# `native` is one proposed GLASS vector; this example window covers 1 mHz.
template = bg.template_on_window(
    native,
    T=duration,
    BW=bandwidth,
    positions=positions,
    bin_lo=7_500,
    n_bins=1_024,
)
```

When integrating with a likelihood, compile the surrounding array calculation
too, and pass duration as an argument to that compiled function. Keep its data,
noise arrays, and cached positions on the device across proposals. The waveform
benchmark does not include this integration.

- Reuse `orbit_cache(T, t0, BW)` across sources on the same observation grid.
- Keep an outer `jax.jit` function alive across calls. Place waveform generation
  and subsequent array operations inside it so intermediate arrays can remain
  on the device. `template_on_window` already has a compiled outer boundary.
- For catalogues, use `jax.jit(jax.vmap(...))` with sources grouped by bandwidth.
  Choose a batch size that fits the device's memory; a larger batch also makes
  derivative arrays larger.
- Pass `T` as a dynamic argument to compiled functions to preserve GLASS's
  carrier-bin rounding. Bandwidth and format are static and changing them can
  require compilation. The README documents the carrier and derivative limits.
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


@partial(jax.jit, static_argnames=("bandwidth",))
def catalogue_channels(sources, duration, epoch, positions, *, bandwidth):
    def one(source):
        wave = bg.ucb_waveform(
            source, T=duration, t0=epoch, BW=bandwidth, positions=positions
        )
        return jnp.stack((wave["A"], wave["E"]))

    return jax.vmap(one)(sources)
```

Here `sources` holds native vectors and the result has shape
`(number_of_sources, 2, bandwidth)` in raw GLASS units. An A/E-only calculation
can discard unused outputs during compilation. Ordinary public calls still
perform eager input validation; traced invalid values retain the documented
NaN behavior.
