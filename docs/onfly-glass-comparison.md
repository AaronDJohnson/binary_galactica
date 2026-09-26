# Why sparse TDI is slower for some small Fourier bands

The September 19, 2026 diagnosis identifies a concrete compiler cost: XLA
fuses expensive response calculations into an overlapping interpolation gather,
repeating work that should be shared between adjacent interpolation windows.
A controlled change that forces the sparse values to be materialized reduces
latency without changing the samples, interpolation degree, or measured output.
Small FFT savings and automatic-grid compilation choices also matter. These
results explain the observed crossover; they do not establish a universal
speed ordering. This investigation does not change the production backend.

The [TDI on the fly paper, section III.1][paper] explicitly constructs a
Fourier-domain galactic-binary response by sparse TDI evaluation, heterodyning,
and an FFT. Its example uses 200 response samples and 512 heterodyned samples.
Its large speedup compares against evaluation at the full data cadence, not
every possible fast-slow reduced-FFT workload. Fourier output is a supported
application of the method; it does not require switching to wavelets.

The source findings below come from local Git object
`079c6a3a6fcf380a7c107f39640fff7fc720d0f1` (May 29, 2026), the revision pinned
by this package. The inspected `/Users/aaron/Documents/lisa/ldasoft` working
checkout instead has HEAD `dbfd2cb2171c1fcf18984f8d7abb03c2983b66d3`
(March 19, 2024): it contains the older `galactic_binary` implementation and
six GSL FFTs, without the spline response. Public source links use the Git
remote `https://github.com/tlittenberg/ldasoft`; source verification used the
local pinned object, since browser retrieval of those URLs failed.

At the pinned revision, the [repository benchmark][benchmark] times
`ucb_waveform_wavelet` at lines 118–131 and Fourier `ucb_waveform` at 158–178.
It fixes a roughly one-year observation and one sky/inclination/mass/distance
configuration, repeats identical inputs 1,000 times, and chooses Fourier
bandwidth with `2*ucb_bandwidth(...,2048)`. Orbit and wavelet initialization
at lines 64 and 97 is outside timing. These outputs and bandwidth choices
differ from our matched Fourier-output diagnostic; the benchmark does not
demonstrate that sparse TDI wins at every fixed `BW`.

The [GLASS response grid][orbit] has `max(20,int(200*T/YEAR))` samples: 199
for that benchmark. `LISA_spline_response` reads precomputed coordinate splines
at lines 979–981. It combines eight retarded source evaluations per channel
using geometry at the reception time, then extracts amplitude and phase.
The Fourier path evaluates six links at each of `BW` times and calls six
complex FFTs ([waveform lines 838–945][waveform]). The benchmark leaves its
orbit callback set to analytic orbits; its spline response uses cached orbit
splines. These are different geometry costs as well as different transforms.

The pinned production spline caller is `ucb_waveform_wavelet`, not a function
named `ucb_waveform_spline`. It builds value-based natural cubic amplitude/phase
interpolants, without automatic differentiation of the response. Its dense
length is `NT*(Nlayers+1)`; each XYZ channel performs a real forward FFT and
one inverse complex FFT per retained layer ([wavelet lines 544–584][wavelet]).
The table variant is disabled: `ucb_waveform_wavelet_tab` exits at waveform
lines 1616–1617, and its benchmark block is commented out. The six Fourier
FFT wrappers allocate/free plans and copy buffers on each call
([math lines 703–729][math]); that cost should not be assumed for JAX FFTs.

Our measurements use JAX 0.7.2, NumPy 2.3.5, arm64 CPU execution, a source
near 3 mHz, `T=7,884,000 s`, and `t0=12,345 s`. The automatic grid chooses
32 points. Each function is explicitly compiled, warmed, and synchronized
for 300 repetitions. These diagnostics return **XYZ only**, with dynamic
duration and epoch, unlike the earlier public calls returning all channels
and bin metadata in [performance.md](performance.md). Their absolute timings
should not be substituted into that earlier table. Source and data are in
[diagnose_onfly.py](../tools/diagnose_onfly.py) and the
[diagnostic record](../tools/performance/onfly-diagnosis.json).

| Precompiled XYZ median, microseconds | BW 128 | BW 512 | BW 2048 |
| --- | ---: | ---: | ---: |
| Fourier | 40.63 | 97.90 | 280.85 |
| On-fly, automatic 32 | 80.27 | 107.56 | 200.65 |
| On-fly, explicit 32 | 78.08 | 87.69 | 138.31 |

Automatic and explicit counts evaluate the same 32-node grid here, but their
compiled control flow differs. Fixing the count helps some scalar cases, but
the small-band slowdown remains. These ablations concern single-source values;
the batch/Jacobian results in performance.md concern the production backends.
Standalone three/six-channel FFT measurements show that halving small FFT
counts saves little compared with the response/interpolation costs.
Standalone raw-response, interpolation, phase, and FFT timings are **not
additive** because separately compiled functions have different fusion and
dispatch behavior.

Optimized HLO gives stronger evidence than component timing alone. With 32
knots, eight-point interpolation gathers 25 overlapping windows: 200 entries
per channel from only 32 unique values. The original `copy_gather_fusion`
produces `c128[200,1,3]` and contains the response exponentials/trigonometry.
Response arithmetic is therefore inside the expanded gather loop, permitting
recomputation for repeated knots instead of reusing a materialized array.
The conditional diagnostic separates a `c128[32,3]` result before that gather.

| Controlled ablation median, microseconds | BW 128 | BW 512 | BW 2048 |
| --- | ---: | ---: | ---: |
| Original explicit-32 path | 78.08 | 87.69 | 138.31 |
| Optimization barrier | 78.29 | 87.35 | 139.83 |
| Conditional materialization | 51.00 | 56.77 | 114.58 |
| Equivalent dense interpolation matrix | 48.56 | 75.29 | 186.44 |

The experiment randomizes execution order over 300 repetitions, including
both backends and all variants in the same timing loop. The conditional has
no observed numerical difference and reduces latency by 35%, 35%, and 17%;
an optimization barrier alone does not help. The matrix
version agrees within 1.9e-15 relative L2 but becomes slower at large bandwidth.
These are diagnostic ablations, not adopted production changes or general
recommendations to add conditionals or dense matrices.

The conditional also changes other compiler decisions, so the ablation does
not assign every saved microsecond to duplicated arithmetic. A production fix
must preserve source derivatives, Hessians, batching and minimum-JAX support;
none of those contracts is established by this diagnostic fixture. Native C
measurements are recorded separately in
[performance.md](performance.md#native-glass-on-fly-measurements--september-19-2026);
they distinguish the response kernel from the complete WDM waveform.

Reproduce the measurements and inspect the optimized HLO with:

```sh
PYTHONPATH=src python tools/diagnose_onfly.py --output /tmp/diagnosis.json --hlo-directory /tmp/onfly-hlo
```

Sparse TDI also has a real per-node cost: eight delayed contributions for each
of three channels versus six links per fast-slow sample. At 32 sparse nodes
and BW 128, these contribution counts already match before interpolation and
dense phase restoration. Counts are not equivalent-cost operations, but they
explain why sparsity alone cannot guarantee a small-band speedup. The HLO and
controlled ablation establish an additional, avoidable implementation cost;
larger bands amortize the sparse response work and favor the current method.

[paper]: https://arxiv.org/html/2506.08093v1#S3.SS1
[benchmark]: https://github.com/tlittenberg/ldasoft/blob/079c6a3a6fcf380a7c107f39640fff7fc720d0f1/apps/src/ucb_waveform_benchmark.c#L118
[orbit]: https://github.com/tlittenberg/ldasoft/blob/079c6a3a6fcf380a7c107f39640fff7fc720d0f1/utils/src/glass_lisa.c#L271
[waveform]: https://github.com/tlittenberg/ldasoft/blob/079c6a3a6fcf380a7c107f39640fff7fc720d0f1/ucb/src/glass_ucb_waveform.c#L738
[wavelet]: https://github.com/tlittenberg/ldasoft/blob/079c6a3a6fcf380a7c107f39640fff7fc720d0f1/utils/src/glass_wavelet.c#L544
[math]: https://github.com/tlittenberg/ldasoft/blob/079c6a3a6fcf380a7c107f39640fff7fc720d0f1/utils/src/glass_math.c#L703
