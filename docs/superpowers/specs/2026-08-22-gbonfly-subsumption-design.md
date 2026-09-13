# Subsuming `gbonfly` — design

**Date:** 2026-08-22
**Package:** `binary-galactica`
**Status:** superseded by [the September 13 migration](2026-09-13-gbonfly-glass-migration.md).

The normalization prescription below refers to older source code. Current
GLASS uses half inclination factors together with `1/(2*BW)`. Following the
older paired correction would preserve the donor's factor-of-two discrepancy.
See the newer specification and committed C reference fixtures for the
implemented conventions; the remainder is retained as historical design context.

## Purpose

Make `binary-galactica` an entire segment of the LISA global fit by subsuming
`gbonfly` (`~/Documents/GitHub/wdm_test/wdm_paper_2/gbonfly`), the JAX
transcription of GLASS's fast-slow galactic-binary TDI generator. The result
carries the whole chain a segment needs to produce a template — source model,
orbit, response, TDI, band selection — together with the prior and proposal
machinery a sampler drives it with.

## Scope

**In:** every module of `gbonfly`, reconciled with this package's conventions;
one bridge between the physical and GLASS-native parameter vectors; a split of
the two constant sets that currently collide; and a written analysis of what a
global-fit segment still lacks.

**Out:** the likelihood, the data/segment container, and the sampler. They are
named in [Gaps](#gaps) with enough precision to be scoped later, and they are
not built here.

**Out, deliberately:** WDM. `gbonfly` contains no wavelet code at all — a grep
for `wdm|wavelet|Nf|Nt|pixel` over all twelve modules returns nothing. The
transform will come from `wdm` (`~/Documents/lisa/wdm`), which is NumPy/SciPy and
requires Python 3.13; wiring it is a segment-layer concern and belongs with the
likelihood. This package gains no dependency on it. The seam is specified in
[The WDM seam](#the-wdm-seam) so that the eventual wiring has a contract to
meet rather than a discovery to make.

**Out, deliberately:** $\ddot{f}$. The public parameter space stays the
standard 8D one, as in the [waveform design](2026-08-07-galactic-binary-waveform-design.md).
GLASS's `d2fdt2` survives only on the GLASS-native path, where it is part of
the transcription.

### Consequence: the package is no longer "the source model only"

The [waveform design](2026-08-07-galactic-binary-waveform-design.md) put
orbits, arm projection, delays and TDI out of scope and delegated them
downstream. There is no longer a downstream to delegate to — this package is
the segment. That scope statement is withdrawn here and in the README.

What survives is the *reason* for it: `polarizations` remains a single,
self-contained entry point for the source physics, and richer source models
still slot in beside it. It moves to `source.py`, because the name
`waveform.py` is needed for the generator that now dominates the package.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope of this work | port and reconcile; document gaps | Merging and correcting conventions is enough work to be its own change |
| Parameter convention | both, one tested bijection | Priors are genuinely flat in $\cos\iota$, $\cos\theta$, $\ln A$; science is not |
| Constants | namespace apart | The collision is a naming problem, not a physics one |
| Validation | freeze goldens, revive `ucb_ref` | The GLASS anchor is missing; a port must not be the thing that hides it |
| Layout | flat, roles named | `gbonfly`'s module names port with least churn; gaps read as absent modules |
| Amplitude prior | GLASS's log-uniform | This is a segment of a GLASS-style global fit; match its prior |
| Intermediate factors | corrected | See [The factor of two](#the-factor-of-two) |
| `numpy` | runtime dependency | It is the host side of the JAX static/traced boundary, not incidental |

## Architecture

```
src/binary_galactica/
├── __init__.py       public API; enables x64 (once, for the whole package)
├── constants.py      PUBLIC, physical: YEAR (LDC), DAY, CLIGHT, AU, PC, TSUN
├── parameters.py     GBParams  <->  (8,) array  <->  GLASS-native vector
├── source.py         phase, frequency, amplitudes, polarizations
├── orbit.py          analytic_orbits, arm_vectors, slow_time_grid
├── response.py       polarization_tensor, detector_tensor
├── waveform.py       ucb_waveform, orbit_cache        -- the fast-slow generator
├── tdi.py            synthesize, xyz_to_ae, xyz_to_aet, FORMATS
├── bandwidth.py      ucb_bandwidth, ucb_bw, analytic_snr, ae_noise
├── alignment.py      Alignment, ucb_alignment
├── convert.py        to_continuous_ft, GLASS_TO_CONTINUOUS
├── compat.py         template_on_window               -- the segment/WDM seam
├── prior.py          UCBPrior, ucb_prior, ucb_fdot, snr_cut
├── proposals.py      FisherProposal, ProposalMix, FisherCache, ...
└── glass/
    └── constants.py  GLASS's literals, bit-pinned
```

`source.py` is the old `waveform.py`, renamed. Everything from `orbit.py`
down is `gbonfly`, with imports rewritten and the house style applied.

### On the host/device boundary

Six of `gbonfly`'s modules import NumPy, and that is deliberate rather than
sloppy. `BW` is simultaneously the slow-time sample count and the output bin
count, so it is an array *shape* and must be a Python `int` before tracing;
`alignment.py` computes `q = int(np.trunc(f0 * T))` on the host for the same
reason; `prior.py` and `proposals.py` evaluate on the host between likelihood
calls. Porting these to `jnp` would move the static/traced line and break the
generator. `numpy` therefore becomes a runtime dependency and the README's
claim that JAX is the only one is withdrawn.

Two properties are load-bearing and must survive the move:

* **`T` is traced, never static.** Given a compile-time divisor XLA rewrites
  `params[0]/T` into `params[0]*(1/T)`, a different float64 result — 377 of
  3000 divisions stop matching IEEE, and the carrier bin moves with them.
* **`BW`, `bin_lo`, `n_bins`, `fmt` are static.** They are shapes and branches.

## The two coordinate systems

`parameters.py` owns the only conversion between them. No other module may
construct or consume a GLASS-native vector.

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **physical** (`GBParams`) | `amp` | `f0` | `fdot` | `phi0` | `iota` | `psi` | `lam` | `beta` |
| **GLASS-native** | $f_0T$ | $\cos\theta$ | $\lambda$ | $\ln A$ | $\cos\iota$ | $\psi$ | $\phi_0$ | $\dot f T^2$ |

Both are plain `(8,)` float arrays and the orderings collide silently: index 3
is `phi0` in one and $\ln A$ in the other, index 6 is `lam` in one and $\phi_0$
in the other. Nothing raises; the waveform is simply wrong. Confining the
native form to two functions, `to_glass_vector(params, T)` and
`from_glass_vector(v, T)`, is what makes that unrepresentable.

The bijection carries `T` because $f_0T$ and $\dot f T^2$ are meaningless
without it, and it is where every remaining convention difference is settled:
$\cos\theta = \sin\beta$; $\ln A$ against `amp`; the two $T$-scalings; and the
sign of $\phi_0$.

### On the sign of $\phi_0$

`ldasoft` carries a comment at `glass_ucb_waveform.c:640` reading
"`-phi0`, hence the `-phi0` in arg2", and the line below it is

```c
double arg2 = PI2*f0*xi[i] - phi0 - df*t;
```

against this package's $\Phi(t) = \phi_0 + 2\pi f_0 t + \pi\dot f t^2$
contracted with $e^{-i\Phi}$. Whether those agree depends on how the envelope's
$e^{-i(\mathrm{arg1}-\mathrm{arg2})}$ composes with the $DP/DC$ contraction,
and the two published conventions (`ldasoft` and `GBGPU`) are reported to
differ here.

This spec deliberately does not assert an answer. The bijection's round-trip
test settles it numerically — perturb $\phi_0$, generate both ways, and read
the sign off the result — and whatever it finds is recorded in `parameters.py`
beside the conversion. An algebraic claim in a design document is exactly the
kind of thing that gets copied forward without being rechecked.

## The factor of two

`gbonfly` builds its polarization amplitudes as

```python
Aplus = amp * (1.0 + cosi * cosi) * 0.5  # waveform.py:187
Across = -amp * cosi  # waveform.py:188
```

under a comment describing them as "GLASS's factors, not this repo's". They
are not GLASS's factors. `glass_ucb_waveform.c:534` reads

```c
Aplus  =  amp*(1.+cosi*cosi);
Across = -amp*(2.0*cosi);
```

which is this package's existing convention, already pinned by
`test_conventions.py`. `gbonfly` halves both, and then scales its reduced DFT
by `1/BW` where the C uses `invBW2 = 1./(double)BW2` with `BW2 = BW*2`, i.e.
$1/(2BW)$. The two factors cancel exactly, which is why the generator validates
against the compiled binary at ratio 1.0 with every intermediate amplitude
wrong by a factor of two.

`gbonfly`'s README records the symptom without the cause: "I could not localise
the compensating factor by reading the C." The compensating factor is not in
the C. It is the halved amplitude `gbonfly` introduced.

Both are corrected here: `ucb_waveform` calls `source.amplitudes()` and the DFT
scale becomes $1/(2BW)$. **In one commit** — either change alone moves the
output by a factor of two — and guarded by a golden captured beforehand, so the
output is proved unchanged while the intermediates become GLASS's for the first
time.

## Constants

Two sets, differing in the eleventh digit, both currently called `YEAR`.

`constants.py` keeps the physical, LDC-facing set the package already
documents: `YEAR = 31558149.763545603`, `DAY`, and the additional values the
generator needs (`CLIGHT`, `AU`, `PC`, `TSUN`).

`glass/constants.py` keeps GLASS's literals bit-for-bit —
`YEAR = 3.15581497632e7`, `SQ3 = 1.73205080757` (truncated, not $\sqrt3$),
`AU`, `VEARTH`, the noise levels — and `orbit.py`, `waveform.py` and `tdi.py`
import from it explicitly. A reader of those modules can see which set they are
in.

No mode switch. The divergence reaches the waveform only through the orbit,
where it moves the spacecraft by 736 m and changes the template by
$1-FF = 4\times10^{-16}$, and through the bandwidth heuristic, where the
Doppler term is rounded to a power of two and cannot move. `gbonfly`'s test
asserting both bounds moves across unchanged, as does its bit-for-bit pin
against the GLASS headers — so "correcting" `SQ3` still fails loudly.

## Testing

`gbonfly` collects 122 tests: 103 pass, 19 skip. The 19 are the two files that
anchor absolute scale, and they point at
`/Users/aaron/Documents/lisa/global_fit_pieces/`, which no longer exists
anywhere on this machine. The headline claim — ratio 1.0000000000 against the
compiled binary — is currently unverifiable.

The move would quietly make that worse: `conftest.REPO = Path(__file__).resolve().parents[2]`
resolves to `wdm_paper_2`, where `toy_models` lives, and would resolve to
`/Users/aaron/Documents/lisa`, where it does not. Tests would go dark by
changing directory.

So validation is rebuilt first, before any code moves:

- **Freeze goldens**, with `gbonfly` still beside `toy_models`: A/E spectra
  over an $f_0 \times BW \times T$ grid, the Sangria PSD, the injected-SNR
  anchor, and the current-generator comparison, vendored to `tests/data/`.
  These become permanent — the package stops needing an external checkout.
- **Revive `ucb_ref`**, built from `/Users/aaron/Documents/lisa/ldasoft`, which
  does exist and whose source is quoted throughout this document. `UCB_REF` is
  repointed and the four compiled-binary tests come back.
- **Delete** the 15 tests reaching for the vanished Python transcription. The
  compiled binary supersedes them; a port cannot settle absolute scale anyway.
- **Fix before moving:** `from conftest import INJECTED_SNR`, which dies under
  `--import-mode=importlib`; and `test_constants.py`'s dependence on a
  `sys.path` side effect from another test file.

Then, unchanged in spirit from this package's existing suite:

- **Conventions:** the amplitude and DFT correction is pinned two ways — the
  golden proves the output did not move, `test_conventions.py` proves the
  intermediates now match `glass_ucb_waveform.c:534`.
- **The bijection** (`test_parameters.py`): round-trip on random draws;
  $\phi_0$'s sign read off numerically; a case asserting that a physical vector
  fed to the native path is caught rather than silently generating.
- **Doctests:** `--doctest-modules` is on and `testpaths` includes `src`, so
  every incoming `>>>` example runs. `gbonfly`'s module docstrings are ported
  with their examples made executable or removed.

## Gaps

What a global-fit segment still needs, and neither package has.

| Gap | State |
|---|---|
| Likelihood | absent |
| Segment / data container | absent; `compat.template_on_window` implies an external `Segment.template` returning `(2, n_bins)` |
| Noise model | analytic `ae_noise` only; no fitted PSD |
| RJMCMC over source count | absent |
| F-statistic proposal | **not ported.** GLASS gives it weight 0.4, of which 0.8 seeds reversible-jump births |
| Differential evolution, parallel tempering | absent |
| `snr_cut` | defined, never called |
| TDI | first-generation Michelson X/Y/Z and the Sangria format; no second-generation |
| Catalogue I/O | absent |
| $\ddot f$ | GLASS-native path only |

`proposals.single_source_mix` is, as its docstring says, "for a run that
samples one binary rather than searching a band". A segment searches a band.
The F-statistic proposal and the birth/death moves are the same gap seen twice.

### The WDM seam

`compat.template_on_window(params, *, T, bin_lo, n_bins, BW, ...) -> (2, n_bins)`
is the single point where a transform would attach. A `WDMTransform` protocol
covering it needs four things:

1. `Nt` and the filter half-support `h`. Bands sit $N_t/2$ rfft bins apart, so
   `bin_lo = (Nt//2)*m_lo - h` and `n_bins = (Nt//2)*m_hi + h - bin_lo + 1`.
   Measured from the transform at runtime, never hard-coded.
2. The input convention. This is the trap: the transform wants a bare rfft,
   while `to_continuous_ft` applies $\sqrt T$. Composed, the segment needs
   $\tilde h_{\rm rfft} = \sqrt{T}\,\tilde h_{\rm GLASS}/\delta t$. Omitting
   the $\delta t$ is a smooth factor of $\delta t^2$ — 25× at the shipped
   cadence — and not a crash.
3. `forward(spectrum) -> (nchan, n_band, Nt)`, with its Parseval statement.
4. Whether it is JAX-traceable, if the segment likelihood is to have gradients.
   `wdm` is NumPy, so today it is not.

Note also that `BW` should be passed explicitly rather than left to
`ucb_bandwidth`, which returns 32 on the shipped segment against the ~512 that
$1-FF < 1/(2\rho^2)$ needs at $\rho = 9.89$.

## Sequence

1. Freeze goldens and revive `ucb_ref`, with `gbonfly` untouched in place.
2. Land the modules, imports rewritten, house style applied; tests green
   against the goldens.
3. `parameters.py` bijection and its round-trip tests; $\phi_0$ settled.
4. The amplitude and $1/(2BW)$ correction, one commit, golden-guarded.
5. The constants split.
6. README and this spec's gap analysis carried into the docs.

## Future work

The gaps above, in the order a segment needs them: the data container and
likelihood first, since everything else is driven through them; then the
F-statistic proposal and reversible jump, which are what turn a single-source
sampler into a band search; then the WDM wiring, against the protocol above.
