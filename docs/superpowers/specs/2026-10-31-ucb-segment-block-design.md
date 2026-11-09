# GLASS-style UCB segment block for Sangria

Sample galactic-binary (UCB) frequency segments of the LDC2a Sangria data the
way GLASS's global fit did, as one block of the user's blocked-Gibbs sampler
`enchilada` (`/Users/aaron/Documents/lisa/enchilada`). Trans-dimensional moves
come from the reversible-jump sampler being added to `impulse`
(`~/Documents/GitHub/impulse_mcmc`, branch `rjmcmc`); none is written here.
Work on branch `ucb-segment-block`, based on `007bd89` (includes the
physical-amplitude fix).

## Decisions

| Topic | Decision |
|---|---|
| Home | New `binary_galactica` subpackage. The README's "likelihoods stay outside this package" statement is amended in the same change. |
| Data | `obs/tdi` minus the true `sky/mbhb/tdi` and `sky/vgb/tdi` streams: galactic binaries (`dgb` + `igb`) plus instrument noise. Built by the application, not this package. |
| Duration | 12 months. The block takes `T` from the L1Data it is given. |
| Coverage | Full UCB band, GLASS-like variable segment widths. |
| Start | Sampling starts at injected values. The start set is supplied as LDC catalogue rows by the user's noise-block package. |
| Dropped from GLASS | GMM/covariance proposals from earlier epochs, the F-statistic grid (births and jumps), mode cloning, time-based load balancing. |
| Noise | Fixed within each Gibbs turn, supplied by enchilada as a native frequency covariance. No noise parameters in the block. |
| Granularity | One enchilada block holding every segment, like GLASS's UCB module. |
| SNR cut | Kept: a state with any active source below SNR 5 has zero posterior, without renormalizing the prior. SNR is measured against a reference PSD fixed for the run (see Segment model). |

## What GLASS does, and what is kept

From the pinned GLASS source (ldasoft `079c6a3`; segment geometry from
`dbfd2cb`, which is self-consistent where `079c6a3`'s `global_fit` is not) and
Littenberg & Cornish, PRD 107, 063004 (2023):

- Segments have an interior plus `qpad` bins of padding per side
  (`globalfit/src/glass_ucb_wrapper.c`). The frequency prior covers the padded
  band; a segment owns only sources with `qmin+qpad < f0*T < qmax-qpad`.
  Owned sources of the two neighbours are subtracted from its data; others are
  ignored; templates are truncated at the segment edges. **Kept.**
- Likelihood: Gaussian over the segment bins with the PSD handed down by the
  noise block each sweep; no per-segment noise parameters in the global fit.
  **Kept**, with enchilada supplying the PSD.
- RJMCMC: 25% of moves are birth/death (`globalfit/src/glass_ucb_wrapper.c`,
  the update loop), births from the prior (0.2) or F-statistic (0.8)
  (`initialize_ucb_proposal`). A fixed-dimension proposal or a birth whose
  moved or new source has SNR < 5 under the current noise model is rejected
  (`ucb/src/glass_ucb_sampler.c`, `ucb_mcmc` and `rj_birth_death`).
  **Kept**, births from the prior only, uniform prior on the count, the cut
  applied to the state (below).
- Fixed-dimension moves on one source (`ucb/src/glass_ucb_proposal.c`,
  `initialize_ucb_proposal`): Fisher 0.3, F-statistic draw 0.2 and jump 0.2,
  prior 0.1, fm shift 0.1, psi-phi jump 0.1. No differential evolution.
  **Kept without the F-statistic**, renormalized.
- 12 tempered chains, ladder starting at `1.2**i`, adaptive. **Kept**, through
  impulse's ladder adaptation.
- Waveform: the Fourier fast-slow generator (`globalfit.c:682`). **Kept as the
  default** (`backend="fourier"`) unless the conventions gate favours the
  on-fly backend.

## The block

One `UCBBlock` implements enchilada's `Block` protocol and registers with
`data_domain="frequency"`. Each `sample()` call receives the conditional
residual (observed data minus every other block's signal; this block's own
signal is still present) in `dt*rfft` units, and a native frequency-domain
`DataCovariance`, from which it takes the A and E PSDs on each segment's bins.
It publishes one `BlockState`:

- `template`: the sum of every segment's owned sources at that segment's
  cold-chain state, each rendered over its full band on the rfft grid, in every
  data channel (T derived from X/Y/Z when the data has a T channel);
- `parameters`: per segment, the owned sources in GLASS-native and LDC form,
  and their count.

The block is constructed from the L1Data grid (`T`, `t0`, channels), the start
set (LDC catalogue rows with the `sky/dgb/cat` fields `Name, Amplitude,
EclipticLatitude, EclipticLongitude, Frequency, FrequencyDerivative,
Inclination, InitialPhase, Polarization`), and a configuration file recorded
verbatim in its output directory. It owns each segment's impulse sampler,
Fisher caches and chain files.

The likelihood uses A and E, as in the paper. T is published but not fitted.
The block does not build L1Data, detrend or taper, choose `T`, compute the
start set or the PSD, or model MBHBs, VGBs or noise. It does not import
enchilada at runtime: `Block` is structural and states are built with
`conditional_residual.block_state(...)`. impulse is an optional extra.

## Segment table

Built once at construction from the start set and `T`, fixed for the run.

- **Band:** covers every start-set source, rounded out to whole segments;
  explicit edges are configurable.
- **Interior widths:** a greedy tiling upward in frequency. Each interior is
  the narrowest width, at least a configurable minimum, holding at most
  `N_target` start-set sources (default 10; GLASS averaged about 14 detections
  per segment). Interiors tile the band without gaps.
- **Padding:** per side, twice the source bandwidth from the Doppler term of
  GLASS's bandwidth rule (`glass/bandwidth.py`),
  `bw = 2T[(4 + 2*pi*f*AU*sin(theta)/c)/yr + |fdot|*T]` bins, evaluated at the
  segment's upper edge with `sin(theta) = 1` and `|fdot|` the largest among
  the start-set sources the segment owns (zero if none). The prior's `fdot`
  envelope is not used: its upper bound is about ten times the GR chirp of a
  0.3 solar-mass binary at every frequency. An interior is never narrower
  than its padding, so padding overlaps only adjacent segments; this
  constraint wins over `N_target`.
- **Ownership:** a source belongs to the segment whose half-open interior
  `[lo, hi)` contains `f0*T`. Each segment starts with its owned start-set
  sources.
- **Prior support:** the frequency prior spans the padded band. A source moved
  or born into the padding is fitted but not published, as in GLASS.
- **Per-segment static settings:** BW is a power of two from GLASS's rule at the
  segment's upper edge, its SNR term set by the brightest start source, capped
  at the padded width. The maximum count is `K = max(10, 2 * initial owned
  count)`.

The table and settings are written to the output directory.

## Segment model

- **Parameters:** each source is a GLASS-native 8-vector
  `[f0*T, cos(colat), lon, ln A, cos(iota), psi, phi0, fdot*T**2]`, with
  periodic `lon` (2*pi), `psi` (pi) and `phi0` (2*pi). No shared parameters.
- **LDC conversion:** LDC catalogue rows are converted to native vectors by one
  tested function; the conventions gate validates it.
- **`lnlike(x, k)`:** render all `K` slots with `template_on_window` on the
  padded bins with the segment's static BW, mask inactive slots with
  `jnp.where`, and sum. With residual `r`, template `h` and coefficient variance
  `C = (T/2)*S` (enchilada's convention), the value is
  `-sum_{A,E} sum_bins |r - h|**2 / C`; the log-determinant term is constant
  within a turn and dropped. Jitted, vectorized over temperatures, float64.
- **SNR cut:** the value is `-inf` if any active source has
  `SNR**2 = 2 * sum_{A,E} sum_bins |h_i|**2 / C_ref < SNR_MIN**2`
  (`SNR_MIN = 5` from `glass/prior.py`), with `h_i` that source's template on
  the padded bins. `C_ref` comes from a reference PSD fixed for the run,
  preferably passed at construction (for example the noise block's starting
  PSD), otherwise the first turn's. GLASS tests the moved
  source against the current noise model instead; a fixed reference keeps the
  truncated target the same in every turn, so a noise update can never leave
  a chain's current state at `-inf` (which would also break swap
  acceptance). Start-set sources below the cut under the reference are
  dropped before sampling and recorded. The cut truncates the component prior
  without renormalizing it, as in GLASS, so the posterior on the count is not
  exactly that of the stated prior. It does not apply in prior-only runs,
  where `lnlike` is off.
- **`lnprior(x, k)`:** `log p(k)` uniform on `0..K`, plus
  `UCBPrior.log_prior` of each active source over the padded band (GLASS's
  uniform box, already normalized).
- **Birth/death:** impulse's built-in move with an
  `IndependentComponentProposal` wrapping the segment's `UCBPrior.sample` and
  `UCBPrior.log_prior`, weighted so about 25% of moves are trans-dimensional.
  Component redraw off.
- **Within-model moves:** custom impulse jumps, each choosing one active source
  uniformly from `chain_stats.current_sample` and returning the current state
  at `k = 0`. Weights: Fisher 0.5; prior redraw of that source, fm shift and
  psi-phi jump 0.167 each. All are symmetric (`qxy = 0`):
  - fm shift, as `fm_shift` at `079c6a3`: `f0*T += s*T/yr` with `s` uniform
    on `±1, ±2, ±3`, and sky longitude `+= U[-0.2, 0.2]`;
  - psi-phi jump, as `psi_phi_jump`: `d ~ U[0, 2*pi)`, `psi += ±d/2`,
    `phi0 += ±d`, independent signs;
  - prior redraw: the source's 8 parameters from `UCBPrior.sample`, symmetric
    because the prior is uniform on its box.

  impulse's AM, SCAM and DE moves are off by default and configurable.
- **Fisher matrices:** at the start of each turn, for every chain and active
  source, `F = 2 Re sum_{A,E} sum_bins (dh/dx)^H (dh/dx) / C` from JAX
  derivatives of the template, stored with the source's `f0*T`; eigenvalues
  are floored so that no unit step exceeds the prior width. A Fisher jump on
  the chosen source takes the stored matrix of the same chain with the nearest
  stored frequency, if within a configurable distance (default 4 bins), and
  moves along one uniformly chosen eigenvector by `N(0,1)/sqrt(eigenvalue)`, as
  `draw_from_fisher` does, without temperature scaling. A jump leaving the
  prior box (such as `|cos(colat)| > 1`) is rejected by the prior; GLASS's
  sign flip at the cosine bounds is not copied, because it is not symmetric.
  The jump is also rejected
  if the proposed source would be assigned a different matrix; a source with
  no matrix in range (for example one born during the turn) is left
  unchanged. The assignment depends only on the source's own frequency and on
  matrices fixed for the turn, so the jump is symmetric however births,
  deaths and swaps rearrange slots. GLASS refreshes the matrix after accepted
  moves, which is not reversible. Here each turn's kernel leaves that turn's
  conditional invariant; refreshing between turns is adaptation that does not
  diminish, as in GLASS.

## One turn

One `sample()` call is one Gibbs turn:

1. Check that the residual and covariance are frequency-domain and on the
   construction grid.
2. Build each segment's residual on its padded bins: the conditional residual
   minus the current owned sources of every other segment overlapping those
   bins. Each segment's rendered owned sources are cached.
3. Sweep even-numbered segments, then odd, so each sees its neighbours' latest
   state. Sequential at first; segments within one parity set do not overlap
   and can later run in a process pool.
4. Per segment: take the PSD slice, refresh Fisher caches, recompute every
   chain's likelihood for the new residual and PSD, and advance `N` steps
   (default 1000, GLASS's 10x100 per update call).
5. Publish the summed owned-source template and parameters.

## Requirements on impulse

Present in the `RJPTSampler` design or in `ChainStats` today: vectorized
`lnlike(x, k)` and `lnprior(x, k)`; custom moves receiving a `ChainStats` with
`chain_index`, `rng` and `current_sample`, so a move object can key per-chain
caches by chain index; NaN or `-inf` likelihoods counted as `-inf`, so such
proposals are rejected; per-temperature `initial_k`.

Missing, and needed for both `PTSampler` and `RJPTSampler`: a continue call
that advances `N` more iterations from the current in-memory state after
recomputing likelihoods for a changed likelihood function, requiring them to
be finite. It must keep the temperature ladder, adaptation history and its
schedule (counted in total iterations across turns), proposal states and RNG
streams, and append to the chain files. Today `sample()` rebuilds state from
`initial_position`, and resume restores likelihood values from a checkpoint of
an unchanged problem.

Until `RJPTSampler` exists, the block runs at a fixed source count on
`PTSampler`, which still needs the continue call.

## Verification

1. **Conventions gate, before sampler work.** For each of the 36 VGBs, convert
   its `sky/vgb/cat` row, render it with both backends on the 12-month grid, and
   compute the A/E match against `dt*rfft` of the noise-free `sky/vgb/tdi`
   stream. Repeat for bright isolated `dgb` and `igb` sources against their own
   streams (`igb` covers negative `fdot`). Matches are recorded and reviewed
   with the user and choose the backend. A low match is a convention bug to
   find (phase, polarization, amplitude, FFT units, orbit phase, BW), not a
   threshold to relax.
2. **Likelihood:** the masked `K`-slot sum equals an explicit sum; inactive
   NaN slots cannot leak; on Sangria's noise-only stream (`obs` minus every sky
   stream) bins whitened by a one-sided Welch PSD of the same stream, an
   estimate independent of the rfft normalization, have unit mean power,
   checking `C = (T/2)*S`. The SNR from the formula above matches
   `sqrt(<h|h>)` from an independent noise-weighted inner product.
3. **Moves:** each custom move leaves the prior invariant in prior-only runs
   (KS tests); the Fisher move with its matrix held fixed.
4. **Fixed count on `PTSampler`:** one isolated bright source (posterior covers
   the injection, widths consistent with its Fisher matrix); then the crowded
   `gbnoisefit` `Segment767` band (18 sources).
5. **Block contract:** `enchilada.testing.check_block` on a toy L1Data; a
   changed residual between turns triggers a likelihood recompute.
6. **With `RJPTSampler`:** the prior-only count distribution, documenting the
   SNR-cut deviation; `Segment767` with birth/death; a handful of real Sangria
   segments compared with the injections and with GLASS's own chains
   (`project-catalog/data/old/glass_h5_files`).
7. **Full-band dry run:** build the table from the start set; measure memory
   and time per turn.

Tests needing impulse, h5py or the Sangria file skip cleanly when absent.

## Out of scope

Catalogue post-processing (GLASS's match clustering and occupancy cut), MBHB,
VGB and noise blocks, data preparation, the start-set computation, epochs other
than 12 months, the F-statistic and GMM proposals, parallel execution of a
parity set, and an incremental (single-source) likelihood. Each can follow
once the block above is validated.
