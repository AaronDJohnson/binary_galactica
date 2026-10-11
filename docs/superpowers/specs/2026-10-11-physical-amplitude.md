# Physical native amplitude

Status: implemented on branch `fix/physical-amplitude` on 2026-10-11 following
user approval; version 0.5.0. This supersedes the amplitude conventions of the
[September 13 migration](2026-09-13-gbonfly-glass-migration.md) and the
normalization paragraph of the
[September 18 on-the-fly design](2026-09-18-tdi-onfly-design.md).

## Finding

Through 0.4.0 every GLASS-native output was exactly half the physical signal
for `params[3] = ln(A)`, with `A` the LDC/Sangria catalogue `Amplitude`. This
held for `ucb_waveform`, `template_on_window`, and `tdi_response`, on both
backends and in `binary_galactica.single`. The physical `GBParams` API was
correct, because its adapter compensated. Three independent checks agreed:
Sangria data, the donor `gbonfly`, and the physical API.

The cause is GLASS commit `b0751bd` (2026-04-21, "finishing touches on wdm-fix
merge"). In the fast-slow `ucb_waveform` of `ucb/src/glass_ucb_waveform.c` it
changed

```c
Aplus  =  amp*(1.+cosi*cosi);       /* 95184e1 and earlier, LDC FastGB */
Across = -amp*(2.0*cosi);
```

to

```c
Aplus  =  amp*(1.+cosi*cosi)*0.5;   /* b0751bd, 079c6a3 */
Across = -amp*(cosi);
```

and kept `invBW2 = 1./(double)BW2` with `BW2 = 2*BW`. The pinned commit
`079c6a3` follows `b0751bd`, so its fast-slow spectra, and the C fixtures in
`tests/data/glass`, are half the physical response.

The package inherited that factor in three steps. The September 13 migration
matched `079c6a3` by keeping the donor's half factors and changing the FFT
scale from `1/BW` to `1/(2*BW)` (`glass/fourier.py`). The September 18 on-fly
design then halved the public spline response and its FFT prefactor
(`sqrt(T)/(4*BW)` in `glass/waveform.py`) to agree, and the float32 path
(`glass/single.py`) later copied both. The physical adapter compensated with
`A_G = 2*amp` (`parameters.py`).

## GLASS amplitude semantics

GLASS itself treats `params[3]` as the logarithm of the physical amplitude:

- `amplitude(Mc, f0, D)` (`utils/src/glass_math.c`) returns
  `2*(G*Mc)**(5/3)*(pi*f0)**(2/3)/D`, the standard LDC strain amplitude.
- The injection readers `UCBInjectVerificationSource` and
  `GetVerificationBinary` (`ucb/src/glass_ucb_data.c`) set
  `amp = amplitude(Mc, f0, D)` and map `ln(amp)` into `params[3]` before
  calling `ucb_waveform`.

After `b0751bd` GLASS's own fast-slow injections are therefore half physical.
The arbitrary-time spline response that `glass/onfly.py` transcribes is
unaffected: its raw output is already the physical Sangria response.

## Measurements

LDC2 Sangria training v2, `sky/vgb`, one year at 5 s. The data are
`dt * rfft(TDI)`, models are `sqrt(T)` times raw spectra with `BW = 512`, and
each entry is the least-squares complex scale `<h, d>/<h, h>` on `X/Y/Z/A/E`.

| Source | Native 0.4.0 | Native 0.5.0, both backends | `gbonfly` |
|---|---|---|---|
| HPLib (1.81 mHz) | 1.9989-1.9992 | 0.99944-0.99960 | 0.99944-0.99960 |
| SDSSJ0651 (2.61 mHz) | 1.9990-2.0000 | 0.99952-0.99999 | 0.99952-0.99995 |
| V407Vul (3.51 mHz) | 1.9990-1.9999 | 0.99951-0.99993 | 0.99953-0.99993 |
| HMCnc (6.22 mHz) | | 0.99983-0.99991 | 0.99984-0.99991 |

The physical API gave these 0.5.0 values before and after the fix. The
time-domain `Re[tdi_response * exp(i*Phi)]` fit to band-limited `X/Y/Z` data
moved from 1.998-2.000 to 0.9992-1.0008, with correlations above 0.99999.
The float32 `template_on_window` gives A/E scales of 0.99944-0.99996 on the same
four sources. Some sources fit less well, but identically for `gbonfly`, so
the residual is waveform-model mismatch rather than normalization: AMCVn
0.9963-0.9966, SDSSJ1235 0.9985-0.9990, and ZTFJ1539 0.9984-0.9996.
HD265435 shares its band with SDSSJ1337 and is not a usable anchor.

On random native vectors the Fourier backend now equals `gbonfly`: across
7/8/9 parameters, all formats, three durations and two bandwidths,
`||gbonfly||/||bg||` differs from one by at most 3.3e-16. The exceptions are
two seven-parameter frequency-format cases, at up to 4.6e-13; 0.4.0 shows the
same deviations at a ratio of exactly two.

## Decision

Restore `A_G = A_phys`. In the fast-slow backend use full polarization factors,
`Aplus = amp*(1 + cos(i)**2)` and `Across = -2*amp*cos(i)`, and keep
`invBW2 = 1/(2*BW)`. Do not instead keep half factors with `1/BW`, as the donor
did. The two choices produce identical outputs. Full factors, however, make the
intermediate polarization amplitudes physical. They match GLASS up to
`95184e1`, LDC FastGB, and this package's source helpers, and confine the
deviation from `079c6a3` to the two lines `b0751bd` changed. The C fixtures
then differ from the backend by an exact factor of two, and the FFT
normalization stays as pinned.

The on-fly backends use `sqrt(T)/(2*BW)` on the raw spline response, and
`tdi_response` returns it unscaled in both precisions. The adapter maps
`A_G = amp` in both directions; `phi0_G = -phi0` is unchanged.
`GLASS_TO_CONTINUOUS` remains `sqrt(T)`, and the on-fly math is unchanged.

## Tests and fixtures

The C fixtures, their NPZ, and its SHA-256 are unchanged. The tests compare
the Fourier backend with `GLASS_079C6A3_FAST_SLOW_SCALE = 2.0` times each
stored spectrum, at the same relative L2 tolerance of 1e-9. Only the
`units` metadata string records the halving. The physical-adapter fixture
test keeps `amp = exp(v[3])/2`, now as the source whose physical response the
halved fixture represents. Dense on-fly references use `sqrt(T)/(2*BW)`.

`tests/test_physical_amplitude.py` pins the scale without those fixtures:

- `ucb_waveform(v)` equals `tdi_waveform` of hand-built physical parameters
  with `amp = exp(v[3])`, and of `from_glass_vector(v, T)`, to 1e-12 relative
  on `X/Y/Z/A/E` for both backends;
- at 0.1-0.3 mHz the year-RMS `|X|`, `|Y|`, `|Z|` of `tdi_response` equals the
  long-wavelength first-generation limit
  `2*x12*x13*|n13.H.n13 - n12.H.n12|` to 1% (measured 3e-4), using actual arm
  lengths from the analytic orbit.

Both fail on 0.4.0, the second at a ratio of 0.500.

The float64 golden archive records 0.4.0 outputs and must be regenerated with
`tools/float64_golden.py` after this change is committed. Before regeneration,
every native-API golden output is 2.000000000000 times the recorded value
(per-element ratios within 1.5e-13; within 3.7e-16 of each output's peak). The
`tdi_waveform` channels move by at most 2e-15 of peak. Its `native` vector
changes only in element 3, by `-ln(2)`. Bins, `qmin`, positions, and the
source-only helpers are bitwise identical.

## Migration

A 0.4.0 native vector `v` gives the same signal in 0.5.0 with `v[3] - ln(2)`.
Equivalently, native outputs for a fixed vector double. Physical `GBParams`
inputs give the same signals; only the intermediate native vector changes,
from `ln(2*amp)` to `ln(amp)`. Native callers change meaning by a factor of
two. One example is `petra_catalogs`, whose `petra/binary_galactica.py` calls
`ucb_waveform` when `parameterization="glass"`. Catalogue vectors built from
LDC `Amplitude`, as GLASS's readers build them, are now correct as they stand.
Saved native templates, SNRs, and amplitude posteriors from 0.4.0 need an
explicit conversion decision. The version becomes 0.5.0, and the README
carries a migration note.
