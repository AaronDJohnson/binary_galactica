# Independent GLASS C reference

These fixtures execute the actual C response from
[GLASS commit 079c6a3a6fcf380a7c107f39640fff7fc720d0f1](https://github.com/nsgs-analysis-center/glass/tree/079c6a3a6fcf380a7c107f39640fff7fc720d0f1).
They do not import a Python waveform or rescale its outputs.

`reference.c` contains 13 complete, unchanged upstream function bodies from
`glass_lisa.c`, `glass_math.c`, and `glass_ucb_waveform.c`, with their original
copyright/license notices. `glass_constants.h` and the four KissFFT source/header
files are byte-for-byte copies from the same commit. `sources.json` records each
upstream file's SHA-256 and Git blob hash, plus extracted function hashes and line
ranges. The pinned upstream tree's Git blob hashes were verified before extraction.

The only local C code is `preamble.h` (minimal declarations) and `driver.c` (input
and output). The driver supplies the upstream analytic orbit with `NI=3`, and
allocates a leading padding element so the upstream `X-1` pointer remains within
its allocation. The generated spectra retain the upstream half polarization
factors and `invBW2 = 1/(2*BW)` normalization. KissFFT uses its upstream default
`double` scalar; no installed library or architecture-specific object is needed.

## Regenerate offline

From the repository root, with Python, NumPy, and a C99 compiler:

```sh
python tools/glass_reference/generate.py
```

The compiler defaults to `$CC` or `cc`; `--cc clang` selects it explicitly.
Compilation happens in an automatically removed temporary directory. Flags are
`-std=c99 -D_DEFAULT_SOURCE -O2 -ffp-contract=off`; libm is linked. Avoid fast-math,
which can change the carrier-bin round trip. No OpenMP, GLASS installation, network,
JAX, or sibling checkout is required. `--output /tmp/glass-reference-check` writes
a separate copy for review. The JSON captures compiler identity, flags, source
hashes, and the NPZ checksum. Different compilers/libm implementations can produce
roundoff differences, so compare spectra numerically rather than requiring binary
identity across platforms.

To reproduce extraction as well, obtain a checkout of the exact pinned GLASS
commit and run:

```sh
python tools/glass_reference/extract.py /path/to/pinned/glass
python tools/glass_reference/generate.py
```

Extraction verifies every listed upstream file hash and all function hashes before
writing `reference.c`. The vendored sources already allow offline regeneration.

## Fixture contract and coverage

`tests/data/glass/reference.json` contains `cases`, each with `id`, native `params`,
`T` and `t0` in seconds, `BW`, `fmt`, C carrier `q`, and first output bin `qmin`.
`reference.npz` has six arrays per case: `{id}_bins` is int64 and
`{id}_X`, `{id}_Y`, `{id}_Z`, `{id}_A`, `{id}_E` are complex128. The arrays have
length `BW`. Frequencies are `bins/T`; continuous Fourier coefficients equal the
stored raw GLASS coefficients times `sqrt(T)`.

The 36 cases preserve the original 18 cases and IDs, then append six varied
sources in each of the three formats. The original cases cover 7/8/9 native
parameters, BW 32 and 512, 1/5/10 mHz carriers, monochromatic and chirping sources,
second derivatives, three-month/year durations, and shifted epochs. Two carrier
cases straddle an awkward binary64 boundary: `31004/T*T` is just below 31004,
whereas the next representable input falls above it. The C output explicitly
records both carrier and every output bin.

The additional sources vary all six sky, amplitude, inclination, polarization,
and phase coordinates, with amplitudes from `2e-23` to `4e-20`, both signs of
inclination cosine and chirp, and distinct nonzero phases. Their fixed values are
specified in `generate.py`, with no random sampling or waveform rescaling:

| Source | Parameters | Duration (years) | BW | f0 (mHz) | fdot (Hz/s) | fddot (Hz/s²) |
| --- | --- | --- | --- | --- | --- | --- |
| South | 7 | 0.25 | 32 | 1.73 | 0 | 0 |
| North, negative chirp | 8 | 1 | 96 | 4.21 | -3e-17 | 0 |
| Equatorial | 9 | 4 | 512 | 8.27 | 1.8e-16 | -2e-25 |
| North, monochromatic | 7 | 4 | 96 | 2.37 | 0 | 0 |
| South, positive chirp | 8 | 0.25 | 512 | 9.61 | 4e-16 | 0 |
| Oblique, negative chirp | 9 | 1 | 32 | 3.43 | -1e-17 | 6e-26 |

Each row produces separate C spectra in `sangria`, `phase`, and `frequency`
formats. BW 96 exercises a non-power-of-two FFT. The corpus coverage test requires
the varied coordinates, all formats, parameter lengths, durations, bandwidths,
and both signs of first and second frequency derivatives.

`tests.test_glass_reference.load_reference_cases()` returns `(metadata, arrays)`
pairs for use by physical-adapter tests. Mandatory tests require exact bin arrays
and relative L2 error at most `1e-9` separately for all 180 channel spectra, without a
fixed absolute floor that could hide these small strain amplitudes. Missing,
empty, or corrupted fixtures fail rather than skip.

Before the backend normalization correction, the original 18 cases failed with
relative error approximately 1. A separate check of all 90 unmodified donor
channels found errors from 0.9999999999921567 to 1.0000000000072011. Halving the
donor solely as a diagnostic gave a maximum relative error of
1.1213450704883488e-10. This diagnostic scaling is not part of fixture generation
or the regression tests.

Separate normalized finite-difference tests compare every Jacobian column for
7-parameter `phase`, 8-parameter `sangria`, and 9-parameter `frequency` sources.
The carriers remain a quarter-bin from truncation boundaries, all derivative
columns must have appreciable norm after waveform normalization, and each column
must agree to relative L2 error below `2e-5`. A reverse-mode phase test compares
real and imaginary quadratures of a normalized complex coefficient, requiring a
unit-magnitude phase derivative and relative error below `1e-6`.

These are analytic-orbit, finite-band comparisons at regular sky positions.
They do not validate numerical orbits, every singular limit, or bandwidth
selection. The Python sinc and pole guards intentionally extend the raw C at
singular inputs; those limits are outside this fixture set.

## Attribution

GLASS routines: copyright 2019 Tyson B. Littenberg and Neil J. Cornish;
`glass_math.c`: copyright 2023 Tyson B. Littenberg and Robert Rosati;
constants: copyright 2019 Tyson B. Littenberg. These files are under Apache-2.0;
the complete upstream license is [LICENSE-GLASS](LICENSE-GLASS).

KissFFT: copyright 2003–2010 Mark Borgerding, BSD-3-Clause. Original headers,
[COPYING](vendor/kissfft/COPYING), and upstream
[license texts](vendor/kissfft/LICENSES/) are retained. Local harness and generation
scripts are covered by binary-galactica's repository license.
