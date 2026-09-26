# Native GLASS UCB timing inputs

This directory contains the reproducibility inputs for the native GLASS timing
report at [tools/performance/glass-native.json](../performance/glass-native.json).
It does not include a GLASS archive, build directory, executable, or measured
output.

The harness targets GLASS commit
`079c6a3a6fcf380a7c107f39640fff7fc720d0f1` (2026-05-29). Given a local GLASS
checkout that retains that object and an external temporary build directory,
build it with:

```sh
sh tools/native_glass_timing/build_minimal.sh /path/to/ldasoft /private/tmp/glass-ucb-build
```

The macOS build script verifies the exact commit object, archives it into the supplied
external build directory, then compiles the pinned, unmodified
`glass_ucb_waveform.c`, `glass_lisa.c`, `glass_wavelet.c`, `glass_math.c`,
`glass_utils.c`, and KissFFT sources. It does not invoke CMake. Local stubs
provide declarations for unavailable MPI, HDF5, OpenMP, LAPACKE, and astrometry
headers. Function/data section dead stripping excludes the unused external
paths; the waveform function bodies are not changed. The release-oriented
flags are `-O3 -march=native`, with `-ffp-contract=off` recorded explicitly.

The program measures the upstream functions that exist at this commit:
`ucb_waveform` (banded Fourier `X/Y/Z/A/E`) and `ucb_waveform_wavelet` (WDM
`X/Y/Z`). It also reports `LISA_spline_response` alone. GLASS does not provide
`ucb_waveform_spline` or a same-output spline-FFT function here, so these
measurements cannot establish spectral equality or directly benchmark
binary-galactica's on-fly FFT path.

Each case uses a native eight-parameter source, analytic interpolated orbit,
epoch 12345 seconds, and the GLASS binary year of 31,457,280 seconds. The
matrix spans 1, 5, and 30 mHz; 0.25, 1, and 4 years; and BW 128, 512, and
2048. Run the full matrix after building with:

```sh
/private/tmp/glass-ucb-build/native_ucb_timing --repeats 100
```

The committed report uses BW512 and three independent 200-call blocks per
source. Reproduce those serialized measurements with:

```sh
for years in 0.25 1.0 4.0; do
  for frequency in 0.001 0.005 0.03; do
    for block in 1 2 3; do
      /private/tmp/glass-ucb-build/native_ucb_timing \
        --case "$frequency" "$years" 512 200
    done
  done
done
```

Each process prints setup messages followed by a JSON object. Group those
objects by frequency and duration and take the median of the three per-call
means, converting seconds to milliseconds. The broader default matrix is
available for separate bandwidth experiments; it is not the recorded run.

Orbit interpolation, WDM initialization, output allocation, and one warm call
per complete waveform path occur before each timed loop. The loops time all
allocations and other work performed internally by the upstream waveform
functions. The response-only stage prebuilds its source splines and reference
phase, then times `LISA_spline_response` including its own internal work.
Checksums and finite-value checks retain output use. The executable emits one
JSON object per case; no measurements are bundled in this directory.
