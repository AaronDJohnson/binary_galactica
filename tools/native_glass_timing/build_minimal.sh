#!/bin/sh
# Build the pinned native UCB/WDM timing harness without GLASS CMake.
set -eu

PIN=079c6a3a6fcf380a7c107f39640fff7fc720d0f1
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECKOUT=${1:?usage: $0 /path/to/ldasoft /external/temp/build-directory}
BUILD_DIR=${2:?usage: $0 /path/to/ldasoft /external/temp/build-directory}
SOURCE="$BUILD_DIR/source-$PIN"
OUTPUT="$BUILD_DIR/native_ucb_timing"
CC=${CC:-cc}

git -C "$CHECKOUT" cat-file -e "$PIN^{commit}"
mkdir -p "$BUILD_DIR"
rm -rf "$SOURCE"
mkdir "$SOURCE"
git -C "$CHECKOUT" archive "$PIN" | tar -x -C "$SOURCE"

exec "$CC" -O3 -march=native -std=gnu99 -include stdbool.h -ffp-contract=off \
    -ffunction-sections -fdata-sections \
    -I"$ROOT/stubs" \
    -I"$SOURCE/external" -I"$SOURCE/utils/src" -I"$SOURCE/ucb/src" -I"$SOURCE/noise/src" \
    "$ROOT/native_ucb_timing.c" \
    "$SOURCE/ucb/src/glass_ucb_waveform.c" \
    "$SOURCE/utils/src/glass_lisa.c" \
    "$SOURCE/utils/src/glass_wavelet.c" \
    "$SOURCE/utils/src/glass_math.c" \
    "$SOURCE/utils/src/glass_utils.c" \
    "$SOURCE/external/kissfft/kiss_fft.c" \
    "$SOURCE/external/kissfft/kiss_fftr.c" \
    -Wl,-dead_strip -lm -o "$OUTPUT"
