# Adopt gbonfly and match current GLASS

Status: implemented and verified on 2026-09-13 following user approval.

The mainline backend choice is superseded by the
[September 18 TDI-on-the-fly design](2026-09-18-tdi-onfly-design.md).
The Fourier implementation and its reference checks remain available through
explicit `backend="fourier"` selection.

This supersedes the implementation choices in the August 22 subsumption
design, especially its amplitude correction and unverified reference claims.

## Result

Make the gbonfly fast-slow Fourier/TDI generator the main package workflow.
Import its self-contained modules under `binary_galactica.glass` and expose
its public entry points from the package root. Retain the existing source
helpers as a supported source-only interface, without presenting them as the
complete GLASS waveform. Include the donor's prior/proposal helpers because
the user requested adopting gbonfly; do not invent a likelihood or sampler.

## Reference and conventions

Target nsgs-analysis-center/glass commit
`079c6a3a6fcf380a7c107f39640fff7fc720d0f1`. The diagnostic comparison found the
donor twice this reference on all five TDI channels. Keep the donor's half
polarization factors and change its FFT normalization to `1/(2*BW)`.
Keep the conversion from raw GLASS to continuous Fourier units as `sqrt(T)`;
do not cancel the correction in that conversion or recalibrate a test SNR.

The native waveform API accepts `(f0*T, sin(beta), lam, log(A_G), cos(iota), psi,
phi0_G, fdot*T**2[, fddot*T**3])`, with 7/8/9 parameters as in GLASS.
The existing `GBParams` amplitude multiplies full inclination factors, so
the explicit physical adapter uses `A_G=2*amp` and `phi0_G=-phi0`.
This relates source amplitude conventions; GLASS response additionally has
first-order amplitude evolution, unlike the constant-amplitude source helper.
The inverse mapping uses canonical inclination/latitude domains.

Use JAX for physical/native conversions so derivatives compose. The donor's
NumPy packing helper remains a host convenience. Make the physical response
entry point distinct from the native-array entry point. `T` remains traced;
`BW`, output-window geometry, and TDI format are static. Band choice is
discrete and is not differentiated. Require explicit `BW` for waveform calls.

## Validation and portability

Ship deterministic C-generated waveforms with exact pin, input, units,
channel ordering, hashes, and reproducible generation instructions. Use the
actual upstream C routines, not a Python reference or rescaled donor output.
Tests must run without sibling repositories, absolute user paths, or silent
skips. Cover formats, frequencies, chirps, carrier bins, nonzero epoch,
window placement, normalization, and scale-aware derivatives. Preserve
intentional sinc/pole guards and document their differences from raw C.

Retain Python >=3.10, JAX >=0.4.30, add NumPy >=1.24 as a runtime dependency.
Preserve GLASS attribution and applicable license notices. Update the README,
example, source convention wording, package metadata, and migration notes.
No WDM generator, fitted noise, or full inference implementation is included.

## Proposal correctness found during review

Preserve the donor's valid Fisher eigenvector draws, but reject the whole
proposal when a cosine crosses its bound or a jump is nonfinite. The donor's
component-wise boundary revert and one-percent NaN fallback do not preserve
proposal symmetry. This intentional helper change is separate from waveform
equivalence. A fixed Fisher decomposition supports the zero proposal-density
correction; scheduled refreshes remain adaptation and do not alone establish
sampler correctness. Document that limitation without adding a sampler.
