# Gbonfly GLASS Migration Implementation Plan

> For agentic workers: use subagent-driven-development to implement and review these tasks.

**Goal:** Make gbonfly the package's main TDI waveform workflow and match the pinned current GLASS implementation.

**Architecture:** The donor modules live in `binary_galactica.glass` with their native parameter convention. Root exports make the generator discoverable; a separate JAX physical-parameter adapter connects `GBParams`. Independent C-generated fixtures establish normalization and waveform equivalence.

**Tech Stack:** Python, JAX float64, NumPy, pytest; C only for fixture regeneration.

**Spec:** `docs/superpowers/specs/2026-09-13-gbonfly-glass-migration.md`.

## Global constraints

- Python >=3.10, JAX >=0.4.30, NumPy >=1.24.
- GLASS pin `079c6a3a6fcf380a7c107f39640fff7fc720d0f1`.
- Raw FFT normalization `1/(2*BW)` and continuous-Fourier conversion `sqrt(T)`.
- Existing source API remains accessible; gbonfly is the primary documented workflow.
- No dependency on external checkout paths during normal tests or use.
- Work on the clean existing `subsume-gbonfly` feature branch; no commits or publication required.

### Task 1: Portable independent reference regression

Files: `tools/glass_reference/`, `tests/data/glass/`, `tests/test_glass_reference.py`.

Produces: a repository NPZ/JSON reference with native inputs, T, t0, BW, format,
qmin and five complex channels per case, plus reproducible source provenance.
Consumes: the compiled extracted C diagnostic at `/tmp/gbonfly-current-validation`
and its pinned originals; tests import `binary_galactica.glass.ucb_waveform`.

- [x] Generate reference spectra by executing the C, including 7/8/9 parameters,
  all three formats, 1/5/10 mHz, BW32/512, shifted epoch and carrier boundaries.
- [x] Add mandatory tests using scale-aware relative L2 error <=1e-9 for every
  channel and exact matching frequency-bin locations.
- [x] Demonstrate the donor fails the same comparisons with relative error ~1.
- [x] Preserve C source hashes, authorship/license, and a portable regeneration
  procedure using source FFT code, not a machine-specific object.
- [x] Once Task 2 lands, run the regression against the migrated package.

### Task 2: Port and correct the donor backend

Files: `src/binary_galactica/glass/*.py`, `tests/glass/`.

Produces: the donor public API under `binary_galactica.glass`, including
`ucb_waveform(params, *, T, BW, t0=0, fmt='sangria', positions=None, q=None)`,
`template_on_window`, alignment, bandwidth, conversion, prior and proposal
helpers. Root owns packaging and physical adapters.

- [x] Copy independent donor tests first with package-local fixtures and new imports;
  omit broken external-transcription comparisons in favor of Task 1's C goldens.
- [x] Observe missing backend failures, then port the twelve donor modules.
- [x] Establish the factor-of-two reference failure before changing the FFT to
  `fftshift(fft(env))/ (2*BW)`. Retain the half polarization factors.
- [x] Fix stale normalization/source claims, retain documented numerical guards,
  and add meaningful input-shape checks without Python validation of tracers.
- [x] Cover fixed windows, all channels/formats, cached orbits, JAX transforms,
  and normalized waveform Jacobians with useful finite-difference tolerances.
- [x] Run backend tests and coordinate with Task 1's regression.

### Task 3: Public integration, documentation, and packaging

Files: root `__init__.py`, `parameters.py`, new `detector.py`, existing source
docstrings, `README.md`, `examples/quickstart.py`, `pyproject.toml`, notices,
`tests/test_detector.py`, and physical conversion tests.

Consumes: Task 2's native backend and Task 1's reference input/output contract.
Produces: root donor entry points plus `to_glass_vector`, `from_glass_vector`,
and `tdi_waveform(theta, *, T, BW, t0=0, fmt='sangria', positions=None)` for
physical GBParams/8-vectors.

- [x] Write explicit expected-vector tests for amplitude/phase/angles/T scaling
  and gradient-compatible batching, then implement conversions with JAX.
- [x] Test the physical response against independently translated C goldens;
  implement the thin physical entry point without changing old source results.
- [x] Export the donor API, add NumPy dependency, and apply repository formatting.
- [x] Make README/example primarily demonstrate TDI generation with explicit BW,
  raw/continuous units, GLASS pin, parameter conventions and validation limits.
- [x] Correct historical source-only equivalence claims and supersede the older
  design's normalization prescription; include required upstream notices.
- [x] Run full pytest including doctests, lint, example, and package build/import.
- [x] Obtain independent final code review, resolve findings, and rerun affected checks.
