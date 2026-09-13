# Progress — 2026-09-13 gbonfly GLASS migration

The subsequent [quality follow-up](../../quality-review.md) records the later
fixes, expanded coverage, compatibility checks, and revised metrics.

User approved implementation after the direct GLASS comparison.

| Tasks | Shared contract | Check |
|---|---|---|
| 1 / 2 | `binary_galactica.glass.ucb_waveform` and native vectors | Same signature and pin; independent file ownership |
| 2 / 3 | Donor public exports; root adapter calls backend | Root owns parent files; backend owns namespace |
| 1 / 3 | C goldens include native input metadata | Physical amplitude mapping explicit, not inferred by a round trip |
| 1 | C expectation vs mandatory regression | No donor-generated goldens |
| 2 | Half amplitudes plus `1/(2*BW)` | Corrects measured factor two |
| 3 | GBParams full amplitudes vs native half factors | `A_G=2*amp`, phase parameter negated |

Decision: work in the existing clean `subsume-gbonfly` feature checkout, keeping
changes directly reviewable in the user's workspace.
Decision: import donor helpers including priors/proposals, as requested, without
claiming they constitute a full inference pipeline.

- [x] Task 1 reference regression
- [x] Task 2 backend port
- [x] Task 3 public integration and documentation
- [x] Final review and verification

## Evidence before final review fixes

- The independent C generator produced 18 cases and 90 channel comparisons.
  Regeneration matched all 108 stored arrays exactly on this machine. The
  uncorrected donor failed every case with relative error approximately 1;
  the corrected backend's maximum channel relative L2 error was
  `1.1213450704883488e-10` (required tolerance `1e-9`).
- The integrated suite passed all 185 tests, including doctests, native
  waveform comparisons, physical adapters, JAX transforms, windows and
  normalized eight-parameter finite-difference Jacobians.
- Ruff lint and formatting passed. The quickstart ran; its physical/native
  adapter agreement was `8.123e-15` relative. The sdist and wheel built,
  and an isolated wheel import matched the first C reference case across
  all five channels (maximum relative error `3.382e-12`).
- Independent reference/adapter review verified all 13 extracted function
  hashes and seven compiled-source hashes, with no findings.
- Final migration review identified correlated Fisher boundary rejection,
  nonfinite Fisher fallback, and overclaimed cache correctness. The fix
  rejects invalid whole-vector draws and qualifies adaptive-cache guidance;
  these are intentional departures from the donor's proposal helpers.

Verification environment: Python 3.14.2, JAX 0.7.2, NumPy 2.3.5, pytest 9.0.2.
The declared minimum dependency versions have not been tested in a separate
environment. Only the extracted response routines and actual upstream FFT
are compiled for reference generation; the full GLASS application is not run.

## Final verification after review fixes

- All three review findings and the README dimensionality correction were
  resolved in one fix pass. The independent scoped re-review reran the
  original reproductions, both cosine boundary signs, correlated forward
  and reverse interior moves, invalid eigenvalues, and copy semantics;
  it found no remaining issue within that scope.
- Five new regression cases failed against the original guard behavior;
  after correction, all 24 proposal tests passed.
- The complete suite passed: **187 tests in 18.40 seconds**, with no skips.
  Ruff lint, formatting (47 files), and `git diff --check` passed.
- The quickstart ran successfully, including a `(2, 512, 8)` native
  Jacobian and batched A/E channels. Physical/native relative agreement
  remained `8.123e-15`.
- The final 0.2.0 sdist and wheel built successfully. Every one of the
  wheel's 17 Python modules matched the current source bytes, and required
  license notices were present. Installed into a separate temporary directory
  and imported from outside the checkout, the wheel matched all five
  channels of the first C case with maximum relative error `3.382e-12`.
- Work remains uncommitted in `subsume-gbonfly`, as authorized. No external
  checkout is required to run the tests or use the installed package.
