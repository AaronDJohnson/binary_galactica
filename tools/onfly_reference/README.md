# Independent GLASS on-the-fly response reference

`upstream.c` contains unmodified functions from GLASS commit
`079c6a3a6fcf380a7c107f39640fff7fc720d0f1`, including its full geometry,
eight retarded evaluations, and signed amplitude/phase extraction. Original
copyright notices are retained; the Apache 2.0 license is at
`../glass_reference/LICENSE-GLASS`. `sources.json` records source and function
hashes. No production Python response is used to generate these fixtures.

`driver.c` supplies storage and replaces spline lookup with exact analytic
orbits and polynomial source evaluation. This isolates response algebra from
spline discretization. It explicitly supports zero-padding seven-parameter
sources and a cubic phase for nine-parameter sources; the upstream UCB spline
constructor itself only supports eight parameters and sets fddot=0.

Run `python tools/onfly_reference/generate.py` to compile with the system C
compiler and regenerate `tests/data/onfly/reference.json`. To verify/extract
the pinned source from a local git repository, add `--extract /path/to/ldasoft`.
Fixtures contain the exact spacecraft positions used by C as well as complex
demodulated XYZ. The 12-source corpus covers 0.1, 1, 5 and 30 mHz, varied sky,
inclination and polarization angles, both signs of frequency derivatives,
7/8/9 parameters, and amplitudes of 1 and 1e-22. Tests normalize comparison
tolerances by each response's peak magnitude. Subtracting large carrier phases
in C limits comparison accuracy; the JAX implementation evaluates polynomial
phase differences directly and uses analytic removable limits.
