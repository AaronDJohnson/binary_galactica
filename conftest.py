"""Enable JAX double precision for the whole pytest session.

Importing ``binary_galactica`` leaves the process-global x64 flag alone, and
its float64 entry points raise until the caller enables it. This root-level
conftest loads before any test module, and before ``--doctest-modules``
imports the package from ``src/``, so the suite and the doctests both run in
float64. Tests of the disabled state use fresh subprocesses rather than
toggling the flag here.
"""

try:
    import jax
except ImportError:  # tools/tdi_zero_crossings documents a JAX-free test run.
    pass
else:
    jax.config.update("jax_enable_x64", True)
