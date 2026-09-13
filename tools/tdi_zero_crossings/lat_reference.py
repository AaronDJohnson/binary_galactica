"""Optional CPU callback using an explicitly supplied LAT source checkout.

The delayed complex X-channel sum is evaluated by LAT's unchanged JAX
``get_tdi_Xf_single`` implementation. Only the orbit and source inputs are
fixtures: a static equilateral constellation and a constant-amplitude,
linearly chirping source. This is a controlled response experiment, not an
orbital ephemeris or a production Galactic-binary waveform validation.

No sibling checkout or JAX module is imported until a factory is called.
The local loader avoids executing LAT's package initializers, which require
the compiled GPUBackendTools stack. It registers no synthetic packages in
``sys.modules``.

Example (after enabling JAX float64)::

    response = make_frozen_chirp_callback("/path/to/LISAanalysistools")
    response(0.0)  # Warm the compiled scalar callback before timing.
    bracket = (-100.0, 100.0)

With the defaults the amplitude minimum is near 8.106 seconds, with a
nonzero residual about 5.26596e-5. Setting ``fdot=0`` produces a persistent
transfer null, which is also not an isolated crossing.
"""

from __future__ import annotations

import builtins
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(
    path: Path, name: str, dependencies: dict[tuple[int, str], ModuleType]
) -> ModuleType:
    """Execute a source file unchanged, resolving just its local imports."""
    if not path.is_file():
        raise FileNotFoundError(f"Required LAT reference module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LAT reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    normal_import = builtins.__import__

    def local_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        dependency = dependencies.get((level, name))
        if dependency is not None:
            return dependency
        return normal_import(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": local_import}
    spec.loader.exec_module(module)
    return module


def load_lat_projection(lat_checkout: str | Path) -> ModuleType:
    """Load the reference kernel without registering synthetic packages.

    ``lat_checkout`` must be the LISAanalysistools repository root containing
    ``src/lisatools/jax/response/projection.py``. This executes trusted local
    Python source; use a checkout you control.
    """
    root = Path(lat_checkout).expanduser().resolve() / "src/lisatools/jax"
    base = _load_module(root / "response/base.py", "_lat_reference.base", {})
    config = _load_module(
        root / "response/tdi_config.py", "_lat_reference.tdi_config", {}
    )
    orbits = _load_module(root / "orbits.py", "_lat_reference.orbits", {})
    return _load_module(
        root / "response/projection.py",
        "_lat_reference.projection",
        {
            (1, "base"): base,
            (1, "tdi_config"): config,
            (0, "lisatools.jax.orbits"): orbits,
        },
    )


def make_frozen_chirp_callback(
    lat_checkout: str | Path,
    *,
    arm_length: float = 2.5e9,
    fdot: float = 1e-7,
    inclination: float = 1.1,
    polarization: float = 0.21,
    longitude: float = 0.63,
    latitude: float = 0.42,
    demodulate: bool = True,
) -> Callable[[float], complex]:
    """Return ``response(t_seconds)`` for the reference complex TDI X sum.

    The source frequency is ``c/(2*arm_length) + fdot*t``. For nonzero chirp,
    a minimum close to the equal-arm transfer null occurs near zero seconds;
    finite chirp generally lifts the exact null. The callback therefore
    supplies a physical near-miss case, without asserting a true crossing.
    A useful initial candidate bracket is ``[-100., 100.]`` for the defaults.

    Enable JAX float64 before calling this factory. The returned callable
    accepts one scalar second value and returns a Python complex number.
    Its first call includes JIT compilation; warm it before timing solvers.
    By default the reference source phase is removed: LAT constructs its
    analytic response with an ``exp(-1j*phase)`` convention, so demodulation
    multiplies by ``exp(+1j*phase_ref)``. This preserves amplitude and zeros
    while making the local complex direction useful for projected solvers.
    Set ``demodulate=False`` to exercise the raw rotating complex response.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    if not jax.config.x64_enabled:
        raise ValueError(
            "Enable JAX float64 before building the LAT reference callback"
        )
    if not np.isfinite(arm_length) or arm_length <= 0:
        raise ValueError("arm_length must be finite and positive")
    if not all(
        np.isfinite(x) for x in (fdot, inclination, polarization, longitude, latitude)
    ):
        raise ValueError("Source parameters must be finite")
    projection = load_lat_projection(lat_checkout)
    c = 1.0 / projection.C_INV
    ltt = arm_length / c

    # LAT link convention: first digit is receiver, second digit is emitter.
    links = np.array([12, 23, 31, 13, 32, 21])
    xyz = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [arm_length, 0.0, 0.0],
            [arm_length / 2.0, np.sqrt(3.0) * arm_length / 2.0, 0.0],
        ]
    )

    class FrozenOrbits:
        """A stationary equilateral constellation with consistent arm delays."""

        sc_r = links // 10
        sc_e = links % 10

        def get_link_ind(self, link: int) -> int:
            """Look up a LAT receiver/emitter link."""
            return int(np.flatnonzero(links == link)[0])

        def get_light_travel_time(self, t: Any, link: int) -> Any:
            """Return the constant light travel time, in seconds."""
            return jnp.zeros_like(t) + ltt

        def get_pos(self, t: Any, sc: int) -> Any:
            """Return a fixed spacecraft position, in meters."""
            return jnp.zeros_like(t)[..., None] + xyz[sc - 1]

    class ChirpingSource:
        """A constant-amplitude source chirping through a transfer minimum."""

        def get_sky_indices(self) -> tuple[int, int, int, int]:
            """Match LAT's Galactic-binary sky parameter ordering."""
            return (5, 6, 7, 8)

        def amplitude(self, t: Any, params: Any) -> Any:
            """Return the source's unit amplitude."""
            return jnp.ones_like(t) * params[0]

        def phase(self, t: Any, params: Any) -> Any:
            """Return the phase of a source with constant frequency derivative."""
            return 2.0 * jnp.pi * (params[1] * t + 0.5 * params[2] * t * t)

    # Exact first-generation X combinations from LAT response/tdiconfig.py.
    combinations = [
        (13, [], +1),
        (31, [13], +1),
        (12, [13, 31], +1),
        (21, [13, 31, 12], +1),
        (12, [], -1),
        (21, [12], -1),
        (13, [12, 21], -1),
        (31, [12, 21, 13], -1),
    ]
    starts, lengths, delays = [], [], []
    for _, delay_links, _ in combinations:
        starts.append(len(delays))
        lengths.append(len(delay_links))
        delays.extend(delay_links)
    config = projection.TDIConfigWrapJAX(
        starts,
        lengths,
        [x[0] for x in combinations],
        delays,
        [x[2] for x in combinations],
        np.zeros(8, dtype=int),
        8,
        1,
    )
    params = jnp.asarray(
        [
            1.0,
            c / (2.0 * arm_length),
            fdot,
            0.0,
            0.0,
            inclination,
            polarization,
            longitude,
            latitude,
        ]
    )
    source, orbit = ChirpingSource(), FrozenOrbits()
    k, u, v = projection.get_sky_vectors(params, source)
    cpu_device = jax.devices("cpu")[0]

    @jax.jit
    def evaluate(t: Any) -> Any:
        response = projection.get_tdi_Xf_single(
            t, params, source, orbit, config, k, u, v
        )[0]
        if demodulate:
            phase_ref = projection.get_phase_ref(t, params, source, orbit)
            response = response * jnp.exp(1j * phase_ref)
        return response

    def scalarresponse(t: float) -> complex:
        if not np.isfinite(t):
            raise ValueError("Response time must be finite")
        return complex(evaluate(jax.device_put(np.float64(t), cpu_device)))

    return scalarresponse
