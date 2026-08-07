# binary-galactica

The gravitational waveform model for LISA galactic binaries, in JAX.

Eight parameters, two polarizations, differentiable. The model is the standard
circular, linearly-chirping one: a monochromatic wave with a quadratic phase and
constant polarization amplitudes. Sign conventions match `ldasoft` and `GBGPU`
exactly, and the test suite pins that agreement.

## Install

```bash
pip install -e .
```

The only runtime dependency is `jax`.

## Use

```python
import jax.numpy as jnp
from binary_galactica import GBParams, YEAR, polarizations

params = GBParams(
    amp=1e-22,     # dimensionless strain amplitude
    f0=1.2e-3,     # GW frequency at t=0 [Hz]
    fdot=3.2e-16,  # frequency derivative [Hz/s]
    phi0=0.7,      # GW phase at t=0 [rad]
    iota=1.1,      # inclination [rad]
    psi=0.4,       # polarization angle [rad]
    lam=4.1,       # ecliptic longitude [rad]
    beta=-0.3,     # ecliptic latitude [rad]
)

t = jnp.arange(0.0, 4 * YEAR, 15.0)
hp, hc = polarizations(params, t)
```

Parameters are defined at `t = 0` of the array you pass. There is no hidden
reference epoch — shift `t` and the parameters mean what they say.

Functions accept either a `GBParams` or a plain `(8,)` array, in the canonical
order `(amp, f0, fdot, phi0, iota, psi, lam, beta)`:

```python
theta = params.to_array()
hp, hc = polarizations(theta, t)
```

### JAX transforms

```python
from jax import grad, jacfwd, jit, vmap

hp, hc = jit(polarizations)(theta, t)              # compiled
jac_hp, jac_hc = jacfwd(polarizations)(theta, t)   # exact (N, 8) Jacobians

hp, hc = vmap(polarizations, in_axes=(0, None))(catalogue, t)   # (K, 8) sources
```

The Jacobian columns are exact — no finite differencing — which is what you want
for Fisher matrices and gradient-based samplers. `GBParams` is a pytree, so
`grad` of a scalar functional returns a `GBParams` of derivatives with the
fields named.

## The model

The wave is monochromatic with a quadratic phase,

$$\Phi(t) = \phi_0 + 2\pi f_0 t + \pi \dot{f} t^2, \qquad f(t) = f_0 + \dot{f} t,$$

with constant amplitudes set by the inclination,

$$\mathcal{A}_+ = \mathcal{A}\,(1 + \cos^2\iota), \qquad \mathcal{A}_\times = -2\mathcal{A}\cos\iota.$$

At polarization angle $\psi = 0$,

$$h_+^{\rm src} = \mathcal{A}_+ \cos\Phi, \qquad h_\times^{\rm src} = \mathcal{A}_\times \sin\Phi,$$

and a rotation carries these to the solar-system barycentre frame,

$$\begin{pmatrix} h_+ \\\\ h_\times \end{pmatrix} = \begin{pmatrix} \cos 2\psi & \sin 2\psi \\\\ -\sin 2\psi & \cos 2\psi \end{pmatrix} \begin{pmatrix} h_+^{\rm src} \\\\ h_\times^{\rm src} \end{pmatrix}.$$

### Scope

This package is the **source model only**. It stops at $h_+$ and $h_\times$ at
the barycentre. LISA's orbits, the arm projection, light-travel delays and TDI
belong to the response code downstream.

That is why `lam` and `beta` do not appear above: sky position enters only
through the propagation direction and the polarization basis, both of which are
the response's business. They are carried in `GBParams` because they are part of
the sampling space and the response needs them — but the `lam` and `beta`
columns of `jacfwd(polarizations)` are identically zero, by construction and by
test.

The parameter space is the standard 8D one. $\ddot{f}$ is deliberately excluded.

### Conventions

Signs and factors are a real hazard here, so they are stated explicitly and
tested against the reference implementations in `tests/test_conventions.py`.

| Quantity | This package | Matches |
|---|---|---|
| $\Phi(t)$ | $\phi_0 + 2\pi f_0 t + \pi \dot f t^2$ | `ldasoft`, `GBGPU` |
| $\mathcal{A}_+$ | $\mathcal{A}(1+\cos^2\iota)$ | `ldasoft`, `GBGPU` |
| $\mathcal{A}_\times$ | $-2\mathcal{A}\cos\iota$ | `ldasoft`, `GBGPU` |
| $\psi$ rotation | $h_+ = h_+^{\rm src}\cos 2\psi + h_\times^{\rm src}\sin 2\psi$ | `ldasoft`, `GBGPU` |
| $f_0$ | GW frequency, twice orbital | LDC |
| Reference epoch | $t = 0$ of the supplied array | — |
| Year | sidereal, 31558149.763545603 s | LDC `YRSID_SI` |

`ldasoft` builds its response from the coefficients

```
DPr =  A+ cos2psi        DPi = -Ax sin2psi
DCr = -A+ sin2psi        DCi = -Ax cos2psi
```

contracted against a wave written as $e^{-i\Phi}$, so the factor multiplying the
plus tensor is `DPr cos(Phi) - DPi sin(Phi)` and likewise for cross. Those are
exactly the two arrays `polarizations` returns.

## Double precision

**Importing `binary_galactica` sets `jax_enable_x64=True` process-wide.**

This is deliberate and it is not a preference. Over a four-year observation the
phase of a millihertz binary accumulates to $\sim10^6$ rad. In float32 that
carries $\sim0.06$ rad of error — enough to destroy a matched filter, and far
too small to notice in a plot. A model that is silently wrong by default is a
worse outcome than a documented global setting, so the package enables float64
before it does anything else.

If your surrounding code depends on JAX defaulting to float32, import this
package last, or in a separate process.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite is built from analytic identities rather than stored waveforms, so it
documents the model as well as checking it: the monochromatic limit reduces to a
closed-form sinusoid; face-on sources are circularly polarized and edge-on
sources have no cross polarization; $\psi$ rotation is orthogonal with period
$\pi$; a phase offset equals a time shift when $\dot f = 0$; `jacfwd` reproduces
central differences on all eight parameters; and the phase stays accurate to
$<10^{-6}$ rad over four years, checked against 50-digit decimal arithmetic.

## References

- Cornish & Littenberg, *Tests of Bayesian model selection techniques for gravitational wave astronomy*, [Phys. Rev. D **76**, 083006 (2007)](https://doi.org/10.1103/PhysRevD.76.083006).
- Littenberg, *A detection pipeline for galactic binaries in LISA data*, [Phys. Rev. D **84**, 063009 (2011)](https://doi.org/10.1103/PhysRevD.84.063009).

## License

MIT.
