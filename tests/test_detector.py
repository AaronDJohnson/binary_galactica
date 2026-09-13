"""The physical detector API uses independently translated GLASS references."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg

DATA = Path(__file__).parent / "data" / "glass"


def _physical_cases():
    cases = json.loads((DATA / "reference.json").read_text())["cases"]
    return [case for case in cases if len(case["params"]) == 8]


@pytest.mark.parametrize("case", _physical_cases(), ids=lambda case: case["id"])
def test_physical_response_matches_independent_glass_case(case):
    v = case["params"]
    duration = case["T"]
    # Independent inverse of GLASS's source conventions, without calling the
    # production converter to construct the inputs for its own test.
    p = bg.GBParams(
        amp=np.exp(v[3]) / 2,
        f0=v[0] / duration,
        fdot=v[7] / duration**2,
        phi0=-v[6],
        iota=np.arccos(v[4]),
        psi=v[5],
        lam=v[2],
        beta=np.arcsin(v[1]),
    )
    result = bg.tdi_waveform(
        p, T=duration, BW=case["BW"], t0=case["t0"], fmt=case["fmt"]
    )
    with np.load(DATA / "reference.npz", allow_pickle=False) as ref:
        np.testing.assert_array_equal(
            result["freq_bin_indices"], ref[f"{case['id']}_bins"]
        )
        for channel in ("X", "Y", "Z", "A", "E"):
            expected = ref[f"{case['id']}_{channel}"]
            residual = np.linalg.norm(np.asarray(result[channel]) - expected)
            assert residual / np.linalg.norm(expected) < 1e-9


def test_physical_detector_composes_with_grad_and_vmap(params):
    duration = 7884000.0
    catalogue = jnp.stack([params.to_array(), params._replace(lam=2.3).to_array()])

    def channels(theta):
        w = bg.tdi_waveform(theta, T=duration, BW=64)
        return jnp.stack([w["A"], w["E"]]) / 1e-22

    batch = jax.jit(jax.vmap(channels))(catalogue)
    assert batch.shape == (2, 2, 64)
    assert not np.allclose(batch[0], batch[1], rtol=1e-4, atol=0)
    derivatives = jax.grad(lambda theta: jnp.sum(jnp.abs(channels(theta)) ** 2))(
        params.to_array()
    )
    assert np.all(np.isfinite(derivatives))
    assert float(derivatives[6]) != 0.0
    assert float(derivatives[7]) != 0.0
