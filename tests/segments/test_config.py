import json

import pytest

from binary_galactica.segments import UCBBlockConfig


def test_defaults_follow_the_spec():
    cfg = UCBBlockConfig()
    assert (cfg.n_target, cfg.steps_per_turn, cfg.ntemps, cfg.k_max_floor) == (10, 1000, 12, 10)
    assert cfg.backend == "onfly"
    weights = cfg.within_model_weights()
    assert weights["fisher"] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_birth_death_weight_gives_the_requested_fraction():
    cfg = UCBBlockConfig()
    total = sum(cfg.within_model_weights().values())
    bd = cfg.birth_death_weight()
    assert bd / (bd + total) == pytest.approx(0.25)
    assert UCBBlockConfig(birth_death=False).birth_death_weight() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_target": 0},
        {"ntemps": 1.5},
        {"steps_per_turn": 10, "thin": 11},
        {"ladder_step": 1.0},
        {"birth_death_fraction": 1.0},
        {"fisher_weight": -0.1},
        {
            "fisher_weight": 0.0,
            "prior_weight": 0.0,
            "fm_shift_weight": 0.0,
            "psi_phi_weight": 0.0,
        },
        {"backend": "wavelet"},
        {"band_bins": (10, 5)},
        {"response_samples": 2},
        {"snr_min": -1.0},
    ],
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        UCBBlockConfig(**kwargs)


def test_json_round_trip(tmp_path):
    cfg = UCBBlockConfig(band_bins=(100, 900), steps_per_turn=50, backend="fourier")
    path = tmp_path / "config.json"
    cfg.write(path)
    assert UCBBlockConfig.read(path) == cfg
    assert json.loads(path.read_text())["band_bins"] == [100, 900]


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="unknown"):
        UCBBlockConfig.from_dict({"steps": 3})


def test_band_bins_accept_numpy_integers():
    import numpy as np

    cfg = UCBBlockConfig(band_bins=(np.int64(10), np.int64(20)), n_target=np.int32(5))
    assert cfg.band_bins == (10, 20)
    assert all(type(x) is int for x in cfg.band_bins)
    with pytest.raises(ValueError):
        UCBBlockConfig(band_bins=(np.True_, 20))
