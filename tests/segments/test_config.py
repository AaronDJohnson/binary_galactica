"""The block configuration has the specified defaults and validation."""

import json

import pytest

from binary_galactica.segments import UCBBlockConfig


def test_defaults_follow_the_spec():
    cfg = UCBBlockConfig()
    assert (cfg.n_target, cfg.steps_per_turn, cfg.ntemps, cfg.k_max_floor) == (
        10,
        1000,
        12,
        10,
    )
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
        {"skip_gaps": 1},
        {"skip_gaps": "yes"},
        {"max_interior_bins": 15},
        {"max_interior_bins": 0},
        {"max_interior_bins": 64.0},
        {"max_interior_bins": True},
        {"min_interior_bins": 32, "max_interior_bins": 31},
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


def test_numpy_integers_become_plain_ints_and_round_trip(tmp_path):
    import numpy as np

    cfg = UCBBlockConfig(
        n_target=np.int32(5),
        response_samples=np.int64(64),
        ntemps=np.int64(3),
        hottest_temperature=10.0,
    )
    for name in ("n_target", "ntemps", "response_samples"):
        assert type(getattr(cfg, name)) is int
    cfg.write(tmp_path / "c.json")
    back = UCBBlockConfig.read(tmp_path / "c.json")
    assert back == cfg and type(back.n_target) is int


def test_tiling_options_default_off_and_round_trip(tmp_path):
    import numpy as np

    assert UCBBlockConfig().skip_gaps is False
    assert UCBBlockConfig().max_interior_bins is None
    cfg = UCBBlockConfig(skip_gaps=True, max_interior_bins=np.int64(16))
    assert type(cfg.max_interior_bins) is int and cfg.max_interior_bins == 16
    cfg.write(tmp_path / "c.json")
    text = json.loads((tmp_path / "c.json").read_text())
    assert text["skip_gaps"] is True and text["max_interior_bins"] == 16
    assert UCBBlockConfig.read(tmp_path / "c.json") == cfg
    assert UCBBlockConfig.from_dict(cfg.to_dict()) == cfg


def test_config_without_the_tiling_options_still_reads(tmp_path):
    old = UCBBlockConfig(steps_per_turn=50).to_dict()
    del old["skip_gaps"], old["max_interior_bins"]
    (tmp_path / "config.json").write_text(json.dumps(old))
    cfg = UCBBlockConfig.read(tmp_path / "config.json")
    assert cfg == UCBBlockConfig(steps_per_turn=50)
    assert cfg.skip_gaps is False and cfg.max_interior_bins is None
