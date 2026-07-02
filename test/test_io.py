import json
import numpy as np
import pytest

import conftest as noise
from noisyvalue.core import (
    NoisyBool, NoisyFloat, NoisyInt,
    noisy_value_sampler, sample_noisy_values,
)
from noisyvalue.io import load, save


def _roundtrip(tmp_path, container):
    p = tmp_path / "data.json"
    save(p, container)
    return load(p)


# ── single NoisyFloat ──────────────────────────────────────────────────────────

def test_single_noisy_float_obs_survives_roundtrip(tmp_path):
    v = NoisyFloat.draw(10.0, noise.gaussian(0, 1), rng=0)
    rt = _roundtrip(tmp_path, v)
    assert isinstance(rt, NoisyFloat)
    assert rt._obs == pytest.approx(v._obs)


def test_single_noisy_float_posterior_survives_roundtrip(tmp_path):
    v = NoisyFloat.draw(10.0, noise.gaussian(0, 1), rng=0)
    rt = _roundtrip(tmp_path, v)
    orig_ci = v.credible_interval(rng=0)
    rt_ci = rt.credible_interval(rng=0)
    assert orig_ci == pytest.approx(rt_ci, abs=0.2)


# ── single NoisyInt / NoisyBool ───────────────────────────────────────────────

def test_single_noisy_int_obs_survives_roundtrip(tmp_path):
    v = NoisyInt.lift(7)
    rt = _roundtrip(tmp_path, v)
    assert isinstance(rt, NoisyInt)
    assert int(rt) == 7


def test_single_noisy_bool_obs_survives_roundtrip(tmp_path):
    v = NoisyBool.lift(True)
    rt = _roundtrip(tmp_path, v)
    assert isinstance(rt, NoisyBool)
    assert bool(rt) is True


# ── binomial noise source ─────────────────────────────────────────────────────

def test_binomial_noise_source_survives_roundtrip(tmp_path):
    v = NoisyInt.binomial(100, 0.3, obs=100)
    rt = _roundtrip(tmp_path, v)
    assert isinstance(rt, NoisyInt)
    orig_ci = v.credible_interval(rng=0)
    rt_ci = rt.credible_interval(rng=0)
    assert orig_ci == pytest.approx(rt_ci, abs=2.0)


# ── shared latent variable: joint structure survives ─────────────────────────

def test_shared_latent_dependency_survives_roundtrip(tmp_path):
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    b = a * 2.0
    rt_a, rt_b = _roundtrip(tmp_path, [a, b])
    draws_a, draws_b = sample_noisy_values(rt_a, rt_b, n=500, rng=2)
    assert np.allclose(draws_b.draws, 2.0 * draws_a.draws)


# ── no name collision after loading ───────────────────────────────────────────

def test_loaded_values_can_be_jointly_sampled_with_new_values(tmp_path):
    v = NoisyFloat.draw(3.0, noise.gaussian(0, 0.5), rng=0)
    rt = _roundtrip(tmp_path, v)
    fresh = NoisyFloat.draw(7.0, noise.gaussian(0, 0.5), rng=1)
    # This would raise if symbol names collided.
    batch_rt, batch_fresh = sample_noisy_values(rt, fresh, n=200, rng=3)
    assert batch_rt.draws.shape == (200,)
    assert batch_fresh.draws.shape == (200,)


# ── ndarray container ─────────────────────────────────────────────────────────

def test_1d_array_shape_and_obs_survive_roundtrip(tmp_path):
    arr = np.array([NoisyFloat.draw(float(i), noise.gaussian(0, 1), rng=i) for i in range(4)])
    rt = _roundtrip(tmp_path, arr)
    assert rt.shape == (4,)
    assert all(isinstance(rt[i], NoisyFloat) for i in range(4))
    assert [rt[i]._obs for i in range(4)] == pytest.approx([v._obs for v in arr])


def test_2d_array_shape_survives_roundtrip(tmp_path):
    arr = np.empty((2, 3), dtype=object)
    for idx in np.ndindex(2, 3):
        arr[idx] = NoisyFloat.draw(float(sum(idx)), noise.gaussian(0, 1), rng=sum(idx))
    rt = _roundtrip(tmp_path, arr)
    assert rt.shape == (2, 3)
    for idx in np.ndindex(2, 3):
        assert isinstance(rt[idx], NoisyFloat)
        assert rt[idx]._obs == pytest.approx(arr[idx]._obs)


# ── list / tuple container ────────────────────────────────────────────────────

def test_list_of_mixed_items_survives_roundtrip(tmp_path):
    v = NoisyFloat.draw(1.0, noise.gaussian(0, 1), rng=0)
    arr = np.array([NoisyInt.lift(i) for i in range(3)])
    rt = _roundtrip(tmp_path, [v, arr])
    assert isinstance(rt, list)
    assert len(rt) == 2
    rt_v, rt_arr = rt
    assert isinstance(rt_v, NoisyFloat)
    assert rt_v._obs == pytest.approx(v._obs)
    assert rt_arr.shape == (3,)
    assert [int(rt_arr[i]) for i in range(3)] == [0, 1, 2]


def test_tuple_container_roundtrips_as_tuple(tmp_path):
    a = NoisyFloat.draw(2.0, noise.gaussian(0, 1), rng=0)
    b = NoisyFloat.draw(4.0, noise.gaussian(0, 1), rng=1)
    rt = _roundtrip(tmp_path, (a, b))
    assert isinstance(rt, tuple)
    assert len(rt) == 2


# ── file format spot-check ────────────────────────────────────────────────────

def test_saved_file_is_valid_json_with_expected_keys(tmp_path):
    v = NoisyFloat.draw(1.0, noise.gaussian(0, 1), rng=0)
    p = tmp_path / "data.json"
    save(p, v)
    doc = json.loads(p.read_text())
    assert doc["version"] == 2
    assert "nodes" in doc
    assert "container" in doc
    assert doc["container"]["kind"] == "value"


def test_load_rejects_unknown_version(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": 99, "nodes": {}, "container": {}}))
    with pytest.raises(ValueError, match="version"):
        load(p)
