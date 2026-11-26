#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GTTE: frozen temporal encoders whose latents are appended to the GR00T state.

These tests exercise the encoder export format, the window construction, the state injection and the policy
wiring without loading the 3B GR00T backbone.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.temporal_encoders import (
    TEMPORAL_ENCODER_FORMAT,
    TemporalEncoderBank,
    _mlp,
    inject_latents_into_state,
    load_temporal_encoder_export,
    zoh_windows,
)
from lerobot.utils.constants import ACTION, OBS_STATE

FT_KEY = "observation.ft_window"
GAZE_KEY = "observation.gaze_window"


def _write_export(
    path,
    *,
    name,
    window_length,
    num_channels,
    embedding_dim,
    hidden=(16, 8),
    seed=0,
    dropout=0.0,
    state_columns=(),
):
    """Write a syncheal_window_ae/v1 export the way syncheal_ae.export does (encoder + decoder weights)."""
    torch.manual_seed(seed)
    encoder = _mlp(window_length * num_channels, hidden, embedding_dim, dropout).eval()
    decoder = _mlp(embedding_dim, hidden[::-1], window_length * num_channels, dropout)
    n_stream = num_channels - len(state_columns)
    state_dict = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    state_dict.update({f"decoder.{k}": v for k, v in decoder.state_dict().items()})
    metadata = {
        "format": TEMPORAL_ENCODER_FORMAT,
        "model": {
            "window_length": window_length,
            "num_channels": num_channels,
            "embedding_dim": embedding_dim,
            "encoder_hidden_dims": list(hidden),
            "decoder_hidden_dims": list(hidden[::-1]),
            "dropout": dropout,
        },
        "stream": {
            "name": name,
            "file": f"{name}.parquet",
            "columns": [f"c{i}" for i in range(n_stream)],
            "state_columns": list(state_columns),
            "rate_hz": 100.0,
            "window_length": window_length,
            "time_column": "recv_time",
        },
        "normalization": {"mean": [0.5] * num_channels, "std": [2.0] * num_channels},
        "latent_stats": {"mean": [0.1] * embedding_dim, "std": [3.0] * embedding_dim},
    }
    (path / "artifacts").mkdir(parents=True)
    torch.save({"model": state_dict, "metadata": metadata}, path / "artifacts" / "model_state.pt")
    return encoder


@pytest.fixture
def exports(tmp_path):
    ft_dir, gaze_dir = tmp_path / "ft", tmp_path / "gaze"
    ft_encoder = _write_export(ft_dir, name="ft", window_length=8, num_channels=6, embedding_dim=4, seed=0)
    gaze_encoder = _write_export(
        gaze_dir, name="gaze", window_length=5, num_channels=3, embedding_dim=2, seed=1
    )
    return {"ft": (ft_dir, ft_encoder), "gaze": (gaze_dir, gaze_encoder)}


def _config(temporal_encoders, state_dim=10, max_state_dim=132, specs=None):
    return GrootConfig(
        device="cpu",
        max_state_dim=max_state_dim,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            FT_KEY: PolicyFeature(type=FeatureType.STATE, shape=(8, 6)),
            GAZE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(5, 3)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        temporal_encoders=temporal_encoders,
        temporal_encoder_specs=specs or {},
    )


def test_zoh_windows_are_causal_and_edge_padded():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.array([[0.0], [1.0], [2.0], [3.0]])

    windows = zoh_windows(times, values, end_times=[0.25, 0.35, 0.3], rate_hz=10.0, length=4)

    # Grid for end 0.25: [-0.05, 0.05, 0.15, 0.25] -> before first sample -> sample 0, then 0, 1, 2.
    np.testing.assert_array_equal(windows[0, :, 0], [0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(windows[1, :, 0], [0.0, 1.0, 2.0, 3.0])
    # A sample received exactly at the end time (the frame's obs_time) is part of the window.
    assert windows[2, -1, 0] == 3.0
    assert windows.dtype == np.float32 and windows.shape == (3, 4, 1)


def test_export_loads_into_bank_and_matches_manual_encoding(exports):
    ft_dir, ft_encoder = exports["ft"]
    spec, state_dict = load_temporal_encoder_export(ft_dir)
    assert spec["stream"]["name"] == "ft" and spec["embedding_dim"] == 4
    assert not any(k.startswith("decoder.") for k in state_dict)

    bank = TemporalEncoderBank({FT_KEY: spec})
    bank.load_encoder_state(FT_KEY, state_dict)
    window = torch.randn(3, 8, 6)

    latents = bank({FT_KEY: window})

    expected = (ft_encoder(((window - 0.5) / 2.0).reshape(3, -1)) - 0.1) / 3.0
    torch.testing.assert_close(latents, expected)
    # Unbatched (T, C) windows are accepted (deployment).
    torch.testing.assert_close(bank({FT_KEY: window[0]}), expected[:1])


def test_export_with_dropout_and_state_context_channels(tmp_path):
    # Gaze-style export: 3 stream channels + 2 joint channels from the 10 Hz state, dropout layers in the MLP.
    encoder = _write_export(
        tmp_path,
        name="gaze",
        window_length=5,
        num_channels=5,
        embedding_dim=2,
        dropout=0.1,
        state_columns=("neck_pitch.pos", "neck_yaw.pos"),
    )
    spec, state_dict = load_temporal_encoder_export(tmp_path)
    assert spec["stream"]["columns"] == ["c0", "c1", "c2"]
    assert spec["stream"]["state_columns"] == ["neck_pitch.pos", "neck_yaw.pos"]

    bank = TemporalEncoderBank({GAZE_KEY: spec})
    bank.load_encoder_state(GAZE_KEY, state_dict)
    window = torch.randn(4, 5, 5)

    expected = (encoder(((window - 0.5) / 2.0).reshape(4, -1)) - 0.1) / 3.0
    torch.testing.assert_close(bank({GAZE_KEY: window}), expected)


def test_export_rejects_channel_count_mismatch(tmp_path):
    _write_export(
        tmp_path, name="gaze", window_length=5, num_channels=5, embedding_dim=2, state_columns=("j",)
    )
    checkpoint = torch.load(tmp_path / "artifacts" / "model_state.pt", weights_only=True)
    checkpoint["metadata"]["stream"]["state_columns"] = []
    torch.save(checkpoint, tmp_path / "artifacts" / "model_state.pt")

    with pytest.raises(ValueError, match="num_channels"):
        load_temporal_encoder_export(tmp_path)


def test_bank_is_frozen_and_ordered(exports):
    specs = {
        key: load_temporal_encoder_export(exports[name][0])[0]
        for key, name in ((GAZE_KEY, "gaze"), (FT_KEY, "ft"))
    }
    bank = TemporalEncoderBank(specs)
    bank.train()

    assert bank.keys == sorted([FT_KEY, GAZE_KEY])
    assert bank.latent_dim == 6
    assert not any(p.requires_grad for p in bank.parameters())
    assert not any(m.training for m in bank.modules())
    out = bank({FT_KEY: torch.zeros(2, 8, 6), GAZE_KEY: torch.zeros(2, 5, 3)})
    assert out.shape == (2, 6) and not out.requires_grad
    with pytest.raises(KeyError, match="observation.gaze_window"):
        bank({FT_KEY: torch.zeros(2, 8, 6)})


def test_inject_latents_into_padded_state_respects_state_dropout():
    state = torch.zeros(3, 1, 8)
    state[0, 0, :3] = torch.tensor([0.1, -0.2, 0.3])
    state[2, 0, :3] = torch.tensor([1.0, 0.0, 0.0])  # row 1 was zeroed by the preprocessor's state dropout
    latents = torch.arange(6, dtype=torch.float32).view(3, 2)

    out = inject_latents_into_state(state, latents, offset=3)

    torch.testing.assert_close(out[..., :3], state[..., :3])
    torch.testing.assert_close(out[0, 0, 3:5], latents[0])
    torch.testing.assert_close(out[1, 0], torch.zeros(8))
    torch.testing.assert_close(out[2, 0, 3:5], latents[2])
    torch.testing.assert_close(out[..., 5:], torch.zeros(3, 1, 3))
    assert state[0, 0, 3] == 0, "input state must not be modified in place"
    with pytest.raises(ValueError, match="exceed"):
        inject_latents_into_state(state, torch.zeros(3, 6), offset=3)


def test_policy_builds_encoders_from_exports_and_injects_latents(exports):
    paths = {FT_KEY: str(exports["ft"][0]), GAZE_KEY: str(exports["gaze"][0])}
    config = _config(paths)
    policy = SimpleNamespace(config=config)

    bank = GrootPolicy._create_temporal_encoders(policy)
    config.validate_features()

    assert set(config.temporal_encoder_specs) == {FT_KEY, GAZE_KEY}
    assert config.temporal_latent_dim == 6
    policy.temporal_encoders = bank
    batch = {FT_KEY: torch.randn(2, 8, 6), GAZE_KEY: torch.randn(2, 5, 3)}
    state = torch.zeros(2, 1, 132)
    state[:, 0, :10] = 0.5
    groot_inputs = {"state": state, "embodiment_id": torch.zeros(2, dtype=torch.int32)}

    out = GrootPolicy._inject_temporal_latents(policy, batch, groot_inputs)

    torch.testing.assert_close(out["state"][:, 0, 10:16], bank(batch))
    assert torch.all(out["state"][:, 0, 16:] == 0)
    assert out["embodiment_id"] is groot_inputs["embodiment_id"]


def test_bank_uses_neutral_latent_for_missing_windows_only_when_allowed(exports):
    specs = {
        key: load_temporal_encoder_export(exports[name][0])[0]
        for key, name in ((GAZE_KEY, "gaze"), (FT_KEY, "ft"))
    }
    bank = TemporalEncoderBank(specs)
    ft_window = torch.randn(3, 8, 6)

    out = bank({FT_KEY: ft_window}, missing_batch_size=3)

    torch.testing.assert_close(out[:, :4], bank.encoders[bank.module_name(FT_KEY)](ft_window))
    torch.testing.assert_close(out[:, 4:], torch.zeros(3, 2))
    torch.testing.assert_close(bank({}, missing_batch_size=2), torch.zeros(2, 6))
    with pytest.raises(KeyError, match="observation.gaze_window"):
        bank({FT_KEY: ft_window})


def test_policy_inference_runs_without_windows_but_training_requires_them(exports):
    paths = {FT_KEY: str(exports["ft"][0]), GAZE_KEY: str(exports["gaze"][0])}
    policy = SimpleNamespace(config=_config(paths))
    policy.temporal_encoders = GrootPolicy._create_temporal_encoders(policy)
    state = torch.zeros(2, 1, 132)
    state[:, 0, :10] = 0.5
    batch = {GAZE_KEY: torch.randn(2, 5, 3)}  # no FT window

    out = GrootPolicy._inject_temporal_latents(policy, batch, {"state": state}, allow_missing=True)

    torch.testing.assert_close(out["state"][:, 0, 10:14], torch.zeros(2, 4))  # FT: neutral latent
    torch.testing.assert_close(
        out["state"][:, 0, 14:16], policy.temporal_encoders.encoders["observation__gaze_window"](batch[GAZE_KEY])
    )
    with pytest.raises(KeyError, match="observation.ft_window"):
        GrootPolicy._inject_temporal_latents(policy, batch, {"state": state})


def test_policy_rebuilds_encoders_from_config_specs_without_exports(exports, tmp_path):
    paths = {FT_KEY: str(exports["ft"][0]), GAZE_KEY: str(exports["gaze"][0])}
    trained = SimpleNamespace(config=_config(paths))
    GrootPolicy._create_temporal_encoders(trained)
    specs = trained.config.temporal_encoder_specs

    # e.g. a fine-tuned checkpoint loaded on the robot: export paths are gone, specs are in config.json.
    missing = {FT_KEY: str(tmp_path / "missing_ft"), GAZE_KEY: str(tmp_path / "missing_gaze")}
    deployed = SimpleNamespace(config=_config(missing, specs=specs))
    bank = GrootPolicy._create_temporal_encoders(deployed)

    assert bank is not None and bank.latent_dim == 6

    with pytest.raises(FileNotFoundError, match="no spec"):
        GrootPolicy._create_temporal_encoders(SimpleNamespace(config=_config(missing)))


def test_config_rejects_mismatched_windows_and_state_overflow(exports):
    paths = {FT_KEY: str(exports["ft"][0]), GAZE_KEY: str(exports["gaze"][0])}
    specs = {key: load_temporal_encoder_export(path)[0] for key, path in paths.items()}

    with pytest.raises(ValueError, match="max_state_dim"):
        _config(paths, state_dim=10, max_state_dim=15, specs=specs).validate_features()

    bad_specs = {**specs, FT_KEY: {**specs[FT_KEY], "window_length": 9}}
    with pytest.raises(ValueError, match="expects windows of shape"):
        _config(paths, specs=bad_specs).validate_features()

    config = _config({**paths, "observation.missing_window": "x"}, specs=specs)
    with pytest.raises(ValueError, match="not input features"):
        config.validate_features()
