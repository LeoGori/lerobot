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

"""Frozen temporal encoders for GR00T ("GTTE": GR00T + Temporal Tactile Encoding).

High-rate sensor streams (force/torque at ~100 Hz, gaze at ~30 Hz, ...) carry information that a 10 Hz
snapshot in ``observation.state`` loses. Following Temporal Tactile Encoding
(https://hsp-iit.github.io/temporal-tactile-bimanual-handover/), each stream is summarized by an
autoencoder that is trained separately and then frozen: the policy receives, for every 10 Hz frame, a
``(T, C)`` window of the raw stream (dataset feature ``observation.<name>_window``), optionally followed by
context channels taken from the 10 Hz ``observation.state`` (e.g. the torso -> head joints for gaze, whose
pixels live in the head-camera frame), encodes it with the
frozen encoder, standardizes the embedding with its train-set statistics, and appends it to the GR00T
state vector, in the zero-padded slots right after the real ``observation.state`` dimensions.

The encoders are exported by the ``syncheal_ae`` trainer (``tactile-encoder-ergocub`` repo, format
``syncheal_window_ae/v1``). Windows are built with :func:`zoh_windows` (causal zero-order hold on a
uniform grid), both when building the training dataset and at deployment time, so the encoder always sees
windows built the same way it was trained on.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

TEMPORAL_ENCODER_FORMAT = "syncheal_window_ae/v1"
_EXPORT_FILE_CANDIDATES = ("artifacts/model_state.pt", "model_state.pt")
# Spec keys that define the encoder architecture/window; must match to reuse exported weights.
_ARCH_KEYS = ("window_length", "num_channels", "embedding_dim", "encoder_hidden_dims", "dropout")


def zoh_windows(
    times: np.ndarray,
    values: np.ndarray,
    end_times: np.ndarray | Sequence[float],
    rate_hz: float,
    length: int,
) -> np.ndarray:
    """Build causal zero-order-hold windows from an irregular stream.

    Args:
        times: ``(N,)`` sorted sample times (same clock as ``end_times``).
        values: ``(N, C)`` samples.
        end_times: ``(M,)`` window end times (e.g. the ``obs_time`` of the 10 Hz frames).
        rate_hz: uniform grid rate of the window.
        length: number of grid steps ``T``; the last step is at the end time.

    Returns:
        ``(M, T, C)`` float32 windows. The value at grid time ``t`` is the last sample with ``time <= t``;
        grid times before the first sample take the first sample.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if len(times) == 0 or len(times) != len(values):
        raise ValueError(f"Need matching non-empty times/values, got {len(times)} and {len(values)}.")
    end_times = np.atleast_1d(np.asarray(end_times, dtype=np.float64))
    grid = end_times[:, None] + (np.arange(length) - (length - 1))[None, :] / float(rate_hz)
    idx = np.clip(np.searchsorted(times, grid.reshape(-1), side="right") - 1, 0, len(times) - 1)
    return values[idx].reshape(len(end_times), length, values.shape[-1])


def resolve_temporal_encoder_file(path: str | Path) -> Path | None:
    """Return the ``model_state.pt`` of an export (directory or file path), or None if it does not exist."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in _EXPORT_FILE_CANDIDATES:
            if (path / candidate).is_file():
                return path / candidate
    return None


def load_temporal_encoder_export(path: str | Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Load an exported window autoencoder.

    Returns:
        ``(spec, state_dict)`` where ``spec`` holds the architecture + stream metadata (JSON-serializable, it
        is stored in ``GrootConfig.temporal_encoder_specs``) and ``state_dict`` matches
        :class:`TemporalWindowEncoder` (encoder weights + normalization buffers).
    """
    file = resolve_temporal_encoder_file(path)
    if file is None:
        raise FileNotFoundError(
            f"No temporal encoder export at '{path}' (expected a model_state.pt or a folder with "
            f"{' or '.join(_EXPORT_FILE_CANDIDATES)})."
        )
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    meta = checkpoint.get("metadata", {})
    if meta.get("format") != TEMPORAL_ENCODER_FORMAT:
        raise ValueError(
            f"'{file}' is not a {TEMPORAL_ENCODER_FORMAT} export (format={meta.get('format')!r})."
        )
    model, stream = meta["model"], meta["stream"]
    state_columns = list(stream.get("state_columns") or [])
    if len(stream["columns"]) + len(state_columns) != int(model["num_channels"]):
        raise ValueError(
            f"'{file}': stream columns {stream['columns']} + state columns {state_columns} do not match "
            f"num_channels={model['num_channels']}."
        )
    spec: dict[str, Any] = {
        "window_length": int(model["window_length"]),
        "num_channels": int(model["num_channels"]),
        "embedding_dim": int(model["embedding_dim"]),
        "encoder_hidden_dims": [int(d) for d in model["encoder_hidden_dims"]],
        "dropout": float(model.get("dropout", 0.0)),
        "stream": {
            "name": stream["name"],
            "file": stream.get("file"),
            "columns": list(stream["columns"]),
            # observation.state entries appended after `columns`, held at the 10 Hz frames' obs_time.
            "state_columns": state_columns,
            "rate_hz": float(stream["rate_hz"]),
            "time_column": stream.get("time_column", "recv_time"),
        },
        "source": str(file),
    }
    # Keep only the encoder half of the autoencoder; the decoder is not needed by the policy.
    state_dict = {k: v for k, v in checkpoint["model"].items() if k.startswith("encoder.")}
    state_dict["input_mean"] = torch.tensor(meta["normalization"]["mean"], dtype=torch.float32)
    state_dict["input_std"] = torch.tensor(meta["normalization"]["std"], dtype=torch.float32)
    state_dict["latent_mean"] = torch.tensor(meta["latent_stats"]["mean"], dtype=torch.float32)
    state_dict["latent_std"] = torch.tensor(meta["latent_stats"]["std"], dtype=torch.float32)
    return spec, state_dict


def same_architecture(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return all(_canonical(a.get(k)) == _canonical(b.get(k)) for k in _ARCH_KEYS)


def _canonical(value: Any) -> Any:
    return [_canonical(v) for v in value] if isinstance(value, (list, tuple)) else value


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> nn.Sequential:
    # Mirrors syncheal_ae.model.mlp so exported `encoder.*` weights load as-is.
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(prev_dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class TemporalWindowEncoder(nn.Module):
    """Frozen encoder: raw ``(B, T, C)`` window -> standardized ``(B, L)`` latent."""

    def __init__(self, spec: Mapping[str, Any]):
        super().__init__()
        self.window_length = int(spec["window_length"])
        self.num_channels = int(spec["num_channels"])
        self.embedding_dim = int(spec["embedding_dim"])
        self.encoder = _mlp(
            self.window_length * self.num_channels,
            spec["encoder_hidden_dims"],
            self.embedding_dim,
            float(spec.get("dropout", 0.0)),
        )
        self.register_buffer("input_mean", torch.zeros(self.num_channels))
        self.register_buffer("input_std", torch.ones(self.num_channels))
        self.register_buffer("latent_mean", torch.zeros(self.embedding_dim))
        self.register_buffer("latent_std", torch.ones(self.embedding_dim))

    def forward(self, window: Tensor) -> Tensor:
        if window.ndim == 2:  # unbatched (T, C)
            window = window.unsqueeze(0)
        expected = (self.window_length, self.num_channels)
        if window.ndim != 3 or tuple(window.shape[1:]) != expected:
            raise ValueError(
                f"Expected a window of shape (B, {expected[0]}, {expected[1]}), got {tuple(window.shape)}."
            )
        x = (window.float() - self.input_mean) / self.input_std.clamp_min(1e-6)
        z = self.encoder(x.reshape(x.shape[0], -1))
        return (z - self.latent_mean) / self.latent_std.clamp_min(1e-6)


class TemporalEncoderBank(nn.Module):
    """The frozen encoders of a policy, applied in a fixed (sorted-key) order.

    The concatenated latents are ``[enc(batch[k]) for k in sorted(specs)]``; this order defines which state
    slots each stream occupies and is the same at training and inference time.
    """

    def __init__(self, specs: Mapping[str, Mapping[str, Any]]):
        super().__init__()
        if not specs:
            raise ValueError("TemporalEncoderBank needs at least one encoder spec.")
        self.keys = sorted(specs)
        self.encoders = nn.ModuleDict(
            {self.module_name(k): TemporalWindowEncoder(specs[k]) for k in self.keys}
        )
        self.requires_grad_(False)
        self.eval()
        self._warned_missing: set[str] = set()

    @staticmethod
    def module_name(key: str) -> str:
        return key.replace(".", "__")

    @property
    def latent_dim(self) -> int:
        return sum(enc.embedding_dim for enc in self.encoders.values())

    def train(self, mode: bool = True) -> TemporalEncoderBank:
        # Frozen: stay in eval mode even when the policy is put in training mode.
        return super().train(False)

    def load_encoder_state(self, key: str, state_dict: Mapping[str, Tensor]) -> None:
        self.encoders[self.module_name(key)].load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def forward(self, batch: Mapping[str, Any], missing_batch_size: int | None = None) -> Tensor:
        """Encode every window of ``batch`` into the concatenated ``(B, latent_dim)`` latents.

        Args:
            batch: holds the ``(B, T, C)`` windows under their feature keys.
            missing_batch_size: when given, a window absent from ``batch`` is replaced by the neutral latent
                (zeros: the encoder's train-set mean after standardization) for that many rows, so the policy
                can run without the stream. When None (training), a missing window is an error.
        """
        latents = []
        for key in self.keys:
            encoder = self.encoders[self.module_name(key)]
            if key not in batch:
                if missing_batch_size is None:
                    raise KeyError(
                        f"Missing temporal window '{key}' in the batch. Build the dataset with the GTTE window "
                        "features (examples/syncheal_gtte/build_gtte_dataset.py) or pass the window at inference."
                    )
                if key not in self._warned_missing:
                    self._warned_missing.add(key)
                    logger.warning(
                        "Temporal window '%s' not provided: using the neutral (train-set mean) latent for it.", key
                    )
                latents.append(encoder.latent_mean.new_zeros(missing_batch_size, encoder.embedding_dim))
                continue
            window = torch.as_tensor(batch[key], device=encoder.input_mean.device)
            latents.append(encoder(window))
        return torch.cat(latents, dim=-1)


def inject_latents_into_state(state: Tensor, latents: Tensor, offset: int) -> Tensor:
    """Write ``latents`` into ``state[..., offset:offset + L]`` (the zero padding after the real state).

    Args:
        state: GR00T state ``(B, H, max_state_dim)`` after the preprocessor (normalized + zero-padded).
        latents: ``(B, L)``.
        offset: number of real (normalized) ``observation.state`` dims.

    Rows the preprocessor zeroed with state dropout keep zero latents, so dropout drops the whole
    proprioceptive/temporal state like in TTE where the embedding is part of the state vector.
    """
    end = offset + latents.shape[-1]
    if end > state.shape[-1]:
        raise ValueError(
            f"observation.state ({offset} dims) + temporal latents ({latents.shape[-1]} dims) exceed the "
            f"GR00T state width {state.shape[-1]}; increase max_state_dim or reduce the state."
        )
    keep = state[..., :offset].abs().sum(dim=-1, keepdim=True) > 0
    state = state.clone()
    lat = latents.to(device=state.device, dtype=state.dtype).unsqueeze(1).expand(-1, state.shape[1], -1)
    state[..., offset:end] = torch.where(keep, lat, torch.zeros_like(lat))
    return state
