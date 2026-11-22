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

"""Build the GTTE training dataset from a SyncHeal LeRobot dataset.

A SyncHeal dataset (e.g. ``SyncHeal/move-handle-test``) is a 10 Hz LeRobot v3 dataset whose repo also holds
``raw_streams/episode_XXXXXX/{frames,ft,gaze}.parquet`` with the force/torque and gaze streams at their native
rate. This script writes a local copy of the dataset where every 10 Hz frame also carries, for each frozen
encoder export given with ``--encoder``, the raw window that encoder expects:

- ``observation.<stream>_window``: float32 ``(T, C)``, built with causal zero-order hold on the stream's
  ``recv_time`` up to the frame's ``obs_time`` (same rule as the encoder training, see
  ``lerobot.policies.groot.temporal_encoders.zoh_windows``). If the encoder declares ``state_columns``
  (e.g. torso/neck joints for gaze), those entries of the *original* 10 Hz ``observation.state``, timestamped
  with ``obs_time``, are held on the same grid and appended as channels. Window length, rate and channels are
  read from the export, so the dataset always matches the encoder.
- ``observation.state``: by default without the 10 Hz ``ft.*`` / ``gaze.*`` snapshots (replaced by the
  temporal latents, as in TTE) and the constant ``ft_tare.*`` calibration; see ``--drop-state-prefix``.

Videos are hard-linked (or copied) unchanged; stats of the modified features are recomputed.

Example:
    python examples/syncheal_gtte/build_gtte_dataset.py \\
        --repo-id SyncHeal/move-handle-test \\
        --encoder ../tactile-encoder-ergocub/outputs/syncheal_ft_ae/export \\
        --encoder ../tactile-encoder-ergocub/outputs/syncheal_gaze_ae/export \\
        --out-root data/move-handle-test-gtte
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.temporal_encoders import load_temporal_encoder_export, zoh_windows
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_STATE

logger = logging.getLogger("build_gtte_dataset")

DEFAULT_DROP_STATE_PREFIXES = ["ft.", "gaze.", "ft_tare."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", default="SyncHeal/move-handle-test")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="Local copy of the source (default: HF_LEROBOT_HOME/<repo-id>).",
    )
    parser.add_argument(
        "--out-repo-id", default=None, help="Id of the output dataset (default: <repo-id>-gtte)."
    )
    parser.add_argument(
        "--out-root", type=Path, default=None, help="Output folder (default: HF_LEROBOT_HOME/<out-repo-id>)."
    )
    parser.add_argument(
        "--encoder",
        action="append",
        default=[],
        required=True,
        help="Encoder export folder (repeat per stream).",
    )
    parser.add_argument(
        "--drop-state-prefix",
        action="append",
        default=None,
        help=f"observation.state names starting with this prefix are dropped (repeatable). Default: {DEFAULT_DROP_STATE_PREFIXES}.",
    )
    parser.add_argument("--keep-full-state", action="store_true", help="Keep all observation.state dims.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_data_parquet(df: pd.DataFrame, path: Path, features: dict) -> None:
    # info.json stores shapes as lists; LeRobot's feature mapping expects tuples (as after `load_info`).
    hf_features = get_hf_features_from_features(
        {k: {**v, "shape": tuple(v["shape"])} for k, v in features.items() if v["dtype"] != "video"}
    )
    table = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features).with_format("arrow")[
        :
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy", use_dictionary=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    out_repo_id = args.out_repo_id or f"{args.repo_id}-gtte"
    src_root = args.src_root or HF_LEROBOT_HOME / args.repo_id
    out_root = args.out_root or HF_LEROBOT_HOME / out_repo_id
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    logger.info("Fetching %s (data, videos, meta, raw_streams) into %s", args.repo_id, src_root)
    snapshot_download(
        args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=src_root,
        allow_patterns=["meta/*", "data/*", "videos/*", "raw_streams/*"],
    )

    info = json.loads((src_root / "meta" / "info.json").read_text())
    features: dict = info["features"]
    src_stats = json.loads((src_root / "meta" / "stats.json").read_text())

    # --- streams / encoders -------------------------------------------------------------------------------
    windows_spec: dict[str, dict] = {}
    for path in args.encoder:
        spec, _ = load_temporal_encoder_export(path)
        key = f"observation.{spec['stream']['name']}_window"
        if key in windows_spec:
            raise ValueError(f"Two encoders produce the same feature '{key}'.")
        windows_spec[key] = {**spec, "export": str(Path(path).resolve())}
        logger.info(
            "%s: %s at %.1f Hz, window (%d, %d) = %.2f s, from %s",
            key,
            spec["stream"]["columns"],
            spec["stream"]["rate_hz"],
            spec["window_length"],
            spec["num_channels"],
            spec["window_length"] / spec["stream"]["rate_hz"],
            spec["stream"]["file"],
        )

    # --- state reduction ------------------------------------------------------------------------------------
    state_names: list[str] = features[OBS_STATE]["names"]
    prefixes = [] if args.keep_full_state else (args.drop_state_prefix or DEFAULT_DROP_STATE_PREFIXES)
    keep_idx = [i for i, n in enumerate(state_names) if not any(n.startswith(p) for p in prefixes)]
    kept_names = [state_names[i] for i in keep_idx]
    dropped = [n for n in state_names if n not in kept_names]
    logger.info("observation.state: %d -> %d dims (dropped %s)", len(state_names), len(kept_names), dropped)

    new_features = dict(features)
    new_features[OBS_STATE] = {**features[OBS_STATE], "shape": [len(kept_names)], "names": kept_names}
    for key, spec in windows_spec.items():
        new_features[key] = {
            "dtype": "float32",
            "shape": [spec["window_length"], spec["num_channels"]],
            "names": None,
        }
    changed_keys = [OBS_STATE, *windows_spec]

    for key, spec in windows_spec.items():
        missing = [c for c in spec["stream"]["state_columns"] if c not in state_names]
        if missing:
            raise KeyError(f"{key}: state columns {missing} are not in the source observation.state.")

    # --- data -----------------------------------------------------------------------------------------------
    episode_stats: dict[int, dict] = {}
    raw_root = src_root / "raw_streams"
    # Original 10 Hz state per episode (before reduction), sorted by frame: source of the state context channels.
    src_df = pd.concat(pd.read_parquet(p) for p in sorted((src_root / "data").glob("*/*.parquet")))
    full_state = {
        int(ep): (g["frame_index"].to_numpy(), np.stack(g[OBS_STATE].to_numpy()).astype(np.float32))
        for ep, g in src_df.sort_values("index").groupby("episode_index")
    }
    for src_path in sorted((src_root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(src_path).sort_values("index").reset_index(drop=True)
        state = np.stack(df[OBS_STATE].to_numpy())[:, keep_idx].astype(np.float32)
        window_cols: dict[str, np.ndarray] = {
            key: np.zeros((len(df), spec["window_length"], spec["num_channels"]), np.float32)
            for key, spec in windows_spec.items()
        }
        for ep in df["episode_index"].unique():
            rows = np.flatnonzero(df["episode_index"].to_numpy() == ep)
            frames = pd.read_parquet(raw_root / f"episode_{ep:06d}" / "frames.parquet").set_index(
                "frame_index"
            )
            obs_time = frames.loc[df["frame_index"].to_numpy()[rows], "obs_time"].to_numpy(np.float64)
            for key, spec in windows_spec.items():
                stream = spec["stream"]
                raw = pd.read_parquet(raw_root / f"episode_{ep:06d}" / stream["file"])
                raw = raw.dropna(subset=[stream["time_column"], *stream["columns"]])
                raw = raw.sort_values(stream["time_column"], kind="stable")
                parts = [
                    zoh_windows(
                        raw[stream["time_column"]].to_numpy(),
                        raw[stream["columns"]].to_numpy(),
                        obs_time,
                        stream["rate_hz"],
                        spec["window_length"],
                    )
                ]
                if stream["state_columns"]:
                    ep_frames, ep_state = full_state[int(ep)]
                    cols = [state_names.index(c) for c in stream["state_columns"]]
                    parts.append(
                        zoh_windows(
                            frames.loc[ep_frames, "obs_time"].to_numpy(np.float64),
                            ep_state[:, cols],
                            obs_time,
                            stream["rate_hz"],
                            spec["window_length"],
                        )
                    )
                window_cols[key][rows] = np.concatenate(parts, axis=-1)
            ep_data = {OBS_STATE: state[rows], **{k: v[rows] for k, v in window_cols.items()}}
            episode_stats[int(ep)] = compute_episode_stats(
                ep_data, {k: new_features[k] for k in changed_keys}
            )

        df[OBS_STATE] = list(state)
        for key, values in window_cols.items():
            df[key] = list(values)
        dst_path = out_root / src_path.relative_to(src_root)
        write_data_parquet(df, dst_path, new_features)
        logger.info("Wrote %s (%d frames)", dst_path.relative_to(out_root), len(df))

    # --- videos, tasks ----------------------------------------------------------------------------------------
    for src_path in sorted((src_root / "videos").rglob("*.mp4")):
        link_or_copy(src_path, out_root / src_path.relative_to(src_root))
    link_or_copy(src_root / "meta" / "tasks.parquet", out_root / "meta" / "tasks.parquet")

    # --- episodes metadata: refresh the per-episode stats columns of the changed features ------------------
    for src_path in sorted((src_root / "meta" / "episodes").glob("*/*.parquet")):
        ep_df = pd.read_parquet(src_path)
        stale = [c for c in ep_df.columns if any(c.startswith(f"stats/{k}/") for k in changed_keys)]
        ep_df = ep_df.drop(columns=stale)
        new_cols: dict[str, list] = {}
        for ep in ep_df["episode_index"]:
            for key in changed_keys:
                for stat, value in episode_stats[int(ep)][key].items():
                    new_cols.setdefault(f"stats/{key}/{stat}", []).append(np.asarray(value).tolist())
        ep_df = pd.concat([ep_df, pd.DataFrame(new_cols, index=ep_df.index)], axis=1)
        dst_path = out_root / src_path.relative_to(src_root)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(dst_path, index=False)

    # --- info + stats -----------------------------------------------------------------------------------------
    info["features"] = new_features
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    new_stats = aggregate_stats([episode_stats[ep] for ep in sorted(episode_stats)])
    stats = {
        k: {s: np.asarray(v) for s, v in src_stats[k].items()} for k in src_stats if k not in changed_keys
    }
    write_stats({**stats, **new_stats}, out_root)
    (out_root / "meta" / "gtte.json").write_text(
        json.dumps(
            {
                "source_repo_id": args.repo_id,
                "source_revision": args.revision,
                "dropped_state_names": dropped,
                "state_names": kept_names,
                "windows": windows_spec,
            },
            indent=2,
        )
    )

    # --- sanity check -------------------------------------------------------------------------------------------
    dataset = LeRobotDataset(out_repo_id, root=out_root)
    item = dataset[len(dataset) // 2]
    logger.info(
        "Built %s: %d episodes, %d frames. Sample: %s",
        out_root,
        dataset.num_episodes,
        dataset.num_frames,
        {k: tuple(item[k].shape) for k in changed_keys},
    )
    # The 10 Hz ft.* / gaze.* state entries are the last raw sample before obs_time, i.e. the last window step;
    # state context channels must match the frame's own state exactly.
    src_state = np.stack(src_df.sort_values("index")[OBS_STATE].to_numpy())
    for key, spec in windows_spec.items():
        stream = spec["stream"]
        last = np.asarray(dataset.hf_dataset.with_format("numpy")[key])[:, -1, :]
        cols = [f"{stream['name']}.{c}" for c in stream["columns"]]
        if all(c in state_names for c in cols):
            match = np.isclose(
                last[:, : len(cols)], src_state[:, [state_names.index(c) for c in cols]], atol=1e-4
            )
            logger.info(
                "%s: last window step == 10 Hz state snapshot on %.1f%% of frames",
                key,
                100 * match.all(-1).mean(),
            )
        if stream["state_columns"]:
            ctx = src_state[:, [state_names.index(c) for c in stream["state_columns"]]]
            if not np.allclose(last[:, len(stream["columns"]) :], ctx):
                raise RuntimeError(
                    f"{key}: state context channels do not match the frames' observation.state."
                )
            logger.info(
                "%s: state context %s matches observation.state on every frame", key, stream["state_columns"]
            )


if __name__ == "__main__":
    main()
