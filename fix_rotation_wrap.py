"""Make the axis-angle end-effector rotation continuous in the reduced dataset, in place.

The rotation is stored as a rotation vector (axis * angle) with norm <= pi. Near a half turn the
vectors `r` and `-r` are the same orientation, so the recorded values flip sign (`wx`: +pi -> -pi)
even though the hand moves smoothly. This script rewrites `ee.wx/wy/wz` of `action` and
`observation.state` so that they are continuous within every episode. The orientation stays
the same; only the representation changes (values may go slightly past pi).

Only the local reduced dataset is modified (videos are not touched, the source dataset is never
read). The original data/meta files are first copied to `<root>.pre_rotfix/`.

Usage:
    python fix_rotation_wrap.py --dry-run   # report only, nothing is written
    python fix_rotation_wrap.py             # overwrite the reduced dataset
    python fix_rotation_wrap.py --push      # ... and push the result to the Hub
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset, aggregate_stats, write_stats
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.io_utils import load_stats

REPO_ID = "LeGo98/graspglasshaptictattile_reduced"
ROOT = Path("data/graspglasshaptictattile_reduced")
ROT_NAMES = ("ee.wx", "ee.wy", "ee.wz")
FEATURES = ("action", "observation.state")
STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
JUMP_THRESHOLD = 2.0  # rad; a real frame-to-frame rotation is far smaller than this.


def rotation_indices(names: list[str]) -> list[int]:
    indices = [names.index(name) for name in ROT_NAMES]
    if indices != list(range(indices[0], indices[0] + 3)):
        raise ValueError(f"Rotation dims must be consecutive, got {indices}")
    return indices


def nearest_branch(rotvec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Equivalent rotation vector (same rotation, angle shifted by k * 2pi) closest to `reference`."""
    norm = np.linalg.norm(rotvec)
    if norm < 1e-9:
        return rotvec
    k = np.arange(-2, 3)[:, None]
    candidates = rotvec[None] * (1.0 + 2.0 * np.pi * k / norm)
    return candidates[np.argmin(np.linalg.norm(candidates - reference, axis=1))]


def unwrap(rotvecs: np.ndarray, episode_index: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Unwrap each episode in time order; the first frame of each episode is matched to `reference`."""
    out = np.empty_like(rotvecs, dtype=np.float64)
    previous = None
    for t in range(len(rotvecs)):
        new_episode = t == 0 or episode_index[t] != episode_index[t - 1]
        out[t] = nearest_branch(rotvecs[t].astype(np.float64), reference if new_episode else previous)
        previous = out[t]
    return out


def rotation_matrices(rotvecs: np.ndarray) -> np.ndarray:
    """Rodrigues formula, used only to check that the orientation is unchanged."""
    theta = np.linalg.norm(rotvecs, axis=1)[:, None, None]
    safe_theta = np.where(theta < 1e-12, 1.0, theta)
    axis = rotvecs[:, None, :] / safe_theta
    ax = np.zeros((len(rotvecs), 3, 3))
    ax[:, 0, 1], ax[:, 0, 2] = -axis[:, 0, 2], axis[:, 0, 1]
    ax[:, 1, 0], ax[:, 1, 2] = axis[:, 0, 2], -axis[:, 0, 0]
    ax[:, 2, 0], ax[:, 2, 1] = -axis[:, 0, 1], axis[:, 0, 0]
    return np.eye(3) + np.sin(theta) * ax + (1.0 - np.cos(theta)) * (ax @ ax)


def count_jumps(rotvecs: np.ndarray, episode_index: np.ndarray) -> tuple[int, int]:
    """Number of discontinuities and of episodes containing at least one."""
    delta = np.linalg.norm(np.diff(rotvecs, axis=0), axis=1)
    bad = (delta > JUMP_THRESHOLD) & (episode_index[1:] == episode_index[:-1])
    return int(bad.sum()), len(set(episode_index[:-1][bad]))


def to_arrow(values: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    flat = pa.array(values.astype(np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, type=arrow_type)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Only report, do not write anything.")
    parser.add_argument("--push", action="store_true", help=f"Push the fixed dataset to {REPO_ID}.")
    args = parser.parse_args()
    root = args.root

    features = json.loads((root / "meta/info.json").read_text())["features"]
    rot_idx = {key: rotation_indices(features[key]["names"]) for key in FEATURES}

    # ---- Load every data file, remembering which rows belong to which file.
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    tables = [pq.read_table(path) for path in data_files]
    columns = {
        key: np.concatenate([np.stack(t[key].to_numpy(zero_copy_only=False)) for t in tables])
        for key in FEATURES
    }
    episode_index = np.concatenate([t["episode_index"].to_numpy() for t in tables])
    frame_index = np.concatenate([t["frame_index"].to_numpy() for t in tables])
    if np.any(np.diff(episode_index) < 0) or np.any(
        (np.diff(frame_index) != 1) & (episode_index[1:] == episode_index[:-1])
    ):
        raise ValueError("Rows are not ordered by (episode_index, frame_index); refusing to guess the order.")
    print(
        f"Loaded {len(episode_index)} frames, {len(set(episode_index))} episodes "
        f"from {len(data_files)} file(s)"
    )

    # ---- Unwrap. The same reference is used for action and state so both end up on the same branch.
    reference = columns["action"][0, rot_idx["action"]].astype(np.float64)
    fixed = {}
    for key in FEATURES:
        old = columns[key][:, rot_idx[key]]
        new = unwrap(old, episode_index, reference)
        jumps_before, eps_before = count_jumps(old, episode_index)
        jumps_after, _ = count_jumps(new, episode_index)
        orientation_error = np.abs(rotation_matrices(old.astype(np.float64)) - rotation_matrices(new)).max()
        changed = int((np.abs(new - old) > 1e-5).any(axis=1).sum())
        print(
            f"{key:18s} jumps: {jumps_before} ({eps_before} episodes) -> {jumps_after} | "
            f"frames changed: {changed} | max orientation change: {orientation_error:.2e} | "
            f"range before {old.min(0).round(3)}..{old.max(0).round(3)} "
            f"after {new.min(0).round(3)}..{new.max(0).round(3)}"
        )
        if orientation_error > 1e-4:
            raise RuntimeError(f"{key}: unwrapping changed the orientation, aborting.")
        if jumps_after:
            raise RuntimeError(f"{key}: {jumps_after} discontinuities remain, aborting.")
        fixed[key] = new

    if all(np.abs(fixed[k] - columns[k][:, rot_idx[k]]).max() < 1e-5 for k in FEATURES):
        print("Rotation is already continuous, nothing to do.")
        return
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    # ---- Everything below is computed in memory first, so a failure cannot leave a half-written dataset.
    new_columns = {key: columns[key].copy() for key in FEATURES}
    for key in FEATURES:
        new_columns[key][:, rot_idx[key]] = fixed[key]

    # Per-episode stats: recompute, then replace only the rotation entries (other dims stay as they were).
    episodes = sorted(set(episode_index))
    episode_stats = {}
    for ep in episodes:
        rows = episode_index == ep
        episode_stats[ep] = compute_episode_stats(
            {key: new_columns[key][rows] for key in FEATURES}, {key: features[key] for key in FEATURES}
        )
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    new_episode_tables = []
    for path in episode_files:
        table = pq.read_table(path)
        row_of_episode = {int(ep): row for row, ep in enumerate(table["episode_index"].to_pylist())}
        for key in FEATURES:
            for stat in STAT_KEYS:
                name = f"stats/{key}/{stat}"
                values = table[name].to_pylist()
                for ep, row in row_of_episode.items():
                    for i in rot_idx[key]:
                        values[row][i] = float(np.asarray(episode_stats[ep][key][stat]).reshape(-1)[i])
                table = table.set_column(
                    table.schema.get_field_index(name),
                    name,
                    pa.array(values, type=table.schema.field(name).type),
                )
        new_episode_tables.append(table)

    # Global stats: aggregate the per-episode stats the same way the dataset writer does.
    global_stats = aggregate_stats([{key: episode_stats[ep][key] for key in FEATURES} for ep in episodes])
    stats = load_stats(root)
    for key in FEATURES:
        for stat in STAT_KEYS:
            for i in rot_idx[key]:
                stats[key][stat][i] = np.asarray(global_stats[key][stat]).reshape(-1)[i]

    new_data_tables, start = [], 0
    for table in tables:
        stop = start + table.num_rows
        for key in FEATURES:
            table = table.set_column(
                table.schema.get_field_index(key),
                key,
                to_arrow(new_columns[key][start:stop], table.schema.field(key).type),
            )
        new_data_tables.append(table)
        start = stop

    # ---- Back up what we are about to overwrite (small: parquet + json only), then write atomically.
    backup = root.with_name(root.name + ".pre_rotfix")
    if backup.exists():
        raise FileExistsError(f"{backup} already exists; move it away first so it is not overwritten.")
    shutil.copytree(root / "data", backup / "data")
    shutil.copytree(root / "meta/episodes", backup / "meta/episodes")
    shutil.copy2(root / "meta/stats.json", backup / "meta/stats.json")
    print(f"Backed up original data/meta files to {backup}")

    to_write = [
        *zip(data_files, new_data_tables, strict=True),
        *zip(episode_files, new_episode_tables, strict=True),
    ]
    for path, table in to_write:
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    write_stats(stats, root)
    print(
        f"Rewrote {len(data_files)} data file(s), {len(episode_files)} episode metadata file(s) "
        f"and stats.json in {root}"
    )

    if args.push:
        LeRobotDataset(REPO_ID, root=root).push_to_hub()
        print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
