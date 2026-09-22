import os
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset

SOURCE_REPO = "TAPTIC/graspglasstattileno"
OUTPUT_REPO = "TAPTIC/graspglasstattileno_reduced"
OUTPUT_ROOT = Path("data/graspglassnohaptictattile_reduced")
# The source has two task entries (episodes 0-49 carry this text, episodes 50-75 an empty string); every
# episode of the reduced dataset gets this single instruction.
TASK = "Pick the glass and place it in the plate"
GENERATED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
NUM_WORKERS = min(8, os.cpu_count() or 1)

# "torchcodec" keeps a cached, open decoder per video across calls, whereas the
# "pyav" fallback reopens and re-seeks the container on every single frame access.
# For a sequential full-dataset read like this one, that's the difference between
# a few open()s per episode and one per frame.
source = LeRobotDataset(SOURCE_REPO, video_backend="torchcodec", revision="main", force_cache_sync=True)


# The source dataset is malformed: episodes 28-33 are missing (44 episode rows and
# 14975 frames on disk) but `episode_index` in the data was not renumbered and
# info.json still claims 50 episodes / 17035 frames. LeRobotDataset looks up
# `meta.episodes[episode_index]` *by row position*, so every episode after the gap
# silently got the video offsets of a different episode (and eventually decoded a
# timestamp past the end of the video). Look episodes up by their real
# `episode_index` instead, and only read the rows that actually exist.
class _EpisodesByIndex:
    def __init__(self, episodes):
        self._episodes = episodes
        self._row_of = {int(ep): row for row, ep in enumerate(episodes["episode_index"])}

    def __len__(self):
        # Used by `meta.get_video_file_path` as an upper bound on valid episode indices.
        return max(self._row_of) + 1

    def __getitem__(self, episode_index):
        return self._episodes[self._row_of[episode_index]]


source.meta.episodes = _EpisodesByIndex(source.meta.episodes)
num_source_frames = len(source._ensure_reader().hf_dataset)
print(f"Source frames on disk: {num_source_frames} (info.json says {len(source)})")

features = deepcopy(source.meta.info.features)
state_feature = features["observation.state"]
original_state_dim = len(state_feature["names"])

keep_names = [
    name for name in state_feature["names"]
    if not name.startswith("xhand.raw_force.")
]
keep_indices = [
    index for index, name in enumerate(state_feature["names"])
    if not name.startswith("xhand.raw_force.")
]

state_feature["names"] = keep_names
state_feature["shape"] = [len(keep_names)]

print(f"Original state dimension: {original_state_dim}")
print(f"Reduced state dimension: {len(keep_names)}")

reduced = LeRobotDataset.create(
    repo_id=OUTPUT_REPO,
    root=OUTPUT_ROOT,
    fps=source.fps,
    robot_type=source.meta.robot_type,
    features=features,
    use_videos=True,
)

current_episode = None

# Reading through a DataLoader lets num_workers subprocesses decode frames (and,
# for multi-camera setups, cameras within a frame) ahead of time in parallel,
# while this process writes them out sequentially. Order is preserved (no
# shuffling), so the written dataset is identical to a plain `for` loop over
# `source`, just produced faster. batch_size=None disables collation, so each
# item comes through exactly as `source[i]` would return it.
loader = DataLoader(
    Subset(source, range(num_source_frames)), batch_size=None, shuffle=False, num_workers=NUM_WORKERS
)

for frame in tqdm(loader, total=num_source_frames, desc="Copying frames"):
    episode_index = int(frame["episode_index"])

    if current_episode is not None and episode_index != current_episode:
        reduced.save_episode()

    current_episode = episode_index

    new_frame = {}
    for key in features:
        if key in GENERATED_FEATURES:
            continue

        value = frame[key]

        if key == "observation.state":
            value = torch.as_tensor(value)[keep_indices]
        elif key.startswith("observation.images."):
            value = torch.as_tensor(value).permute(1, 2, 0)

        new_frame[key] = value

    new_frame["task"] = TASK
    reduced.add_frame(new_frame)

if reduced.has_pending_frames():
    reduced.save_episode()

reduced.finalize()
# Not pushed here: the source's axis-angle rotation wraps around at +-pi, so fix it first and push once:
#   python fix_rotation_wrap.py --push
print(f"Saved reduced dataset to {OUTPUT_ROOT}")
print(f"Next: python fix_rotation_wrap.py --push   (makes ee.wx/wy/wz continuous, then pushes {OUTPUT_REPO})")