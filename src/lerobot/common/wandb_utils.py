#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Training run tracking with Weights & Biases (https://wandb.ai), next to (or instead of) Trackio.

Enabled with ``--wandb.enable=true``; needs ``pip install 'lerobot[wandb]'`` and a W&B login
(``wandb login``, ``~/.netrc`` or ``WANDB_API_KEY``). Restored from the logger LeRobot used before it
switched to Trackio, with the same interface as :class:`lerobot.common.trackio_utils.TrackioLogger`.
"""

import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import _wandb_available, require_package

if TYPE_CHECKING or _wandb_available:
    import wandb

WANDB_RESUME_MODES = frozenset({"allow", "must", "never", "auto"})
WANDB_MODES = frozenset({"online", "offline", "disabled"})


def cfg_to_group(
    cfg: TrainPipelineConfig, return_list: bool = False, truncate_tags: bool = False, max_tag_length: int = 64
) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters (W&B rejects tags longer than 64 characters)."""
        return tag if len(tag) <= max_tag_length else tag[:max_tag_length]

    trainable_kind = "reward_model" if cfg.is_reward_model_training else "policy"
    lst = [f"{trainable_kind}:{cfg.trainable_config.type}", f"seed:{cfg.seed}"]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    """Return the id of the W&B run stored under ``log_dir`` (used to resume it)."""
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    return match.group(1)


def get_safe_wandb_artifact_name(name: str) -> str:
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """Logs training metrics, evaluation videos and checkpoints to Weights & Biases."""

    def __init__(self, cfg: TrainPipelineConfig) -> None:
        require_package("wandb", "wandb")

        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        os.environ["WANDB_SILENT"] = "True"

        wandb_run_id: str | None = None
        if self.cfg.run_id:
            wandb_run_id = self.cfg.run_id
        elif cfg.resume:
            # `validate()` always resolves `output_dir` on resume; None here means it was skipped.
            if self.log_dir is None:
                raise ValueError("Resuming a WandB run requires `cfg.output_dir` to locate the previous run.")
            wandb_run_id = get_wandb_run_id_from_filesystem(Path(self.log_dir))
        resume = self.cfg.resume or ("must" if cfg.resume else None)
        if resume is not None and resume not in WANDB_RESUME_MODES:
            raise ValueError(f"Unsupported WandB resume mode: {resume!r}")
        mode = self.cfg.mode or "online"
        if mode not in WANDB_MODES:
            raise ValueError(f"Unsupported WandB mode: {mode!r}")
        run = wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            dir=self.log_dir,
            config=cfg.to_dict(),
            save_code=False,
            job_type="train_eval",
            resume=cast(Literal["allow", "must", "never", "auto"] | None, resume),
            mode=cast(Literal["online", "offline", "disabled"], mode),
            settings=wandb.Settings(
                console=self.cfg.console,
                console_multipart=self.cfg.console_multipart,
                console_chunk_max_seconds=self.cfg.console_chunk_max_seconds,
            ),
        )
        # Persist the W&B run id in the config so a resumed training reattaches to the same run.
        self.cfg.run_id = run.id
        # Custom step keys, for asynchronous training where several step counters coexist.
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        # `run.url` is None for offline runs (recent wandb versions removed `run.get_url()`).
        where = run.url or f"offline run, upload it with `wandb sync {self.log_dir}/wandb/offline-run-*`"
        logging.info(f"Track this run --> {colored(where, 'yellow', attrs=['bold'])}")

    def log_policy(self, checkpoint_dir: Path) -> None:
        """Upload the policy weights of a checkpoint as a W&B model artifact (unless disabled)."""
        if self.cfg.disable_artifact:
            return

        artifact_name = get_safe_wandb_artifact_name(f"{self._group}-{checkpoint_dir.name}")
        artifact = wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT model: the adapter plus the configs needed to load it.
            artifact.add_file(adapter_model_file)
            for name in ("adapter_config.json", "config.json"):
                if (pretrained_model_dir / name).exists():
                    artifact.add_file(pretrained_model_dir / name)
        elif standard_model_file.exists():
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # W&B's step must increase monotonically with each log call; asynchronous training (e.g. RL)
        # has several step counters, so those metrics are logged against a custom step key instead.
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                wandb.define_metric(new_custom_key, hidden=True)

        batch_data = {}
        for k, v in d.items():
            if custom_step_key is not None and k == custom_step_key:
                continue
            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue
            batch_data[f"{mode}/{k}"] = v

        if batch_data:
            if custom_step_key is not None:
                batch_data[f"{mode}/{custom_step_key}"] = d[custom_step_key]
                wandb.log(batch_data)
            else:
                wandb.log(data=batch_data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        wandb.log({f"{mode}/video": wandb.Video(video_path, fps=self.env_fps, format="mp4")}, step=step)

    def finish(self) -> None:
        wandb.finish()
