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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

import draccus
import pytest

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import TrackioConfig, WandBConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers `--policy.type=act`)


@pytest.fixture
def wandb(monkeypatch):
    module = MagicMock()
    module.Settings.return_value = sentinel.settings
    module.init.return_value.id = "run-id"
    module.init.return_value.url = "https://wandb.ai/test/run-id"
    monkeypatch.setattr("lerobot.common.wandb_utils.wandb", module, raising=False)
    monkeypatch.setattr("lerobot.common.wandb_utils.require_package", lambda *args, **kwargs: None)
    monkeypatch.setenv("WANDB_SILENT", "False")
    return module


def _cfg(wandb_cfg: WandBConfig, output_dir=None, resume: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        wandb=wandb_cfg,
        output_dir=output_dir,
        job_name="test-run",
        env=None,
        trainable_config=SimpleNamespace(type="test-policy"),
        is_reward_model_training=False,
        seed=42,
        dataset=None,
        resume=resume,
        to_dict=lambda: {},
    )


def test_wandb_config_defaults():
    cfg = WandBConfig()

    assert cfg.enable is False
    assert cfg.resume is None
    assert cfg.disable_artifact is True
    assert cfg.console == "wrap"
    assert cfg.console_multipart is False
    assert cfg.console_chunk_max_seconds == 0


def test_wandb_logger_forwards_resume_and_console_settings(wandb, tmp_path):
    wandb_cfg = WandBConfig(
        project="proj",
        run_id="run-id",
        resume="allow",
        console="off",
        console_multipart=True,
        console_chunk_max_seconds=60,
    )

    WandBLogger(_cfg(wandb_cfg, output_dir=tmp_path))

    wandb.Settings.assert_called_once_with(console="off", console_multipart=True, console_chunk_max_seconds=60)
    kwargs = wandb.init.call_args.kwargs
    assert kwargs["resume"] == "allow"
    assert kwargs["settings"] is sentinel.settings
    assert kwargs["project"] == "proj"
    assert kwargs["mode"] == "online"
    assert kwargs["tags"] == ["policy:test-policy", "seed:42"]


def test_wandb_logger_resumes_with_checkpoint(wandb, tmp_path):
    WandBLogger(_cfg(WandBConfig(run_id="run-id"), output_dir=tmp_path, resume=True))

    assert wandb.init.call_args.kwargs["resume"] == "must"


def test_wandb_logger_resume_without_output_dir_raises(wandb):
    with pytest.raises(ValueError, match=r"cfg\.output_dir"):
        WandBLogger(_cfg(WandBConfig(), output_dir=None, resume=True))

    wandb.init.assert_not_called()


def test_wandb_logger_rejects_unknown_mode(wandb, tmp_path):
    with pytest.raises(ValueError, match="mode"):
        WandBLogger(_cfg(WandBConfig(mode="sometimes"), output_dir=tmp_path))


def test_wandb_logger_stores_run_id_and_logs_prefixed_metrics(wandb, tmp_path):
    wandb_cfg = WandBConfig()
    logger = WandBLogger(_cfg(wandb_cfg, output_dir=tmp_path))
    logger.log_dict({"loss": 0.5, "lr": 1e-4, "skipped": [1, 2]}, step=10)
    logger.log_dict({"eval_loss": 0.4}, step=10, mode="eval")
    logger.finish()

    assert wandb_cfg.run_id == "run-id"
    assert wandb.log.call_args_list[0].kwargs == {"data": {"train/loss": 0.5, "train/lr": 1e-4}, "step": 10}
    assert wandb.log.call_args_list[1].kwargs == {"data": {"eval/eval_loss": 0.4}, "step": 10}
    wandb.finish.assert_called_once()


def test_train_config_loads_wandb_section_next_to_trackio(tmp_path):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/repo",
            "--policy.type=act",
            "--wandb.enable=true",
            "--wandb.project=proj",
            "--wandb.entity=team",
        ],
    )
    cfg.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / TRAIN_CONFIG_NAME).read_text())
    assert payload["wandb"]["project"] == "proj"

    reloaded = TrainPipelineConfig.from_pretrained(tmp_path)

    assert reloaded.wandb == WandBConfig(enable=True, project="proj", entity="team")
    assert reloaded.trackio == TrackioConfig()
