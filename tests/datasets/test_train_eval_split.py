#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import split_episodes_by_task


def test_split_holds_out_last_episodes_by_default():
    train, eval_ = split_episodes_by_task({"a": list(range(10))}, 0.2)
    assert train == list(range(8))
    assert eval_ == [8, 9]


def test_split_rounds_the_held_out_count_up_per_task():
    train, eval_ = split_episodes_by_task({"a": [0, 1, 2], "b": [3, 4, 5, 6, 7]}, 0.1)
    # ceil(3 * 0.1) = 1 and ceil(5 * 0.1) = 1
    assert eval_ == [2, 7]
    assert train == [0, 1, 3, 4, 5, 6]


def test_shuffled_split_draws_a_random_subset_of_the_same_size():
    episodes = {"a": list(range(70))}
    train, eval_ = split_episodes_by_task(episodes, 0.1, shuffle=True, seed=42)

    assert len(eval_) == 7
    assert eval_ != list(range(63, 70))  # not simply the last episodes
    assert set(train).isdisjoint(eval_)
    assert sorted(train + eval_) == list(range(70))
    # both lists keep the original episode order
    assert train == sorted(train)
    assert eval_ == sorted(eval_)


def test_shuffled_split_is_reproducible_and_depends_on_the_seed():
    episodes = {"a": list(range(70))}
    first = split_episodes_by_task(episodes, 0.1, shuffle=True, seed=42)
    assert split_episodes_by_task(episodes, 0.1, shuffle=True, seed=42) == first
    assert split_episodes_by_task(episodes, 0.1, shuffle=True, seed=43) != first


def test_shuffled_split_is_drawn_per_task():
    episodes = {"a": list(range(0, 20)), "b": list(range(20, 30))}
    _, eval_ = split_episodes_by_task(episodes, 0.1, shuffle=True, seed=0)

    assert len([ep for ep in eval_ if ep < 20]) == 2
    assert len([ep for ep in eval_ if ep >= 20]) == 1


def test_dataset_config_eval_split_shuffle_is_off_by_default():
    assert DatasetConfig(repo_id="user/repo").eval_split_shuffle is False
    assert DatasetConfig(repo_id="user/repo", eval_split=0.1, eval_split_shuffle=True).eval_split_shuffle
