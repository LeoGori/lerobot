# GTTE: GR00T N1.7 + frozen temporal encoders on SyncHeal data

Fine-tune GR00T N1.7 on [`SyncHeal/move-handle-test`](https://huggingface.co/datasets/SyncHeal/move-handle-test)
while also feeding it the high-rate force/torque (~96 Hz) and gaze (~25 Hz) streams that the 10 Hz LeRobot
frames only snapshot. Following Temporal Tactile Encoding
([project page](https://hsp-iit.github.io/temporal-tactile-bimanual-handover/)), each stream is summarized by
an autoencoder trained separately and frozen; its latent is appended to the GR00T state vector. Both
encoders see the last 2 s of their stream (FT at 100 Hz; gaze + torso/neck joints at 30 Hz).

```
raw_streams/ft.parquet   (~96 Hz) --causal ZOH--> observation.ft_window   (200, 6) --frozen enc--> 32 latents --+
raw_streams/gaze.parquet (~25 Hz) --causal ZOH--> observation.gaze_window (60, 9)  --frozen enc-->  8 latents --+
  + torso/neck joints of the 10 Hz state (gaze pixels depend on the head pose)                                  |
observation.state (46: poses, apertures, joints; raw ft/gaze snapshots dropped) -- GR00T min-max norm --+        |
                                                                                                        v        v
                                                            GR00T state (132) = [ state 0:46 | latents 46:86 | 0 ]
```

The latents are standardized with the encoders' train-set statistics and written into the zero-padded state
slots, so they reach GR00T's (fine-tuned) state encoder like the paper's TTE embedding. The encoders are
part of the policy (`policy.temporal_encoders`, frozen, saved in `model.safetensors`), and their architecture
is saved in `config.json` (`temporal_encoder_specs`), so a checkpoint is self-contained.

## 1. Train the encoders (`tactile-encoder-ergocub` repo)

```bash
cd ../tactile-encoder-ergocub
git switch syncheal/ft-ae   && python -m syncheal_ae.train --config configs/syncheal_ft.yaml
git switch syncheal/gaze-ae && python -m syncheal_ae.train --config configs/syncheal_gaze.yaml
# -> outputs/syncheal_ft_ae/export, outputs/syncheal_gaze_ae/export   (add --set device=cpu to train on CPU)
```

## 2. Build the GTTE dataset

```bash
python examples/syncheal_gtte/build_gtte_dataset.py \
  --encoder ../tactile-encoder-ergocub/outputs/syncheal_ft_ae/export \
  --encoder ../tactile-encoder-ergocub/outputs/syncheal_gaze_ae/export \
  --out-root data/move-handle-test-gtte
```

Window length, rate and channels come from each export, so dataset and encoder always agree. Videos are
hard-linked/copied unchanged; `meta/gtte.json` records the windows and the dropped state names. By default the
10 Hz `ft.*`, `gaze.*` (replaced by the latents) and constant `ft_tare.*` entries are dropped from
`observation.state` (62 -> 46); see `--drop-state-prefix` / `--keep-full-state`.

## 3. Train GR00T

```bash
tmux new -s gtte
./examples/syncheal_gtte/train_gtte.sh            # STEPS=40000 BATCH_SIZE=8 by default, see the script header
```

Key flags: `--policy.temporal_encoders='{"observation.ft_window": "<ft export>", "observation.gaze_window":
"<gaze export>"}'`, `--policy.chunk_size=16 --policy.n_action_steps=8` (paper: 1.6 s chunks, re-plan after 8).
`--dataset.eval_split=0.1 --dataset.eval_split_shuffle=true` with `--seed=42` holds out episodes
`[1, 7, 15, 17, 40, 47]`, the same episodes the encoders used for validation.

Full fine-tuning (projector + diffusion head) needs most of a 48 GB GPU at batch 8.

## Deployment

Feed the policy, at each 10 Hz tick, the same windows the dataset was built with
(`lerobot.policies.groot.temporal_encoders.zoh_windows`):

```python
from lerobot.policies.groot.temporal_encoders import zoh_windows

# ft_t, ft_v: recent (recv_time, [fx..tz]) samples; gaze_t, gaze_v: (recv_time, [u, v, score]);
# tick_t, joints: obs_time and [torso_roll, torso_pitch, torso_yaw, neck_pitch, neck_roll, neck_yaw]
# of the recent 10 Hz policy ticks (not the high-rate encoders, to match training).
ft_window = zoh_windows(ft_t, ft_v, [now], rate_hz=100.0, length=200)[0]                 # (200, 6)
gaze_window = np.concatenate(
    [zoh_windows(gaze_t, gaze_v, [now], 30.0, 60)[0], zoh_windows(tick_t, joints, [now], 30.0, 60)[0]], axis=-1
)                                                                                         # (60, 9)
observation["observation.ft_window"] = torch.from_numpy(ft_window)
observation["observation.gaze_window"] = torch.from_numpy(gaze_window)
```

Keep a bit more than 2 s of FT, gaze and tick history in ring buffers. The frozen encoders run inside the policy;
only the raw windows are needed.
