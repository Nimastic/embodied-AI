# Software

Nothing is implemented yet. This file fixes the layout and the data schema up front, because the
schema is the one decision that is expensive to change later — every trained model, every
experiment and every plot depends on it.

## Intended layout

```text
software/
├── sim/                  Phase 0: MuJoCo mecanum base with the real action interface
│   ├── mecanum_env.py        (vx, vy, omega) in, camera render + wheel velocities out, 20 Hz
│   ├── slip_model.py         deliberate per-wheel slip + actuator lag, so FK is wrong in sim too
│   └── teleop_sim.py         keyboard teleop, writes the real dataset schema
├── robot/                M0: runs on the Pi 5
│   ├── teleop.py             gamepad/keyboard -> (vx, vy, omega)
│   ├── serial_link.py        20 Hz command/telemetry to the Pico
│   ├── camera.py             picamera2 capture, timestamp at capture
│   ├── logger.py             synchronized writer, one row per control tick
│   └── run_episode.py        episode orchestration, manifest, safety watchdog
├── pico/                 M0: runs on the Pico, 100 Hz
│   └── main.py               quadrature decode, mecanum IK, per-wheel velocity PID, watchdog
├── kinematics/           M1
│   ├── mecanum.py            closed-form FK and IK
│   └── calibrate.py          commanded vs achieved sweeps, slip + lag fitting
├── groundtruth/          M1
│   ├── aruco_pose.py         overhead marker -> arena-frame pose
│   └── calibrate_overhead.py intrinsics + floor homography
├── models/               M2-M3
│   ├── analytic.py           FK baseline as a predictor, same interface as the learned models
│   ├── state_model.py        MLP/GRU, residual-on-analytic and direct variants
│   └── latent_model.py       CNN encoder/decoder + latent dynamics
├── planning/             M4
│   └── mppi.py               batched rollouts through any predictor
├── eval/                 M2-M6
│   ├── error_vs_horizon.py
│   ├── rollout_compare.py    CURRENT | PREDICTED | ACTUAL filmstrips
│   └── closed_loop.py        success rate, time-to-goal, latency sweeps
└── data/                 gitignored (see root .gitignore: **/data/)
```

Every predictor — analytic, state, latent — implements the same interface, so MPPI and every eval
script work against all of them without special-casing:

```python
predict(state, action_sequence) -> predicted_state_sequence
```

That one constraint is what makes "learned versus analytic" a one-line config change instead of a
rewrite, and it is why `analytic.py` is a model rather than a utility.

---

## Dataset schema

One row per 20 Hz control tick. Parquet for everything numeric, mp4 alongside for frames with the
row carrying a frame index into it. No images in the Parquet.

| Column | Type | Units | Notes |
|---|---|---|---|
| `episode_id` | str | | matches the manifest |
| `tick` | int | | monotonic within episode, 20 Hz |
| `t_mono` | float64 | s | single monotonic clock, **stamped at capture** |
| `action_cmd` | float32[3] | m/s, m/s, rad/s | `(vx, vy, omega)` as commanded by teleop or policy |
| `wheel_cmd` | float32[4] | rad/s | setpoints after mecanum IK |
| `wheel_meas` | float32[4] | rad/s | measured from encoders on the Pico |
| `encoder_counts` | int32[4] | ticks | raw cumulative, for re-deriving velocity differently later |
| `imu_gyro` | float32[3] | rad/s | |
| `imu_accel` | float32[3] | m/s² | |
| `pose_gt` | float32[3] | m, m, rad | arena-frame `(x, y, theta)` from the overhead fiducial |
| `pose_gt_valid` | bool | | false when the marker is occluded or lost |
| `pose_odom` | float32[3] | m, m, rad | integrated mecanum FK; **drifts by design** |
| `body_vel_avg` | float32[3] | m/s, m/s, rad/s | **the learning target.** Average body velocity over the tick *ending* at this row's `pose_gt` |
| `frame_idx` | int32 | | index into the episode mp4, −1 if no frame this tick |
| `t_frame_mono` | float64 | s | actual frame capture time; will not equal `t_mono` exactly |
| `batt_v` | float32 | V | pack voltage; sag correlates with slip and belongs in the model input |
| `cpu_temp` | float32 | °C | catches thermal throttling after the fact |
| `surface` | str | | `hard`, `mat` — the M6 condition label |
| `shell` | bool | | body attached or not — the other M6 condition |

### The one alignment that will silently ruin everything

`body_vel_avg[t]` is the velocity over the tick that **ended** at `pose_gt[t]` — it describes the
past, not the future. To predict forward from `pose_gt[t]`, integrate `body_vel_avg[t+1 ...]`.

Get this off by one and the model learns to predict the tick it has already seen, which trains to
a beautiful loss and is worthless. Measured both ways: with the correct offset, integrating
`body_vel_avg` reproduces `pose_gt` 1.0 s ahead to **0.000 mm**; off by one, to 14.6 mm.

Use the exact-reproduction property as a unit test on any new dataset, sim or real. If integrating
the logged velocity does not return the logged pose, something upstream is wrong and no model will
fix it.

### Schema rules that are not negotiable

- **One clock.** Every timestamp from `time.monotonic()` on the Pi. The Pico reports its own tick
  counter, which gets mapped to Pi time on receipt; never mix two wall clocks.
- **Stamp at capture, not at write.** A timestamp applied when the logger gets around to writing
  the row is a timestamp of the logger, not of the world.
- **Keep raw encoder counts** as well as derived velocities. Every velocity-estimation choice you
  make now is one you will want to revisit, and counts let you do that without recollecting.
- **Keep `pose_gt_valid`.** Occluded-marker rows silently become zeros otherwise, and zeros look
  like a stationary robot to a training loop.
- **Average, never instantaneous.** The velocity logged for a tick is the average across it, not
  the value sampled at the end. A real encoder reports counts accumulated over an interval, so
  this is also the honest thing to record. Sampling instantaneously instead costs 8.9 mm of pure
  discretisation error over a 1 s horizon — two thirds of the analytic baseline, and nothing to
  do with world modelling.
- **`batt_v` is an input, not metadata.** Pack sag changes achieved wheel velocity for identical
  commands. Leaving it out means part of your dynamics is unobservable and the learned model gets
  blamed for it.

### Run manifest

One JSON per episode, next to the Parquet: git commit of the robot code, wheel radius and chassis
geometry, PID gains, camera intrinsics and overhead homography, surface, shell state, lighting,
time of day, operator notes, and anything that got bumped. The manifest is what makes an
experiment reproducible three months later when you have forgotten which gains were loaded.

---

## Conventions

- Body frame: `x` forward, `y` left, `theta` counter-clockwise positive. Write it in every module
  that touches geometry and never rely on remembering it.
- Actions are clamped to the training distribution before being sent, both in teleop and in
  planning. A planner that commands velocities no human ever demonstrated is extrapolating, and
  the model has nothing to say about it.
- Images are stored at capture resolution and downsampled at load time. Resampling in the logger
  throws away data you cannot get back.
- Every eval script writes a `metrics.json` into the matching `experiments/` folder. Plots are
  regenerated from it, never hand-edited.
