# Phase 0 — MuJoCo mecanum base

The robot, in simulation, with the same action interface the real one will have.
Build M0–M4 here while parts ship; when hardware lands, the training code points at a different
dataset and nothing else changes.

## Run it

Everything is already installed on this machine (MuJoCo 3.14.0, numpy, pandas, pyarrow,
matplotlib, torch 2.8.0+cpu). From `Project/arkham-batmobile/software/sim/`:

```bash
python ../kinematics/mecanum.py
```

Sanity-checks the kinematics — forward/strafe/spin wheel patterns, `fk(ik(v)) == v`, orthogonal
columns, the null-space scrub direction. Start here; if this fails nothing downstream is
meaningful.

```bash
python slip_model.py
```

Prints how wrong the analytic model is per surface and per mode. Strafe should be worst, `mat`
worse than `hard`. That ordering is the structure M2 has to rediscover.

```bash
python mecanum_env.py
```

Full characterisation: throughput, actuator time constant, delivered-velocity fraction, and the
analytic baseline. Writes plots and `metrics.json` to `experiments/P0-sim-baseline/assets/`.
Takes about a minute.

```bash
python mecanum_env.py --viewer
```

Opens the interactive MuJoCo viewer and drives the robot with random actions. **Watch this
first** — it makes the mecanum motion modes obvious in a way no diagram does. Drag to orbit,
scroll to zoom, double-click a body to track it.

```bash
python collect.py --minutes 10
```

Collects a dataset in the real schema. About 45× real time with frames, 500× without.

| Flag | Effect |
|---|---|
| `--mode random` | temporally correlated random actions (**default, best for dynamics**) |
| `--mode expert` | P controller to random goals; narrow action coverage, useful for M4 |
| `--mode teleop` | drive it yourself: `WASD` translate, `QE` rotate, `SPACE` stop |
| `--surface hard\|mat` | the M6 domain-shift condition |
| `--no-video` | state only, ~10× faster |
| `--mp4` | also write a viewable mp4 beside the lossless `.npy` |

Output lands in `software/data/sim/<surface>/<mode>/` — gitignored by the root `.gitignore`
(`**/data/`).

## What the pieces are

| File | Role |
|---|---|
| `../kinematics/mecanum.py` | IK, FK, scrub, the achievable-set clamp. **Runs unchanged on the Pico.** Derivation in `notes/just-in-time/mecanum-kinematics.md` |
| `mecanum_base.xml` | the MuJoCo model: 2 × 2 m arena, landmarks, gate posts, front + overhead cameras |
| `slip_model.py` | the deliberate imperfection — where the learnable residual comes from |
| `mecanum_env.py` | the env, the action clamp, and the analytic baseline ladder |
| `collect.py` | dataset collection in the schema defined in `../README.md` |

## Three design decisions worth understanding

**The wheels are visual only.** Simulating 36 roller bodies with contacts would be slow, fragile,
and would answer the wrong question. The chassis is a planar body driven by velocity actuators;
slip is injected in the command path by `slip_model.py`, which is where it happens on real
hardware. The goal is a plant whose analytic model is almost-but-not-quite right, not a faithful
contact model of mecanum hardware.

**Encoders see motor-side effects; they do not see contact slip.** Droop, per-wheel gain spread
and roller ripple are in `wheel_meas`. Contact slip is applied afterwards, to the body only. So
wheel odometry *over-reads* motion by 3–27% depending on mode and surface, exactly as it does in
reality — and that is why the overhead fiducial, not odometry, is the pose label.

**The action space is a box but the achievable set is a polytope.** `|vx| + |vy| + L|ω|` all draw
on one wheel-speed budget: `(0.5, 0.5, 2.0)` needs 33 rad/s from a wheel that tops out at 14. Every
action goes through `clamp_action()`, which scales all three components together — preserving the
direction of motion. Clipping wheels individually instead turns a straight strafe into a curve and
dumps the discarded motion into the scrub direction. (Measured: individual clipping drove mean
`|scrub|` to 1.004 rad/s against a proper value of 0.07.)

## Measured baseline

From `python mecanum_env.py` on this machine:

| | hard floor | mat |
|---|---:|---:|
| Actuator time constant | 59 ms translate, 60 ms rotate | — |
| Delivered fraction, forward / strafe / rotate | 92% / 81% / 87% | — |
| **Naive analytic**, 1.0 s pose error (median) | 31.7 mm | 49.1 mm |
| **Calibrated analytic**, 1.0 s pose error (median) | **4.7 mm** | **4.7 mm** |

Throughput: ~9,600 ticks/s physics-only (480× real time), ~940 renders/s at 64 × 64.

**The 4.7 mm figure is M2's target.** Beating the naive number proves nothing — fitting four
scalars is an afternoon's work. (This read 13.4 mm until 2026-10-13; the lag filter was
cold-started. See the correction in `experiments/P0-sim-baseline/README.md`.)

Two things to be careful about when reading that table:

- The calibrated fit recovers gains of 0.83 strafe / 0.93 forward on hard and 0.68 / 0.88 on mat,
  which matches the ground-truth ordering built into `slip_model.py`. That is a consistency check
  on the whole pipeline, not a result.
- Mat and hard show near-identical absolute error after the correction, which is not mat being
  equally easy — the robot simply
  travels less far per second when it slips more. Absolute millimetres are confounded with path
  length. **For M1 and M2, normalise error by distance travelled** before comparing across
  surfaces.

## Next

1. Watch `--viewer` for two minutes.
2. `python collect.py --minutes 30` on each surface — that is the M0 dataset.
3. M2: train `(state, actions) → future states`, predicting the **residual on top of the
   calibrated analytic model**. Beat 4.7 mm.

Training on CPU torch is fine for M2 (small MLP/GRU). M3's latent video model wants a GPU — use
Colab.
