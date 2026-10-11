# Milestones

Seven milestones, M0 to M6. Each one ships code, a README, reproducible instructions, a
quantitative metric, a visual artifact, and a short failure analysis — the same completion
discipline the root `AGENTS.md` imposes on roadmap stages.

Milestones are **not** roadmap stages. `ROADMAP.md` remains the SO-101 line and is untouched.
The "roadmap analogue" column only says which stage's *methods* each milestone practises.

| | Milestone | Roadmap analogue | Blocked on hardware? |
|---|---|---|---|
| M0 | Teleop and logging loop | Stage 02 | no — do it in sim first |
| M1 | Mecanum FK/IK plus measured slip | Stage 01 / 05 | partly |
| M2 | Action-conditioned state predictor | Stage 06 | no |
| M3 | Action-conditioned video prediction | Stage 06–07 | no |
| M4 | MPPI planning in the learned model | Stage 09 | no |
| M5 | Closed-loop under inference latency | Stage 10 | yes |
| M6 | Domain shift: body shell and floor change | Stage 11 analogue | yes |

---

## Phase 0: start today, spend nothing

Before any parts arrive, build a **mecanum base in MuJoCo with exactly the real action
interface**: a 3-vector `(vx, vy, omega)` in, a camera render and wheel velocities out, 20 Hz.

This mirrors `ROADMAP.md` Stage 0 philosophy — simulate before buying — and it means M0 through
M4 can be fully developed, debugged and benchmarked in simulation while parts ship. When hardware
lands, the same training code points at a different dataset.

Make the sim deliberately *imperfect*: add per-wheel slip coefficients and a little actuator lag,
so the analytic model is wrong in sim too. Otherwise M2 trivially succeeds in sim and then
collapses on real floors, and you learn nothing from the comparison.

**Deliverable:** `software/sim/` with a mecanum MuJoCo env, a keyboard teleop script, and a
10-minute logged dataset in the real schema.

**Status: done.** See [`software/sim/README.md`](software/sim/README.md) and
[`experiments/P0-sim-baseline/`](experiments/P0-sim-baseline/README.md). Calibrated analytic
baseline: 13.4 mm median 1.0 s pose error on hard floor, 11.8 mm on mat. Three bugs caught and
documented in that write-up.

---

## M0 — Teleop and logging loop

**Goal.** A human can drive the robot, and every tick is recorded in a form a model can train on.
This is the whole project in miniature: if the data is bad, nothing downstream is salvageable.

**Build.**
- Keyboard or gamepad → `(vx, vy, omega)` → mecanum IK → four wheel setpoints → PID on the Pico.
- Synchronized logger: one row per 20 Hz control tick, with camera frame index, four encoder
  counts, IMU sample, commanded action, overhead-fiducial pose, and a single monotonic timestamp
  per stream.
- Episode boundaries, a run manifest, and a loader that reconstructs `(obs, action)` pairs.

**Metrics.** Timestamp jitter distribution per stream; dropped-frame rate; achieved control-loop
period histogram; total minutes logged per surface.

**Visual artifact.** Jitter and loop-period histograms. A short teleop clip with the commanded
action vector overlaid on the frame.

**Gate.** Do not proceed to M2 until jitter is under ~5 ms and dropped frames under 1%. Skewed
timestamps teach a model blurred dynamics, and the failure is invisible later — it looks like
"world models just do not work here."

**Just-in-time learning.** Nothing new. This is engineering, and it is where most robot-learning
projects actually die.

---

## M1 — Mecanum FK/IK and measured slip

**Goal.** Build the baseline the learned model has to beat, and know exactly how wrong it is.

**Build.**
- Closed-form mecanum inverse kinematics: body velocity to four wheel angular velocities, with
  the wheel radius and the half-width-plus-half-length term derived from your actual chassis.
- Forward kinematics: four measured wheel velocities back to body velocity.
- Calibration sweeps: commanded velocity versus achieved velocity from the overhead fiducial,
  across forward, strafe, diagonal and spin, on each surface.
- Fit a per-wheel slip coefficient and an actuator lag constant per surface.

**Metrics.** Analytic 1-step and 1.0 s pose-prediction error (position in mm, heading in degrees)
on held-out data, per surface. Report mean and 95th percentile, not just mean.

**Visual artifact.** Commanded-versus-achieved velocity scatter per direction. A top-down plot of
analytic-predicted trajectory against fiducial ground truth for a few held-out episodes.

**This milestone produces the number in success criterion S2, and S3 is measured against it.**

**Just-in-time learning.** Mecanum kinematics derivation only. One afternoon, not a course.

---

## M2 — Action-conditioned state predictor

**Goal.** `(state_t, a_{t:t+H}) → state_{t+1:t+H}` using proprioception only — no pixels yet.

**Build.**
- State: body pose and velocity, four wheel velocities, IMU angular rate and acceleration.
- A small MLP or GRU. Start by predicting the **residual on top of analytic FK**, not the raw next
  state. This is the single highest-leverage design choice in the project: the network only has to
  learn what physics homework got wrong, which is a far smaller and better-conditioned function
  than the whole dynamics.
- Train on multi-step rollouts, not just one-step, or the model will look great at H=1 and
  diverge at H=10.

**Metrics.** Error-vs-horizon curve for H = 1 to 40 ticks (0.05 s to 2 s), three models on the
same axes: analytic, learned-residual, learned-direct. Held-out episodes and a held-out surface.

**Visual artifact.** The error-vs-horizon curve. A top-down overlay of predicted versus actual
trajectory for a hard episode.

**Gate.** Success criterion S3: at least 30% lower 1.0 s pose error than analytic. If it fails,
investigate *where* — transients, direction changes, specific surfaces — and write that up. A
documented negative result closes this milestone legitimately.

**Status: done in simulation.** Residual 1.40 mm vs calibrated 5.27 mm on held-out hard floor
(73% better, S3 target 30%). Residual beats direct, 1.40 vs 2.57 mm. **Transfer to an unseen
surface fails** — 1.1% better than analytic on `mat`. Write-up:
[`experiments/M2-state-predictor/`](experiments/M2-state-predictor/README.md). Still to do on real
hardware.

**Just-in-time learning.** Compounding error in learned dynamics; teacher forcing versus
multi-step training objectives. Skim the model-learning sections of `papers/world-models/
hafner-2023-dreamerv3.pdf`, not the whole paper.

---

## M3 — Action-conditioned video prediction

**Goal.** Imagine what the camera will see. `(image_t, a_{t:t+H}) → image_{t+1:t+H}`.

**Build.**
- Downsample frames to 64×64. Train a small convolutional encoder/decoder to a latent of 32–64
  dimensions. Keep it small; this is not a video-generation project.
- Latent dynamics conditioned on the action sequence, trained with multi-step rollouts in latent
  space and a reconstruction loss at the decoder.
- Predict 1–2 seconds ahead given a candidate action sequence.

**Metrics.** Latent and pixel prediction error vs. horizon. Whether image prediction quality
correlates with the state-space accuracy from M2 — one of the open research questions listed in
`ROADMAP.md` Stage 09.

**Visual artifact.** The `CURRENT ∣ PREDICTED ∣ ACTUAL` filmstrip from the roadmap, plus an
animated GIF of a one-second imagined rollout beside the real one.

**Just-in-time learning.** Latent world models and why latent prediction usually beats pixel
prediction for control. `papers/world-models/hafner-2023-dreamerv3.pdf`, properly this time.

---

## M4 — MPPI planning inside the learned model

**Goal.** The payoff. Stop predicting and start *acting on* predictions.

**Task.** Pick something where the analytic model is predictably wrong: drive through a narrow
taped gate at an angle requiring diagonal strafe, on a surface with uneven slip. Or reach a target
pose within a tight tolerance in minimum time. Needs a pose target and a clear success test.

**Build.**
- Sample K = 256–1024 action sequences over a horizon chosen off the M2 error curve, not guessed.
- Roll them all through the learned model, batched on GPU.
- Score against a cost function, take the softmax-weighted average action sequence, execute the
  first chunk, replan.
- Run the same planner against the analytic model as a control condition.

**Metrics.** Success rate and time-to-goal across ≥ 20 trials for four conditions: reactive
baseline, MPPI-in-analytic, MPPI-in-learned, and teleop human reference.

**Visual artifact.** Overhead-view trajectory bundles showing the sampled candidates, the chosen
plan, and what actually happened. Video of a successful and a failed gate traversal.

**Watch for model exploitation.** MPPI will find action sequences the learned model loves and
reality rejects. Log planned-versus-actual divergence every episode; if the gap is large, penalize
action magnitude and jerk and clamp samples to the training action distribution.

**Just-in-time learning.** `papers/foundations/williams-2017-information-theoretic-mpc.pdf` is the
MPPI paper and is directly implementable. `papers/foundations/levine-2018-control-as-inference.pdf`
for why this is planning as inference.

---

## M5 — Closed-loop behaviour under inference latency

**Goal.** Quantify what delay does to a closed loop, on real hardware. This is where the
Pi-plus-laptop architecture pays for itself: the latency is genuinely there, not simulated.

**Build.**
- Measure the real end-to-end delay of offboard mode: capture, encode, Wi-Fi, inference, return,
  execute.
- Inject additional controlled delay to sweep 20, 50, 100, 200 ms.
- Implement action chunking and asynchronous inference: execute a chunk while the next is being
  computed, instead of stalling on each round trip.
- Compare against onboard mode, where a smaller model runs on the Pi with no network in the loop.

**Metrics.** Success rate and tracking error versus injected delay, with and without chunking.
Measured wall-clock latency distribution, not nominal.

**Visual artifact.** Success-rate-versus-delay curve. Side-by-side video of the same task at 20 ms
and 200 ms.

**Just-in-time learning.** `papers/realtime/black-2025-real-time-chunking.pdf`.

**Write-up.** This is the roadmap Stage 10 report, retargeted: *Closed-Loop World Models Under
Inference Latency, on a $450 Robot.*

---

## M6 — Domain shift: the body shell and the floor

**Goal.** Break the model on purpose, in two controlled ways, and explain what broke.

**Build.**
- Train everything on the bare chassis on surface A.
- Evaluate unchanged on: bare chassis on surface B; body shell attached on surface A; shell on B.
- The shell changes mass, mass distribution, and camera height and angle. The surface changes slip.
  These are different kinds of distribution shift and should fail differently.
- Then measure how little new data fine-tuning needs to recover.

**Metrics.** M2 prediction error and M4 planning success rate in all four conditions. Episodes of
fine-tuning data required to return to within 10% of baseline performance.

**Visual artifact.** A 2×2 condition grid of prediction error. The demo video — an Arkham Knight
Batmobile planning through a gate using its own imagined futures.

**Failure analysis.** The most valuable page in the project. Which shift hurt more, and why; what
the model was secretly relying on; whether residual-on-analytic transferred better than direct
prediction.

---

## Deliberately not milestones

- **Projectile launcher.** No autonomous targeting, and nothing aimed at people. If a batarang
  launcher is added it is decoration on finished work: static targets, manual trigger, never wired
  to a person detector.
- **SLAM or mapping.** Localization is an overhead fiducial in a taped arena. Nothing more.
- **Outdoor driving, speed, jumps.** Different project, and it destroys data repeatability.
- **A second roadmap.** `ROADMAP.md` stays the single canonical roadmap. New topics discovered
  here go to `papers/backlog/` or `notes/just-in-time/`.
