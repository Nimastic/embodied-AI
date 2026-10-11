# PRD: Arkham Batmobile World-Model Testbed

**Status:** draft · **Owner:** Jeriel · **Track:** parallel hardware track to `ROADMAP.md`

---

## 1. Why this exists

The original version of this document ([`raw/prd-original.md`](raw/prd-original.md)) specified a
*screen-accurate RC toy*: 1:8 scale, brushless motors, 15–25 km/h, 5000 mAh 4S LiPo. That is a
coherent project, but it is a **model-making** project. Almost nothing in it teaches world models.

This rewrite keeps the Batmobile and inverts the priority:

| | Original PRD | This PRD |
|---|---|---|
| Primary goal | visual fidelity | measurable learning results |
| Success metric | "recognizable as the Batmobile" | prediction error, planning success rate |
| Speed | 15–25 km/h | 0.5 m/s, closed-loop velocity control |
| Motors | 4× brushless + ESC | 4× geared DC **with encoders** |
| Scale | 1:8 (~75 cm) | ~1:12 (~40 cm) |
| Body | the deliverable | a shell bolted onto a working robot, and a domain-shift experiment |

The shell still gets built. It just stops being the thing that determines whether the project
succeeded.

---

## 2. Objectives

1. **A real robot-learning data loop on physical hardware.** Teleoperate, log synchronized
   observation/action streams, train on them, evaluate closed-loop. The same loop as `ROADMAP.md`
   Stage 02, on wheels instead of an arm.
2. **An action-conditioned world model that beats an analytic baseline.** The mecanum kinematic
   model is free and nearly right; the learned model must demonstrably predict the residual.
3. **Planning inside the learned model.** MPPI over imagined rollouts, closed-loop, replanning.
4. **A measured latency story.** How much inference delay a closed loop absorbs before it
   degrades, and whether action chunking recovers it.
5. **A Batmobile that looks the part**, for the demo video and because it is more fun to debug a
   robot you want to look at.

### Non-objectives

- Autonomous targeting of people, or firing at people under any control scheme.
- SLAM, map building, or navigation in unstructured spaces.
- Top speed, jumps, outdoor durability, 1:8 scale fidelity.

---

## 3. Success criteria

The project has succeeded when all of the following are true and written up:

| # | Criterion | Measurement |
|---|---|---|
| S1 | A reproducible dataset exists | ≥ 60 min of teleop across ≥ 2 floor surfaces, < 5 ms timestamp jitter, < 1% dropped frames |
| S2 | The analytic baseline is characterized | 1-step and 1.0 s pose-prediction error of mecanum FK on held-out data, per surface |
| S3 | The learned state model beats it | ≥ 30% lower 1.0 s pose error than analytic, on held-out data |
| S4 | Imagined video is legible | `CURRENT ∣ PREDICTED ∣ ACTUAL` filmstrip over a 1 s horizon that a human reads as the right future |
| S5 | Planning works | MPPI-in-learned-model success rate on a gate-traversal task, vs. reactive baseline and vs. MPPI-in-analytic-model |
| S6 | Latency is quantified | success-rate-vs-delay curve at 20/50/100/200 ms, with and without action chunking |
| S7 | Domain shift is measured | the same model evaluated after the body shell is attached and the surface changed, with a failure analysis of what broke |
| S8 | It is a Batmobile | recognizable Arkham Knight shell, working headlights and tail lights, 60 s demo video |

S3 is the load-bearing claim. If the learned model *cannot* beat mecanum FK, that is a legitimate
negative result and the write-up explains why — per `ROADMAP.md`, "mostly no, and here are six
reasons why" counts as a result.

---

## 4. System architecture

```text
        ┌──────────────── ROBOT (Raspberry Pi 5) ────────────────┐
        │                                                        │
 gamepad│  teleop ──┐                                            │
   or   │           ├──> action (vx, vy, w) ──> mecanum IK ──> 4x PWM + encoder PID
 policy │  policy ──┘                                            │        │
        │                                                        │        ▼
        │  camera (30 Hz) ─┐                                     │    motors
        │  encoders (100Hz)├──> timestamped observation ─────────┼──> logger (Parquet + mp4)
        │  IMU (100 Hz) ───┘                                     │
        └────────────────────────┬───────────────────────────────┘
                                 │  Wi-Fi
        ┌────────────────────────▼───────────────────────────────┐
        │              LAPTOP / GPU (trainer + planner)          │
        │  dataset ──> world model ──> MPPI ──> action chunk ────┼──> back to robot
        └────────────────────────────────────────────────────────┘
```

Two execution modes, deliberately:

- **Onboard mode** — a small model runs on the Pi; no network in the control loop.
- **Offboard mode** — the Pi streams observations and executes returned action chunks; the heavy
  model runs on the laptop. Network delay is real, measurable, and is the subject of S6.

---

## 5. Requirements

### 5.1 Mechanical

| Req | Spec |
|---|---|
| Scale | ~1:12, 380–450 mm long, ≤ 2.5 kg all-in |
| Drive | 4× mecanum wheels, 60–80 mm, one motor per wheel |
| Motors | geared brushed DC, 12 V, with quadrature or hall encoders |
| Top speed | 0.5 m/s, hard-limited in software |
| Chassis | flat 3–4 mm aluminium or acrylic plate, drilled grid so sensors can be remounted |
| Body | removable shell on ≥ 4 M3 standoffs, removable without unplugging anything |

Rationale for slow: at 0.5 m/s a one-second prediction horizon is 50 cm, which fits inside a
2 × 2 m taped arena and inside a camera frame. At 20 km/h the same horizon is 5.5 m — the robot
leaves the arena mid-episode, the dynamics are dominated by violent slip, and data collection
stops being repeatable. Encoders matter more than power: without them there is no velocity ground
truth, and therefore no world model.

### 5.2 Compute and sensing

| Req | Spec |
|---|---|
| Onboard compute | Raspberry Pi 5, 8 GB |
| Camera | Pi Camera Module 3 Wide (IMX708, 120° FOV), 640×480 @ 30 Hz capture, downsampled to 64×64 for models |
| Proprioception | 4× wheel encoders @ 100 Hz, 6-axis IMU @ 100 Hz |
| Ground-truth pose | overhead webcam plus an ArUco fiducial on the robot roof, ~30 Hz |
| Control loop | 20 Hz action rate, 100 Hz wheel PID |
| Clock | one monotonic clock; every stream timestamped at capture, never at write |

Ground-truth pose from an overhead fiducial is what makes S2 and S3 honest. Encoder odometry
drifts precisely *because* of the slip being studied, so it cannot also serve as the label.

### 5.3 Software

| Req | Spec |
|---|---|
| Robot runtime | Python, `picamera2`, `gpiozero`/`pigpio`, one process per sensor, shared-memory ring buffer |
| Dataset format | Parquet for state and action, mp4 for frames, one row per control tick — schema in [`software/README.md`](software/README.md) |
| Models | PyTorch: analytic baseline, MLP/GRU state model, CNN-encoder latent video model |
| Planner | MPPI, vectorized rollouts, K ≈ 256–1024 samples, horizon 10–20 ticks |
| Safety | software velocity clamp, 500 ms command watchdog that stops the robot on signal loss, hardware kill switch |

### 5.4 Aesthetics

| Req | Spec |
|---|---|
| Body v1 | foamboard or EVA foam plus matte black paint; cheap, fast, ~200 g |
| Body v2 | 3D-printed modular shell — plan preserved in [`hardware/build-notes.md`](hardware/build-notes.md) § Body shell |
| Lighting | WS2812 strip, white front and red rear, animation patterns on a spare GPIO |
| Sound | small I²S or PWM amp and speaker, engine and boost samples triggered on command |

Body v1 exists so that attaching it becomes a **measurable domain shift** (S7) rather than a
six-week blocker. Mass and camera height both change; the world model should notice.

---

## 6. Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Timestamp skew between camera and encoders | silently destroys action-conditioned learning; the model learns blurred dynamics | single monotonic clock, stamp at capture, log and plot jitter in M0 before any training |
| Learned model does not beat analytic FK | S3 fails | legitimate result; measure *where* it fails (acceleration transients, surface changes) and report that |
| Compounding rollout error | imagined futures useless past ~0.5 s | measure error-vs-horizon explicitly in M2, pick the planning horizon off that curve rather than off hope |
| Model exploitation by the planner | MPPI finds action sequences the model wrongly loves | penalize action magnitude and jerk, clamp to the training action distribution, compare planned against actual rollouts every episode |
| Pi 5 thermal throttling under camera plus inference | control rate drifts mid-episode | active cooler; log CPU temperature and loop period in every run metrics file |
| LiPo fire | real hazard, first-time owner | balance charger, charge inside a LiPo bag on a hard surface, never unattended, never below 3.3 V per cell, storage-charge if idle more than 3 days |
| Mecanum slip on soft carpet | rollers bind, odometry becomes garbage | characterize per surface in M1; use hard flooring or a thin mat as the primary arena |
| Scope creep into the body shell | the fun part eats the project | body v1 is foam and timeboxed to one weekend; v2 only after M4 passes |

---

## 7. Deliverables

- Working robot: mecanum base, Pi 5, camera, encoders, IMU, kill switch.
- Dataset: ≥ 60 min teleop, ≥ 2 surfaces, documented schema, reproducible loader.
- Four models: analytic baseline, state predictor, latent video predictor, MPPI planner.
- Seven experiment write-ups under `experiments/`, one per milestone, each with `metrics.json`.
- One article: *What a Learned World Model Knows That Mecanum Kinematics Does Not.*
- A 60-second demo video of an Arkham Knight Batmobile planning inside its own imagination.
