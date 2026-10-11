# Arkham Batmobile — Physical World-Model Testbed

A small mecanum-drive Batmobile replica used as a **physical testbed for action-conditioned
world models and planning**. The Batman shell is the reward for doing the hard part; the hard
part is the learning loop underneath it.

This is a **parallel hardware track** to the SO-101 manipulation line in
[`ROADMAP.md`](../../ROADMAP.md). It does not add stages to that roadmap and does not replace it.
It reuses the *methods* of Stages 06–10 (action-conditioned prediction → imagined rollouts →
planning → latency) on a cheap robot that physically exists months before an SO-101 arm would.

---

## The research question

A mecanum base is unusual among hobby robots: it has an **exact analytic model**. The map from
body velocity `(vx, vy, ω)` to four wheel speeds is closed-form textbook algebra.

On a real floor that model is wrong. Rollers slip. Carpet differs from tile. Batteries sag under
acceleration. Mass shifts when you bolt a body shell on top.

> **Can a learned action-conditioned model predict what the analytic model cannot — well enough
> to plan trajectories the analytic model gets wrong?**

This is a good first world-model question because the baseline is free and non-trivial. You are
not competing against nothing; you are competing against physics homework. Every claim is a
number: prediction error at horizon *H*, learned vs. analytic, on held-out data.

---

## How it maps to the canonical roadmap

| Milestone | What it builds | Roadmap analogue |
|---|---|---|
| M0 | Teleop → synchronized `(observation, action)` logging | Stage 02 — the data loop |
| M1 | Mecanum FK/IK + measured slip; the baseline to beat | Stage 01 / Stage 05 |
| M2 | `(state, actions) → future states` predictor | Stage 06 |
| M3 | Action-conditioned **video** prediction in latent space | Stage 06–07 |
| M4 | MPPI planning inside the learned model | Stage 09 |
| M5 | Closed-loop behaviour under 20/50/100/200 ms delay | Stage 10 |
| M6 | Domain shift: bolt on the body, change the floor | Stage 11 analogue |

Full detail in [`MILESTONES.md`](MILESTONES.md).

---

## Scope boundaries

Deliberately **in** scope:

- mecanum omnidirectional drive, slow and precisely controlled
- wheel encoders, IMU, one forward camera, overhead fiducial ground truth
- action-conditioned prediction, latent dynamics, MPPI / planning as inference
- inference latency, action chunking, asynchronous execution
- an Arkham Knight body shell, LEDs, sound

Deliberately **out** of scope:

- **Autonomous targeting of people.** The original notes sketched a robot that detects a person,
  drives at them and fires projectiles. That is not getting built. If a batarang launcher goes on
  later it is decoration: static targets, human finger on the trigger, never wired to a person
  detector. There is no world-model learning content in the launcher anyway.
- **SLAM, mapping, LiDAR, autonomous navigation.** Localization comes from encoder odometry plus
  an overhead fiducial marker in a taped arena. See `ROADMAP.md` → *What Is NOT on the Roadmap*.
- **Speed.** The original PRD specced 15–25 km/h brushless. Wrong tool — see
  [`hardware/BOM.md`](hardware/BOM.md) for why slow, repeatable and instrumented beats fast.
- Weapons, hydraulics, or a full-size drivable car.

---

## Current status

**Phase 0 running.** Parts not ordered — and nothing needs ordering yet.

A MuJoCo mecanum base with the real action interface is working: slip model, dataset collection in
the production schema, and a calibrated analytic baseline. See
[`software/sim/README.md`](software/sim/README.md) to run it and
[`experiments/P0-sim-baseline/`](experiments/P0-sim-baseline/README.md) for the numbers.

**M2 is done in simulation.** A learned residual model predicts pose 1.0 s ahead to 1.40 mm on
held-out hard-floor episodes, against 5.27 mm for a calibrated analytic baseline — a 73% gain,
well past the 30% target in S3. Predicting the residual beats predicting dynamics outright
(1.40 mm vs 2.57 mm), which was the M2 design claim.

**It does not transfer.** On a surface it never trained on, the learned model is indistinguishable
from the analytic baseline. It learned hard-floor dynamics, not dynamics. See
[`experiments/M2-state-predictor/`](experiments/M2-state-predictor/README.md).

Next: test whether a short history of (action, achieved velocity) lets the model identify the
surface and close that gap — cheaper and more informative than M3, and it changes what the real
robot needs to log.

---

## Layout

```text
├── README.md               ← you are here
├── PRD.md                  ← requirements, rewritten learning-first
├── MILESTONES.md           ← M0–M6, deliverables and metrics
├── hardware/
│   ├── BOM.md              ← phased budget, exact parts, buy order
│   └── build-notes.md      ← chassis, power, wiring, safety, body shell plan
├── software/               ← code (see software/README.md for layout + data format)
├── experiments/            ← one folder per run: metrics.json, plots, failure notes
├── notes/                  ← just-in-time concept notes
└── raw/                    ← the original unedited brain-dump, kept for provenance
```

## Operating rule

Inherited from the root [`AGENTS.md`](../../AGENTS.md): **70% build · 20% just-in-time learning ·
10% papers.** Each milestone ships code, a README, reproducible instructions, a quantitative
metric, a visual artifact, and a short failure analysis. A milestone is not complete because
reading happened.
