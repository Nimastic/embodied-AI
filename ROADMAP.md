# Canonical Embodied AI / WAM Roadmap

## North-Star Artifact

**Mini World-Action Model for SO-101**

```text
current observation + language instruction
                ↓
        world/action model
                ↓
 candidate actions + predicted futures
                ↓
       evaluate / select
                ↓
           execute
                ↓
       observe real result
                ↓
         replan / recover
```

The final system should exist in **both simulation and reality**.

Research niche:

> **world models + generative robot policies + planning/control + real-time physical execution**

---

## Stage 0: Simulated Robot, Immediately

### Goal

Get a simulated manipulation loop running before buying hardware.

### Stack

**Python + MuJoCo + LeRobot/robosuite where useful**

```text
observation
    ↓
policy/controller
    ↓
action
    ↓
MuJoCo
    ↓
next observation
```

You need to understand only:

| Concept | Required depth |
|---|---|
| state / observation | know exactly what information the robot receives |
| action | understand joint/action representation |
| dynamics | understand that environment maps state + action → next state |
| reward / cost | enough to specify desired behavior |
| control loop | observe → act → observe |

### Artifact: `00-sim-so101`

A virtual arm performs a simple reaching / push / pick-place task.

Publish:

- README
- architecture diagram
- 30–60 second video
- success rate
- one failure analysis

**No giant robotics course first.**

---

## Stage 1: Minimal Robotics Fundamentals

Learn only enough robotics to understand what your simulated and eventual physical arm is actually doing.

From Stanford CS223A, extract only:

**coordinate frames → transforms → forward kinematics → inverse kinematics → Jacobian**

Do not complete CS223A.

### Build

```text
joint angles
     ↓ FK
end-effector pose

desired pose
     ↓ IK
joint configuration
```

Verify your calculations against MuJoCo.

### Artifact: `01-so101-kinematics`

Interactive visualizer:

- joint state
- end-effector pose
- FK / IK
- target reaching
- Jacobian / singularity example

---

## Stage 2: Learn the Robot-Learning Data Loop

**teleoperation → demonstrations → dataset → policy → rollout → evaluation**

In simulation first. Then, once you buy the SO-101, use exactly the same pipeline physically.

### Build

Collect demonstrations:

```text
(o₀, a₀), (o₁, a₁), ..., (oT, aT)
```

Train simple **behavior cloning**, then progress to **ACT / action chunking**.

### Artifact: `02-imitation-learning`

Measure:

- number of demonstrations
- train success
- held-out scene success
- failure modes

---

## Stage 3: Physical SO-101

**Only now does the hardware purchase become justified.**

You've already built the virtual counterpart. Now the new learning comes from reality:

- calibration, cameras, servo noise, timing
- physical tolerances, bad demonstrations
- lighting, object variation
- USB/hardware failures
- sim-to-real mismatch

### Build

Replicate Stage 2 physically.

> Example: *Pick the red cube and place it into the bowl.*

### Artifact: `03-real-so101`

```text
SIMULATION        REAL ROBOT
    ↓                  ↓
same task          same task
same policy?       same policy?
    ↓                  ↓
            compare
```

---

## Stage 4: Modern Generative Robot Policy

Learn **Diffusion Policy**. Harold's "generative models" recommendation becomes operational.

Read the paper, understand what you need, implement/reproduce.

### Build

Compare on your manipulation task:

**BC vs ACT vs Diffusion Policy**

### Artifact: `04-generative-policies`

| Model | Success | OOD success | Inference latency |
|---|---:|---:|---:|
| BC | | | |
| ACT | | | |
| Diffusion | | | |

---

## Stage 5: Harold's Classical-Control Block

Only now stop briefly and go underneath modern methods.

**MPC + MPPI + planning as inference**

Use **one environment**. Compress into a single stage.

### Learn

```text
MPC → MPPI → planning as inference
```

> Given a current state, reason over possible futures and choose actions that produce desirable ones.

### Papers

- **MPC:** targeted AA203/Tedrake material
- **MPPI:** Williams et al., *Information Theoretic Model Predictive Control*
- **Planning as inference:** Levine, *Reinforcement Learning and Control as Probabilistic Inference*

### Artifact: `05-planning-foundations`

One interactive environment comparing **MPC vs MPPI vs inference-style planning**.

Write one article:

> **From MPC to Generative Robot Policies: Why Modern Embodied AI Keeps Rediscovering Planning**

---

## Stage 6: Your First World Model

So far your policy: o_t → a_t

Now build: (o_t, a_t) → ô_{t+1}

### Start embarrassingly small

Using MuJoCo:

1. collect trajectories
2. record images, robot state, actions
3. train a small action-conditioned predictor
4. predict next observation/state
5. compare against simulator's actual next state

### Artifact: `06-action-conditioned-world-model`

```text
CURRENT | PREDICTED NEXT | ACTUAL NEXT
```

plus prediction error.

---

## Stage 7: Multi-Step Imagined Futures

Upgrade: (o_t, a_{t:t+H}) → ô_{t+1:t+H}

Now your model predicts trajectories.

Learn: autoregressive prediction, diffusion/video models, latent world models, rollout error, model exploitation.

### Read: DreamerV3

### Artifact: `07-imagined-rollouts`

```text
candidate action sequence → predicted future rollout → actual MuJoCo rollout
```

Measure how error compounds with horizon.

---

## Stage 8: World Action Models

Deeply study DreamDojo / DreamZero / current WAM work.

Target changes from (o_t, a_t) → o_{t+1} to:

**(o_t, instruction) → (a_{t:t+H}, o_{t+1:t+H})**

Actions and world evolution modeled together.

### Artifact: `08-mini-wam`

On a constrained tabletop domain:

> language instruction → action sequence + predicted future → simulated execution

---

## Stage 9: Use the World Model for Planning

Generate candidate action sequences, world model predicts futures, choose best, execute, observe, replan.

### Artifact: `09-world-model-planning`

Research questions appear naturally:

- Does world-model planning outperform reactive policy execution?
- How long can imagined rollouts remain useful?
- How many candidate trajectories are enough?
- Does image prediction quality correlate with task success?
- Do latent predictions work better than pixel predictions?

---

## Stage 10: Systems / Real-Time Execution

Measure: **20 / 50 / 100 / 200 ms inference delay**

Study: asynchronous inference, action buffering, action chunk execution, control frequency, stale observations.

### Artifact: `10-real-time-wam`

Paper-style report:

> **Closed-Loop World Models Under Inference Latency**

---

## Stage 11: Sim-to-Real WAM

Take the most interesting result from simulation to your SO-101.

### North-star artifact: `11-so101-wam`

> **Can a low-cost robot use learned imagined futures to improve physical manipulation?**

Even if the answer is "Mostly no, and here are six reasons why," that is far more valuable than 95 hours of completed lectures.

---

## What Is NOT on the Roadmap

Do not spontaneously add unless the current artifact requires it:

| Topic | Status |
|---|---|
| Full ENGR108 | Reference only |
| Full CS223A | Reference only |
| Full CS231N | Reference only |
| EE259 / LiDAR / IMU | Ignore |
| SLAM | Ignore |
| Navigation | Ignore |
| Quadrupeds | Ignore |
| Humanoid locomotion | Ignore |
| PPO/SAC benchmark grinding | Ignore |
| Full CS234 | Reference only |
| Generic ROS mastery | Just-in-time |
| Generic CV curriculum | Just-in-time |
| Full RL curriculum | Just-in-time |
| Random robotics papers | Backlog |
| New roadmap creation | **Forbidden** |

---

## Operating Rule

**70% build / experiment — 20% just-in-time learning — 10% papers / exploration**

> **No new learning branch unless you can name the current experiment it unlocks.**

---

## Stage-Completion Discipline

Each stage must produce:

- working code
- README
- reproducible instructions
- experiment/result
- quantitative metric where applicable
- visual artifact (plot, GIF, or video)
- short failure analysis

A stage is not complete merely because papers or lectures were consumed.

---

## Current Status

**Active stage:** 00 — Simulated SO-101

**Next action:** Establish a simulated manipulation environment with an observation → action → physics → next observation loop and produce the first measurable artifact.
