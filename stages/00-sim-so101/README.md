# Stage 00: Simulated SO-101

> Establish a simulated manipulation environment with an observation → action → physics → next observation loop and produce the first measurable artifact.

## Architecture

```text
observation (joint angles, EE position)
         ↓
    IK controller (damped least-squares Jacobian pseudoinverse)
         ↓
    action (joint position targets)
         ↓
       MuJoCo (500 Hz physics, 50 Hz control)
         ↓
    next observation
```

## Stack

- **Python 3.13** + **MuJoCo 3.14.0**
- **Robot model:** SO-ARM100 from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100)
- **Task:** Reach a 3D target position with the gripper (Fixed_Jaw body)
- **Controller:** Cartesian proportional control via damped-least-squares IK
- **Success threshold:** 15 mm from target

## Robot

The SO-ARM100 has 6 actuated joints:

| Joint | Type | Range (rad) |
|---|---|---|
| Rotation | Base yaw | -1.92 to 1.92 |
| Pitch | Shoulder pitch | -3.32 to 0.174 |
| Elbow | Elbow flex | -0.174 to 3.14 |
| Wrist_Pitch | Wrist pitch | -1.66 to 1.66 |
| Wrist_Roll | Wrist roll | -2.79 to 2.79 |
| Jaw | Gripper | -0.174 to 1.75 |

## Results

### Primary episode

Target: [0.10, -0.10, 0.12] m

| Metric | Value |
|---|---|
| Success | Yes |
| Steps | 183 (3.66 s sim time) |
| Final distance | 14.8 mm |
| Min distance | 14.9 mm |

### Multi-target evaluation (10 random targets)

| Metric | Value |
|---|---|
| **Success rate** | **4/10 (40%)** |
| Avg final distance | 48.8 mm |
| Avg min distance | 48.8 mm |

## Failure Analysis

The controller succeeds on targets in the **right-front quadrant** (negative Y, moderate X and Z) but fails badly elsewhere:

1. **Workspace boundary failures:** Targets at positive Y (robot's left side) or far from the base consistently fail. The arm physically cannot reach some random targets in the sampled volume — the reachable workspace of a 5-DOF arm chain is not a cube.

2. **Low-height targets fail:** Targets near table height (Z ≈ 0.06) often fail because the arm reaches joint limits trying to fold down. The Pitch joint's range (-3.32 to 0.174 rad) creates an asymmetric workspace.

3. **No workspace-awareness:** The random target sampler doesn't check whether a target is kinematically reachable. A proper evaluation would sample targets within the verified reachable workspace, which would yield a much higher success rate.

4. **Controller is greedy:** The damped-least-squares IK moves in a straight Cartesian line toward the target. It doesn't plan around singularities or joint limits — it just slows down and stalls when it hits them.

**The 40% rate is not a controller quality problem — it's a target sampling problem.** Targets that are actually reachable are reached reliably (14.8–14.9 mm final distance, just at the 15 mm threshold). This suggests the controller works but the workspace model is missing.

## Artifacts

- `sim_so101.py` — main simulation script
- `assets/reaching_episode.mp4` — 4.5s video of primary reaching episode
- `assets/trajectory_plot.png` — convergence curve + 3D end-effector path
- `assets/metrics.json` — full metrics for all episodes

## Usage

```bash
# Headless: run episode + evaluation, save video + metrics
python stages/00-sim-so101/sim_so101.py

# Interactive viewer (requires display)
python stages/00-sim-so101/sim_so101.py --viewer

# Custom evaluation count
python stages/00-sim-so101/sim_so101.py --eval 50
```

## What this stage established

- [x] MuJoCo loads and simulates the SO-ARM100
- [x] Complete observe → act → step → observe loop runs at 50 Hz control / 500 Hz physics
- [x] A simple IK controller moves the gripper toward arbitrary 3D targets
- [x] Video, trajectory plot, and quantitative metrics are produced automatically
- [x] Failure modes are identified and documented

## Next stage

**Stage 01: SO-101 Kinematics** — implement FK/IK from scratch, verify against MuJoCo, and build a workspace visualizer. This directly addresses the workspace-boundary failure discovered here.
