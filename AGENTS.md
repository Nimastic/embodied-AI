# Agent Governance Rules

## Roadmap Authority

`ROADMAP.md` is the sole authoritative roadmap.

Archived roadmaps in `archive/legacy-roadmaps/` are historical material only.

Never use an archived roadmap to decide:

- what to work on next
- what to study next
- prerequisites
- stage ordering
- project priorities

## Build-First Rule

Default effort allocation:

- **70%** implementation / experiments
- **20%** just-in-time learning
- **10%** paper reading / exploration

Do not recommend broad prerequisite study before implementation.

When a knowledge gap occurs:

1. identify the exact blocker
2. learn the minimum concept required
3. immediately apply it
4. return to the artifact

## No Curriculum Expansion

Do NOT create a new roadmap because another useful topic is discovered.

New topics go into:

- `papers/backlog/`
- `notes/just-in-time/`

unless they are required by the current stage.

Do not add stages without explicit user approval.

## Current Specialization

Optimize decisions for:

**World Action Models / world models for robotics**

Particularly the intersection of:

- manipulation
- video/world models
- generative action policies
- planning
- MPC / MPPI / planning as inference
- action-conditioned prediction
- closed-loop execution
- inference latency
- sim-to-real
- SO-101

Do not redirect the project toward generic:

- SLAM
- autonomous navigation
- quadruped locomotion
- humanoid locomotion
- LiDAR
- IMU curricula
- ROS mastery
- generic reinforcement-learning exercises
- broad computer-vision coursework

unless a current experiment explicitly requires them.

## Stage-Completion Discipline

Each stage must produce something observable. At minimum:

- working code
- README
- reproducible instructions
- experiment/result
- quantitative metric where applicable
- visual artifact (plot, GIF, or video)
- short failure analysis

A stage is not complete merely because papers or lectures were consumed.

## Current Active Stage

Stage 02: Imitation Learning (The Robot-Learning Data Loop)

The immediate objective is: build the data pipeline (teleoperation / scripted expert demonstrations → dataset → policy training → evaluation in simulation). Collect demonstrations for reach/pick-and-place, train Behavior Cloning and Action Chunking (ACT) baselines, and measure success rates on held-out scenes.

Do not start Stage 03 until Stage 02's completion criteria are satisfied.
