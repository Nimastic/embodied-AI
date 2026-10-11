# World Action Lab

Building toward a simulated-to-physical Mini World Action Model on SO-101, using artifact-first, just-in-time learning.

## Canonical Roadmap

**[ROADMAP.md](ROADMAP.md)** — the sole source of truth for project sequencing.

## Stages

| Stage | Artifact | Status |
|---|---|---|
| 00 | [Simulated SO-101](stages/00-sim-so101/) | **COMPLETE** |
| 01 | [SO-101 Kinematics](stages/01-so101-kinematics/) | **COMPLETE** |
| 02 | [Imitation Learning](stages/02-imitation-learning/) | **ACTIVE** |
| 03 | [Real SO-101](stages/03-real-so101/) | — |
| 04 | [Generative Policies](stages/04-generative-policies/) | — |
| 05 | [Planning Foundations](stages/05-planning-foundations/) | — |
| 06 | [Action-Conditioned World Model](stages/06-action-conditioned-world-model/) | — |
| 07 | [Imagined Rollouts](stages/07-imagined-rollouts/) | — |
| 08 | [Mini WAM](stages/08-mini-wam/) | — |
| 09 | [World-Model Planning](stages/09-world-model-planning/) | — |
| 10 | [Real-Time WAM](stages/10-real-time-wam/) | — |
| 11 | [SO-101 WAM](stages/11-so101-wam/) | — |

## North-Star Artifact

A World Action Model for SO-101 that:

- conditions on observations and language instructions
- models actions and future world evolution jointly
- uses predicted futures for planning/replanning
- is evaluated first in simulation, then on the physical robot

## Research Niche

**world models + generative robot policies + planning/control + real-time physical execution**

## Parallel Hardware Track

[**Project/arkham-batmobile**](Project/arkham-batmobile/) — a mecanum-drive Batmobile used as a
cheap physical testbed for the Stage 06–10 methods (action-conditioned prediction, imagined
rollouts, MPPI planning, inference latency). Not a roadmap stage; `ROADMAP.md` remains the sole
canonical roadmap and the SO-101 line is unchanged.

## Repository Layout

```text
├── README.md
├── ROADMAP.md              ← single canonical roadmap
├── AGENTS.md               ← agent governance rules
├── stages/                 ← one folder per stage
├── Project/                ← parallel hardware tracks (not roadmap stages)
├── papers/                 ← active readings + backlog
├── notes/just-in-time/     ← concept notes pulled as needed
├── sources/                ← original conversation / provenance
├── resources/              ← codebase entry points
├── templates/              ← paper notes, experiment reports
├── scripts/                ← utilities
└── archive/legacy-roadmaps/ ← previous roadmaps (historical only)
```

## Operating Rule

**70% build** · **20% just-in-time learning** · **10% papers**

> No new learning branch unless you can name the current experiment it unlocks.
