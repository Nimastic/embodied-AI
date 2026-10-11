> **ARCHIVED:** Historical reference only. This document is not an active roadmap. `ROADMAP.md` is the sole source of truth for project sequencing and priorities.

# Embodied AI: from control foundations to a research artifact

Start with [00 - Math and control essentials](roadmap/00-math-and-control-basics/README.md), then follow the numbered core stages. The destination is a reproducible study of **generative robot policies, planning/control and real-time inference**.

This roadmap combines Harold's foundations-first advice with Yuejiang's learn-the-stack, reproduce, then investigate advice. The folder was organized from the [supplied conversation](sources/README.md); the project specifications are a proposed curriculum, not quotes from either professor.

## Follow this order

MPC -> MPPI -> control as inference -> diffusion planning -> imitation baselines -> Diffusion Policy -> ManiSkill -> latency study -> your research question.

| Stage | What to work on | Track |
| --- | --- | --- |
| 00 | [Math and control essentials](roadmap/00-math-and-control-basics/README.md) | Core; start here |
| 01 | [MPC from scratch](roadmap/01-mpc-from-scratch/README.md) | Core |
| 02 | [MPPI from scratch](roadmap/02-mppi-from-scratch/README.md) | Core |
| 03 | [Control as probabilistic inference](roadmap/03-control-as-inference/README.md) | Core |
| 04 | [Diffusion trajectory planning](roadmap/04-diffusion-planning/README.md) | Core |
| 05 | [Robot-learning stack and imitation baselines](roadmap/05-robomimic-baselines/README.md) | Core |
| 06 | [Diffusion Policy reproduction](roadmap/06-diffusion-policy-reproduction/README.md) | Core |
| 07 | [ManiSkill simulation and manipulation](roadmap/07-maniskill-robot-learning/README.md) | Core |
| 08 | [Inference latency and real-time execution](roadmap/08-realtime-diffusion-policy/README.md) | Core research direction |
| 09 | [Choose one research question](roadmap/09-own-research-question/README.md) | Core capstone |
| 10 | [Autoregressive action models](roadmap/10-autoregressive-action-models/README.md) | Extension; after 06 or after the capstone |
| 11 | [World models and world action models](roadmap/11-world-models-and-wams/README.md) | Extension; after 06, preferably after 09 |

The conversation suggested more than one reading order. This version starts with tangible controllers, moves robomimic before the first serious reproduction, and keeps autoregressive policies and world models as explicit extensions. You can start simulator familiarization early; a full ManiSkill transfer is not required to begin the Push-T latency experiment.

## Start today

1. Read the stage 00 brief and Tedrake's introductory dynamics material.
2. Write the state, action, time step and goal cost for a 2D double integrator in stage 00 `notes.md`.
3. Implement one rollout in its `src/` folder and save a trajectory plot in `assets/`.
4. Advance to MPC when the stage's completion condition is met. Fill math gaps as they appear.

## Download status

Downloaded and verified **12 papers plus Tedrake's course notes** (13 PDFs, about 104 MB). One of the 13 named papers, Mayne et al. (2000), remains available as a source link because the located downloads failed. See the [library status](papers/README.md#one-outstanding-download).

## How to use this repository

- [Paper library](papers/README.md): all named readings, local PDFs, source links and a download manifest.
- [Progress tracker](PROGRESS.md): check off reading, implementation, experiment and shareable output separately.
- [Official codebases](resources/codebases.md): entry points for the reproduction stages.
- [Templates](templates/README.md): paper notes, experiment reports and publication checklist.
- `roadmap/`: one folder per stage, with its own instructions, notes, results table, source and assets slots.
- `scripts/download_papers.py`: fetch missing PDFs and verify saved copies.

Every completed stage should produce **implementation + technical notes + measured comparison + figure or video**. Use the same small control task across the early stages and carry the same trained policy into the latency study. This keeps the work cumulative and the comparisons interpretable.

No experiments are marked complete, and no results are invented. Plan by completion conditions rather than a fixed calendar. The first few stages can be small CPU experiments; training/reproduction stages need a deliberate GPU, runtime and storage budget. Check each upstream project's current requirements before installing. Keep separate environments for upstream stacks and record their versions.

PDFs are available locally but ignored by Git; the manifest and downloader make the library recoverable without automatically redistributing third-party papers. The original personal conversation is also local-only. This folder structure is ready for a public learning repo, but nothing has been committed, pushed or published.
