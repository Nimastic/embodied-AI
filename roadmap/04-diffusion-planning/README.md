# 04 - Diffusion trajectory planning

**Track:** Core  
**Prerequisites:** 03

Build a small Diffuser-inspired planner and compare it against your classical controllers.

## Read

- [Planning with Diffusion for Flexible Behavior Synthesis](../../papers/generative-control/janner-2022-diffuser.pdf) ([source](https://arxiv.org/abs/2205.09991)).

- Read the trajectory representation, denoising objective, conditioning and guided sampling in Diffuser.
- Keep trajectory generation distinct from predicting only an action chunk in Diffusion Policy.

## Build

1. Generate a fixed set of trajectories from your earlier task; separate train and held-out initial/goal conditions.
2. Train a small diffusion model over trajectories and condition on a goal.
3. Execute generated plans through the same environment; log dynamics inconsistency and constraint violations instead of assuming generated plans are feasible.

## Experiment

Compare MPC, MPPI and the learned planner on identical held-out starts/goals. Report success, cost, planning latency, training data and training compute.

## Publish

MPC to diffusion planning: one task, three planners, with comparison videos and failure cases.

## Move on when

The report calls this a toy implementation inspired by Diffuser, and clearly identifies which original components were omitted.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
