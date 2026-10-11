# 06 - Diffusion Policy reproduction

**Track:** Core  
**Prerequisites:** 05

Make this your first substantial reproduction: state-based Push-T first, then vision or one manipulation task.

## Read

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](../../papers/generative-control/chi-2023-diffusion-policy.pdf) ([source](https://arxiv.org/abs/2303.04137v4)).

- Read the RSS 2023 version of Diffusion Policy saved in the library.
- Study observation, prediction and execution horizons; conditioning; normalization; and the exact benchmark metric.

## Build

1. Use the official state-based notebook for a smoke run, then record a local or GPU-host environment and a pinned source commit.
2. Train on the prescribed demonstrations and run the matching held-out evaluation. Save predictions, actions and videos.
3. Compare with a simple BC baseline, then vary execution horizon or denoising steps one at a time.

## Experiment

Report the benchmark metric by its actual definition (for example Push-T overlap score if used), not an invented success rate. Record seeds, episode count, uncertainty and deviation from the matching published setup.

## Publish

A reproduction report with configuration, curves, videos, failed attempts and an honest paper-versus-own-results table.

## Move on when

Another person can rerun your evaluation; claims are limited to the observation mode, task and budget actually reproduced.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
