# 11 - World models and world action models

**Track:** Extension; after 06, preferably after 09  
**Prerequisites:** 06

Connect learned dynamics and imagined trajectories to later embodied-AI work.

## Read

- [Mastering Diverse Domains through World Models (DreamerV3)](../../papers/world-models/hafner-2023-dreamerv3.pdf) ([source](https://arxiv.org/abs/2301.04104v1)).
- [World Action Models are Zero-shot Policies (DreamZero)](../../papers/world-models/dreamzero-2026-world-action-models.pdf) ([source](https://dreamzero0.github.io/)).

- DreamerV3: latent dynamics, imagined rollouts and actor-critic learning. Start with the saved original 2023 version.
- DreamZero: map observations, generated video/action outputs and execution mechanisms. Treat this as advanced reading before deciding on implementation.

## Build

1. Draw separate training and execution diagrams for DreamerV3 and DreamZero. Identify what each model predicts and how actions are chosen.
2. Optionally run a small official DreamerV3 example with a fixed compute cap, or fit a tiny learned dynamics model to your existing control dataset.
3. Measure multi-step prediction error before using a learned dynamics model for planning.

## Experiment

For a toy learned model, report error versus rollout horizon and performance under model mismatch. A toy dynamics model is not a DreamerV3 or DreamZero reproduction.

## Publish

A comparison article and small learned-dynamics demo linking world-model assumptions back to earlier control experiments.

## Move on when

You can distinguish latent-model imagination, diffusion trajectory planning, imitation action generation and joint video/action modeling.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
