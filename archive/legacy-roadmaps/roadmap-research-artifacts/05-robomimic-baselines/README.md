# 05 - Robot-learning stack and imitation baselines

**Track:** Core  
**Prerequisites:** 04; can begin dataset inspection earlier

Learn the dataset and evaluation workflow before attempting a substantial policy reproduction.

## Read

- [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](../../papers/robot-learning/mandlekar-2021-robomimic.pdf) ([source](https://arxiv.org/abs/2108.03298)).

- Read the robomimic paper with attention to dataset quality, observation choices and evaluation.
- Use the official project documentation linked in resources/codebases.md for installation and data acquisition.

## Build

1. Inspect one demonstration: observations, actions, episode boundaries, normalization and train/validation split.
2. Run a small behavior-cloning baseline and a recurrent BC baseline on the same simple task and dataset.
3. Record dependency versions, source commit, configuration, checkpoint rule and evaluation seeds.

## Experiment

Report success counts and episode counts for each baseline. Keep dataset, observation mode, training budget and evaluation protocol matched.

## Publish

Learning robomimic: BC versus BC-RNN, with an annotated data example, table and rollout videos.

## Move on when

You can train and evaluate a baseline from recorded commands and explain how action and observation conventions are defined.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
