# 09 - Choose one research question

**Track:** Core capstone  
**Prerequisites:** 08

Turn a repeatable failure or tradeoff into a narrow hypothesis.

## Read

- [Real-Time Execution of Action Chunking Flow Policies](../../papers/realtime/black-2025-real-time-chunking.pdf) ([source](https://arxiv.org/abs/2506.07339)).

- Revisit the relevant papers and search new literature for the specific proposed change before claiming novelty.

## Build

1. Choose one question: Does adapting execution horizon to measured delay improve performance over a fixed horizon?
2. Write the proposed mechanism, strongest simple baseline, expected failure mode, compute cap and stopping rule before running a sweep.
3. Change one component, run multiple seeds, inspect failures and reproduce the most important plot from clean outputs.

## Experiment

Report the baseline, ablation and uncertainty. Reserve held-out task conditions; show negative results and the range where the idea fails.

## Publish

A small research artifact: code, benchmark configuration, measured results, short technical report and demonstration video.

## Move on when

Your conclusion follows from the evidence; novelty and generalization claims are scoped to the literature checked and conditions tested.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
