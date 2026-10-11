# 10 - Autoregressive action models

**Track:** Extension; after 06 or after the capstone  
**Prerequisites:** 06

Learn action tokenization and transformer policies without treating a small imitation model as a full RT-1 reproduction.

## Read

- [RT-1: Robotics Transformer for Real-World Control at Scale](../../papers/generative-control/brohan-2022-rt1.pdf) ([source](https://arxiv.org/abs/2212.06817)).

- Read RT-1 for its tokenized robot-policy architecture and training setup. Inspect its decoding design rather than assuming all transformers decode autoregressively.
- RT-1 is a tokenized-action transformer reference; a causal action decoder is a deliberate educational variant.

## Build

1. Use one existing dataset and train an MLP BC baseline.
2. Quantize actions and train a small causal decoder that predicts action tokens conditioned on state or images.
3. Measure action-quantization error and compare with a continuous-output baseline.

## Experiment

Match data and evaluation protocol; report closed-loop performance, token error and inference time. Do not equate low token loss with successful control.

## Publish

A small tokenized-action policy demo comparing causal transformer and BC, with clear differences from RT-1.

## Move on when

You can explain tokenization, decoding order, exposure bias and the tradeoff between action precision and vocabulary size.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
