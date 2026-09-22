# 08 - Inference latency and real-time execution

**Track:** Core research direction  
**Prerequisites:** 06-07

Study how delayed inference changes closed-loop performance using the policy and task you already understand.

## Read

- [Real-Time Execution of Action Chunking Flow Policies](../../papers/realtime/black-2025-real-time-chunking.pdf) ([source](https://arxiv.org/abs/2506.07339)).
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](../../papers/generative-control/chi-2023-diffusion-policy.pdf) ([source](https://arxiv.org/abs/2303.04137v4)).

- Read RTC: asynchronous chunk generation, committed actions and overlap/inpainting.
- A basic background worker is an asynchronous baseline; call it an RTC reproduction only if the paper-specific algorithm is implemented and checked.

## Build

1. Create a timestamped observation/inference/action queue. Let environment time advance while inference is pending.
2. Inject 0, 20, 50, 100 and 200 ms of delay; map these explicitly to the task control period. Separate synthetic delay from measured inference time.
3. Compare synchronous execution, naive queued chunks and an asynchronous overlap strategy. Fix checkpoint, observations and evaluation starts.

## Experiment

Measure task performance, action discontinuity, observation age, missed deadlines and p50/p95 latency. Sweep delay and execution horizon with paired evaluation seeds; preserve a zero-delay reference.

## Publish

What happens to diffusion robot policies under inference latency? A benchmark, plots, video and reproducible report.

## Move on when

Your timing model is explicit, stale observations are modeled, and any benefit survives matched task conditions and repeated runs.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
