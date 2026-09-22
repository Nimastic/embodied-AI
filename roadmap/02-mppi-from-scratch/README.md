# 02 - MPPI from scratch

**Track:** Core  
**Prerequisites:** 01

Replace numerical trajectory optimization with sampled action sequences while retaining the same closed-loop task.

## Read

- [Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving](../../papers/foundations/williams-2017-information-theoretic-mpc.pdf) ([source](https://arxiv.org/abs/1707.02342)).
- [Path Integrals and Symmetry Breaking for Optimal Control Theory](../../papers/foundations/kappen-2005-path-integrals.pdf) ([source](https://arxiv.org/abs/physics/0505066)).

- Williams: derivation, sampling distribution, importance weights and control update. Match the cost and correction terms to the formulation you implement.
- Kappen is optional theory after a working demo: study the assumptions that permit the path-integral construction.

## Build

1. Sample perturbed control sequences, simulate them, compute stable normalized weights, and update the nominal sequence.
2. Apply the first action and shift the remaining sequence. Handle bounds and log the random seed.
3. Start on the shared double integrator, then optionally demonstrate pendulum swing-up.

## Experiment

Sweep sample count, horizon and temperature. Compare MPC and MPPI on the same task, cost and starts; report runtime as well as performance.

## Publish

MPPI from scratch: disturbance-recovery GIF, cost-versus-compute plots and an explanation of weighted samples.

## Move on when

The update is numerically stable, deterministic under a fixed seed, and its differences from the paper are listed.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
