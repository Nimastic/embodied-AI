# 01 - MPC from scratch

**Track:** Core  
**Prerequisites:** 00

Understand receding-horizon optimization by building it on the double integrator before moving to nonlinear dynamics.

## Read

- [Underactuated Robotics](../../papers/foundations/tedrake-underactuated-robotics.pdf) ([source](https://underactuated.csail.mit.edu/)).
- [Constrained Model Predictive Control: Stability and Optimality](https://doi.org/10.1016/S0005-1098(99)00214-9) ([source](https://doi.org/10.1016/S0005-1098(99)00214-9)).

- Tedrake: Trajectory Optimization, direct shooting/transcription and Model-Predictive Control.
- Mayne: problem setup, constraints and the role of terminal ingredients. Treat the stability proofs as a second pass; they should not block your first working controller.

Mayne download status: local PDF unavailable; the link above opens its DOI. Begin with Tedrake's saved notes and return to Mayne when an accessible copy is available.

## Build

1. Form a finite-horizon quadratic objective with state dynamics and bounded acceleration. Use an optimization solver while writing the model and replanning loop yourself.
2. Execute one optimized action, observe the new state, shift the plan and solve again.
3. Display the predicted trajectory alongside the executed one. Add a fixed-time disturbance and a solver-failure fallback.

## Experiment

Compare open-loop and MPC on identical starts and disturbances. Sweep horizon; record cost, constraint violations and median/p95 solve time.

## Publish

An MPC animation or interactive visualizer plus a concise explanation of horizon, constraints and replanning.

## Move on when

A saved configuration reproduces a rollout; constraints and solver status are logged, and the disturbance comparison is documented.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
