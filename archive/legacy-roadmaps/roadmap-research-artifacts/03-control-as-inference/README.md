# 03 - Control as probabilistic inference

**Track:** Core  
**Prerequisites:** 00-02

Connect the controllers you built to probabilistic objectives without assuming every stochastic control problem is ordinary Bayesian inference.

## Read

- [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](../../papers/foundations/levine-2018-control-as-inference.pdf) ([source](https://arxiv.org/abs/1805.00909)).
- [Robot Trajectory Optimization using Approximate Inference](../../papers/foundations/toussaint-2009-trajectory-optimization.pdf) ([source](https://icml.cc/2009/papers/271.pdf)).
- [Path Integrals and Symmetry Breaking for Optimal Control Theory](../../papers/foundations/kappen-2005-path-integrals.pdf) ([source](https://arxiv.org/abs/physics/0505066)).

- Levine: optimality variables, entropy regularization, soft Bellman recursion, and the distinction between deterministic exact inference and stochastic variational formulations.
- Toussaint: factorization of a trajectory problem and approximate inference; sketch where dynamics and costs enter.
- Use Kappen for a deeper bridge to stochastic control if the MPPI derivation still feels opaque.

## Build

1. Build a small gridworld with a fixed transition model. Implement ordinary and entropy-regularized value iteration.
2. Visualize soft action distributions and sample rollouts for multiple temperatures.
3. Derive the deterministic finite-horizon soft backup in your own notation; explain the chosen reward-to-optimality likelihood scaling.

## Experiment

Compare reward, policy entropy and goal-reaching rate. Check the small-temperature policy against ordinary value iteration, allowing ties.

## Publish

Control as inference from scratch: derivation, heatmaps and trajectory animations.

## Move on when

You can explain what is inferred, where the dynamics are fixed, and how temperature changes behavior.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
