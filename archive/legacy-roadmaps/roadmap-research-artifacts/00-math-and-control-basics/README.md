# 00 - Math and control essentials

**Track:** Core; start here  
**Prerequisites:** None

Learn only the prerequisites you need while building. Real analysis is not a prerequisite for this route.

## Read

- [Underactuated Robotics](../../papers/foundations/tedrake-underactuated-robotics.pdf) ([source](https://underactuated.csail.mit.edu/)).

- Vectors, matrices, quadratic forms, gradients and Jacobians.
- Gaussian noise, expectation, conditional probability, log probabilities, entropy and KL divergence.
- Discrete-time dynamics, state versus observation, action, cost, feedback and finite-horizon trajectories.
- Tedrake: introductory systems and pendulum material, then Dynamic Programming, Linear Quadratic Regulators and the opening Trajectory Optimization sections. Return to Robust and Stochastic Control / Output Feedback for LQG when needed.

## Build

1. Simulate a 2D double integrator with position and velocity as state, acceleration as action, a fixed time step and bounded actions.
2. Write a quadratic goal-reaching cost. Visualize open-loop motion and a simple feedback controller.
3. Keep this environment and its evaluation starts reusable for stages 01, 02 and 04.

## Experiment

Plot position, action and total cost for the same initial conditions; explain each term and distinguish simulated time from wall-clock time.

## Publish

A short illustrated notebook: From dynamics to a feedback loop.

## Move on when

You can compute a rollout, explain a quadratic cost, take its gradient and explain why feedback helps after a disturbance.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
