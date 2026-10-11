# 07 - ManiSkill simulation and manipulation

**Track:** Core  
**Prerequisites:** 06

Transfer your understanding to a robotics simulator and learn how controllers and task definitions affect results.

## Read

- [ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI](../../papers/robot-learning/tao-2024-maniskill3.pdf) ([source](https://arxiv.org/abs/2410.00425)).

- Read ManiSkill3 for its simulator and benchmark design. The paper first appeared in 2024 and was published at RSS 2025.
- Consult current official installation and task documentation; choose a supported environment for the available GPU and operating system.

## Build

1. Run three documented small manipulation tasks; inspect observation/action shapes, control modes, reset conditions and rewards.
2. Pick one task and reproduce a provided baseline before adapting your policy workflow.
3. Modify one task property such as initial-object distribution and document the exact difference.

## Experiment

Keep control mode and action rate fixed across policy comparisons. Report nominal and modified-task results separately; include simulation throughput and evaluation seeds.

## Publish

A ManiSkill playground with three videos, task-interface notes and one controlled environment change.

## Move on when

You can diagnose an action/controller mismatch and reproduce the chosen baseline on one task.

- [ ] Read and explain the central idea.
- [ ] Implement the scoped build.
- [ ] Run and record the comparison.
- [ ] Write the result and save a representative video or plot.

Use `src/` for implementation, `notes.md` for derivations, `results.csv` for measurements and `assets/` for selected figures/videos. Keep large raw data and checkpoints outside Git. Start from the [templates](../../templates/README.md). No implementation or experiment results have been created yet.

[Back to roadmap](../../README.md)
