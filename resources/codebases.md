# Official project entry points

Use these when reaching the corresponding stage. This setup downloads papers only; it does not install simulator stacks, clone all repositories, or download datasets/checkpoints. Follow current upstream requirements and pin the commit you actually use.

| Resource | Official entry point | Use here |
| --- | --- | --- |
| Underactuated Robotics | [Course notes](https://underactuated.csail.mit.edu/) | Dynamics, LQR, trajectory optimization and MPC |
| Diffuser | [Project](https://diffusion-planning.github.io/) | Trajectory diffusion reference and linked code |
| Diffusion Policy | [Project, code, data and notebooks](https://diffusion-policy.cs.columbia.edu/) | State-based Push-T first; use RSS 2023 configuration when comparing to the saved paper |
| robomimic | [Project](https://arise-initiative.github.io/robomimic-web/) | Demonstration datasets, BC and recurrent BC |
| ManiSkill | [Documentation](https://maniskill.readthedocs.io/en/latest/) | Installation, demos, controller/task definitions and baselines |
| RT-1 | [Project](https://robotics-transformer1.github.io/) | Tokenized-action robot transformer reference |
| RTC | [Author research page](https://www.pi.website/research/real_time_chunking) | Real-time action-chunking explanation and resources |
| DreamerV3 | [Paper](https://arxiv.org/abs/2301.04104) | Follow the authors' code link; pin version before running |
| DreamZero | [Project](https://dreamzero0.github.io/) | Later WAM reading and linked implementation resources |

For each reproduction, record: upstream URL and commit, dependency lock/environment export, dataset version and split, training configuration, checkpoint-selection rule and exact evaluation command. Begin with a tiny smoke run before committing to training.
