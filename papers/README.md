# Paper library

Named readings from the supplied conversation. Each PDF is stored once; stages link here.

Source URLs, verification dates, byte sizes, page counts and SHA-256 hashes are in [manifest.json](manifest.json).

| ID | Reading | Local copy | Source |
| --- | --- | --- | --- |
| tedrake | Underactuated Robotics (2023 PDF snapshot) | [PDF](foundations/tedrake-underactuated-robotics.pdf) | [Source](https://underactuated.csail.mit.edu/) |
| mayne2000 | Constrained Model Predictive Control: Stability and Optimality (2000) | Unavailable: see manifest | [Source](https://doi.org/10.1016/S0005-1098(99)00214-9) |
| williams2017 | Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving (2017) | [PDF](foundations/williams-2017-information-theoretic-mpc.pdf) | [Source](https://arxiv.org/abs/1707.02342) |
| levine2018 | Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review (2018) | [PDF](foundations/levine-2018-control-as-inference.pdf) | [Source](https://arxiv.org/abs/1805.00909) |
| toussaint2009 | Robot Trajectory Optimization using Approximate Inference (2009) | [PDF](foundations/toussaint-2009-trajectory-optimization.pdf) | [Source](https://icml.cc/2009/papers/271.pdf) |
| kappen2005 | Path Integrals and Symmetry Breaking for Optimal Control Theory (2005) | [PDF](foundations/kappen-2005-path-integrals.pdf) | [Source](https://arxiv.org/abs/physics/0505066) |
| janner2022 | Planning with Diffusion for Flexible Behavior Synthesis (2022) | [PDF](generative-control/janner-2022-diffuser.pdf) | [Source](https://arxiv.org/abs/2205.09991) |
| chi2023 | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (2023) | [PDF](generative-control/chi-2023-diffusion-policy.pdf) | [Source](https://arxiv.org/abs/2303.04137v4) |
| brohan2022 | RT-1: Robotics Transformer for Real-World Control at Scale (2022) | [PDF](generative-control/brohan-2022-rt1.pdf) | [Source](https://arxiv.org/abs/2212.06817) |
| mandlekar2021 | What Matters in Learning from Offline Human Demonstrations for Robot Manipulation (2021) | [PDF](robot-learning/mandlekar-2021-robomimic.pdf) | [Source](https://arxiv.org/abs/2108.03298) |
| tao2024 | ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI (2024) | [PDF](robot-learning/tao-2024-maniskill3.pdf) | [Source](https://arxiv.org/abs/2410.00425) |
| black2025 | Real-Time Execution of Action Chunking Flow Policies (2025) | [PDF](realtime/black-2025-real-time-chunking.pdf) | [Source](https://arxiv.org/abs/2506.07339) |
| hafner2023 | Mastering Diverse Domains through World Models (DreamerV3) (2023) | [PDF](world-models/hafner-2023-dreamerv3.pdf) | [Source](https://arxiv.org/abs/2301.04104v1) |
| dreamzero2026 | World Action Models are Zero-shot Policies (DreamZero) (2026) | [PDF](world-models/dreamzero-2026-world-action-models.pdf) | [Source](https://dreamzero0.github.io/) |

## One outstanding download

Mayne et al. (2000) is catalogued but its PDF could not be retrieved: the indexed CiteSeerX copy returns 404, its older link returns HTML, and the publisher PDF returns 403. Use the source DOI through institutional access or an author-provided copy. Place it at `papers/foundations/mayne-2000-constrained-mpc.pdf` and rerun the downloader to verify it. Tedrake's local notes cover the initial MPC stage in the meantime.

## Version notes

- Diffusion Policy is pinned to arXiv v4, the RSS 2023 version linked by the authors.
- DreamerV3 is pinned to arXiv v1 (2023), matching the original reading recommendation.
- Other unversioned arXiv URLs retrieve the version available on the download date; hashes identify the exact saved bytes.
- Underactuated Robotics is course material, not a paper. Its official PDF release is dated 2023-03-24; prefer the live HTML for updated chapters and interactive examples.
- DreamZero is included because it is explicitly named in the conversation. Generic WAM/VLA references do not identify additional specific papers.

To fetch missing files or reverify saved PDFs: `python -m pip install pypdf`, then `python scripts/download_papers.py` from the repository root.
Existing PDFs are verified and retained. To deliberately update a version, change its URL and move the old file aside first.
