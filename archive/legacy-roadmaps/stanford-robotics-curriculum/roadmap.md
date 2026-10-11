> **ARCHIVED:** Historical reference only. This document is not an active roadmap. `ROADMAP.md` is the sole source of truth for project sequencing and priorities. The link tables below remain useful as a reference library for just-in-time learning.

# Stanford's Robotics Degree, for Free
## The 6-Month Path to Robot Learning Engineer

> **Source:** [@0x_meden on X](https://x.com/0x_meden/status/2103423701349781606) — Sep 25, 2026
> **Author:** Martin ([@0x_meden](https://x.com/0x_meden))

---

## Overview

| Metric | Value |
|---|---|
| Total Lectures | 119 across 7 Stanford courses & seminar series |
| Total Video | ~95 hours (~4 hours/week) |
| Free Textbooks | 2 (written by Stanford faculty) |
| Practice Projects | 6 (run on a normal laptop, no robot, no GPU) |
| Optional Hardware | 1 × $122 robot arm (final month) |
| Stanford Tuition (for comparison) | ~$70,000/year |

---

## Why Robot Learning & Why Stanford

Every company in the humanoid race is hiring for the same thing: people who can make robots learn from data instead of being programmed line by line. Tesla Optimus, Figure, 1X, Physical Intelligence, Skild — different robots, same job description.

### Key Instructors

| Name | Role | Course |
|---|---|---|
| **Chelsea Finn** | Stanford Professor, Co-founder of Physical Intelligence | CS224R Deep Reinforcement Learning |
| **Marco Pavone** | Stanford Professor, Former NASA JPL, Senior Director of AV Research at NVIDIA | AA203 Optimal and Learning-Based Control |
| **Oussama Khatib** | Stanford Professor, trained generations of robotics engineers | CS223A Introduction to Robotics |
| **Ashish Kumar** | AI Lead, Tesla Optimus | Guest lecture in CS224R |
| **Daniele Gammelli** | Stanford / NVIDIA | AA203 co-instructor |

### What a Robot Learning Engineer Does

A robot learning engineer operates at the intersection of two domains:

1. **Classical Robotics** — kinematics, control, planning (the math that has enabled robot arms in factories for decades)
2. **Machine Learning** — neural networks, imitation learning, reinforcement learning (the part that empowers robots to fold laundry and manipulate unfamiliar objects)

> Learning will be applied at the top of the stack, while control theory will be utilized at the base.

---

## Month 1: The Math That Everything Else Stands On

**Goal:** Become comfortable with vectors, matrices and least squares, because every later course assumes it.

### Course

| # | Course | Link | Details |
|---|---|---|---|
| 1 | **ENGR108: Introduction to Applied Linear Algebra** — Stephen Boyd | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMz-WbFQtNUsUElIh2cPmN9) | 54 short lectures (10–30 min each), ~18.5 hours total |
| 2 | **Vectors, Matrices, and Least Squares** — Boyd & Vandenberghe (textbook) | [Free PDF](https://web.stanford.edu/~boyd/vmls/) | Companion textbook with exercises and code companions |

### What to Focus On

- Vectors, norms, distances and angles (lectures 2–12)
- Matrix-vector multiplication as "a function that moves things" (lectures 17–23)
- Least squares, the single most used idea in robotics (lectures 33–36)
- Linear dynamical systems — first look at how a robot's state changes over time (lecture 26)
- Linear quadratic control and state estimation (lectures 48–49) — preview of Month 4

### What to Skip

- Lecture 47 on portfolio optimization (unless curious)
- k-means lectures (13 and 14) can wait

### Practice Task

Install Python with NumPy and Jupyter, then:

1. **Least squares:** Fit a line through 50 noisy points using least squares — first by hand with the formula, then with `np.linalg.lstsq` — check answers match
2. **Rotation matrix:** Write a function that rotates a 2D point by any angle using a rotation matrix, plot a square rotating in 10° steps
3. **Linear dynamical system:** Simulate `x(t+1) = A x(t)` and plot what happens when you change A

> All three come back later: least squares in sensor calibration, rotations in kinematics, and linear systems in control.

### Month 1 Milestone

- [ ] Multiply a matrix by a vector in your head for 2×2 cases and explain what it did geometrically
- [ ] Solve a least squares problem and explain why it's "the best fit"
- [ ] Rotate points with a rotation matrix in code
- [ ] Simulate a linear system in a loop and plot it

---

## Month 2: The Robot's Body (from Oussama Khatib)

**Goal:** Understand how a robot arm moves, where its hand is, and how much force its motors need.

> This is the part robot learning people skip most often, and it's the part that breaks their policies later.

### Course

| # | Course | Link | Details |
|---|---|---|---|
| 1 | **CS223A: Introduction to Robotics** — Oussama Khatib | [YouTube Playlist](https://www.youtube.com/playlist?list=PL65CC0384A1798ADF) | 16 lectures, ~19 hours total |
| 2 | **CS223A Course Page** (Stanford Engineering Everywhere) | [SEE Page](https://see.stanford.edu/Course/CS223A) | Handouts, problem sets, exams — free download |

### What to Focus On

- **Spatial descriptions:** How to describe where something is and how it's turned
- **Forward kinematics:** Given joint angles → where is the hand?
- **Inverse kinematics:** Given where the hand should be → what are the joint angles?
- **The Jacobian:** How joint speeds turn into hand speeds (shows up in almost every robot controller)
- **Dynamics:** Why motors need different torque depending on arm pose
- **Position and force control** — preview of Month 4

### Practice Task

Install [MuJoCo](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100) (free physics simulator from Google DeepMind). Load the SO-ARM100 model, then:

1. Read joint angles from the simulator, compute gripper position yourself with forward kinematics, check against simulator's answer
2. Write a simple inverse kinematics solver that moves the gripper to a target point
3. Find a pose where the arm loses a degree of freedom (singularity) and explain what happened using the Jacobian

### Month 2 Milestone

- [ ] Describe the position and orientation of any object with a transform
- [ ] Compute forward kinematics for a simple arm by hand and in code
- [ ] Solve inverse kinematics numerically
- [ ] Explain what the Jacobian is to someone who has never heard of it
- [ ] Load a real robot model in MuJoCo and move it with your own code

---

## Month 3: How a Robot Sees and Senses

**Goal:** Understand how cameras, IMUs and lidar turn the physical world into numbers, and how neural networks read those numbers.

### Courses

| # | Course | Link | Details |
|---|---|---|---|
| 1 | **CS231N: Deep Learning for Computer Vision** (Spring 2025) | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) | 18 lectures total; need 10 this month (~11.5 h) |
| 2 | **EE259: Principles of Sensing for Autonomy** | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOhE007XQu707Dy52qXiZGV) | 21 lectures total; need 5 this month (~5.5 h) |

### CS231N — Watch These

- Lectures 1–6: Classification, neural networks, backpropagation, CNNs
- Lecture 8: Attention and transformers
- Lecture 9: Detection and segmentation
- Lecture 15: 3D vision
- Lecture 17: Robot Learning (Yunzhu Li, Columbia) — bridge to everything that follows

### EE259 — Watch These

- Lectures 5–6: How accelerometers and gyroscopes actually work
- Lecture 7: Inertial navigation and dead reckoning (why position estimates drift)
- Lecture 15: How lidar works
- Lecture 19: How a camera forms an image

### What to Focus On

- How a CNN turns pixels into a prediction, and what backpropagation actually does
- Why transformers took over vision
- Detection and segmentation — a robot needs to know *where* things are, not only *what*
- Sensor noise and drift — why a robot's estimate of where it is gets worse over time
- The difference between what a camera sees and what a lidar sees

### Practice Task

Add a camera to your MuJoCo scene from Month 2, place three colored objects on the table, then:

1. Render images from the simulated camera and train a small CNN in PyTorch to find the position of one object
2. Use Month 2 inverse kinematics to move the gripper above the predicted position
3. Add random noise and lighting changes to watch accuracy drop

> That last step is your first taste of the **sim-to-real gap**.

### Month 3 Milestone

- [ ] Train and evaluate a CNN in PyTorch
- [ ] Explain attention in one paragraph without equations
- [ ] Explain why an IMU drifts and why lidar and cameras fail in different ways
- [ ] Connect a vision model to a robot arm in simulation (sees object → moves toward it)

---

## Month 4: Planning and Control (from the NVIDIA Autonomy Lab)

**Goal:** Learn how a robot decides what to do next, mathematically, when it knows how the world works.

### Course

| # | Course | Link | Details |
|---|---|---|---|
| 1 | **AA203: Optimal and Learning-Based Control** (Spring 2026) — Pavone & Gammelli | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNsMOMF1LG3ohsTqZVJVeVp) | 19 lectures total; watch 1–12 this month (~15.5 h) |
| 2 | **Principles of Robot Autonomy** — Gammelli et al. (textbook) | [Free Online](https://porabook.com/) | Companion textbook with Python/Jupyter exercises |
| 3 | **AA203 Course Page** | [Stanford ASL](https://stanfordasl.github.io/aa203/sp2526/) | Slides for every lecture and full schedule |

### Lecture-by-Lecture Guide

| Lecture | Topic |
|---|---|
| 1 | Course overview |
| 2–6 | Optimization — two families of methods for computing the best trajectory |
| 7 | Dynamic programming — the idea underneath all of reinforcement learning |
| 8 | LQR — the most famous controller in robotics (previewed in ENGR108 L48) |
| 9 | What happens when the world is random |
| 10 | Reachability analysis — proving a robot will never enter a dangerous state |
| 11–12 | Model Predictive Control (MPC) — drives most real autonomous systems today |

### What to Focus On

- **Cost function:** Every behavior you want becomes a number to minimize
- **Dynamic programming & Bellman equation** — Month 5 is built on it
- **LQR** — until you can implement it from memory
- **MPC:** Plan ahead, take one step, re-plan. Robots and self-driving cars do this dozens of times per second

### Practice Task

Use the [MuJoCo cart-pole model](https://gymnasium.farama.org/environments/mujoco/) or write a simple cart-pole simulation in Python, then:

1. Linearize the system around the upright position and balance the pole with LQR
2. Replace LQR with a simple MPC controller and add a limit on how hard the cart can push
3. Push the pole with a random force and compare how both controllers recover

### Month 4 Milestone

- [ ] Write a robot's goal as a cost function
- [ ] Explain dynamic programming and the Bellman equation
- [ ] Implement LQR and MPC from scratch
- [ ] Explain why MPC re-plans at every step
- [ ] Explain what reachability analysis guarantees and why safety teams care

---

## Month 5: Teaching Robots Instead of Programming Them

**Goal:** Learn imitation learning and reinforcement learning — the two ways robots learn skills from data.

> This is the month everything connects. Reinforcement learning is dynamic programming from Month 4, done without knowing the rules of the world in advance.

### Courses

| # | Course | Link | Details |
|---|---|---|---|
| 1 | **AA203, Lectures 13–19** | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNsMOMF1LG3ohsTqZVJVeVp) | ~9.5 hours |
| 2 | **CS224R: Deep Reinforcement Learning** (Spring 2025) — Chelsea Finn | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL) | 19 videos total; watch 7 this month (~7.5 h) |
| 3 | **CS224R Course Page** | [Stanford](https://cs224r.stanford.edu/) | |
| — | *Optional:* **CS234 Reinforcement Learning** — Emma Brunskill | [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX) | Slower, more mathematical — if AA203/CS224R move too fast |

### AA203 Lectures 13–19

| Lecture | Topic |
|---|---|
| 13 | Intro to learning |
| 14 | Intro to imitation learning and RL |
| 15 | Imitation learning |
| 16 | Fundamentals of RL |
| 17 | Value-based methods (Q-learning) |
| 18 | Policy optimization |
| 19 | Model-based RL + summary of the whole stack |

### CS224R — Watch These

| Lecture | Topic |
|---|---|
| 2 | Imitation learning |
| 3 | Policy gradients |
| 4 | Actor-critic methods |
| 6 | Q-learning |
| 7 | Offline RL (learning from a fixed dataset) |
| 11 | Model-based RL |
| 16 | RL for Robots (why robots still aren't autonomous) |

### What to Focus On

- **Behavior cloning:** Copying demonstrations, and why it fails when the robot drifts
- **Policy gradients and PPO** — "probably the most popular RL algorithm" (Gammelli)
- **Q-learning** and why it can learn from old data
- **Offline RL** — real robots are expensive; most data is collected once and reused
- **Model-based RL** — why an optimizer will exploit every mistake in a learned world model

### Practice Task

Install [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), then:

1. Train PPO on a MuJoCo environment (e.g., Hopper) and plot the learning curve
2. Train the same environment with SAC and compare steps needed
3. Record 50 episodes of your LQR controller balancing cart-pole, train a small neural network with behavior cloning
4. Push the cloned policy off its usual path — watch it fail, explain why using AA203 L15

### Month 5 Milestone

- [ ] Explain the difference between imitation learning and reinforcement learning
- [ ] Train PPO and SAC and read a learning curve
- [ ] Explain why behavior cloning drifts and how DAgger fixes it
- [ ] Explain model-free vs. model-based RL and when each wins
- [ ] Connect RL back to the dynamic programming from Month 4

---

## Month 6: The Frontier & Your First Real Robot

**Goal:** Learn where robot learning is heading right now, and build one project on a real or simulated arm that proves everything you learned.

### Guest Lectures & Seminars (~8.5 hours total)

| # | Talk | Speaker | Link |
|---|---|---|---|
| 1 | CS231N L17: Robot Learning | Yunzhu Li (Columbia) | [YouTube](https://www.youtube.com/watch?v=XSfmOH_xVSU) |
| 2 | CS224R L17: Advancing Robot Intelligence | Ashish Kumar (Tesla Optimus) | [YouTube](https://www.youtube.com/watch?v=Hp1WBWghrak) |
| 3 | CS224R L18: Frontiers | Chelsea Finn | [YouTube](https://www.youtube.com/watch?v=FacJ_1tTSx4) |
| 4 | Seminar: Gen Control, Action Chunking, Moravec's Paradox | Max Simchowitz (CMU) | [YouTube](https://www.youtube.com/watch?v=D7e7oUbtkFM) |
| 5 | Seminar: Ψ₀ — Open Foundation Model for Humanoid Loco-manipulation | Yue Wang (USC) | [YouTube](https://www.youtube.com/watch?v=popjDuA-whA) |
| 6 | Seminar: Manipulation Data Pyramid | Yang Gao (Tsinghua) | [YouTube](https://www.youtube.com/watch?v=99wIpauDEmY) |
| 7 | Seminar: Embodied Foundation Models | Jonas Frey (Stanford) | [YouTube](https://www.youtube.com/watch?v=5uWtpDON7Vs) |
| 8 | Seminar: Robot Learning from Human Experience | Danfei Xu (Georgia Tech / NVIDIA) | [YouTube](https://www.youtube.com/watch?v=NUtaN10cVDc) |
| 9 | AI and Robotics (conversation) | Chelsea Finn & Andrew Ng | [YouTube](https://www.youtube.com/watch?v=IT734HriiHQ) |

**Full Stanford Robotics Seminar Archive:** [stanfordasl.github.io/robotics_seminar](https://stanfordasl.github.io/robotics_seminar/)

### What to Focus On

- **Action chunking:** Predicting a short sequence of actions at once instead of one at a time
- **Foundation models for robots** — why they need huge and varied datasets
- **The data problem:** Internet video, teleoperation, robot experience — what each is good for
- **Sim-to-real:** Why it works well for walking and much less well for grasping

### Final Project — Three Budget Tiers

| Tier | Cost | Setup |
|---|---|---|
| **Simulation only** | $0 | [LeRobot (Hugging Face)](https://huggingface.co/docs/lerobot/index) — train imitation learning on simulated tasks |
| **One follower arm** | $122 | [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (3D-print parts) |
| **Full leader + follower pair** | $230 | Both arms — teleoperate leader, follower copies, LeRobot records training data |

### Deliverable

A **GitHub repository** with:
- README
- 30-second video
- One honest paragraph about what failed
- Three numbers:
  1. Success rate on the trained task
  2. Success rate when you change one thing in the scene
  3. How many demonstrations or training steps it took

> That second number is the one interviewers care about most.

### Month 6 Milestone

- [ ] Explain what a robot foundation model is and what data it needs
- [ ] Explain action chunking and why it helped imitation learning
- [ ] Train and evaluate an imitation learning policy with LeRobot
- [ ] Present one robot project with numbers, a video, and an honest failure analysis
- [ ] Follow new robot learning research on your own via the Stanford seminar series

---

## Full Curriculum Summary

| Month | Course(s) | Lectures | Hours | Instructor(s) |
|---|---|---|---|---|
| 1 | ENGR108 Applied Linear Algebra | 54 | ~18.5 | Stephen Boyd |
| 2 | CS223A Introduction to Robotics | 16 | ~19 | Oussama Khatib |
| 3 | CS231N Computer Vision + EE259 Sensing | 14 | ~17 | — |
| 4 | AA203 Optimal Control (L1–12) | 12 | ~15.5 | Marco Pavone & Daniele Gammelli |
| 5 | AA203 (L13–19) + CS224R Deep RL | 14 | ~17 | Pavone, Gammelli, Chelsea Finn |
| 6 | Seminars (Tesla, CMU, USC, etc.) | 9 | ~8.5 | Various |
| **Total** | | **119** | **~95** | **2 textbooks, 6 projects, $0–$230 hardware** |

---

## How to Actually Finish This

1. **Watch at 1.25× speed** — saves ~19 hours over the whole path
2. **Never watch more than 2 lectures without writing code** — practice tasks ARE the course
3. **Keep the free textbooks open** next to the video — when a professor goes too fast, the book explains it slower
4. **Put every project on GitHub from day one** — with a README explaining what broke and how you fixed it
5. **Budget 2 hours/day:** ~1 hour lectures, ~1 hour code. Keep your job while you do this

---

## What This Won't Give You

- **A diploma** — if a job requires a degree on paper, this doesn't replace it
- **Feedback** — nobody will grade your code; the numbers in your final project are your grade
- **A network** — you'll have to build that online by sharing what you build

> The people who post their progress publicly while working through material like this get noticed by the same companies that hire from Stanford.

---

## All Links (Quick Reference)

### YouTube Playlists (Courses)

| Course | Playlist URL |
|---|---|
| ENGR108 Linear Algebra | https://www.youtube.com/playlist?list=PLoROMvodv4rMz-WbFQtNUsUElIh2cPmN9 |
| CS223A Intro to Robotics | https://www.youtube.com/playlist?list=PL65CC0384A1798ADF |
| CS231N Computer Vision (2025) | https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16 |
| EE259 Sensing for Autonomy | https://www.youtube.com/playlist?list=PLoROMvodv4rOhE007XQu707Dy52qXiZGV |
| AA203 Optimal Control (2026) | https://www.youtube.com/playlist?list=PLoROMvodv4rNsMOMF1LG3ohsTqZVJVeVp |
| CS224R Deep RL (2025) | https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL |
| CS234 RL (optional) | https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX |

### Individual Talks (Month 6)

| Talk | URL |
|---|---|
| Robot Learning — Yunzhu Li | https://www.youtube.com/watch?v=XSfmOH_xVSU |
| Advancing Robot Intelligence — Ashish Kumar | https://www.youtube.com/watch?v=Hp1WBWghrak |
| Frontiers — Chelsea Finn | https://www.youtube.com/watch?v=FacJ_1tTSx4 |
| Gen Control & Action Chunking — Max Simchowitz | https://www.youtube.com/watch?v=D7e7oUbtkFM |
| Ψ₀ Foundation Model — Yue Wang | https://www.youtube.com/watch?v=popjDuA-whA |
| Manipulation Data Pyramid — Yang Gao | https://www.youtube.com/watch?v=99wIpauDEmY |
| Embodied Foundation Models — Jonas Frey | https://www.youtube.com/watch?v=5uWtpDON7Vs |
| Robot Learning from Human Experience — Danfei Xu | https://www.youtube.com/watch?v=NUtaN10cVDc |
| AI and Robotics — Finn & Ng | https://www.youtube.com/watch?v=IT734HriiHQ |

### Free Textbooks

| Book | URL |
|---|---|
| Vectors, Matrices, and Least Squares (Boyd & Vandenberghe) | https://web.stanford.edu/~boyd/vmls/ |
| Principles of Robot Autonomy (Gammelli et al.) | https://porabook.com/ |

### Course Pages & Resources

| Resource | URL |
|---|---|
| CS223A (Stanford Engineering Everywhere) | https://see.stanford.edu/Course/CS223A |
| AA203 Course Page | https://stanfordasl.github.io/aa203/sp2526/ |
| CS224R Course Page | https://cs224r.stanford.edu/ |
| Stanford Robotics Seminar Archive | https://stanfordasl.github.io/robotics_seminar/ |

### Tools & Libraries

| Tool | URL |
|---|---|
| MuJoCo SO-ARM100 Model | https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100 |
| Gymnasium MuJoCo Envs | https://gymnasium.farama.org/environments/mujoco/ |
| Stable-Baselines3 | https://stable-baselines3.readthedocs.io/ |
| LeRobot (Hugging Face) | https://huggingface.co/docs/lerobot/index |
| SO-ARM100 Build Guide | https://github.com/TheRobotStudio/SO-ARM100 |
