# Bill of Materials, Budget, and Compute Decision

You own none of this, so this document decides rather than presents options. Prices are
approximate USD at time of writing and drift — verify at purchase. (If you are sourcing in
Singapore, Cytron, SGBotic and Sim Lim Tower are usually cheaper than Pi/Adafruit shipping, and
AliExpress is cheapest for motors, wheels and drivers if you can wait three weeks.)

---

## The headline numbers

| Phase | What it gets you | Cost |
|---|---|---|
| **Phase 0** | Milestones M0–M4 in simulation. Start this week. | **$0** |
| **Phase 1** | The learning robot. Everything needed for M0–M5 on real hardware. | **~$450** |
| **Phase 2** | Foam Arkham body, lights, sound. Finishes M6 and the demo video. | **~$70** |
| **Phase 3** | Optional: onboard AI accelerator, better IMU, second camera. | **$70–250** |
| **Phase 4** | Optional: 3D printer for the screen-accurate v2 shell. | **$200–350** |

**Spend to a complete, written-up project: about $520.** That sits inside the $650–700 your
original PRD estimated, and it buys a robot that learns rather than a robot that is fast.

Do not buy anything until Phase 0 runs. If the simulated loop bores you, you have learned
something valuable for free.

---

## The compute decision

**Verdict: Raspberry Pi 5 (8 GB) onboard + a Raspberry Pi Pico as a real-time motor controller +
your laptop or Colab as the trainer.**

### Why not a Jetson Orin Nano

It is the obvious "edge AI" answer and it is the wrong first answer, for three reasons.

1. **Your bottleneck is data, not inference.** M0 through M3 are dominated by collecting clean
   synchronized data and training on a GPU. Neither happens on the robot. A Jetson accelerates the
   one part that is not currently slow.
2. **JetPack is a time tax.** Pinned CUDA versions, specific PyTorch wheels, L4T base images,
   kernel modules for cameras. Expect to lose a week to the toolchain, and a week here is a week
   not spent on the actual research question.
3. **$250 versus $80.** The $170 difference is the entire Phase 2 body plus a spare battery.

Buy a Jetson later, if M5 shows that onboard inference is the real constraint. That is a decision
made with data instead of vibes, and nothing in Phase 1 is wasted when you do.

### Why not ESP32-only with everything offboard

$10 and fast to start, but every control decision then crosses Wi-Fi, so you can never run the
no-network condition in M5 — you lose the comparison that makes the latency study interesting.

### Why the Pico matters

The Pi 5 has no hardware quadrature decoder. Decoding four encoders in Python under a
non-realtime Linux scheduler, while also capturing 30 Hz video, is exactly how you end up with the
timestamp jitter that M0 gates on. A $5 Pico running a tight 100 Hz loop — quadrature decode plus
per-wheel velocity PID, talking to the Pi over USB serial at 20 Hz — removes that whole class of
bug. This is the standard architecture for a reason: Linux for perception, microcontroller for
real time.

### Prerequisite to check before ordering

Training needs a GPU somewhere. A laptop with a CUDA GPU, or free Colab, or a few dollars of
cloud. The models here are small — a 64×64 latent dynamics model is minutes-to-hours on a T4, not
days — but CPU-only training will make M3 miserable. Confirm you have one of those three.

---

## Phase 1: the learning robot (~$450)

### Drive

| Item | Qty | ~USD | Notes |
|---|---:|---:|---|
| Mecanum wheel set, 60–80 mm, with couplings | 1 set of 4 | 60 | Buy wheels and couplings together; mismatched hub bores are a classic waste of a week. |
| Geared DC motor **with encoder**, 12 V (JGA25-371 class) | 4 | 56 | **The encoder is non-negotiable.** No encoders means no velocity ground truth means no world model. |
| Motor mounting brackets | 4 | 10 | Often bundled with the motors; check before ordering separately. |
| Dual H-bridge driver (TB6612FNG class) | 2 | 10 | One board drives two motors. Avoid L298N: large voltage drop, runs hot, wastes battery. |
| Chassis plate, 3–4 mm acrylic or aluminium | 1 | 15 | Drill a grid of M3 holes so sensors can be moved later. Many mecanum kits include a plate. |

**Shortcut worth considering:** a pre-built mecanum chassis kit (Yahboom, Hiwonder, Waveshare and
similar) bundles plate, motors, wheels and driver for roughly $150–200, which is close to buying
separately but removes fit risk. **Verify the motors have encoders** — many kits ship encoder-less
motors, and those kits are useless here.

### Compute and sensing

| Item | Qty | ~USD | Notes |
|---|---:|---:|---|
| Raspberry Pi 5, 8 GB | 1 | 80 | 8 GB not 4 GB; latent models and video buffers eat RAM. |
| Active cooler | 1 | 5 | Mandatory. A throttling Pi silently changes your control rate mid-episode. |
| 27 W USB-C power supply | 1 | 12 | For bench work. |
| microSD, 64 GB, A2 | 1 | 12 | A2 rating matters for write throughput while logging. |
| Pi Camera Module 3 Wide (IMX708, 120°) | 1 | 35 | Better sensor than the IMX219 in the original notes, and the wide FOV keeps the arena in frame. |
| Raspberry Pi Pico (or Pico 2) | 1 | 5 | Real-time motor controller. |
| MPU-6050 IMU breakout | 1 | 5 | Good enough to start. Upgrade to a BNO085 in Phase 3 if drift annoys you. |
| USB webcam, 1080p | 1 | 25 | Mounted overhead for ArUco ground-truth pose. Any cheap one works. |

### Power

| Item | Qty | ~USD | Notes |
|---|---:|---:|---|
| 3S LiPo, 2200–3000 mAh, XT60 | 1 | 25 | Motors only. |
| Balance charger (SkyRC/HOTA/ISDT entry class) | 1 | 35 | One-time cost that serves every future build. Do not use the charger that came free with anything. |
| LiPo safe bag | 1 | 10 | Non-negotiable. |
| USB-C PD power bank, 20 W+, 10000 mAh | 1 | 25 | **Powers the Pi separately from the motors.** Motor current spikes browning out the Pi mid-episode is the single most common beginner failure, and two isolated supplies with a common ground eliminates it outright. Also saves buying a UBEC. |
| XT60 pigtails, inline fuse, rocker kill switch | 1 set | 12 | The kill switch is a safety requirement, not a nicety. |

### Glue

| Item | ~USD | Notes |
|---|---:|---|
| M3 standoffs, screws, nuts assortment | 12 | |
| Dupont jumpers, JST connectors, heatshrink, 22 AWG wire | 15 | |
| Zip ties, double-sided foam tape, cable sleeve | 8 | |
| USB gamepad | 20 | Skip if you own one. Keyboard teleop works fine for mecanum: WASD plus QE. |

**Phase 1 subtotal: ~$450** (~$430 without the gamepad).

### Things in the original PRD not to buy

| Item | Why not |
|---|---|
| 4× brushless motors + 4× 30 A ESC (~$200) | Brushless gives speed and no position feedback. You want slow, repeatable, encoder-instrumented motion. This is the single biggest saving and the single biggest technical improvement. |
| 6–8 channel 2.4 GHz RC system (~$100) | A gamepad over USB into the Pi logs cleanly and timestamps in the same clock as everything else. An RC receiver gives you PWM you then have to decode and cannot easily log. |
| 4S 5000 mAh LiPo (~$50) | Oversized for 0.5 m/s. 3S 2200 mAh runs for well over an hour of teleop at this speed. |
| 100–120 mm mecanum wheels | Too big for a ~40 cm chassis and for a 2×2 m arena. |
| 5–6 kg of filament (~$100) | You have no printer, and body v1 is foam. |

---

## Phase 2: body, lights, sound (~$70)

| Item | ~USD | Notes |
|---|---:|---|
| Foamboard / 5 mm EVA foam sheets | 15 | Body v1. One weekend, ~200 g, looks genuinely good on video in matte black. |
| Matte black spray paint + filler primer | 25 | |
| WS2812B LED strip, 1 m | 12 | White front, red rear, animations on one GPIO. |
| I²S amp (MAX98357A) + 3 W speaker | 15 | Engine and boost samples. |

Body v1 is foam on purpose. It makes attaching the shell a **one-weekend domain-shift experiment**
(M6) instead of a six-week CAD-print-sand-paint blocker. The modular 3D-print plan from your
original PRD is preserved in [`build-notes.md`](build-notes.md) § Body shell for when v2 happens.

---

## Phase 3: optional upgrades, only when measured need appears

| Item | ~USD | Buy when |
|---|---:|---|
| Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS) | 70 | M5 shows network latency is the binding constraint and you want onboard inference without changing platform. |
| Jetson Orin Nano Super 8 GB dev kit | 250 | You want to run a genuinely large world model onboard, and you have accepted the toolchain cost with open eyes. |
| BNO085 9-axis IMU with onboard fusion | 25 | MPU-6050 heading drift is polluting your pose labels. |
| Second camera (rear or downward) | 35 | Multi-view prediction becomes interesting after M3 works with one. |
| Spare 3S LiPo | 25 | Data collection sessions start ending because of charging, not fatigue. |

## Phase 4: optional 3D printer ($200–350)

Only after M4 passes. If the printed v2 shell matters to you, an entry FDM printer is cheaper than
paying a print service for a ~40 cm multi-part body, and useful forever afterwards. A 180 mm bed
turns a 40 cm shell into a frustrating number of modules; a ~256 mm bed is the comfortable floor.
Buying a printer *before* M4 is how this becomes a model-making project with a learning folder
attached.

---

## Buy order

1. **Nothing.** Run Phase 0 in simulation first. Confirm you have a GPU for training.
2. Drive train and chassis — longest shipping, most fit risk. Get it in hand early.
3. Pi 5, cooler, SD, camera, Pico, IMU.
4. Power and the kill switch. Do not run motors before the kill switch exists.
5. Overhead webcam, glue, gamepad.
6. Phase 2 only after M4 passes.
