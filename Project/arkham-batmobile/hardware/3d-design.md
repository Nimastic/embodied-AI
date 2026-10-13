# 3D Design Spec — Arkham Knight Batmobile shell

Design intent, dimensioned envelope, and build plan for the body. Written before any modelling,
because the shell has three hard constraints from the robot underneath it and getting those wrong
means reprinting everything.

---

## 1. The mode decision: build battle mode

The Arkham Knight Batmobile has two forms, and you sent reference for both.

| | Pursuit mode | Battle mode |
|---|---|---|
| Stance | low, long, narrow, wheels tucked | raised, splayed, wheel pods rotated outward |
| Steering | conventional, front wheels steer | **omnidirectional — strafes sideways while facing forward** |
| Silhouette | streamlined wedge | hunkered, armoured, turret deployed |

**Build battle mode.** Not as a compromise — because of a genuine convergence:

> Battle mode's defining mechanic is **holonomic motion**: the car crabs sideways and diagonally
> while keeping its nose pointed at the target. That is exactly and only what mecanum wheels do.

Our drivetrain is not an approximation of battle mode. It is a functionally faithful
implementation of it. A pursuit-mode replica on mecanum wheels would be the dishonest one — it
would look right and move wrong.

Two further arguments, both practical:

**Battle mode hides the mecanum wheels.** This is the single biggest aesthetic problem with the
build: mecanum wheels are unmistakably not car tyres — 9 angled rollers per wheel, visible from
every angle. Battle mode's armoured wheel pods are large, boxy, and already cover most of the
wheel in the source design. Pursuit mode's tucked-in wheels would leave them exposed.

**Pursuit mode is a different robot.** It needs Ackermann steering: a steering servo, a linkage,
and non-driven front wheels. That is a second drivetrain, not a second body.

### What transformation would actually cost

Worth being concrete, since you asked whether it is too difficult.

**The real transformation** rotates each wheel pod ~90° while the wheel stays driven, and changes
ride height. That means per corner: a rotating pod, a drive path that survives rotation (flexible
coupling or in-pod motor with a slip ring), and a height actuator. Four corners, so ~8 extra
actuators plus the linkages. It is harder than the entire rest of this build combined, and it is a
mechanism project rather than a robotics one.

It would also damage the research. Two drivetrain configurations means two dynamics regimes plus a
transition, so every dataset splits three ways and the world model has to learn a mode-switch it
has almost no data for. M6 already supplies domain shift more cheaply.

**The cosmetic version is very achievable** and worth doing as a stretch goal: servo-deployed
armour panels, a rising turret, extending side skirts, lighting change. The drivetrain never
changes — the wheels are already in battle configuration — so it is pure theatre bolted to a
working robot. Budget 2 servos and a weekend, after M4 passes on hardware.

**Verdict: battle mode geometry, static. Cosmetic deploy animation as a later stretch goal.**

---

## 2. Hard constraints from the robot

These are not style choices. Violating any one means a reprint.

### 2.1 Wheel clearance envelope

From `MecanumGeometry` (`software/kinematics/mecanum.py`) and `mecanum_base.xml`:

| | Value |
|---|---|
| Wheel centres | ±90 mm longitudinal, ±75 mm lateral |
| Wheel radius | 40 mm |
| Wheel width | 24 mm |
| Outer wheel face | ±87 mm lateral → **174 mm track width** |
| Hull | 200 × 130 × 50 mm |

Mecanum rollers protrude slightly beyond the nominal radius as they rotate into contact. **Allow
6 mm radial and 4 mm lateral clearance** around each wheel inside the fender. A fender that fouls
a roller does not merely rub — it changes the effective slip, which silently invalidates every
calibration in M1.

### 2.2 Camera aperture — the one that breaks experiments

The forward camera sits at `(95, 0, 55) mm` in body frame, tilted 15° down, 78° vertical FOV.

- The aperture must clear the **full 78° cone** with margin. Partial occlusion at the frame edge is
  worse than no shell, because it is a silent, asymmetric change to every observation.
- The camera must return to **the same pose** every time the shell comes off and goes back on. Use
  two dowel pins plus the M3 screws, not screws alone.
- The shell must not cast a hard shadow into the lower frame under arena lighting.

M6 compares model performance with and without the shell. That comparison is only meaningful if
the camera geometry is either identical or *deliberately* changed by a measured amount.

### 2.3 Mass and centre of gravity

| | Target |
|---|---|
| Shell mass | ≤ 400 g, ≤ 250 g preferred |
| CoG rise vs bare chassis | ≤ 15 mm |
| Mounting | 4 × M3 standoffs, same positions as foam v1 |

Mass above the wheel axis changes load transfer under acceleration, which changes slip, which is
the thing being modelled. A heavy shell is not just slow — it moves the robot off the dynamics the
model was trained on. Keeping v1 (foam) and v2 (printed) on **identical standoff positions** is
what makes the M6 comparison clean.

### 2.4 Serviceability

- Battery swap without removing the shell, or with one captive fastener.
- Kill switch reachable with the shell on.
- Pi USB and power accessible; vent above the Pi's active cooler.
- Wire channels moulded in, not drilled later.

---

## 3. Dimensioned envelope

Scale is set by the chassis, not the other way round. Proportions come from reference images.

| Dimension | Target | Driven by |
|---|---|---|
| Overall length | 400 mm | hull 200 mm + nose + rear deck |
| Overall width | 230 mm | 174 mm track + fender walls + armour |
| Overall height | 115 mm | wheel top 80 mm + deck + turret base |
| Ground clearance | 12 mm | underbody skirt must not scrape |
| Wheelbase (visual) | 180 mm | fixed by the drivetrain |

Nominally **~1:14** against a ~5.6 m reference vehicle. I do not have authoritative in-game
dimensions, so treat the scale figure as approximate and set proportions from orthographic
reference rather than from a scale factor.

**Proportion note.** Battle mode reads as *wide and hunkered* — width close to 60% of length, with
the wheel pods as the widest point and the canopy set well back. The common failure in fan models
is making it too long and too narrow, which reads as pursuit mode in battle-mode clothing.

---

## 4. Modular breakdown

Sized for a 220 mm bed. On a 256 mm bed, modules 2 and 3 merge.

| # | Module | Pieces | Notes |
|---|---:|---|---|
| 1 | Nose and front splitter | 2 | split on centreline; headlight LED pockets |
| 2 | Left wheel pod + fender | 2 | **drives the whole look**; must clear the roller envelope |
| 3 | Right wheel pod + fender | 2 | mirror of 2 |
| 4 | Cockpit canopy | 1 | removable; clear PETG or cut acrylic glazing |
| 5 | Rear deck and afterburner | 2 | LED pocket for the glow |
| 6 | Underbody skirt | 1 | wire channels; **camera aperture lives here** |
| 7 | Turret base | 1 | static now; servo mount pre-modelled for the stretch goal |
| 8 | Detail parts | — | vents, armour plates, exhausts — printed separately for painting |

**Print module 2 first, on its own.** It is the hardest surface, it carries the roller clearance
risk, and it tells you whether the proportions read correctly before you commit 20 hours of
printing.

Joinery: M3 bolts into printed bosses at seams, dowel pins for alignment, CA glue only where a
seam will never need to separate.

---

## 5. Toolchain

The shell splits into two genuinely different modelling problems, and one tool is bad at each.

| Part | Needs | Tool |
|---|---|---|
| Chassis plate, standoffs, camera aperture, mount bosses, clearance envelopes | exact dimensions, parametric, driven by `MecanumGeometry` | **CadQuery / build123d** (Python, versionable, diffable) |
| Body surfaces — nose, pods, canopy, armour | organic curved panels, visual judgement | **Blender** |

Recommended split: generate the **mounting skeleton and clearance envelopes** parametrically from
the same constants the simulator uses, export as STEP, and model the aesthetic surfaces around
that in Blender. The skeleton then updates automatically if the geometry changes, and the body
cannot drift out of tolerance with the robot.

Neither is installed. See § 7.

---

## 6. Simulator coupling

The shell should exist in MuJoCo too, so M6's domain shift can be rehearsed before the printer
runs: a low-poly convex-decomposed version as a visual mesh on the chassis body, plus the mass and
inertia change. That makes "attach the shell" a sim experiment first and a hardware one second —
same as everything else here.

---

## 7. Open decisions

1. **CAD toolchain.** CadQuery is `pip install cadquery` but pulls OCP, a ~300 MB dependency. Say
   the word and I will install it and build the parametric skeleton.
2. **Blender MCP.** Not connected in this session. If you connect it I can drive the body
   surfacing directly; otherwise I can produce a dimensioned build sheet and reference-plane setup
   for you to model against.
3. **Reference images.** The two you sent are die-cast toys, which are already interpretations.
   For real accuracy, orthographic screenshots from the game (front / side / top / rear at
   consistent FOV) are far better. Toy proportions are usually exaggerated and chunky.

---

## 8. Note on the design

This is a personal fan replica. The Arkham Knight Batmobile design is owned by DC/Warner Bros, so
keep it to personal use — do not sell prints or distribute the model files.
