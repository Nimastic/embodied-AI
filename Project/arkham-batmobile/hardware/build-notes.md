# Build Notes

Assembly, wiring, power, safety, arena setup, and the body shell plan.
Parts and prices live in [`BOM.md`](BOM.md).

---

## Layout

Flat plate, everything accessible from above, body shell removable without unplugging anything.

```text
        FRONT
   ┌───────────────┐
   │ [cam]   LEDs  │   camera on a fixed, repeatable mount — height and
   │               │   angle are part of the observation, so a knocked
   │  M1       M2  │   camera invalidates your dataset
   │   ┌─────────┐ │
   │   │ Pi 5    │ │
   │   │ Pico    │ │
   │   └─────────┘ │
   │  [LiPo] [PB]  │   LiPo for motors, power bank for the Pi
   │  M3       M4  │
   │ [kill switch] │   reachable without reaching over the wheels
   └───────────────┘
         REAR
```

Mecanum wheel orientation is not optional: the rollers form an **X pattern when viewed from
above**. Get this wrong and strafing becomes spinning. Dry-run each wheel direction with the robot
up on blocks before the first drive, and write the sign convention into the code as a comment.

Drill the plate as an M3 grid rather than point-drilling for today's parts. You will move the
camera and the IMU, and M6 depends on the camera mount being *repeatable* when the shell goes on
and comes off.

---

## Power

Two isolated supplies, one common ground:

```text
3S LiPo ──> inline fuse ──> kill switch ──> 2x TB6612 ──> 4x motor
                                      └──> (GND shared with Pi)

USB-C power bank ──> Pi 5 ──> Pico (over USB) ──> encoder + PWM lines
```

The Pi never draws from the LiPo. Motor stall currents sag a shared rail enough to brown out a
Pi 5 mid-episode, which looks like random data corruption and costs days to diagnose. Two supplies
cost $25 more and remove the entire failure mode.

Ground the LiPo negative to the Pi ground. Without a common ground the PWM and encoder logic
levels float and the motors behave erratically.

### LiPo safety, first-time owner edition

- Charge only on a balance charger, in a LiPo bag, on a hard non-flammable surface.
- Never charge unattended. Never charge a puffed or damaged pack — dispose of it.
- Never discharge below 3.3 V per cell (9.9 V on a 3S). Add a low-voltage buzzer, or log pack
  voltage and stop the session at 10.5 V.
- Storage-charge to ~3.8 V per cell if the pack will sit more than a few days.
- A fused, switched main lead is a build requirement, not a refinement.

---

## Motor control architecture

| Runs where | Rate | Responsibility |
|---|---|---|
| Pi 5 (Linux, Python) | 20 Hz | camera capture, logging, policy or planner, teleop input, body-velocity command |
| Pico (bare metal) | 100 Hz | quadrature decode on 4 encoders, per-wheel velocity PID, PWM output, command watchdog |

The Pi sends `(vx, vy, omega)` over USB serial at 20 Hz. The Pico does mecanum IK, closes the loop
on each wheel, and returns four measured wheel velocities plus a tick counter.

**Watchdog on the Pico:** if no command arrives for 500 ms, ramp all motors to zero. This covers a
crashed Python process, a Wi-Fi drop in offboard mode, and a disconnected USB cable. Implement it
before the first drive, not after the first runaway.

**Sign conventions:** write down, in code, the positive direction of each wheel and each encoder,
and verify each one individually on blocks. Four motors times two possible encoder polarities
times two possible motor polarities is a lot of ways to be subtly wrong, and a single flipped
encoder produces a world model that is confidently wrong in one direction only.

---

## Arena and ground truth

A taped 2 × 2 m square on hard flooring. Everything depends on this being boring and repeatable.

- **Overhead camera** on a tripod, shelf or ceiling mount, looking straight down, fixed for the
  life of the dataset. Moving it means recalibrating and arguably invalidating earlier episodes.
- **ArUco marker** flat on the robot roof, as large as fits. This is the pose label for M1 through
  M6. Print a second marker taped at a known arena corner as the world origin.
- **Calibrate** the overhead camera intrinsics (a printed checkerboard, OpenCV, twenty minutes)
  and record the homography from image plane to floor plane. Store both in the run manifest.
- **Two surfaces** for M6: hard floor as surface A, a thin low-pile mat as surface B. Carpet deep
  enough to bind the rollers is a different project.
- **Lighting:** keep it constant and avoid daylight, or the camera observation distribution shifts
  across sessions and you will mistake it for a dynamics problem. Log the time of day anyway.

---

## Body shell

Two versions, deliberately.

### v1 — foam, one weekend

Foamboard or 5 mm EVA foam over the plate, on four M3 standoffs, matte black. Target ~200 g.
Purpose: make shell attachment a **measurable mass-and-camera-geometry shift** for M6 while
costing one weekend. It photographs well, which is most of what the demo video needs.

Keep the camera aperture oversized so the shell never partly occludes the lens, and keep the
standoff positions identical between removals.

### v2 — printed, after M4 passes

The modular print plan below is salvaged from the original PRD
([`../raw/prd-original.md`](../raw/prd-original.md)), rescaled from 1:8 to ~1:12. The original
assumed a 75 cm body on a 220 mm bed, which is why it needed eight modules split into twenty-odd
pieces. At ~40 cm on a 256 mm bed, most of that disappears.

| Module | Pieces | Notes |
|---|---:|---|
| Front nose and splitter | 2 | LED slots for headlights; split on the centreline |
| Cockpit canopy | 1–2 | Removable for access; clear PETG or cut acrylic for glazing |
| Side fenders, left and right | 2 | Must clear full mecanum roller travel — check with the wheels fitted, not from CAD |
| Rear engine housing | 2 | LED slot for the afterburner glow |
| Underbody skirt | 1 | Wire channels; leave the plate underside open for airflow to the Pi |
| Spoiler and fins | 2–3 | Print separately for painting |
| Detail parts (vents, armour plates, exhausts) | — | Separate, painted off the body |

Joinery: M3 bolts into printed bosses at module seams, snap-fit pegs for alignment, CA glue only
where a seam will never need to come apart. Mount to the chassis plate on the same four standoff
positions as v1 so M6 comparisons stay valid.

Finishing: sand, filler primer, matte black, silver dry-brush for edge wear, red tail LEDs,
Arkham insignia decals.

Keep the shell under ~400 g. Mass above the wheel axis raises the centre of gravity and changes
the slip behaviour you spent M1 characterising — which is interesting as an experiment and
annoying as a surprise.

### Concept reference

`../raw/images/concept-front-quarter.png` and `../raw/images/concept-rear-quarter.png` are the
original generated concept renders. Treat them as mood, not as orthographic references — they are
not dimensionally consistent. For real reference, pull orthographic screenshots from the game or
from a fan-made model and scale from the wheelbase you actually built.

---

## Deliberately not built

**No projectile launcher wired to a person detector.** The original notes sketched detect-a-person,
drive-at-them, fire. That is not getting built here, and the YOLO-plus-relay snippet in
`../raw/notes-dump.txt` is kept only as provenance of where this project started.

If a batarang launcher is added for the look, the rules are: static targets only, a human finger
on a physical button, no wiring between any detector and the trigger, and foam darts. There is no
world-model learning content in the launcher — it is a prop, and it should be built like one.
