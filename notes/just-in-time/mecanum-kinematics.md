# Mecanum Kinematics

**Pulled for:** `Project/arkham-batmobile` M1 (the analytic baseline the learned world model has to
beat) and Phase 0 (the MuJoCo mecanum base).
**Minimum concept required:** the closed-form map between body velocity and four wheel speeds, and
an honest account of where that map stops being true.

---

## 1. Frames and conventions

Fix these once and never re-derive them:

- Body frame: **x forward, y left, θ counter-clockwise positive** (right-handed, z up).
- Wheel angular velocity `ω_i` positive when the wheel drives the robot forward.
- `r` — wheel radius.
- `l` — half the wheelbase (front-to-rear distance / 2).
- `w` — half the track width (left-to-right distance / 2).
- Wheels indexed FL, FR, RL, RR at body-frame positions `(+l, +w)`, `(+l, -w)`, `(-l, +w)`,
  `(-l, -w)`.

Every sign error in a mecanum build traces back to one of these being assumed rather than written
down.

---

## 2. Velocity at each wheel hub

The chassis is a rigid body. For a point at body-frame offset `(x_i, y_i)`:

```
v_i = v_origin + ω × r_i
```

With `ω = ω ẑ` and `r_i = (x_i, y_i, 0)`, the cross product is `ω(-y_i, x_i, 0)`, so:

```
v_ix = vx − ω·y_i
v_iy = vy + ω·x_i
```

That is all the rigid-body mechanics this needs. The mecanum-specific part is next.

---

## 3. Where the 45° enters

A conventional wheel enforces a hard constraint: no sideways motion at the contact point. That is
why a car has to turn before it can move sideways.

A mecanum wheel replaces the tyre with free-spinning rollers whose axes sit at 45° to the axle. The
roller can spin freely, so the wheel offers **no resistance along the roller axis** — it only
transmits force perpendicular to it. One constraint per wheel is removed, which is exactly what
buys the extra degree of freedom.

Projecting the hub velocity onto the direction the wheel can actually drive, for a roller at angle
γ to the axle:

```
r·ω_i = v_ix + (1 / tan γ)·v_iy
```

At γ = 45°, `tan γ = 1`, so the lateral term enters with unit weight and a sign set by which way
that wheel's rollers lean:

```
r·ω_i = v_ix ± v_iy
```

The mirrored **X pattern** — FL and RR leaning one way, FR and RL the other — is what makes the
signs work out so the lateral terms can be made to cancel (forward) or reinforce (strafe).

---

## 4. Inverse kinematics

Substituting the hub velocities and the wheel positions:

```
ω_FL = (1/r)·( vx − vy − (l+w)·ω )
ω_FR = (1/r)·( vx + vy + (l+w)·ω )
ω_RL = (1/r)·( vx + vy − (l+w)·ω )
ω_RR = (1/r)·( vx − vy + (l+w)·ω )
```

In matrix form, with `L = l + w`:

```
⎡ω_FL⎤       ⎡ 1  −1  −L ⎤ ⎡vx⎤
⎢ω_FR⎥ = 1/r ⎢ 1   1   L ⎥ ⎢vy⎥
⎢ω_RL⎥       ⎢ 1   1  −L ⎥ ⎢ω ⎥
⎣ω_RR⎦       ⎣ 1  −1   L ⎦
```

Call that matrix `J`, so `ω_wheels = (1/r)·J·v_body`. Six lines of code, no iteration, no
singularities. This is the entire "control system" of a mecanum base.

### Sanity checks — run these before trusting any implementation

| Command | Expected wheel pattern |
|---|---|
| `vx > 0` (forward) | all four forward, equal |
| `vy > 0` (strafe left) | FL back, FR forward, RL forward, RR back |
| `vy < 0` (strafe right) | FL forward, FR back, RL back, RR forward |
| `ω > 0` (spin CCW) | left pair back, right pair forward |
| `ω < 0` (spin CW) | left pair forward, right pair back |

If strafing makes the robot spin, two wheels are swapped or two rollers lean the wrong way. Check
the X pattern before touching the code.

---

## 5. Forward kinematics

Four measurements, three unknowns — overdetermined, so solve least-squares. `J` turns out to have
**orthogonal columns**:

```
Jᵀ J = diag(4, 4, 4L²)
```

(Check: columns are `(1,1,1,1)`, `(−1,1,1,−1)`, `L·(−1,1,−1,1)`; every pairwise dot product is
zero.) So the pseudoinverse is diagonal and the FK has a clean closed form:

```
vx = (r/4)·(  ω_FL + ω_FR + ω_RL + ω_RR )
vy = (r/4)·( −ω_FL + ω_FR + ω_RL − ω_RR )
ω  = (r/4L)·( −ω_FL + ω_FR − ω_RL + ω_RR )
```

Orthogonality is not a coincidence worth skipping past: it means the three body degrees of freedom
are **independently observable** from wheel speeds, and that each one is a simple signed average.
Forward is the mean, strafe is one diagonal difference, rotation is the other.

---

## 6. The fourth direction: scrub

`J` is 4×3, so one combination of wheel speeds lies in its left null space and produces **no body
motion at all**. Solving `Jᵀu = 0`:

```
u ∝ (1, 1, −1, −1)
```

Front pair forward, rear pair backward. The wheels fight each other, the rollers scrub, and the
robot sits still burning current.

This is a free diagnostic. Define:

```
scrub = ( ω_FL + ω_FR − ω_RL − ω_RR ) / 4
```

On a correctly calibrated robot driving freely, `scrub ≈ 0`. When it is not, something specific is
wrong — a mis-signed encoder, one PID loop tuned differently, a wheel not touching the floor, or
an external push. Log it from day one in M0; it is the cheapest health check available and it
costs one line.

---

## 7. Where this model stops being true

The point of M1 is not to implement the above — that is an afternoon. It is to **measure how wrong
it is**, because that error is what the learned world model in M2 exists to predict. The derivation
assumes all of the following, and reality grants none of them:

| Assumption | Reality |
|---|---|
| Rollers never slip along their axis | They slip constantly, and more on soft or dusty floors |
| All four wheels bear equal load | Weight shifts under acceleration; a 3-point-contact chassis loads two wheels harder |
| Rollers are exactly at 45° | Manufacturing tolerance, and the effective angle changes as a roller wears |
| Contact is a point at the wheel centreline | It is a patch that migrates as each roller rotates into and out of contact — which is why mecanum drive has a characteristic small periodic ripple |
| Commanded wheel speed is achieved | PID has finite bandwidth; motors saturate; pack voltage sags under current |
| Strafing is as good as driving | It is not. Lateral motion loads the rollers in their compliant direction, so strafe loses noticeably more to slip than forward motion does |

The last row is the one to expect to dominate. If a learned model beats analytic FK anywhere, it
will be on strafe and on direction changes.

### The structure that makes this learnable

The residual is **not** noise. It is a function of state and action — speed, direction, surface,
pack voltage, acceleration. That is precisely why M2 can beat M1 instead of just fitting noise, and
it is why `batt_v` is an input in the dataset schema rather than metadata. If the residual were
zero-mean white noise, no model could beat FK and the honest result would be to say so.

---

## 8. How this gets used

- **M1** — implement `J` and the FK, calibrate `r`, `l`, `w` against the overhead fiducial, fit a
  per-wheel slip gain per surface, and report analytic prediction error at 1 step and 1.0 s.
- **M2** — have the network predict the **residual on top of this model**, not the raw next state.
  Learning a correction to a nearly-right model is a far smaller and better-conditioned problem
  than learning the dynamics from scratch.
- **M4** — the analytic model doubles as a control condition: MPPI planning inside FK versus MPPI
  inside the learned model, same planner, same cost, one line of config apart.

## Implementation

`Project/arkham-batmobile/software/kinematics/mecanum.py` — `J`, the FK, the scrub projection, and
the sanity checks above as runnable assertions.
