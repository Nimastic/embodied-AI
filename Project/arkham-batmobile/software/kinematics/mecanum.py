"""
Mecanum kinematics — the analytic baseline.

Closed-form map between body velocity (vx, vy, omega) and four wheel speeds.
Derivation and the list of assumptions this model makes:
    notes/just-in-time/mecanum-kinematics.md

This module is the M1 baseline AND the model the learned world model must beat.
It is deliberately dependency-free (numpy only) so the identical code runs in
simulation, on the Raspberry Pi, and inside the MPPI planner.

Conventions (fixed once, never re-derived):
    body frame      x forward, y left, theta counter-clockwise positive
    wheel order     FL, FR, RL, RR
    wheel positions FL(+l, +w)  FR(+l, -w)  RL(-l, +w)  RR(-l, -w)
    omega_i > 0     drives the robot forward

Usage:
    python mecanum.py           # run the sanity checks
"""

from dataclasses import dataclass

import numpy as np

WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class MecanumGeometry:
    """Physical geometry of the base. Calibrate these in M1, do not trust the datasheet."""

    wheel_radius: float = 0.040      # r, metres (80 mm wheel)
    half_wheelbase: float = 0.090    # l, metres (front-to-rear / 2)
    half_track: float = 0.075        # w, metres (left-to-right / 2)

    @property
    def lever(self) -> float:
        """L = l + w, the rotation lever arm that appears throughout the kinematics."""
        return self.half_wheelbase + self.half_track


class MecanumKinematics:
    """
    Analytic forward/inverse kinematics for a four-wheel mecanum base.

    The inverse-kinematics matrix J (4x3) maps body velocity to wheel speeds:

        omega_wheels = (1/r) * J @ (vx, vy, omega)

                [ 1  -1  -L ]
        J  =    [ 1   1   L ]
                [ 1   1  -L ]
                [ 1  -1   L ]

    J has orthogonal columns (J.T @ J is diagonal), so the forward kinematics
    pseudoinverse collapses to three signed averages and the three body degrees
    of freedom are independently observable from wheel speeds.
    """

    def __init__(self, geometry: MecanumGeometry | None = None):
        self.geom = geometry or MecanumGeometry()
        L = self.geom.lever
        self.J = np.array(
            [
                [1.0, -1.0, -L],   # FL
                [1.0, +1.0, +L],   # FR
                [1.0, +1.0, -L],   # RL
                [1.0, -1.0, +L],   # RR
            ]
        )

    # ── Inverse kinematics: body velocity → wheel speeds ────────────────────

    def ik(self, body_vel: np.ndarray) -> np.ndarray:
        """
        (vx, vy, omega) in m/s, m/s, rad/s  →  four wheel speeds in rad/s.

        Exact and singularity-free: a mecanum base can realise any body velocity
        from any configuration, which is what "holonomic" means in practice.
        """
        body_vel = np.asarray(body_vel, dtype=float)
        return (self.J @ body_vel) / self.geom.wheel_radius

    # ── Forward kinematics: wheel speeds → body velocity ────────────────────

    def fk(self, wheel_vel: np.ndarray) -> np.ndarray:
        """
        Four wheel speeds in rad/s  →  (vx, vy, omega).

        Least-squares solution. Because J has orthogonal columns this reduces to:
            vx = (r/4)  * ( +FL +FR +RL +RR )
            vy = (r/4)  * ( -FL +FR +RL -RR )
            om = (r/4L) * ( -FL +FR -RL +RR )

        Note this *discards* the fourth, unobservable combination of wheel speeds
        (see scrub() below) rather than letting it corrupt the estimate.
        """
        w = np.asarray(wheel_vel, dtype=float)
        r, L = self.geom.wheel_radius, self.geom.lever
        return np.array(
            [
                (r / 4.0) * (+w[0] + w[1] + w[2] + w[3]),
                (r / 4.0) * (-w[0] + w[1] + w[2] - w[3]),
                (r / (4.0 * L)) * (-w[0] + w[1] - w[2] + w[3]),
            ]
        )

    # ── The achievable set ──────────────────────────────────────────────────

    def peak_wheel_speed(self, body_vel: np.ndarray) -> float:
        """Fastest wheel required to realise this body velocity, in rad/s."""
        return float(np.max(np.abs(self.ik(body_vel))))

    def clamp_achievable(self, body_vel: np.ndarray, wheel_speed_max: float) -> np.ndarray:
        """
        Scale a body velocity down until every wheel is within its speed limit.

        The achievable set of a mecanum base is NOT a box - it is a polytope,
        because |vx| + |vy| + L|omega| all draw on the same wheel budget. A
        command can respect every per-axis limit and still be impossible: with
        r=40 mm and L=165 mm, (0.5, 0.5, 2.0) needs 33 rad/s from a wheel that
        tops out at 14.

        Scaling all three components by one factor preserves the DIRECTION of
        motion and only reduces its magnitude. Clipping wheels individually
        instead would break the mirrored symmetry and turn a straight strafe
        into a curve - and it dumps the discarded motion into the scrub
        direction, which is both wasted current and a corrupted dataset.

        This is what real mecanum firmware does, and it belongs here rather than
        in the simulator because the Pico will need exactly this function.
        """
        peak = self.peak_wheel_speed(body_vel)
        if peak <= wheel_speed_max or peak < 1e-9:
            return np.asarray(body_vel, dtype=float)
        return np.asarray(body_vel, dtype=float) * (wheel_speed_max / peak)

    # ── The fourth direction: wasted motion ─────────────────────────────────

    def scrub(self, wheel_vel: np.ndarray) -> float:
        """
        Project wheel speeds onto the left null space of J, u = (1, 1, -1, -1)/4.

        This combination produces *no body motion* — the front pair drives forward
        while the rear pair drives backward, the rollers scrub, and the robot burns
        current standing still.

        On a healthy, calibrated robot driving freely this is ~0. When it is not,
        suspect a mis-signed encoder, one badly tuned wheel PID, a wheel off the
        floor, or an external push. Log it from M0 onward — cheapest health check
        available, and it costs one line.
        """
        w = np.asarray(wheel_vel, dtype=float)
        return float((w[0] + w[1] - w[2] - w[3]) / 4.0)

    # ── Dead reckoning ──────────────────────────────────────────────────────

    def integrate_pose(self, pose: np.ndarray, body_vel: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance a world-frame pose (x, y, theta) by a body-frame velocity for dt.

        Midpoint heading is used for the rotation, which keeps the error second
        order in dt instead of first — it matters once you integrate for a few
        hundred ticks and compare against the overhead fiducial.
        """
        x, y, th = np.asarray(pose, dtype=float)
        vx, vy, om = np.asarray(body_vel, dtype=float)

        th_mid = th + 0.5 * om * dt
        c, s = np.cos(th_mid), np.sin(th_mid)

        return np.array(
            [
                x + (vx * c - vy * s) * dt,
                y + (vx * s + vy * c) * dt,
                _wrap_angle(th + om * dt),
            ]
        )

    def integrate_batch(self, pose0: np.ndarray, vels: np.ndarray, dt: float) -> np.ndarray:
        """
        Vectorised midpoint integration over N trajectories at once.

            pose0  (N, 3)      starting poses
            vels   (N, H, 3)   body velocities per step
            returns (N, H, 3)  pose after each step

        Numerically identical to calling integrate_pose in a loop, with no
        Python loop over either N or H. Heading integrates independently of
        position, so both reduce to cumulative sums:

            theta_mid[k] = theta0 + dt * (cumsum(omega)[k] - omega[k] / 2)
            xy[k]        = xy0 + dt * cumsum(R(theta_mid[k]) @ v_xy[k])

        MPPI in M4 scores hundreds of candidate rollouts per control tick, so
        this is the function that makes planning real-time rather than a
        research curiosity. Evaluation gets every horizon in one pass too.
        """
        pose0 = np.asarray(pose0, dtype=float)
        vels = np.asarray(vels, dtype=float)

        om = vels[..., 2]
        cum_om = np.cumsum(om, axis=1)
        th_mid = pose0[:, None, 2] + dt * (cum_om - 0.5 * om)

        c, s = np.cos(th_mid), np.sin(th_mid)
        dx = (vels[..., 0] * c - vels[..., 1] * s) * dt
        dy = (vels[..., 0] * s + vels[..., 1] * c) * dt

        out = np.empty(vels.shape)
        out[..., 0] = pose0[:, None, 0] + np.cumsum(dx, axis=1)
        out[..., 1] = pose0[:, None, 1] + np.cumsum(dy, axis=1)
        out[..., 2] = _wrap_angle_array(pose0[:, None, 2] + dt * cum_om)
        return out

    def predict(self, pose: np.ndarray, action_sequence: np.ndarray, dt: float) -> np.ndarray:
        """
        The analytic model as a *predictor*, with the same signature as the learned
        models in models/. This is what makes "analytic vs learned" a config change
        instead of a rewrite, and why this file is a model rather than a utility.

        pose             (3,)     current (x, y, theta)
        action_sequence  (H, 3)   body velocities to apply
        returns          (H, 3)   predicted pose after each action

        It assumes commanded velocity is achieved exactly. That assumption is the
        whole research question — see the note, section 7.
        """
        poses = np.empty((len(action_sequence), 3))
        p = np.asarray(pose, dtype=float)
        for i, action in enumerate(action_sequence):
            p = self.integrate_pose(p, action, dt)
            poses[i] = p
        return poses


def _wrap_angle_array(a: np.ndarray) -> np.ndarray:
    """Vectorised form of _wrap_angle."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]. Forgetting this is how heading error silently becomes 2*pi."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def body_to_world(body_vel: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate a body-frame velocity into the world frame.

    Needed because the robot is commanded in its own frame but MuJoCo's slide
    joints — and the overhead camera — live in the world frame.
    """
    vx, vy, om = np.asarray(body_vel, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([vx * c - vy * s, vx * s + vy * c, om])


# ── Sanity checks (section 4 of the note, as runnable assertions) ───────────

def _sanity_checks() -> None:
    kin = MecanumKinematics()
    r = kin.geom.wheel_radius

    fl, fr, rl, rr = kin.ik([1.0, 0.0, 0.0])
    assert np.allclose([fl, fr, rl, rr], 1.0 / r), "forward: all four wheels equal and positive"

    fl, fr, rl, rr = kin.ik([0.0, 1.0, 0.0])
    assert fl < 0 and fr > 0 and rl > 0 and rr < 0, "strafe left: FL back, FR fwd, RL fwd, RR back"

    fl, fr, rl, rr = kin.ik([0.0, -1.0, 0.0])
    assert fl > 0 and fr < 0 and rl < 0 and rr > 0, "strafe right: FL fwd, FR back, RL back, RR fwd"

    fl, fr, rl, rr = kin.ik([0.0, 0.0, 1.0])
    assert fl < 0 and rl < 0 and fr > 0 and rr > 0, "spin CCW: left pair back, right pair fwd"

    # FK inverts IK exactly for any realisable body velocity
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.uniform(-1.0, 1.0, size=3)
        assert np.allclose(kin.fk(kin.ik(v)), v), "fk(ik(v)) must be the identity"

    # J has orthogonal columns -> J.T @ J is diagonal
    gram = kin.J.T @ kin.J
    assert np.allclose(gram, np.diag(np.diag(gram))), "columns of J must be orthogonal"

    # The null-space direction produces no body motion, and reads as pure scrub
    null = np.array([1.0, 1.0, -1.0, -1.0])
    assert np.allclose(kin.fk(null), 0.0), "(1,1,-1,-1) must produce zero body velocity"
    assert np.isclose(kin.scrub(null), 1.0), "(1,1,-1,-1) must read as unit scrub"
    for _ in range(200):
        v = rng.uniform(-1.0, 1.0, size=3)
        assert np.isclose(kin.scrub(kin.ik(v)), 0.0), "clean driving must produce zero scrub"

    # Pure rotation integrates to pure rotation; pure strafe moves sideways only
    pose = np.zeros(3)
    for _ in range(100):
        pose = kin.integrate_pose(pose, [0.0, 0.0, 1.0], 0.01)
    assert np.allclose(pose[:2], 0.0, atol=1e-9), "spinning in place must not translate"
    assert np.isclose(pose[2], 1.0), "1 rad/s for 1 s must be 1 rad"

    pose = kin.integrate_pose(np.zeros(3), [0.0, 0.5, 0.0], 1.0)
    assert np.isclose(pose[0], 0.0) and np.isclose(pose[1], 0.5), "strafe left is +y at theta=0"

    # Batched integration must be numerically identical to the scalar loop
    rng2 = np.random.default_rng(5)
    pose0 = rng2.uniform(-1, 1, size=(32, 3))
    vels = rng2.uniform(-0.5, 0.5, size=(32, 20, 3))
    batch = kin.integrate_batch(pose0, vels, 0.05)
    for i in range(32):
        p = pose0[i].copy()
        for k in range(20):
            p = kin.integrate_pose(p, vels[i, k], 0.05)
            assert np.allclose(batch[i, k], p, atol=1e-12), "integrate_batch must match the loop"

    print("mecanum kinematics: all sanity checks passed")
    print(f"  geometry      r={kin.geom.wheel_radius:.3f} m  l={kin.geom.half_wheelbase:.3f} m  "
          f"w={kin.geom.half_track:.3f} m  L={kin.geom.lever:.3f} m")
    print(f"  J.T @ J       diag({np.diag(gram).round(4)})")
    print(f"  max wheel speed at 0.5 m/s forward: {kin.ik([0.5, 0, 0]).max():.2f} rad/s "
          f"({kin.ik([0.5, 0, 0]).max() * 60 / (2 * np.pi):.0f} rpm)")


if __name__ == "__main__":
    _sanity_checks()
