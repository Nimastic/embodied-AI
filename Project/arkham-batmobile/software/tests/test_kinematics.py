"""
Kinematics conventions and batched integration.

Protects against
----------------
BUG 1  Sign/convention errors in the mecanum wheel patterns. Every one of these
       produces a robot that drives *somewhere*, and plots that look fine. P0
       measured forward motion as 7.8% delivered and strafe as 76.8% - exactly
       inverted - and the numbers were plausible enough to believe.
       (experiments/P0-sim-baseline/README.md, "Step response measured against
       a wall")

BUG 2  `integrate_batch` drifting away from the scalar `integrate_pose` loop.
       MPPI scores every candidate rollout with the batched version while every
       evaluation script and the analytic `predict()` use the loop. If the two
       disagree, the planner optimises a trajectory the evaluator will never
       reproduce, and both halves still report sensible millimetres.

`mecanum.py` carries these checks in `_sanity_checks()` under
`if __name__ == "__main__"`, which means they only run when somebody remembers
to run that file by hand. These are the same invariants as collectable tests,
plus the exact wheel patterns (not just their signs) and the integration
conventions the dataset schema depends on.
"""

import numpy as np

from _harness import main
from kinematics.mecanum import MecanumGeometry, MecanumKinematics, body_to_world

KIN = MecanumKinematics()
R = KIN.geom.wheel_radius
L = KIN.geom.lever
# wheel order is FL, FR, RL, RR everywhere in this project
FL, FR, RL, RR = 0, 1, 2, 3


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# ── Geometry ────────────────────────────────────────────────────────────────

def test_lever_is_half_wheelbase_plus_half_track():
    geom = MecanumGeometry(wheel_radius=0.05, half_wheelbase=0.1, half_track=0.2)
    assert np.isclose(geom.lever, 0.3), geom.lever


# ── BUG 1: wheel patterns, exact values not just signs ──────────────────────

def test_ik_forward_drives_all_four_wheels_equally_forward():
    w = KIN.ik([1.0, 0.0, 0.0])
    assert np.allclose(w, 1.0 / R), f"forward: expected all {1.0/R:.3f}, got {w}"


def test_ik_strafe_left_pattern():
    """+y is LEFT. FL back, FR forward, RL forward, RR back."""
    w = KIN.ik([0.0, 1.0, 0.0])
    expect = np.array([-1.0, +1.0, +1.0, -1.0]) / R
    assert np.allclose(w, expect), f"strafe left: expected {expect}, got {w}"


def test_ik_strafe_right_is_the_mirror_of_strafe_left():
    assert np.allclose(KIN.ik([0.0, -1.0, 0.0]), -KIN.ik([0.0, 1.0, 0.0]))
    w = KIN.ik([0.0, -1.0, 0.0])
    assert w[FL] > 0 and w[FR] < 0 and w[RL] < 0 and w[RR] > 0, w


def test_ik_spin_ccw_drives_the_left_pair_backwards():
    """theta counter-clockwise positive: left wheels back, right wheels forward."""
    w = KIN.ik([0.0, 0.0, 1.0])
    expect = np.array([-L, +L, -L, +L]) / R
    assert np.allclose(w, expect), f"spin CCW: expected {expect}, got {w}"
    assert w[FL] < 0 and w[RL] < 0 and w[FR] > 0 and w[RR] > 0, w


def test_fk_inverts_ik_for_any_body_velocity():
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = rng.uniform(-2.0, 2.0, size=3)
        back = KIN.fk(KIN.ik(v))
        assert np.allclose(back, v, atol=1e-12), f"fk(ik({v})) = {back}"


def test_fk_inverts_ik_for_a_different_geometry():
    """The identity must hold from the geometry, not from the default numbers."""
    kin = MecanumKinematics(MecanumGeometry(0.0325, 0.12, 0.105))
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.uniform(-2.0, 2.0, size=3)
        assert np.allclose(kin.fk(kin.ik(v)), v, atol=1e-12)


def test_J_has_orthogonal_columns():
    """
    J.T @ J diagonal is what makes FK three signed averages and the three body
    modes independently observable. The slip model's per-mode efficiencies and
    the calibrated baseline's three fitted gains both depend on it.
    """
    gram = KIN.J.T @ KIN.J
    off = gram - np.diag(np.diag(gram))
    assert np.allclose(off, 0.0, atol=1e-12), f"J.T @ J not diagonal:\n{gram}"
    assert np.allclose(np.diag(gram), [4.0, 4.0, 4.0 * L * L]), np.diag(gram)


def test_null_space_direction_produces_no_body_motion_and_unit_scrub():
    null = np.array([1.0, 1.0, -1.0, -1.0])
    assert np.allclose(KIN.fk(null), 0.0, atol=1e-12), KIN.fk(null)
    assert np.isclose(KIN.scrub(null), 1.0), KIN.scrub(null)


def test_clean_driving_produces_zero_scrub():
    """
    Scrub is the cheapest health check on the robot. If IK leaks into the null
    space, scrub stops meaning "something is wrong" and starts meaning nothing.
    """
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(500):
        v = rng.uniform(-2.0, 2.0, size=3)
        worst = max(worst, abs(KIN.scrub(KIN.ik(v))))
    assert worst < 1e-12, f"max |scrub| over clean IK commands was {worst:.3e}"


def test_scrub_is_blind_to_body_motion_and_sees_only_the_null_direction():
    """A real base's scrub must not be contaminated by how fast it is driving."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        v = rng.uniform(-1.0, 1.0, size=3)
        amount = rng.uniform(-3.0, 3.0)
        wheels = KIN.ik(v) + amount * np.array([1.0, 1.0, -1.0, -1.0])
        assert np.isclose(KIN.scrub(wheels), amount, atol=1e-12)
        assert np.allclose(KIN.fk(wheels), v, atol=1e-12)


# ── Integration conventions (the dataset schema depends on these) ───────────

def test_spinning_in_place_does_not_translate():
    pose = np.zeros(3)
    for _ in range(100):
        pose = KIN.integrate_pose(pose, [0.0, 0.0, 1.0], 0.01)
    assert np.allclose(pose[:2], 0.0, atol=1e-9), pose
    assert np.isclose(pose[2], 1.0), pose[2]


def test_strafe_left_moves_in_plus_y_at_zero_heading():
    pose = KIN.integrate_pose(np.zeros(3), [0.0, 0.5, 0.0], 1.0)
    assert np.isclose(pose[0], 0.0) and np.isclose(pose[1], 0.5), pose


def test_forward_at_ninety_degrees_moves_in_plus_y():
    pose = KIN.integrate_pose([0.0, 0.0, np.pi / 2], [0.5, 0.0, 0.0], 1.0)
    assert np.isclose(pose[0], 0.0, atol=1e-12) and np.isclose(pose[1], 0.5), pose


def test_body_to_world_rotates_translation_and_leaves_omega_alone():
    out = body_to_world([1.0, 0.0, 0.7], np.pi / 2)
    assert np.allclose(out, [0.0, 1.0, 0.7], atol=1e-12), out
    out = body_to_world([0.0, 1.0, 0.0], np.pi / 2)
    assert np.allclose(out, [-1.0, 0.0, 0.0], atol=1e-12), out


def test_heading_wraps_rather_than_accumulating():
    pose = np.array([0.0, 0.0, 3.0])
    pose = KIN.integrate_pose(pose, [0.0, 0.0, 10.0], 0.05)
    assert -np.pi < pose[2] <= np.pi, pose[2]


# ── BUG 2: batched integration must equal the scalar loop ───────────────────

def _assert_batch_matches_loop(pose0, vels, dt, atol=1e-12):
    batch = KIN.integrate_batch(pose0, vels, dt)
    n, h = vels.shape[0], vels.shape[1]
    assert batch.shape == (n, h, 3), batch.shape
    worst = 0.0
    for i in range(n):
        p = np.asarray(pose0, dtype=float)[i].copy()
        for k in range(h):
            p = KIN.integrate_pose(p, vels[i, k], dt)
            dxy = np.abs(batch[i, k, :2] - p[:2]).max()
            dth = abs(float(_wrap(batch[i, k, 2] - p[2])))
            worst = max(worst, dxy, dth)
    assert worst < atol, f"integrate_batch deviates from the loop by {worst:.3e}"
    return worst


def test_integrate_batch_matches_the_scalar_loop():
    rng = np.random.default_rng(5)
    pose0 = rng.uniform(-1.0, 1.0, size=(32, 3))
    vels = rng.uniform(-0.5, 0.5, size=(32, 20, 3))
    _assert_batch_matches_loop(pose0, vels, 0.05)


def test_integrate_batch_matches_the_loop_for_a_single_step():
    rng = np.random.default_rng(6)
    pose0 = rng.uniform(-1.0, 1.0, size=(8, 3))
    vels = rng.uniform(-0.5, 0.5, size=(8, 1, 3))
    _assert_batch_matches_loop(pose0, vels, 0.05)


def test_integrate_batch_matches_the_loop_through_a_heading_wrap():
    """
    The batched form integrates heading as a cumulative sum and wraps once at
    the end; the loop wraps every step. Saturated spin commands over a 15-step
    MPPI horizon cross +-pi routinely, so this is the case that would diverge.
    """
    pose0 = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]])
    vels = np.tile(np.array([0.3, -0.2, 2.0]), (2, 60, 1))
    vels[1] *= -1.0
    _assert_batch_matches_loop(pose0, vels, 0.05)


def test_integrate_batch_matches_the_loop_at_zero_velocity():
    pose0 = np.array([[0.4, -0.2, 1.1]])
    vels = np.zeros((1, 10, 3))
    batch = KIN.integrate_batch(pose0, vels, 0.05)
    assert np.allclose(batch - pose0[:, None, :], 0.0, atol=1e-15), batch


def test_analytic_predict_matches_integrate_batch():
    """
    `predict()` (the loop) is the analytic model used in evaluation;
    `integrate_batch` is what MPPI rolls out. They must be the same model.
    """
    rng = np.random.default_rng(7)
    pose = rng.uniform(-1.0, 1.0, size=3)
    seq = rng.uniform(-0.5, 0.5, size=(15, 3))
    loop = KIN.predict(pose, seq, 0.05)
    batch = KIN.integrate_batch(pose[None], seq[None], 0.05)[0]
    assert np.abs(loop[:, :2] - batch[:, :2]).max() < 1e-12
    assert np.abs(_wrap(loop[:, 2] - batch[:, 2])).max() < 1e-12


if __name__ == "__main__":
    raise SystemExit(main(globals()))
