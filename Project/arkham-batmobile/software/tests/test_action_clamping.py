"""
Action clamping and the achievable polytope.

Protects against
----------------
BUG 4  Commands outside the achievable set, and the per-wheel clipping that
       used to "fix" them.

       `ACTION_LIMITS` is a BOX; the achievable set is a POLYTOPE, because
       |vx|, |vy| and L|omega| all draw on one wheel-speed budget. The action
       box (0.5, 0.5, 2.0) asks 33 rad/s of a 14 rad/s wheel, and in P0 72% of
       ticks saturated. Clipping each wheel independently destroyed the
       mirrored roller pattern and dumped the discarded motion into the
       null-space scrub direction: mean |scrub| read 1.004 rad/s against a
       proper 0.07. The dataset still looked like a dataset.

       Fixed two ways, both of which this file pins: scale all four wheels by
       ONE factor (`clamp_achievable`, which preserves direction), and clamp
       sampled actions into the polytope before they are ever commanded
       (`clamp_action`) - so saturation in collected data is 0%.
       (experiments/P0-sim-baseline/README.md, "Individual wheel clipping broke
       the mecanum symmetry")

Every producer of actions is tested here, because the invariant is only worth
anything if it holds at every entry point: the random sampler, the wall-avoider,
the scripted expert, the reactive controller, and the MPPI planner.
"""

import numpy as np

from _harness import import_or_skip, main, sim_episode
from kinematics.mecanum import MecanumKinematics

KIN = MecanumKinematics()
TOL = 1.0 + 1e-9        # same relative tolerance SlipModel uses, for exactly the
                        # same reason: callers clamp to exactly WHEEL_SPEED_MAX


def _env():
    import_or_skip("mujoco", " (the Phase 0 sim needs it)")
    return import_or_skip("sim.mecanum_env")


def _peak(action):
    return float(np.max(np.abs(KIN.ik(action))))


def _assert_achievable(action, limit, what):
    peak = _peak(action)
    assert peak <= limit * TOL, (
        f"{what} emitted {np.asarray(action).round(3)}, which needs {peak:.2f} rad/s "
        f"from a {limit:.2f} rad/s wheel. Actions must be inside the achievable "
        f"polytope before they are commanded, or the dataset records commands the "
        f"robot never executed."
    )


# ── The invariant, at every producer ────────────────────────────────────────

def test_clamp_action_output_is_always_achievable():
    env = _env()
    rng = np.random.default_rng(0)
    # include the box corners, which are the worst case and the P0 example
    corners = np.array(np.meshgrid(*[[-1.0, 1.0]] * 3)).T.reshape(-1, 3) * env.ACTION_LIMITS
    samples = np.vstack([corners, rng.uniform(-3.0, 3.0, size=(600, 3)) * env.ACTION_LIMITS])
    for a in samples:
        _assert_achievable(env.clamp_action(a), env.WHEEL_SPEED_MAX, "clamp_action")


def test_clamp_action_leaves_achievable_commands_untouched():
    env = _env()
    for a in ([0.1, 0.0, 0.0], [0.0, -0.05, 0.3], [0.2, 0.1, 0.0]):
        assert np.allclose(env.clamp_action(a), a), (a, env.clamp_action(a))


def test_clamp_achievable_preserves_direction():
    """
    The whole point of scaling over clipping: magnitude drops, direction does
    not. A clamp that bends a straight strafe into a curve teaches the world
    model a correlation that does not exist on the robot.
    """
    rng = np.random.default_rng(1)
    for _ in range(300):
        v = rng.uniform(-1.0, 1.0, size=3) * np.array([1.0, 1.0, 4.0])
        out = KIN.clamp_achievable(v, 14.0)
        scale = np.linalg.norm(out) / np.linalg.norm(v)
        assert scale <= 1.0 + 1e-12
        assert np.allclose(out, v * scale, atol=1e-12), f"{v} -> {out} is not parallel"


def test_smoothed_random_actions_are_achievable():
    env = _env()
    actions = env.smoothed_random_actions(1500, np.random.default_rng(2))
    peaks = np.max(np.abs(actions @ KIN.J.T) / KIN.geom.wheel_radius, axis=1)
    assert peaks.max() <= env.WHEEL_SPEED_MAX * TOL, (
        f"random sampler peak wheel speed {peaks.max():.3f} rad/s"
    )
    # and it must actually use the budget, or the test is vacuous
    assert peaks.max() > 0.8 * env.WHEEL_SPEED_MAX, (
        f"random actions never approach the wheel limit (peak {peaks.max():.2f}); "
        f"this test would not notice a broken clamp"
    )


def test_wall_avoiding_action_is_achievable():
    env = _env()
    rng = np.random.default_rng(3)
    base = env.smoothed_random_actions(200, rng)
    for i, a in enumerate(base):
        # poses well outside the margin, which is where the push term bites
        obs = {"pose_gt": np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9),
                                    rng.uniform(-np.pi, np.pi)])}
        _assert_achievable(env.wall_avoiding_action(obs, a), env.WHEEL_SPEED_MAX,
                           "wall_avoiding_action")


def test_scripted_expert_action_is_achievable():
    env = _env()
    collect = import_or_skip("sim.collect")
    rng = np.random.default_rng(4)
    for _ in range(400):
        obs = {"pose_gt": rng.uniform(-0.9, 0.9, size=3)}
        goal = rng.uniform(-0.9, 0.9, size=3)
        _assert_achievable(collect.expert_action(obs, goal), env.WHEEL_SPEED_MAX,
                           "collect.expert_action")


def test_reactive_controller_action_is_achievable():
    env = _env()
    mppi = import_or_skip("planning.mppi", " (it imports models/, which may be mid-edit)")
    rng = np.random.default_rng(5)
    for _ in range(400):
        pose = rng.uniform(-0.9, 0.9, size=3)
        goal = rng.uniform(-0.9, 0.9, size=3)
        _assert_achievable(mppi.reactive_action(pose, goal, KIN), env.WHEEL_SPEED_MAX,
                           "reactive_action")


def test_mppi_plan_is_achievable():
    """
    The planner is the newest action producer and the one most likely to leave
    the distribution: MPPI is an optimiser, so it pushes toward saturation.
    """
    env = _env()
    mppi = import_or_skip("planning.mppi", " (it imports models/, which may be mid-edit)")
    gate = import_or_skip("planning.gate_task")

    rng = np.random.default_rng(6)
    task = gate.GateTask(rng)
    predictor = mppi.AnalyticPredictor(KIN, gains=[1.0, 0.88, 0.88], tau=0.12)
    planner = mppi.MPPI(predictor, task.cost, KIN, seed=0)
    hist = np.zeros((1, 14), dtype=np.float32)
    hist[0, 3:6] = [0.3, -0.1, 0.4]           # a moving start, not a resting one

    pose = task.start.copy()
    for _ in range(5):
        plan = planner.plan(pose, hist)
        for a in plan:
            _assert_achievable(a, env.WHEEL_SPEED_MAX, "MPPI.plan")
        pose = KIN.integrate_pose(pose, plan[0], env.CONTROL_DT)
        planner.shift(1)


# ── The incident itself: clipping vs scaling ────────────────────────────────

def test_independent_per_wheel_clipping_is_what_broke_the_symmetry():
    """
    Pins the mechanism, not just the symptom. If anyone reaches for
    `np.clip(wheels, -max, max)` again, this is the number that says no.
    """
    v = np.array([0.5, 0.5, 2.0])                     # the P0 example: needs 33 rad/s
    wheels = KIN.ik(v)
    assert np.abs(wheels).max() > 30.0, np.abs(wheels).max()

    clipped = np.clip(wheels, -14.0, 14.0)
    scaled = KIN.ik(KIN.clamp_achievable(v, 14.0))

    assert abs(KIN.scrub(clipped)) > 1.0, (
        f"independent clipping should dump motion into the scrub direction "
        f"(got {KIN.scrub(clipped):.3f})"
    )
    assert abs(KIN.scrub(scaled)) < 1e-12, (
        f"one-factor scaling must stay out of the null space (got {KIN.scrub(scaled):.3e})"
    )
    # Scaling keeps the MIX of the three modes exactly; clipping does not, so
    # the commanded diagonal-plus-spin comes out as a different manoeuvre.
    ratio_scaled = KIN.fk(scaled) / v
    ratio_clipped = KIN.fk(clipped) / v
    assert ratio_scaled.max() - ratio_scaled.min() < 1e-12, (
        f"one-factor scaling must deliver every mode at the same fraction, got "
        f"{ratio_scaled}"
    )
    assert ratio_clipped.max() / ratio_clipped.min() > 1.05, (
        f"independent clipping should distort the mode mix, got {ratio_clipped}"
    )


def test_mean_scrub_over_the_action_box_stays_near_zero_when_clamped():
    """The headline number from the incident: 1.004 rad/s against a proper 0.07."""
    env = _env()
    rng = np.random.default_rng(7)
    box = rng.uniform(-1.0, 1.0, size=(500, 3)) * env.ACTION_LIMITS
    clamped = np.mean([abs(KIN.scrub(KIN.ik(env.clamp_action(a)))) for a in box])
    clipped = np.mean([abs(KIN.scrub(np.clip(KIN.ik(a), -14.0, 14.0))) for a in box])
    assert clamped < 1e-9, f"mean |scrub| over clamped actions was {clamped:.3e}"
    assert clipped > 0.3, (
        f"per-wheel clipping only produced {clipped:.3f} rad/s of mean scrub; this "
        f"test can no longer tell the two policies apart"
    )


def test_slip_model_saturation_scales_all_four_wheels_together():
    slip_mod = import_or_skip("sim.slip_model")
    slip = slip_mod.SlipModel(surface="hard", kinematics=KIN, seed=0)
    out = slip.step(np.array([0.5, 0.5, 2.0]), 0.05)
    assert out["saturated"] is True
    assert abs(KIN.scrub(out["wheel_meas"])) < 0.25, (
        f"|scrub| {abs(KIN.scrub(out['wheel_meas'])):.3f} rad/s on a saturating "
        f"command - the saturation handler is clipping wheels independently again"
    )
    assert np.abs(out["wheel_meas"]).max() < 14.0 * 1.1


def test_an_exactly_clamped_action_does_not_register_as_saturated():
    """
    The floating-point half of the same bug, recorded in slip_model.py: callers
    clamp to exactly WHEEL_SPEED_MAX, so an exact-equality saturation test
    reported ~43% of ticks as saturated when none were.
    """
    env = _env()
    slip_mod = import_or_skip("sim.slip_model")
    slip = slip_mod.SlipModel(surface="hard", kinematics=KIN, seed=0)
    action = env.clamp_action([0.5, 0.5, 2.0])
    assert np.isclose(_peak(action), env.WHEEL_SPEED_MAX), _peak(action)
    assert slip.step(action, 0.05)["saturated"] is False


# ── End to end: what lands in a collected episode ──────────────────────────

def test_collected_episode_has_no_saturated_ticks():
    df = sim_episode(n_ticks=300, surface="hard", seed=11).df
    rate = float(df["saturated"].mean())
    assert rate < 0.01, (
        f"{rate:.1%} of collected ticks saturated. P0 ran at 72% before actions "
        f"were clamped into the polytope, and reported 0% after."
    )


def test_collected_episode_scrub_is_small():
    df = sim_episode(n_ticks=300, surface="hard", seed=11).df
    mean_scrub = float(np.abs(df["scrub"].to_numpy()).mean())
    assert mean_scrub < 0.25, (
        f"mean |scrub| = {mean_scrub:.3f} rad/s. Proper value is ~0.07 (per-wheel "
        f"gain asymmetry only); the per-wheel-clipping bug read 1.004."
    )
    assert mean_scrub > 1e-4, (
        f"mean |scrub| = {mean_scrub:.2e} is too clean - the slip model is supposed "
        f"to inject wheel gain asymmetry, which shows up here"
    )


if __name__ == "__main__":
    raise SystemExit(main(globals()))
