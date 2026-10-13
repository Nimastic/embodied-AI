"""
The M4 gate cost function.

Protects against
----------------
BUG 6  A barrier margin wider than the gate.

       `BARRIER_MARGIN` must be strictly smaller than (narrowest gate
       half-width - CLEARANCE). At margin 0.05 the barrier activated below
       137 mm while gate half-widths are 105-140 mm, so a PERFECT centre-line
       pass was already inside the penalty zone - from both posts at once. The
       two gradients cancelled, the planner had no steering signal through the
       gate, and it optimised the goal term alone. With the cost saturated, no
       dynamics model could matter.

       The run reported the learned model as 2.3x more accurate and marginally
       worse than naive (72.5% vs 75%): "a striking, publishable-sounding
       result". Fixing the margin made success rates monotonic in model quality
       immediately.
       (experiments/M4-world-model-planning/README.md, "a bug that nearly
       became a finding")

       "A clean negative result and a broken experiment produce identical
       summary tables. The difference only appears when you ask why."

This is a pure-numpy module, so these tests need neither MuJoCo nor torch.
"""

import numpy as np

from _harness import main
from planning.gate_task import (BARRIER_MARGIN, CLEARANCE, POST_RADIUS, ROBOT_HALF_WIDTH,
                                GateTask)

N_SAMPLES = 2000          # enough to pin the sampled half-width range


def _sampled_tasks(n=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    return [GateTask(rng) for _ in range(n)]


def _barrier(task, poses, margin=BARRIER_MARGIN):
    """The barrier term of `GateTask.cost`, isolated."""
    d = task.post_distances(np.asarray(poses))
    return float(np.sum(np.clip(CLEARANCE + margin - d, 0.0, None) ** 2))


def _centre_line(task, n=21, span=0.35, offset=0.0):
    """Poses along the gate normal, `offset` metres off the centre line."""
    ts = np.linspace(-span, span, n)
    return np.array([
        [*(task.gate_xy + task.normal * t + task.along * offset), task.gate_angle]
        for t in ts
    ])


# ── The structural invariant ───────────────────────────────────────────────

def test_clearance_is_the_chassis_half_width_plus_the_post_radius():
    assert np.isclose(CLEARANCE, ROBOT_HALF_WIDTH + POST_RADIUS), CLEARANCE


def test_barrier_margin_fits_inside_the_narrowest_sampled_gate():
    """
    The invariant the comment in gate_task.py states, asserted over the gate
    distribution the task actually samples rather than over the numbers that
    happened to be in the comment.
    """
    half_widths = np.array([t.half_width for t in _sampled_tasks()])
    slack = half_widths.min() - CLEARANCE
    assert BARRIER_MARGIN < slack, (
        f"BARRIER_MARGIN = {BARRIER_MARGIN*1000:.1f} mm but the narrowest sampled "
        f"gate leaves only {slack*1000:.1f} mm between the chassis and the post "
        f"({half_widths.min()*1000:.1f} mm half-width - {CLEARANCE*1000:.1f} mm "
        f"clearance). The two posts' barrier zones then overlap across the whole "
        f"opening and the planner loses its steering signal through the gate."
    )
    assert slack > 0.0, "the narrowest sampled gate is narrower than the chassis"


def test_every_sampled_gate_admits_an_unpenalised_pass():
    """
    Per-instance version of the above: for EVERY gate the task can generate, a
    perfect centre-line traversal must cost zero barrier. This is the exact
    property that was false in the broken M4 run.
    """
    worst = None
    for task in _sampled_tasks(500, seed=1):
        cost = _barrier(task, _centre_line(task))
        if cost > 0.0:
            worst = (task.half_width, cost)
            break
    assert worst is None, (
        f"a centre-line pass through a {worst[0]*1000:.1f} mm half-width gate is "
        f"already penalised (barrier {worst[1]:.5f}); every trajectory then pays a "
        f"constant toll it cannot avoid"
    )


def test_the_broken_margin_would_be_caught_by_the_test_above():
    """
    Teeth. With the margin that caused the incident (0.05), the narrowest gates
    penalise a perfect pass - so the invariant above is doing real work and not
    passing vacuously.
    """
    tasks = sorted(_sampled_tasks(500, seed=1), key=lambda t: t.half_width)
    narrowest = tasks[0]
    assert _barrier(narrowest, _centre_line(narrowest), margin=0.05) > 0.0, (
        "the historical 0.05 margin no longer penalises a centre-line pass; has "
        "the gate distribution been widened? If so, this suite is no longer "
        "testing the configuration M4 ran."
    )
    assert _barrier(narrowest, _centre_line(narrowest)) == 0.0


# ── The barrier still has to speak where it matters ────────────────────────

def test_the_barrier_grows_as_the_robot_approaches_a_post():
    """
    A margin small enough to be silent at the centre line must still be loud
    near a post, or it is not a barrier at all.
    """
    task = sorted(_sampled_tasks(200, seed=2), key=lambda t: t.half_width)[0]
    offsets = np.linspace(0.0, task.half_width - 0.002, 8)
    costs = [_barrier(task, _centre_line(task, offset=o)) for o in offsets]
    assert costs[0] == 0.0, costs[0]
    assert costs[-1] > 0.0, f"grazing the post costs nothing: {costs}"
    assert all(b >= a for a, b in zip(costs, costs[1:])), (
        f"barrier is not monotonic in lateral offset: {np.round(costs, 6)}"
    )


def test_a_collision_course_costs_more_than_a_centre_line_pass():
    """
    The end-to-end ranking MPPI depends on, through the real `cost()` including
    the goal and heading terms.
    """
    task = sorted(_sampled_tasks(200, seed=3), key=lambda t: t.half_width)[0]
    centre = _centre_line(task)[None]
    into_post = np.array([
        [*(task.gate_xy + task.normal * t + task.along * (task.half_width - 0.005)),
         task.gate_angle]
        for t in np.linspace(-0.35, 0.35, 21)
    ])[None]
    actions = np.zeros((1, centre.shape[1], 3))
    cost_centre = float(task.cost(centre, actions)[0])
    cost_post = float(task.cost(into_post, actions)[0])
    assert cost_post > cost_centre, (
        f"a path that grazes a post ({cost_post:.3f}) must cost more than the "
        f"centre line ({cost_centre:.3f})"
    )


def test_collision_is_scored_against_the_same_clearance_the_barrier_uses():
    """
    If `collided()` and the barrier disagree about where the posts are, the
    planner optimises one geometry and the scorer grades another.
    """
    task = sorted(_sampled_tasks(200, seed=4), key=lambda t: t.half_width)[0]
    just_outside = np.array([[*(task.post_l + task.normal * (CLEARANCE + 1e-4)),
                              task.gate_angle]])
    just_inside = np.array([[*(task.post_l + task.normal * (CLEARANCE - 1e-3)),
                             task.gate_angle]])
    assert not bool(task.collided(just_outside)[0])
    assert bool(task.collided(just_inside)[0])
    assert _barrier(task, just_inside) > _barrier(task, just_outside)


def test_the_gate_distribution_is_still_the_one_the_sweep_was_run_with():
    """
    M4's numbers only mean something for the gate range it reports: 18-53 mm of
    clear air per side. Two earlier, wider settings "measured nothing" - every
    condition including model-free reactive scored 100%. If the range moves,
    the recorded success rates stop being comparable.
    """
    half_widths = np.array([t.half_width for t in _sampled_tasks()])
    clear_air_mm = (half_widths - CLEARANCE) * 1000.0
    assert 17.0 < clear_air_mm.min() < 20.0, clear_air_mm.min()
    assert 51.0 < clear_air_mm.max() < 54.0, clear_air_mm.max()


def test_start_pose_is_before_the_gate_and_goal_is_beyond_it():
    for task in _sampled_tasks(300, seed=5):
        assert not task.past_gate(task.start), "start pose is already past the gate"
        assert task.past_gate(task.goal), "goal is not beyond the gate"
        assert not task.collided(task.start[None])[0], "start pose is in collision"


# ── Noise mistaken for signal ──────────────────────────────────────────────
#
# The other way M4 nearly produced a finding: "A 20-trial run showed the
# learned model uniquely failing at 400 ms (85%, 15% collisions). At 40 trials
# it was 95%. Pure noise, and it would have made a confident story about model
# exploitation."
#
# These three tests encode which of M4's reported gaps the sample size can
# actually support, so the next sweep cannot quietly promote one trial into a
# conclusion. They are arithmetic, not a measurement - they run in microseconds
# and they are the cheapest available defence against the failure mode.

def _two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided two-proportion z-test, normal approximation, no scipy."""
    import math

    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 1.0
    z = abs(k1 / n1 - k2 / n2) / se
    return math.erfc(z / math.sqrt(2.0))


def test_the_learned_over_calibrated_gap_is_not_significant_at_forty_trials():
    """M4 at 750 ms: calibrated 70.0% vs learned 72.5% - one trial in forty."""
    p = _two_proportion_p(29, 40, 28, 40)
    assert p > 0.2, (
        f"p = {p:.3f}: M4 reports the +2.5 point learned-over-calibrated gap as "
        f"within noise, and the write-up's conclusion depends on it being read "
        f"that way"
    )


def test_the_calibration_gap_is_significant_at_forty_trials():
    """M4 at 750 ms: naive 47.5% vs calibrated 70.0% - this one is real."""
    p = _two_proportion_p(19, 40, 28, 40)
    assert p < 0.05, f"p = {p:.3f}: the +22.5 point calibration gap should survive"


def test_the_twenty_trial_scare_was_within_noise():
    """17/20 vs 38/40 at the 400 ms replan interval: the retracted finding."""
    p = _two_proportion_p(17, 20, 38, 40)
    assert p > 0.05, f"p = {p:.3f}"


if __name__ == "__main__":
    raise SystemExit(main(globals()))
