"""
Slip-model structure: the residual must be signal, not noise.

Protects against
----------------
BUG 7  The simulator's imperfection degenerating into zero-mean noise, or its
       per-mode ordering silently inverting.

       slip_model.py's one rule: "The residual must be a structured function of
       state and action - speed, direction, surface, pack voltage, wheel angle -
       and NOT zero-mean noise. A learned model can only beat the analytic
       baseline if there is signal to find." If the structure goes, every M2/M4
       result becomes "no model can beat FK here" while still producing a full
       table of plausible millimetres.

       The ordering is load-bearing in three write-ups: strafe is the worst
       mode (rollers loaded in their compliant direction, 80.9% delivered vs
       91.8% forward), and `mat` is worse than `hard` everywhere. P0's very
       first run reported forward at 7.8% and strafe at 76.8% - the ordering
       inverted - because the step response was measured against a wall.
       (experiments/P0-sim-baseline/README.md)

Also pins the asymmetry the whole project rests on: the encoders cannot see
contact-side loss, so wheel odometry OVER-reads, which is why the overhead
fiducial is the only honest pose label.
"""

from dataclasses import replace

import numpy as np

from _harness import import_or_skip, main
from kinematics.mecanum import MecanumKinematics

KIN = MecanumKinematics()
DT = 0.05
MODES = {
    "forward": np.array([1.0, 0.0, 0.0]) * 0.4,
    "strafe": np.array([0.0, 1.0, 0.0]) * 0.4,
    "rotate": np.array([0.0, 0.0, 1.0]) * 1.5,
}


def _slip_mod():
    return import_or_skip("sim.slip_model")


def _model(surface="hard", seed=0, noise=True):
    mod = _slip_mod()
    slip = mod.SlipModel(surface=surface, kinematics=KIN, seed=seed)
    if not noise:
        # Rebind on the instance; SURFACES holds shared dataclass objects and
        # mutating one in place would leak into every other test.
        slip.params = replace(slip.params, noise_std=0.0)
    return slip


def _delivered(surface, cmd, n=60, seed=0):
    """Mean fraction of the commanded body velocity actually delivered."""
    slip = _model(surface, seed=seed)
    got = [np.linalg.norm(slip.step(cmd, DT)["body_vel"]) for _ in range(n)]
    return float(np.mean(got) / np.linalg.norm(cmd))


# ── The per-mode ordering ──────────────────────────────────────────────────

def test_strafe_is_the_worst_mode_on_both_surfaces():
    for surface in ("hard", "mat"):
        delivered = {m: _delivered(surface, cmd) for m, cmd in MODES.items()}
        assert delivered["strafe"] < delivered["rotate"] < delivered["forward"], (
            f"{surface}: delivered fractions {({k: round(v, 3) for k, v in delivered.items()})} "
            f"- expected forward > rotate > strafe. P0 measured 91.8% / 86.7% / 80.9%, "
            f"and its first (wall-contact) run inverted them."
        )


def test_mat_is_worse_than_hard_in_every_mode():
    for mode, cmd in MODES.items():
        hard, mat = _delivered("hard", cmd), _delivered("mat", cmd)
        assert mat < hard, (
            f"{mode}: mat delivered {mat:.3f} vs hard {hard:.3f}. `mat` is the M6 "
            f"domain-shift condition and has to be the harder surface everywhere."
        )


def test_mat_is_disproportionately_worse_at_strafing():
    """Rollers bind in soft pile: the gap must be widest in the worst mode."""
    gaps = {m: _delivered("hard", c) - _delivered("mat", c) for m, c in MODES.items()}
    assert gaps["strafe"] == max(gaps.values()), {k: round(v, 3) for k, v in gaps.items()}


def test_surface_parameters_themselves_encode_the_ordering():
    """
    Catches a transposed tuple before it ever reaches a measurement: the mode
    order is (forward, strafe, rotate) in both `eta0` and `eta_speed`.
    """
    mod = _slip_mod()
    for name, params in mod.SURFACES.items():
        fwd, strafe, rot = params.eta0
        assert strafe < rot < fwd, f"{name}: eta0 = {params.eta0}"
        fwd_s, strafe_s, rot_s = params.eta_speed
        assert strafe_s > fwd_s and strafe_s > rot_s, f"{name}: eta_speed = {params.eta_speed}"
        assert all(0.0 < e <= 1.0 for e in params.eta0), params.eta0
    hard, mat = mod.SURFACES["hard"], mod.SURFACES["mat"]
    assert all(m < h for m, h in zip(mat.eta0, hard.eta0)), (mat.eta0, hard.eta0)
    assert all(m > h for m, h in zip(mat.eta_speed, hard.eta_speed)), (
        mat.eta_speed, hard.eta_speed
    )


def test_characterise_reports_the_same_ordering():
    """The helper P0 actually ran, so the published table stays reproducible."""
    mod = _slip_mod()
    for surface in ("hard", "mat"):
        rows = {r["mode"]: r["mean_rel_error"] for r in mod.characterise(surface, n=400)["modes"]}
        assert rows["strafe"] > rows["rotate"] > rows["forward"], (
            {k: round(v, 4) for k, v in rows.items()}
        )


# ── Signal, not noise ──────────────────────────────────────────────────────

def test_the_residual_is_deterministic_given_a_seed():
    """"Deterministic given a seed, so an episode is exactly reproducible from
    its manifest." Without this, no experiment in this project is reproducible."""
    cmds = [np.array([0.3, -0.2, 0.8]) * s for s in np.linspace(0.2, 1.0, 10)]
    a = [_model(seed=4).step(c, DT) for c in cmds]
    b = [_model(seed=4).step(c, DT) for c in cmds]
    for x, y in zip(a, b):
        for key in ("wheel_meas", "body_vel", "odom_vel"):
            assert np.array_equal(x[key], y[key]), key


def test_the_residual_is_structured_not_zero_mean_noise():
    """
    The rule from the module docstring, as a number: the systematic shortfall
    for a fixed command must dwarf the spread across random seeds. If noise
    ever dominates, "no model can beat FK here" becomes the honest answer and
    the training environment is useless.
    """
    cmd = np.array([0.0, 0.4, 0.0])
    achieved = np.array([_model(seed=s).step(cmd, DT)["body_vel"] for s in range(24)])
    signal = float(np.linalg.norm(achieved.mean(axis=0) - cmd))
    noise = float(np.linalg.norm(achieved.std(axis=0)))
    assert signal > 20.0 * noise, (
        f"systematic shortfall {signal*1000:.2f} mm/s vs seed-to-seed spread "
        f"{noise*1000:.2f} mm/s - the residual has become noise"
    )
    assert signal > 0.02, f"shortfall {signal:.4f} m/s is too small to learn from"


def test_slip_grows_with_speed():
    """
    `eta_speed`: efficiency falls with speed, which is what a learned model
    conditioned on velocity can represent and a single fitted scalar cannot.
    This is reason 2 in P0's "why the calibrated model still fails".
    """
    slow = _delivered("hard", np.array([0.0, 0.08, 0.0]))
    fast = _delivered("hard", np.array([0.0, 0.45, 0.0]))
    assert fast < slow - 0.02, (
        f"strafe delivered {slow:.3f} at 0.08 m/s and {fast:.3f} at 0.45 m/s - "
        f"slip must grow with speed, or error-vs-horizon curves stop steepening "
        f"on fast episodes"
    )


def test_roller_ripple_is_periodic_in_wheel_angle_not_random():
    """
    Ripple is predictable from integrated encoder counts and invisible to a
    memoryless model - reason 3 in P0's residual analysis. With noise off, a
    constant command must still produce a bounded oscillation, not a constant
    and not a random walk.
    """
    mod = _slip_mod()
    slip = _model("hard", seed=0, noise=False)
    cmd = np.array([0.25, 0.0, 0.0])
    speeds = np.array([slip.step(cmd, DT)["wheel_meas"][0] for _ in range(200)])
    spread = float(speeds.max() - speeds.min())
    scale = float(np.abs(speeds).mean())
    assert spread > 0.005 * scale, f"no ripple at all (spread {spread:.5f} rad/s)"
    assert spread < 4.0 * mod.SURFACES["hard"].ripple_amp * scale, (
        f"ripple spread {spread:.4f} rad/s exceeds what ripple_amp "
        f"{mod.SURFACES['hard'].ripple_amp} can explain - something else is drifting"
    )
    # periodic, so it keeps returning near its own mean rather than wandering off
    assert abs(speeds[:100].mean() - speeds[100:].mean()) < 0.05 * scale


def test_pack_voltage_changes_what_the_wheels_deliver():
    """
    "batt_v is an input, not metadata. Pack sag changes achieved wheel velocity
    for identical commands. Leaving it out means part of your dynamics is
    unobservable and the learned model gets blamed for it."
    """
    cmd = np.array([0.35, 0.0, 0.0])
    full = _model("hard", seed=0, noise=False)
    full.reset(pack_soc=1.0)
    flat = _model("hard", seed=0, noise=False)
    flat.reset(pack_soc=0.1)
    v_full = np.linalg.norm(full.step(cmd, DT)["body_vel"])
    v_flat = np.linalg.norm(flat.step(cmd, DT)["body_vel"])
    assert v_flat < v_full, f"flat pack delivered {v_flat:.4f} vs full {v_full:.4f} m/s"
    assert flat.step(cmd, DT)["batt_v"] < full.step(cmd, DT)["batt_v"]


def test_encoders_cannot_see_the_contact_loss_so_odometry_over_reads():
    """
    The asymmetry the whole project rests on: slip is injected between wheel
    and floor, so `fk(wheel_meas)` is larger than the achieved body velocity.
    If this ever flips, `pose_odom` stops drifting and the overhead fiducial
    stops being necessary.
    """
    slip = _model("hard", seed=0)
    over = []
    for mode, cmd in MODES.items():
        out = slip.step(cmd, DT)
        over.append(np.linalg.norm(out["odom_vel"]) - np.linalg.norm(out["body_vel"]))
        assert over[-1] > 0.0, f"{mode}: odometry under-read by {over[-1]:.4f}"
    assert np.mean(over) > 1e-3


def test_an_unknown_surface_is_rejected_rather_than_silently_defaulted():
    mod = _slip_mod()
    try:
        mod.SlipModel(surface="carpet")
    except ValueError:
        return
    raise AssertionError("an unknown surface name must not silently fall back to 'hard'")


if __name__ == "__main__":
    raise SystemExit(main(globals()))
