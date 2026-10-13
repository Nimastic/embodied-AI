"""
The dataset alignment rule.

Protects against
----------------
BUG 3  The off-by-one that trains a model to predict the past.

       `body_vel_avg[t]` is the average body velocity over the tick that
       ENDED at `pose_gt[t]`. To predict forward from `pose_gt[t]` you must
       integrate `body_vel_avg[t+1 ...]`. Off by one and the model learns to
       predict a tick it has already observed: the loss is beautiful and the
       model is worthless.

       Measured in software/README.md: correct offset reproduces `pose_gt`
       1.0 s ahead to 0.000 mm, off by one to 14.6 mm. A 14 mm error is
       indistinguishable from "our analytic baseline is 13.4 mm" by eye, which
       is exactly why this needs to be a test and not a habit.

BUG 3b The "average, never instantaneous" rule. Logging the end-of-tick
       instantaneous velocity instead of the tick average costs 8.9 mm of pure
       discretisation error over a 1 s horizon - two thirds of the P0 analytic
       baseline, and nothing whatsoever to do with world modelling.

The README asks for exactly this as a unit test on any new dataset, sim or
real: "If integrating the logged velocity does not return the logged pose,
something upstream is wrong and no model will fix it."
"""

import numpy as np

from _harness import main, reset_boundaries, sim_episode
from kinematics.mecanum import MecanumKinematics

KIN = MecanumKinematics()
HORIZON = 20          # 1.0 s at 20 Hz, the M2 prediction horizon
DT = 0.05


def _episode():
    ep = sim_episode(n_ticks=300, surface="hard", seed=11)
    pose = np.stack(ep.df["pose_gt"].to_numpy()).astype(np.float64)
    vel_avg = np.stack(ep.df["body_vel_avg"].to_numpy()).astype(np.float64)
    vel_inst = np.stack(ep.df["body_vel_gt"].to_numpy()).astype(np.float64)
    return pose, vel_avg, vel_inst


def _usable_starts(pose):
    """Window starts that do not span an in-episode `env.reset()` teleport."""
    bad = reset_boundaries(pose)
    return [j for j in range(len(pose) - HORIZON)
            if not any((j + k) in bad for k in range(HORIZON))]


def _integrate(pose0, vels):
    p = np.asarray(pose0, dtype=float).copy()
    for v in vels:
        p = KIN.integrate_pose(p, v, DT)
    return p


def _errors(pose, vel, offset):
    """
    Position error, in mm, of integrating `HORIZON` velocity rows starting at
    index `j + offset`, against the logged pose at `j + HORIZON`.

    offset = 1 is the documented rule. offset = 0 is the bug.
    """
    out = []
    for j in _usable_starts(pose):
        pred = _integrate(pose[j], vel[j + offset: j + offset + HORIZON])
        out.append(np.linalg.norm(pred[:2] - pose[j + HORIZON][:2]))
    return np.array(out) * 1000.0


# ── The episode has to actually contain motion, or everything below is vacuous

def test_the_test_episode_moves():
    pose, _, _ = _episode()
    starts = _usable_starts(pose)
    assert len(starts) > 100, f"only {len(starts)} usable windows"
    travel = np.array([
        np.linalg.norm(pose[j + HORIZON][:2] - pose[j][:2]) for j in starts
    ]) * 1000.0
    assert np.median(travel) > 50.0, (
        f"median 1.0 s displacement only {np.median(travel):.1f} mm - a stationary "
        f"robot makes every alignment check pass for free"
    )


# ── BUG 3 ───────────────────────────────────────────────────────────────────

def test_logged_velocity_integrates_back_to_the_logged_pose():
    pose, vel_avg, _ = _episode()
    err = _errors(pose, vel_avg, offset=1)
    assert err.max() < 0.1, (
        f"integrating body_vel_avg[t+1 ...] from pose_gt[t] missed pose_gt[t+H] by "
        f"median {np.median(err):.4f} mm / max {err.max():.4f} mm. The schema "
        f"guarantees ~0.000 mm; anything else means the velocity target and the "
        f"pose label disagree and no model can fix it."
    )


def test_logged_velocity_integrates_back_to_the_logged_heading():
    pose, vel_avg, _ = _episode()
    worst = 0.0
    for j in _usable_starts(pose):
        pred = _integrate(pose[j], vel_avg[j + 1: j + 1 + HORIZON])
        worst = max(worst, abs((pred[2] - pose[j + HORIZON][2] + np.pi)
                               % (2 * np.pi) - np.pi))
    assert worst < 1e-4, f"heading reproduction off by {worst:.2e} rad"


def test_off_by_one_alignment_is_detectably_wrong():
    """
    The teeth of the test above. If this one ever starts reporting a small
    number, the exact-reproduction check has stopped being able to catch the
    bug it exists for (e.g. because the episode stopped moving).
    """
    pose, vel_avg, _ = _episode()
    good = _errors(pose, vel_avg, offset=1)
    bad = _errors(pose, vel_avg, offset=0)
    assert np.median(bad) > 5.0, (
        f"off-by-one integration only misses by {np.median(bad):.2f} mm median - "
        f"this dataset cannot distinguish the correct alignment from the bug"
    )
    assert np.median(bad) > 50.0 * max(np.median(good), 1e-6), (
        f"off-by-one {np.median(bad):.3f} mm vs aligned {np.median(good):.6f} mm"
    )


# ── BUG 3b: average, never instantaneous ────────────────────────────────────

def test_instantaneous_velocity_is_a_worse_integration_target_than_the_average():
    """
    `body_vel_gt` is the end-of-tick instantaneous velocity and `body_vel_avg`
    is the tick average. Only the average inverts the integrator exactly. The
    schema says sampling instantaneously costs ~8.9 mm over 1 s; this pins that
    the two columns have not been swapped.
    """
    pose, vel_avg, vel_inst = _episode()
    avg_err = _errors(pose, vel_avg, offset=1)
    inst_err = _errors(pose, vel_inst, offset=1)
    assert np.median(inst_err) > 2.0, (
        f"instantaneous velocity integrates to {np.median(inst_err):.2f} mm - "
        f"suspiciously good. Are body_vel_gt and body_vel_avg the same column?"
    )
    assert np.median(inst_err) > 100.0 * max(np.median(avg_err), 1e-6), (
        f"instantaneous {np.median(inst_err):.2f} mm vs average "
        f"{np.median(avg_err):.6f} mm"
    )


def test_odometry_drifts_away_from_ground_truth_rather_than_tracking_it():
    """
    `pose_odom` is integrated wheel FK and drifts BY DESIGN - the wheels turn
    further than the robot travels. If odometry ever matches `pose_gt` closely,
    the slip model has stopped injecting contact-side loss and every learned-vs
    -analytic result in this project becomes meaningless.
    """
    ep = sim_episode(n_ticks=300, surface="hard", seed=11)
    pose = np.stack(ep.df["pose_gt"].to_numpy()).astype(np.float64)
    odom = np.stack(ep.df["pose_odom"].to_numpy()).astype(np.float64)
    bad = reset_boundaries(pose)
    end = min(bad) if bad else len(pose) - 1          # odometry is not re-seeded
    drift = np.linalg.norm(odom[end][:2] - pose[end][:2]) * 1000.0
    assert drift > 10.0, (
        f"wheel odometry drifted only {drift:.1f} mm over {end} ticks - FK is "
        f"supposed to over-read, and the overhead fiducial is supposed to be the "
        f"only honest pose label"
    )


if __name__ == "__main__":
    raise SystemExit(main(globals()))
