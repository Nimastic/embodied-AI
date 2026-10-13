"""
Baseline fairness: the analytic predictor must be handed the same measured
velocity the learned model gets.

Protects against
----------------
BUG 5  `first_order_lag(..., v0=None)` starts its filter at zero, which asserts
       the robot is AT REST at the start of every prediction window. It is not,
       for almost all of them - while the learned model is handed the measured
       current velocity as an input.

       M2's first run reported the calibrated baseline at 13.22 mm and the
       learned model at 3.79 mm: a 71% win. Passing `v0` moved the baseline to
       5.27 mm, a 2.5x gain from one line. The headline survived but the
       *shape* of the finding changed - from "learned models win big" to
       "learned models win big in-distribution and not at all out of it".
       (experiments/M2-state-predictor/README.md, "the bug that nearly produced
       a fake result")

The trap worth understanding, and the reason the first test below refits:
a calibration FITTED under the v0=None handicap hides it. The lag constant
inflates to absorb the missing initial velocity (P0 fitted 110-130 ms against a
true actuator tau of ~60 ms), after which swapping v0 back in changes almost
nothing. The handicap is only visible when each variant is allowed its own best
parameters - which is exactly the comparison M2 ran.

"Before trusting any M4/M5 result, check that every predictor receives exactly
the same information."
"""

import inspect

import numpy as np

from _harness import SOFTWARE_DIR, expected_failure, import_or_skip, main, prediction_windows
from kinematics.mecanum import MecanumKinematics

KIN = MecanumKinematics()
DT = 0.05


def _lag():
    import_or_skip("mujoco", " (mecanum_env imports it at module level)")
    return import_or_skip("sim.mecanum_env")


# ── The mechanism, as a unit ────────────────────────────────────────────────

def test_v0_none_is_identical_to_asserting_the_robot_is_at_rest():
    env = _lag()
    seq = np.tile([0.3, -0.1, 0.5], (10, 1))
    cold = env.first_order_lag(seq, 0.12)
    rest = env.first_order_lag(seq, 0.12, v0=np.zeros(3))
    assert np.allclose(cold, rest), "v0=None must be documented as 'starts from rest'"


def test_v0_is_actually_used_by_the_filter():
    env = _lag()
    seq = np.tile([0.3, 0.0, 0.0], (10, 1))
    moving = env.first_order_lag(seq, 0.12, v0=np.array([0.3, 0.0, 0.0]))
    assert np.allclose(moving, seq), (
        "a robot already travelling at the commanded velocity should stay there; "
        f"got first step {moving[0]}"
    )
    cold = env.first_order_lag(seq, 0.12)
    alpha = 1.0 - np.exp(-DT / 0.12)
    assert np.allclose(cold[0], alpha * seq[0]), cold[0]
    assert np.linalg.norm(cold[0] - moving[0]) > 0.1, (
        "the two initialisations must differ materially on the first step"
    )


def test_zero_tau_ignores_v0_entirely():
    """tau<=0 is the 'naive' predictor: no lag, so no initial condition."""
    env = _lag()
    seq = np.tile([0.3, -0.1, 0.5], (5, 1))
    assert np.allclose(env.first_order_lag(seq, 0.0), seq)
    assert np.allclose(env.first_order_lag(seq, 0.0, v0=np.ones(3)), seq)


# ── The consequence, end to end ────────────────────────────────────────────

_FITS: dict[tuple, tuple] = {}


def _fit_and_score(windows, use_v0):
    """
    Coarse version of `mecanum_env.fit_calibration`: three per-mode gains plus
    one lag constant, by coordinate descent. Deliberately the same shape as the
    real fit, just on a coarser grid so it runs in under a second. Cached, so
    the two tests below share one fit each.
    """
    key = (use_v0, len(windows))
    if key in _FITS:
        return _FITS[key]
    env = _lag()

    def mean_err(gains, tau):
        total = 0.0
        for p0, v0, seq, p1 in windows:
            vels = env.first_order_lag(seq * gains, tau, v0=(v0 if use_v0 else None))
            pred = KIN.predict(p0, vels, DT)[-1]
            total += float(np.linalg.norm(pred[:2] - p1[:2]))
        return 1000.0 * total / len(windows)

    gains, tau = np.ones(3), 0.08
    for _ in range(2):
        for m in range(3):
            best, best_err = gains[m], mean_err(gains, tau)
            for cand in np.arange(0.60, 1.101, 0.04):
                trial = gains.copy()
                trial[m] = cand
                err = mean_err(trial, tau)
                if err < best_err:
                    best, best_err = cand, err
            gains[m] = best
        best_tau, best_err = tau, mean_err(gains, tau)
        for cand in np.arange(0.0, 0.241, 0.02):
            err = mean_err(gains, cand)
            if err < best_err:
                best_tau, best_err = cand, err
        tau = best_tau
    _FITS[key] = (gains, float(tau), mean_err(gains, tau))
    return _FITS[key]


def test_a_fairly_initialised_calibrated_baseline_is_far_stronger():
    windows = prediction_windows()
    assert len(windows) > 30, f"only {len(windows)} windows"
    _, tau_cold, err_cold = _fit_and_score(windows, use_v0=False)
    _, tau_warm, err_warm = _fit_and_score(windows, use_v0=True)
    assert err_warm < err_cold / 1.5, (
        f"calibrated baseline: {err_cold:.2f} mm starting from rest vs "
        f"{err_warm:.2f} mm given the measured velocity. M2 measured 13.22 -> "
        f"5.27 mm for exactly this one line. If these two are now close, check "
        f"whether the windows still start in motion."
    )


def test_withholding_v0_inflates_the_fitted_lag_constant():
    """
    The reason the bug is invisible once calibration has been fitted around it:
    tau stops being the actuator time constant and becomes a fudge for the
    missing initial condition. P0 recorded this as "the fit lands on 110-130 ms
    against a true actuator tau of ~60 ms ... the model is misspecified, so the
    fitted parameters are biased."
    """
    windows = prediction_windows()
    _, tau_cold, _ = _fit_and_score(windows, use_v0=False)
    _, tau_warm, _ = _fit_and_score(windows, use_v0=True)
    assert tau_cold > tau_warm, (
        f"fitted tau was {tau_cold*1000:.0f} ms without v0 and "
        f"{tau_warm*1000:.0f} ms with it"
    )


def test_the_windows_do_not_start_from_rest():
    """Without this, every test above would pass for the wrong reason."""
    windows = prediction_windows()
    speeds = np.array([np.linalg.norm(v0[:2]) for _, v0, _, _ in windows])
    assert np.median(speeds) > 0.05, (
        f"median window-opening speed {np.median(speeds):.3f} m/s - these windows "
        f"really do start at rest, so v0 would not matter"
    )


# ── The code paths ─────────────────────────────────────────────────────────

def test_planning_predictors_are_given_the_measured_velocity():
    """
    Behavioural, not textual: the M4 analytic predictor's rollout must change
    when the history's measured velocity changes. If it does not, the planner
    is steering with a cold-start model while the learned condition is not, and
    the M4 comparison is rigged the same way M2's first run was.
    """
    env = _lag()
    mppi = import_or_skip("planning.mppi", " (it imports models/, which may be mid-edit)")
    predictor = mppi.AnalyticPredictor(KIN, gains=[1.0, 0.88, 0.88], tau=0.12)
    actions = np.tile([0.3, 0.0, 0.0], (1, 12, 1))
    pose0 = np.zeros(3)

    at_rest = np.zeros((1, 14), dtype=np.float32)
    moving = np.zeros((1, 14), dtype=np.float32)
    moving[0, 3:6] = [0.3, 0.0, 0.0]

    a = predictor.rollout(pose0, at_rest, actions)[0, -1]
    b = predictor.rollout(pose0, moving, actions)[0, -1]
    assert np.linalg.norm(a[:2] - b[:2]) > 0.005, (
        f"AnalyticPredictor.rollout ignored the measured start velocity "
        f"(difference {np.linalg.norm(a[:2]-b[:2])*1000:.2f} mm over 0.6 s)"
    )


def test_learned_predictor_also_passes_v0_into_its_analytic_prior():
    """
    `LearnedPredictor` needs a trained checkpoint to instantiate, so this one
    is a source check: its analytic prior must be warm-started too, or the
    residual it learns is a correction to a different model than the one the
    analytic condition uses.
    """
    source = (SOFTWARE_DIR / "planning" / "mppi.py").read_text(encoding="utf-8")
    body = source.split("class LearnedPredictor")[1]
    assert "first_order_lag(" in body, "LearnedPredictor no longer builds an analytic prior"
    prior_call = body.split("first_order_lag(")[1].split(")")[0]
    assert "v0=" in prior_call, (
        f"LearnedPredictor's analytic prior is cold-started: first_order_lag({prior_call})"
    )


# Was an expected_failure when this suite was written: the P0 evaluation path
# had no way to pass v0, so experiments/P0-sim-baseline reported a cold-started
# 13.4 mm as its "calibrated analytic" baseline while M2 reported 5.27 mm for
# the same predictor. Fixed 2026-10-13 (v0 threaded through collect_windows,
# fit_calibration, predict_calibrated and analytic_baseline, then refitted);
# P0's corrected figure is 4.7 mm. Marker removed - this now guards the fix.
def test_p0_baseline_path_can_be_given_the_measured_velocity():
    env = _lag()
    params = inspect.signature(env.predict_calibrated).parameters
    assert "v0" in params, (
        f"predict_calibrated{tuple(params)} cannot receive a measured start velocity"
    )


if __name__ == "__main__":
    raise SystemExit(main(globals()))
