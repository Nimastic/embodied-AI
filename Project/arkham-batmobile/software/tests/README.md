# Regression suite

Every bug this project has hit so far produced **plausible-looking numbers rather than a crash**.
A wall-contact step response reported strafe and forward motion exactly inverted; an off-by-one in
the dataset trained a model to predict the past and converged beautifully; a cold-started baseline
turned a 73% win into a 71% one and changed what the experiment could conclude; a barrier margin
wider than the gate made success rate independent of model quality and looked like a publishable
negative result.

None of those would be caught by "does it run". This suite is the set of invariants that each of
those incidents violated, written so that the same mistake fails loudly the second time.

## Running it

```bash
cd Project/arkham-batmobile
python -m pytest software/tests                 # primary
python -m pytest software/tests -rxX            # also list the known-bug markers
python -m pytest software/tests/test_schema.py  # one module
```

No pytest (e.g. on the Pi):

```bash
python software/tests/run_all.py                # same tests, same semantics
python -m software.tests.run_all
python software/tests/run_all.py test_kinematics test_slip_model
python software/tests/test_kinematics.py        # any module runs standalone
```

**84 tests, ~4 s**, all passing (82 passed, 2 `xfail` — see *Live findings*). Nothing depends on a
collected dataset: where a test needs data it collects a 120–300 tick episode into a temp
directory at test time (~0.1 s) through the real `sim/collect.py` writer and reads it back through
Parquet. Anything needing MuJoCo, torch or `software/models/` **skips with the actual import
error** rather than passing quietly.

## What each file protects, and which incident motivated it

| File | Tests | Protects against | Incident |
|---|--:|---|---|
| `test_kinematics.py` | 21 | Wheel-pattern sign/convention errors; `fk(ik(v)) == v`; `J` with orthogonal columns; the `(1,1,-1,-1)` null space producing zero body motion and unit scrub; clean driving producing zero scrub; body-frame conventions; `integrate_batch` drifting from the scalar `integrate_pose` loop | P0 "Step response measured against a wall" — forward read 7.8% delivered and strafe 76.8%, exactly inverted. `integrate_batch` is what MPPI and every eval script rely on |
| `test_dataset_alignment.py` | 6 | The off-by-one that trains a model to predict the past; "average, never instantaneous"; odometry that stops drifting | `software/README.md` "The one alignment that will silently ruin everything" — correct offset reproduces pose 1.0 s ahead to 0.000 mm, off by one to 14.6 mm |
| `test_schema.py` | 10 | Missing or retyped dataset columns; `tick`/`t_mono` restarting mid-episode; frame indices that do not line up with the saved frame stack | `software/README.md` schema contract — "the one decision that is expensive to change later"; `pose_gt_valid` dropped turns occluded rows into a stationary robot |
| `test_action_clamping.py` | 14 | Actions outside the achievable polytope at **any** producer (random sampler, wall-avoider, scripted expert, reactive controller, MPPI); per-wheel clipping; saturation in collected data; float-point saturation detection | P0 "Individual wheel clipping broke the mecanum symmetry" — mean \|scrub\| 1.004 rad/s against a proper 0.07, 72% of ticks saturated |
| `test_baseline_fairness.py` | 9 | A baseline handed less information than the learned model: `first_order_lag(v0=None)` asserting the robot starts at rest | M2 "the bug that nearly produced a fake result" — calibrated baseline 13.22 → 5.27 mm from one line; the finding's *shape* changed |
| `test_planning_cost.py` | 12 | `BARRIER_MARGIN` wider than (narrowest gate half-width − `CLEARANCE`), which overlaps the two posts' barrier zones across the whole opening and leaves the planner no steering signal; plus the gate distribution the sweep was run with; plus which of M4's reported gaps the sample size supports | M4 "a bug that nearly became a finding" — a 2.3× more accurate model scored the same, because the cost function was the binding constraint. And the 20-trial run that showed a 15% collision rate which was pure noise at 40 trials |
| `test_slip_model.py` | 12 | The simulator's residual degenerating into zero-mean noise; the per-mode ordering inverting (strafe worst, `mat` worse than `hard`); ripple/battery/speed structure disappearing; odometry under-reading | `slip_model.py`'s "ONE RULE"; P0's delivered fractions 91.8 / 86.7 / 80.9%; M2b's whole argument rests on `mat` being the harder surface |

A few tests exist only to keep the others honest — `test_off_by_one_alignment_is_detectably_wrong`,
`test_the_broken_margin_would_be_caught_by_the_test_above`,
`test_the_windows_do_not_start_from_rest`, `test_the_test_episode_moves`. An invariant that would
also pass on a stationary robot or a widened gate distribution is not protecting anything, so each
of those asserts that the *bug* is still detectable by the test above it.

## Live findings

Two tests are marked `expected_failure` (pytest `xfail(strict=True)`): the suite stays green while
the bug exists and turns **red the moment the test starts passing**, which is the reminder to
delete the marker. Neither was fixed here — this suite does not own that code.

**1. The P0 analytic baseline is still the handicapped one.**
`test_p0_baseline_path_can_be_given_the_measured_velocity`

`software/sim/mecanum_env.py:predict_calibrated()` has no `v0` parameter, and
`collect_windows()` does not return the measured start velocity, so neither `fit_calibration()`
nor `analytic_baseline()` *can* pass one. This is exactly the bug M2 found and fixed in its own
code path — `planning/mppi.py` passes `v0=v0` in both predictors — but the P0 path was never
updated. Consequence: the **13.4 mm** "calibrated analytic, 1.0 s median error" in
`experiments/P0-sim-baseline/README.md` is the cold-started number. M2 measures the same predictor
on the same surface at **5.27 mm**.

Reproduced here (59 windows, hard floor, per-mode gains and `tau` refit for each variant, which is
the comparison M2 ran):

| Calibrated analytic | fitted gains (fwd/strafe/rot) | fitted τ | mean 1.0 s error |
|---|---|--:|--:|
| `v0=None` (starts from rest) | 1.00 / 0.88 / 0.88 | 120 ms | 12.56 mm |
| `v0=measured` | 0.92 / 0.84 / 0.80 | 80 ms | **6.23 mm** |

Note the second half of the story, which `test_withholding_v0_inflates_the_fitted_lag_constant`
pins: a calibration *fitted* under the handicap hides it. τ inflates to absorb the missing initial
condition — and then passing `v0` into those same parameters makes the error **worse**
(12.56 → 15.36 mm), because the inflated lag is now double-counting. So this is not a one-line
change at evaluation time; the calibration has to be refit, which is also what makes the bug
survive a calibration step unnoticed. P0 recorded the symptom without connecting it: "the fit lands
on 110–130 ms against a true actuator τ of ~60 ms … the model is misspecified, so the fitted
parameters are biased."

**Not fixed here.** The fix is to thread a measured `v0` through `collect_windows` →
`fit_calibration` → `predict_calibrated` (a handful of lines), refit, and regenerate P0's table and
`assets/analytic_baseline.png`. Until then the P0 and M2 "calibrated analytic" rows are not the
same predictor and should not be compared.

**2. `cpu_temp` is in the documented schema and in no episode.**
`test_cpu_temp_column_exists`

`software/README.md` documents `cpu_temp` (float32, °C, "catches thermal throttling after the
fact") and `sim/collect.py:_row()` does not write it. Every other documented column is present
with the right shape and dtype. The sim has no CPU temperature to read, but the column has to
exist — as NaN if need be — or the sim and robot schemas have already diverged before
`robot/logger.py` is written, and the first real dataset will not load with the same reader.

`test_every_documented_column_is_present` tolerates exactly this one column via a `KNOWN_MISSING`
set, so any *other* missing column fails immediately.

## Files

```text
tests/
├── _harness.py    sys.path setup; skip/expected_failure that work under pytest AND run_all;
│                  the cached 300-tick episode and prediction windows the suite shares
├── conftest.py    pytest: load _harness before any test module
├── run_all.py     dependency-free runner (pytest is not required to use this suite)
├── pytest.ini     keeps .pytest_cache inside this directory
└── test_*.py      one module per bug class, each with the incident in its docstring
```

## Adding to it

The rule that makes this suite worth keeping: **a bug that produced plausible numbers gets a test
before it gets a fix.** Write the invariant it violated, check the invariant fails against the old
behaviour (see the "teeth" tests above for the pattern), then fix the code. Put the incident in the
test's docstring with a pointer to the write-up — the test's job is not only to fail, it is to say
what went wrong last time.

Keep it fast. The whole suite is a few seconds because every sim artifact is short and cached; a
test that needs 60 min of collected data belongs in `experiments/`, not here.
