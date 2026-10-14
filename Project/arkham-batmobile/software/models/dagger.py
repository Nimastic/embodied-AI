"""
M5: DAgger-style iteration on the learned world model.

M4's finding, and the reason this file exists: the M2 residual GRU is ~4x more
accurate than the calibrated analytic model on its TRAINING action distribution
(0.87 mm vs 5.03 mm at 0.75 s) and only 35% better under MPPI-GENERATED actions
(2.90 mm vs 4.44 mm). The planner is an optimiser, so it hunts for action
sequences the model was never trained on. Net task gain over calibrated: +2.5
success points, i.e. one trial in forty.

The textbook fix is DAgger (Ross et al. 2011): collect data with the current
policy driving, aggregate, retrain, repeat.

    round 0   the existing random-exploration dataset + the M2 checkpoint
    round N   run MPPI on the gate task with round N-1's model, append the
              episodes to the aggregate, retrain FROM SCRATCH, re-evaluate

Per round we report
    (a) 0.75 s open-loop pose error under BOTH action distributions, and
    (b) the M4 success sweep at chunk 8 (400 ms) and chunk 15 (750 ms).

TWO THINGS THAT MAKE THIS HONEST, both learned the hard way on this project.

1. The collection loop IS the deployment loop. `run_m4.run_trial` drives a thin
   recording proxy around the environment, so the collected action distribution
   is the deployment action distribution by construction, not by a
   reimplementation that drifts.

2. The ORACLE condition. Before spending an hour on DAgger we ask what a
   PERFECT model is worth: `OraclePredictor` plans through the simulator's own
   slip model, reading privileged internal state no learned model can see. Its
   success rate is the ceiling. If the ceiling is already at the calibrated
   model's score, model accuracy is not the binding constraint and no amount of
   DAgger can convert into task success. That check costs two minutes and
   decides whether the rest of the experiment means anything.

Usage:
    python dagger.py --stage diag              # effect-size checks first
    python dagger.py --stage loop --rounds 3
    python dagger.py --stage all --rounds 3
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models.state_model as sm                                      # noqa: E402
from kinematics.mecanum import MecanumKinematics                     # noqa: E402
from planning.gate_task import GateTask                              # noqa: E402
from planning.mppi import (                                          # noqa: E402
    AnalyticPredictor,
    LearnedPredictor,
    MPPI,
    reactive_action,
)
from planning.run_m4 import (                                        # noqa: E402
    MAX_TICKS,
    _feat as m4_feat,
    _summarise,
    run_trial,
)
from sim.collect import _row, save_episode                           # noqa: E402
from sim.mecanum_env import (                                        # noqa: E402
    ACTION_LIMITS,
    CONTROL_DT,
    MecanumEnv,
    WHEEL_SPEED_MAX,
    clamp_action,
)
from sim.slip_model import SlipModel, SlipState                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAGGER_DATA = ROOT / "data" / "dagger"          # gitignored via root **/data/
CKPT_DIR = ROOT / "checkpoints"
BASE_CKPT = CKPT_DIR / "state_model-residual-h1.pt"
ASSETS = ROOT.parent / "experiments" / "M5-dagger" / "assets"

# sm.HORIZON is 20 ticks (1.0 s). M4 quotes 0.75 s because that is the MPPI
# horizon, so index 14 of the per-horizon error curve is the headline number.
H075 = 14
HORIZON_S = (H075 + 1) * CONTROL_DT

# Episodes are collected at a mix of replan intervals so the aggregate covers
# the whole deployment sweep, not just one point on it. Long chunks are also
# cheaper to collect (fewer solves per episode).
COLLECT_CHUNKS = (1, 3, 8, 15)

# Task seeds. run_m4 evaluates on seed..seed+trials-1 with seed=11, so training
# collection lives far away from it and the two never overlap.
EVAL_SEED = 11
TRAIN_SEED0 = 100_000
EVAL_WINDOW_SEEDS = tuple(range(EVAL_SEED, EVAL_SEED + 24))   # 11..34, never trained on

STRIDE_RANDOM = 2       # as in M2, so round 0 reproduces the published checkpoint
STRIDE_DAGGER = 1       # planner episodes are ~20x shorter; this is how you get
                        # comparable window counts without collecting for an hour


# ── Recording proxy ─────────────────────────────────────────────────────────

class RecordingEnv:
    """
    Transparent wrapper that remembers every observation.

    The point is that `run_m4.run_trial` can drive this instead of a raw
    MecanumEnv, so DAgger collection and M4 evaluation execute literally the
    same control loop. A hand-written collection loop would be one edit away
    from collecting a distribution the planner never actually produces, which
    is the exact bug this experiment is supposed to be measuring.
    """

    def __init__(self, env: MecanumEnv):
        self.__dict__["_env"] = env
        self.__dict__["rows"] = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)
        self.rows.clear()
        return obs

    def step(self, action):
        obs = self._env.step(action)
        self.rows.append(obs)
        return obs


# ── The oracle predictor ────────────────────────────────────────────────────

class OraclePredictor:
    """
    Plans through the simulator's OWN dynamics. The ceiling condition.

    This is not a baseline, it is an upper bound: what is a PERFECT dynamics
    model worth on this task at this replan interval? If the answer is "no more
    than the calibrated analytic one", then model accuracy is not the binding
    constraint and no amount of DAgger can convert into task success. That is
    worth knowing before spending an hour collecting data.

    It reproduces the env's full chain, and reads privileged state no learned
    model or real robot could have:

      1. the real SlipModel, re-seeded each rollout from `env.slip.state` -
         integrated wheel angle (which sets the roller ripple phase) and pack
         state of charge;
      2. the MuJoCo velocity actuator, which is exactly first order in the
         WORLD frame. tau = M_ii / (kv_i + damping_i) read off the model
         (60.3 / 60.3 / 61.2 ms, matching P0's measured 59.3 / 60.2 ms), with
         the closed-form tick-AVERAGE as well as the end-of-tick value, since
         the average is what moves the robot;
      3. `env.data.qvel`, the true world-frame velocity at plan time.

    Low-passing in the body frame instead of the world frame is wrong and not
    subtly so: at 1-2 rad/s the heading turns 45-90 degrees inside one 0.75 s
    horizon. An earlier version did that and left 24 mm of divergence, i.e. no
    better than the learned model, which would have made the oracle useless as
    a ceiling. Divergence is the evidence that this one earns the name.
    """

    def __init__(self, kin: MecanumKinematics, env: MecanumEnv, name: str = "oracle"):
        import mujoco

        self.kin = kin
        self.env = env
        self.name = name
        self.slip = SlipModel(surface=env.surface, kinematics=kin, seed=99991)

        # mj_forward first, or the mass matrix comes back all zeros and tau is
        # silently infinite - which produces a confident, useless oracle.
        mujoco.mj_forward(env.model, env.data)
        m = np.zeros((env.model.nv, env.model.nv))
        mujoco.mj_fullM(env.model, env.data, m)
        assert np.all(np.diag(m) > 1e-9), "degenerate mass matrix"
        kv = np.asarray(env.model.actuator_gainprm[:, 0], dtype=float)
        damping = np.asarray(env.model.dof_damping, dtype=float)
        tau = np.diag(m) / (kv + damping)
        self.dc_gain = kv / (kv + damping)
        self.a_end = np.exp(-CONTROL_DT / tau)
        self.a_avg = (tau / CONTROL_DT) * (1.0 - self.a_end)
        self.tau_ms = (tau * 1000).round(2).tolist()

    def rollout(self, pose0: np.ndarray, hist: np.ndarray, actions: np.ndarray) -> np.ndarray:
        del hist                                   # the oracle uses true state instead
        w0 = self.env.slip.state.wheel_angle.copy()
        soc = float(self.env.slip.state.pack_soc)
        qvel0 = np.asarray(self.env.data.qvel[:3], dtype=float).copy()
        k, h = actions.shape[0], actions.shape[1]
        a_end, a_avg, dc = self.a_end, self.a_avg, self.dc_gain

        # Scalar inner loop on purpose: K*H = 5760 iterations per solve, and
        # numpy on 3-vectors costs more in dispatch than the arithmetic saves.
        # This is 3-4x faster than the array version and bit-identical.
        dt = CONTROL_DT
        av0, av1, av2 = a_avg
        en0, en1, en2 = a_end
        g0, g1, g2 = dc
        step = self.slip.step
        out = np.empty((k, h, 3))
        for i in range(k):
            self.slip.state = SlipState(wheel_angle=w0.copy(), pack_soc=soc)
            seq = actions[i]
            x, y, th = float(pose0[0]), float(pose0[1]), float(pose0[2])
            wx, wy, wt = qvel0[0], qvel0[1], qvel0[2]
            row = out[i]
            for j in range(h):
                bv = step(seq[j], dt)["body_vel"]
                c, s = math.cos(th), math.sin(th)   # ctrl uses start-of-tick heading
                t0 = (bv[0] * c - bv[1] * s) * g0
                t1 = (bv[0] * s + bv[1] * c) * g1
                t2 = bv[2] * g2
                x += (t0 + (wx - t0) * av0) * dt
                y += (t1 + (wy - t1) * av1) * dt
                th += (t2 + (wt - t2) * av2) * dt
                wx = t0 + (wx - t0) * en0
                wy = t1 + (wy - t1) * en1
                wt = t2 + (wt - t2) * en2
                row[j, 0], row[j, 1], row[j, 2] = x, y, th
        out[..., 2] = (out[..., 2] + np.pi) % (2 * np.pi) - np.pi
        return out


# ── Collection ──────────────────────────────────────────────────────────────

def run_trial_fixed(env, task, condition, predictor, kin, seed, n_execute, record=False):
    """
    `run_m4.run_trial` with ONE line changed, and nothing else.

    run_m4 indexes the predicted path at `len(actions) - 1`, i.e. the end of
    the chunk it INTENDED to execute. But the execution loop breaks early the
    moment the goal is reached, so on the last replan of nearly every episode a
    15-step prediction is compared against a 3-step outcome. Measured with the
    oracle predictor, whose true 0.75 s open-loop error is 0.27 mm:

        planned 15, executed 15  ->  0.22 mm
        planned 15, executed 15  ->  0.16 mm
        planned 15, executed 15  ->  0.25 mm
        planned 15, executed  8  ->  98.31 mm     <- the artifact
                                      mean 24.7 mm

    That single entry is where M4's "20.18 mm divergence at chunk 15" comes
    from. It is a termination artifact, not model error, and it grows with
    chunk size - which is why the divergence column looked like it was tracking
    model quality. The fix is to index at the number of actions ACTUALLY
    executed. run_m4 is not mine to edit (see the M5 write-up); this is the
    corrected copy, and `verify_runner` proves it changes nothing else.

    Success, collision and timing are untouched, so M4's headline table stands.
    """
    obs = env.reset(pose=task.start)
    env.data.mocap_pos[0] = [*task.post_l, 0.10]
    env.data.mocap_pos[1] = [*task.post_r, 0.10]

    planner = None
    if condition != "reactive":
        planner = MPPI(predictor, task.cost, kin, seed=seed)
        planner.reset()

    hist_buf = [m4_feat(obs)]
    traj, frames, divergences = [obs["pose_gt"].copy()], [], []
    collided = False
    plan_times = []
    t = 0

    while t < MAX_TICKS:
        if condition == "reactive":
            actions = [reactive_action(obs["pose_gt"], task.goal, kin)]
        else:
            t0 = time.perf_counter()
            hist = np.asarray(hist_buf[-1:], dtype=np.float32)
            plan = planner.plan(obs["pose_gt"], hist)
            plan_times.append(time.perf_counter() - t0)
            actions = plan[:n_execute]
            predicted_path = planner.last_diagnostics["predicted_path"]

        n_done = 0
        for a in actions:
            obs = env.step(clamp_action(a))
            hist_buf.append(m4_feat(obs))
            traj.append(obs["pose_gt"].copy())
            n_done += 1
            if task.collided(obs["pose_gt"]):
                collided = True
            if record:
                frames.append(env.render("overhead", 320))
            t += 1
            if task.reached(obs["pose_gt"]) or obs["out_of_bounds"] or t >= MAX_TICKS:
                break

        if planner is not None:
            planner.shift(n_done)
            # THE FIX: compare against the prediction for the steps actually taken.
            divergences.append(
                float(np.linalg.norm(predicted_path[n_done - 1][:2] - obs["pose_gt"][:2]))
            )

        if task.reached(obs["pose_gt"]) or obs["out_of_bounds"]:
            break

    success = task.reached(obs["pose_gt"]) and task.past_gate(obs["pose_gt"]) and not collided
    return {
        "success": bool(success),
        "collided": bool(collided),
        "reached": bool(task.reached(obs["pose_gt"])),
        "out_of_bounds": bool(obs["out_of_bounds"]),
        "ticks": t,
        "time_s": round(t * CONTROL_DT, 2),
        "final_dist_mm": round(float(np.linalg.norm(obs["pose_gt"][:2] - task.goal[:2]) * 1000), 1),
        "min_post_dist_mm": round(float(np.min(task.post_distances(np.array(traj))) * 1000), 1),
        "mean_plan_divergence_mm": (
            round(float(np.mean(divergences) * 1000), 1) if divergences else None),
        "plan_ms": round(float(np.mean(plan_times) * 1000), 1) if plan_times else None,
        "traj": np.array(traj),
        "frames": frames,
    }


def verify_runner(env, kin, predictor, trials: int = 10) -> dict:
    """
    Assert that run_trial_fixed reproduces run_m4.run_trial trial for trial on
    success, collision and tick count. The two differ only in a diagnostic
    index, so the trajectories must be identical; if they are not, something
    else drifted and the M5 numbers are not comparable to M4's.
    """
    bad, m4_div, fixed_div = 0, [], []
    for chunk in (8, 15):
        for i in range(trials):
            seed = EVAL_SEED + i
            a = run_trial(env, GateTask(np.random.default_rng(seed)), "mppi-learned",
                          predictor, kin, seed, chunk)
            b = run_trial_fixed(env, GateTask(np.random.default_rng(seed)), "mppi-learned",
                                predictor, kin, seed, chunk)
            if (a["success"], a["collided"], a["ticks"]) != (
                    b["success"], b["collided"], b["ticks"]):
                bad += 1
            m4_div.append(a["mean_plan_divergence_mm"])
            fixed_div.append(b["mean_plan_divergence_mm"])
    if bad:
        raise SystemExit(f"run_trial_fixed diverged from run_m4.run_trial on {bad} trials")
    out = {"n_trials_checked": trials * 2, "trajectory_mismatches": 0,
           "mean_divergence_run_m4_mm": round(float(np.mean(m4_div)), 2),
           "mean_divergence_fixed_mm": round(float(np.mean(fixed_div)), 2)}
    print(f"    {out['n_trials_checked']} trials, 0 trajectory mismatches. "
          f"Divergence as run_m4 reports it {out['mean_divergence_run_m4_mm']:.2f} mm, "
          f"corrected {out['mean_divergence_fixed_mm']:.2f} mm")
    return out


def collect_episodes(
    env: MecanumEnv, predictor, kin: MecanumKinematics, seeds, out_dir: Path,
    label: str, chunks=COLLECT_CHUNKS,
) -> dict:
    """Run MPPI on the gate task and log every tick in the dataset schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = RecordingEnv(env)
    manifest = {
        "mode": "mppi-dagger",
        "surface": env.surface,
        "control_hz": 1.0 / CONTROL_DT,
        "action_limits": ACTION_LIMITS.tolist(),
        "predictor": getattr(predictor, "name", "?"),
        "source": "sim",
        "collected_by": "models/dagger.py",
    }

    t0 = time.perf_counter()
    successes, ticks = [], 0
    for i, seed in enumerate(seeds):
        chunk = chunks[i % len(chunks)]
        task = GateTask(np.random.default_rng(seed))
        res = run_trial_fixed(rec, task, "mppi-learned", predictor, kin, seed, chunk)
        successes.append(res["success"])
        episode_id = f"{label}-{i:04d}"
        df = pd.DataFrame([_row(episode_id, o, -1) for o in rec.rows])
        ticks += len(df)
        save_episode(df, [], out_dir, episode_id,
                     {**manifest, "chunk": chunk, "task_seed": int(seed)})

    return {
        "label": label,
        "n_episodes": len(seeds),
        "n_ticks": int(ticks),
        "collect_success_rate": round(float(np.mean(successes)), 3),
        "wall_s": round(time.perf_counter() - t0, 1),
        "chunks": list(chunks),
    }


def load_dir(path: Path) -> list[dict]:
    """Episode dicts in the shape sm.build_windows expects."""
    out = []
    for p in sorted(path.glob("*.parquet")):
        df = pd.read_parquet(p)
        if len(df) < sm.HORIZON + 2:          # too short to yield a single window
            continue
        out.append({
            "path": p.name,
            "pose": np.stack(df["pose_gt"].to_numpy()).astype(np.float64),
            "vel": np.stack(df["body_vel_avg"].to_numpy()).astype(np.float32),
            "action": np.stack(df["action_cmd"].to_numpy()).astype(np.float32),
            "wheel": np.stack(df["wheel_meas"].to_numpy()).astype(np.float32),
            "gyro": np.stack(df["imu_gyro"].to_numpy()).astype(np.float32),
            "batt": df["batt_v"].to_numpy().astype(np.float32),
        })
    if not out:
        raise FileNotFoundError(f"no usable episodes in {path}")
    return out


# ── The alignment invariant, as a gate ──────────────────────────────────────

def check_alignment(kin: MecanumKinematics, data: dict, label: str) -> float:
    """
    body_vel_avg[t] is the average velocity over the tick ENDING at pose_gt[t],
    so integrating vel[t+1 .. t+H] from pose[t] must land exactly on pose[t+H].

    If it does not, the dataset is misaligned and every model trained on it is
    learning to predict the past. Off by one measures 15.2 mm here; correct
    measures 0.0000 mm. Run on every new dataset, sim or real.
    """
    pred = sm.integrate_all(kin, data["pose0"], data["vels"])
    err = np.linalg.norm(pred[:, -1, :2] - data["poseH"][:, :2], axis=1) * 1000.0
    worst = float(err.max())
    print(f"  [invariant] {label:24s} {len(err):6d} windows   median "
          f"{float(np.median(err)):.4f} mm   max {worst:.4f} mm")
    if worst > 0.01:
        raise SystemExit(f"ALIGNMENT BROKEN on {label}: {worst:.3f} mm. Stop and fix the "
                         f"dataset before training anything.")
    return worst


# ── Windows ─────────────────────────────────────────────────────────────────

def concat_windows(parts: list[dict]) -> dict:
    keys = ("hist", "actions", "vels", "pose0", "poseH")
    return {
        **{k: np.concatenate([p[k] for p in parts], axis=0) for k in keys},
        "dropped": int(sum(p["dropped"] for p in parts)),
    }


def random_splits(stride: int = STRIDE_RANDOM) -> tuple[dict, dict]:
    """
    The round-0 dataset, split BY EPISODE exactly as M2 did, so round 0
    reproduces the published checkpoint's training set.
    """
    hard = sm.load_episodes("hard")
    n_test = max(4, len(hard) // 5)
    return (sm.build_windows(hard[:-n_test], stride, 1),
            sm.build_windows(hard[-n_test:], stride, 1))


# ── Train / evaluate ────────────────────────────────────────────────────────

def train_model(data: dict, gains: np.ndarray, tau: float, epochs: int, seed: int,
                variant: str = "residual") -> sm.StatePredictor:
    model = sm.StatePredictor(data["hist"].shape[2], variant=variant)
    prior = sm.analytic_prior(data["actions"], gains, tau, sm.start_vel(data))
    t0 = time.perf_counter()
    sm.train(model, data, prior, epochs, seed)
    print(f"    trained on {len(data['hist']):,} windows in "
          f"{time.perf_counter() - t0:.0f}s")
    return model


def save_ckpt(model: sm.StatePredictor, feat_dim: int, gains: np.ndarray, tau: float,
              path: Path, note: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feat_dim": feat_dim,
        "hidden": 96,
        "variant": model.variant,
        "history": 1,
        "horizon": sm.HORIZON,
        "gains": np.asarray(gains).tolist(),
        "tau": float(tau),
        "trained_on": note,
    }, path)
    return path


def err075(kin, data: dict, gains, tau, model, label: str) -> dict:
    """
    0.75 s open-loop pose error for naive / calibrated / learned on one window
    set. Every predictor gets the measured start velocity (sm.analytic_prior
    passes v0) - withholding it once inflated the learned model's apparent
    margin from 42% to 71%.
    """
    ev = sm.evaluate(kin, data, gains, tau, model, label)
    out = {"split": label, "n_windows": ev["n_windows"],
           "mean_path_m": ev["mean_path_m"], "mm_at_0.75s": {}, "mm_at_1.0s": {}}
    for name in ("naive", "calibrated", "learned"):
        out["mm_at_0.75s"][name] = round(ev["error_vs_horizon_mm"][name][H075], 3)
        out["mm_at_1.0s"][name] = ev["predictors"][name]["median_mm"]
    out["curves_mm"] = ev["error_vs_horizon_mm"]
    cal, lrn = out["mm_at_0.75s"]["calibrated"], out["mm_at_0.75s"]["learned"]
    out["learned_advantage_pct"] = round((cal - lrn) / cal * 100, 1)
    return out


def sweep(env, kin, predictor, chunks, trials: int, seed: int = EVAL_SEED,
          condition: str = "mppi-learned") -> dict:
    """The M4 success sweep, same seeds, same task instances, same summary."""
    out = {}
    for chunk in chunks:
        results = []
        for i in range(trials):
            task = GateTask(np.random.default_rng(seed + i))
            r = run_trial_fixed(env, task, condition, predictor, kin, seed + i, chunk)
            r.pop("traj"), r.pop("frames")
            results.append(r)
        out[f"chunk{chunk}"] = _summarise(results, trials)
        s = out[f"chunk{chunk}"]
        print(f"    {getattr(predictor, 'name', condition):11s} chunk {chunk:<3d} "
              f"success {s['success_rate']:6.1%}  collisions {s['collision_rate']:6.1%}  "
              f"divergence {str(s['mean_plan_divergence_mm']):>6s} mm")
    return out


# ── Action-distribution diagnostic ──────────────────────────────────────────

def action_stats(episodes: list[dict], label: str) -> dict:
    """
    How different IS the planner's action distribution? This is the mechanism
    behind M4's result, so it gets measured rather than asserted.

    `wheel_saturated_frac` is the one that matters: the fraction of ticks where
    the command sits on the achievable polytope boundary. The planner wants to
    go as fast as the gate allows; a random walk almost never saturates.
    """
    kin = MecanumKinematics()
    acts = np.concatenate([e["action"] for e in episodes], axis=0)
    peak = np.max(np.abs(acts @ kin.J.T) / kin.geom.wheel_radius, axis=1)
    jerk = np.concatenate([np.diff(e["action"], axis=0) for e in episodes], axis=0)
    return {
        "label": label,
        "n_ticks": int(len(acts)),
        "mean_abs": [round(float(v), 4) for v in np.abs(acts).mean(0)],
        "p95_abs": [round(float(v), 4) for v in np.percentile(np.abs(acts), 95, axis=0)],
        "mean_speed_m_s": round(float(np.linalg.norm(acts[:, :2], axis=1).mean()), 4),
        "mean_peak_wheel_rad_s": round(float(peak.mean()), 3),
        "wheel_saturated_frac": round(float((peak > 0.98 * WHEEL_SPEED_MAX).mean()), 4),
        "mean_abs_jerk": [round(float(v), 4) for v in np.abs(jerk).mean(0)],
    }


# ── Stage 1: effect size, before spending an hour ───────────────────────────

def validate_oracle(env, kin, oracle, trials: int = 8) -> dict:
    """
    An oracle is only a ceiling if it is actually accurate. Measure its
    plan-vs-actual divergence at chunk 15 on task seeds OUTSIDE the evaluation
    range: that is the same 0.75 s open-loop error the learned model is judged
    on (20.18 mm in M4), so it has to come out near zero.
    """
    div = []
    for i in range(trials):
        seed = 900_000 + i
        task = GateTask(np.random.default_rng(seed))
        r = run_trial_fixed(env, task, "mppi-learned", oracle, kin, seed, 15)
        if r["mean_plan_divergence_mm"]:
            div.append(r["mean_plan_divergence_mm"])
    out = {"actuator_tau_ms": oracle.tau_ms, "n_trials": trials,
           "mean_divergence_mm": round(float(np.mean(div)), 3),
           "max_divergence_mm": round(float(np.max(div)), 3)}
    print(f"    oracle actuator tau {oracle.tau_ms} ms   0.75 s divergence "
          f"{out['mean_divergence_mm']:.2f} mm mean / {out['max_divergence_mm']:.2f} mm worst")
    return out


def stage_diag(args) -> dict:
    kin = MecanumKinematics()
    env = MecanumEnv(surface="hard", seed=args.seed)
    base = LearnedPredictor(kin, BASE_CKPT)
    gains, tau = base.gains, base.tau
    report: dict = {"calibration": {"gains": gains.round(4).tolist(), "tau_s": round(tau, 4)}}

    print("-- D1. Reproduce M4's distribution-shift table -------------------")
    d_train, d_rand = random_splits()
    check_alignment(kin, d_rand, "random held-out")

    eval0 = DAGGER_DATA / "eval-r0"
    if args.recollect or not eval0.exists():
        report["collect_eval_r0"] = collect_episodes(
            env, base, kin, EVAL_WINDOW_SEEDS, eval0, "eval-r0")
    d_mppi = sm.build_windows(load_dir(eval0), STRIDE_DAGGER, 1)
    check_alignment(kin, d_mppi, "MPPI (round 0 model)")

    report["d1_random"] = err075(kin, d_rand, gains, tau, base.model, "random held-out")
    report["d1_mppi"] = err075(kin, d_mppi, gains, tau, base.model, "MPPI, round-0 model")
    for key in ("d1_random", "d1_mppi"):
        r = report[key]
        print(f"  {r['split']:22s} n={r['n_windows']:5d}   calibrated "
              f"{r['mm_at_0.75s']['calibrated']:6.2f} mm   learned "
              f"{r['mm_at_0.75s']['learned']:6.2f} mm   advantage "
              f"{r['learned_advantage_pct']:5.1f}%")

    report["action_stats"] = [
        action_stats(sm.load_episodes("hard"), "random exploration"),
        action_stats(load_dir(eval0), "MPPI, round-0 model"),
    ]
    for a in report["action_stats"]:
        print(f"  {a['label']:22s} mean speed {a['mean_speed_m_s']:.3f} m/s   "
              f"peak wheel {a['mean_peak_wheel_rad_s']:5.2f} rad/s   "
              f"saturated {a['wheel_saturated_frac']:5.1%}")

    print("\n-- D2. The ceiling: MPPI through the simulator's own dynamics ----")
    print("  verifying the corrected trial runner against run_m4's")
    report["runner_verification"] = verify_runner(env, kin, base)
    oracle = OraclePredictor(kin, env)
    report["oracle_validation"] = validate_oracle(env, kin, oracle)
    report["d2_oracle"] = sweep(env, kin, oracle, args.chunks, args.trials)
    report["d2_calibrated"] = sweep(
        env, kin, AnalyticPredictor(kin, gains, tau, "calibrated"), args.chunks, args.trials)
    report["d2_learned_r0"] = sweep(env, kin, base, args.chunks, args.trials)

    print("\n-- D3. The floor: can ANY model fit the MPPI distribution? -------")
    r1 = DAGGER_DATA / "round1"
    if args.recollect or not r1.exists():
        seeds = range(TRAIN_SEED0, TRAIN_SEED0 + args.episodes)
        report["collect_round1"] = collect_episodes(env, base, kin, seeds, r1, "r1")
        print(f"    {report['collect_round1']}")
    d_r1 = sm.build_windows(load_dir(r1), STRIDE_DAGGER, 1)
    check_alignment(kin, d_r1, "MPPI round-1 train")
    print(f"    training on PLANNER DATA ONLY ({len(d_r1['hist']):,} windows)")
    only = train_model(d_r1, gains, tau, args.epochs, args.seed)
    report["d3_planner_only_mppi"] = err075(kin, d_mppi, gains, tau, only,
                                            "MPPI, planner-only model")
    report["d3_planner_only_random"] = err075(kin, d_rand, gains, tau, only,
                                              "random, planner-only model")
    for key in ("d3_planner_only_mppi", "d3_planner_only_random"):
        r = report[key]
        print(f"  {r['split']:28s} calibrated {r['mm_at_0.75s']['calibrated']:6.2f} mm   "
              f"learned {r['mm_at_0.75s']['learned']:6.2f} mm")

    env.close()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "metrics-diag.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n  {ASSETS / 'metrics-diag.json'}")
    return report


# ── Stage 2: the DAgger loop ────────────────────────────────────────────────

def stage_loop(args) -> dict:
    kin = MecanumKinematics()
    env = MecanumEnv(surface="hard", seed=args.seed)
    base = LearnedPredictor(kin, BASE_CKPT)
    gains, tau = base.gains, base.tau

    # The calibration is FROZEN at the round-0 fit. It is both the residual
    # model's prior and the analytic baseline column, so freezing it makes the
    # learned model the only moving part in the table. The "does recalibrating
    # on planner data do the same job?" control runs at the end.
    report: dict = {
        "horizon_s": HORIZON_S,
        "epochs": args.epochs,
        "episodes_per_round": args.episodes,
        "trials": args.trials,
        "chunks": args.chunks,
        "calibration_frozen": {"gains": gains.round(4).tolist(), "tau_s": round(tau, 4)},
        "stride": {"random": STRIDE_RANDOM, "dagger": STRIDE_DAGGER},
        "rounds": {},
        "collection": {},
    }

    print("-- Round 0: existing random dataset + existing checkpoint --------")
    d_train, d_rand = random_splits()
    check_alignment(kin, d_rand, "random held-out")
    agg = [d_train]

    eval0 = DAGGER_DATA / "eval-r0"
    if not eval0.exists():
        report["collection"]["eval-r0"] = collect_episodes(
            env, base, kin, EVAL_WINDOW_SEEDS, eval0, "eval-r0")
    d_fixed = sm.build_windows(load_dir(eval0), STRIDE_DAGGER, 1)
    check_alignment(kin, d_fixed, "MPPI fixed eval set")

    report["rounds"]["0"] = _round_report(
        kin, env, base.model, base, gains, tau, d_rand, d_fixed, d_fixed,
        args, n_train=len(d_train["hist"]), n_dagger=0, ckpt=str(BASE_CKPT.name))

    # Reference rows: unchanged across rounds, but re-run here so the M5 table
    # is self-contained rather than quoting M4 across experiments.
    print("\n-- Reference conditions ------------------------------------------")
    report["reference"] = {
        "mppi-naive": sweep(env, kin, AnalyticPredictor(kin, name="naive"),
                            args.chunks, args.trials),
        "mppi-calibrated": sweep(env, kin, AnalyticPredictor(kin, gains, tau, "calibrated"),
                                 args.chunks, args.trials),
    }
    react = []
    for i in range(args.trials):
        task = GateTask(np.random.default_rng(EVAL_SEED + i))
        r = run_trial_fixed(env, task, "reactive", None, kin, EVAL_SEED + i, 1)
        r.pop("traj"), r.pop("frames")
        react.append(r)
    report["reference"]["reactive"] = _summarise(react, args.trials)
    print(f"    reactive (no model)  success "
          f"{report['reference']['reactive']['success_rate']:.1%}")

    predictor = base
    for rnd in range(1, args.rounds + 1):
        print(f"\n=== Round {rnd} ===================================================")
        out_dir = DAGGER_DATA / f"round{rnd}"
        if not out_dir.exists():
            seeds = range(TRAIN_SEED0 + (rnd - 1) * args.episodes,
                          TRAIN_SEED0 + rnd * args.episodes)
            info = collect_episodes(env, predictor, kin, seeds, out_dir, f"r{rnd}")
            report["collection"][f"round{rnd}"] = info
            print(f"  collected {info['n_episodes']} episodes, {info['n_ticks']} ticks in "
                  f"{info['wall_s']}s  (task success during collection "
                  f"{info['collect_success_rate']:.0%})")

        d_new = sm.build_windows(load_dir(out_dir), STRIDE_DAGGER, 1)
        check_alignment(kin, d_new, f"MPPI round-{rnd} train")
        agg.append(d_new)
        d_agg = concat_windows(agg)
        n_dagger = int(sum(len(a["hist"]) for a in agg[1:]))
        print(f"  aggregate: {len(d_agg['hist']):,} windows "
              f"({n_dagger:,} planner = {n_dagger / len(d_agg['hist']):.0%})")

        model = train_model(d_agg, gains, tau, args.epochs, args.seed)
        ckpt = save_ckpt(model, d_agg["hist"].shape[2], gains, tau,
                         CKPT_DIR / f"dagger-r{rnd}.pt", f"hard random + {rnd} MPPI rounds")
        predictor = LearnedPredictor(kin, ckpt)

        eval_dir = DAGGER_DATA / f"eval-r{rnd}"
        if not eval_dir.exists():
            report["collection"][f"eval-r{rnd}"] = collect_episodes(
                env, predictor, kin, EVAL_WINDOW_SEEDS, eval_dir, f"eval-r{rnd}")
        d_on = sm.build_windows(load_dir(eval_dir), STRIDE_DAGGER, 1)
        check_alignment(kin, d_on, f"MPPI on-policy round {rnd}")

        report["rounds"][str(rnd)] = _round_report(
            kin, env, model, predictor, gains, tau, d_rand, d_fixed, d_on,
            args, n_train=len(d_agg["hist"]), n_dagger=n_dagger, ckpt=ckpt.name)
        report["rounds"][str(rnd)]["action_stats"] = action_stats(
            load_dir(eval_dir), f"MPPI, round-{rnd} model")

    # Control: does refitting the four analytic scalars ON PLANNER DATA do the
    # same job? M4's finding #2 was that calibration, not learning, carried
    # most of the benefit, so this is the question a reader will ask.
    print("\n-- Control: recalibrate the analytic model on the aggregate ------")
    d_agg = concat_windows(agg)
    g2, t2 = sm.fit_calibration_fast(kin, d_agg["pose0"], d_agg["actions"],
                                     d_agg["poseH"], sm.start_vel(d_agg))
    print(f"  refit gains {g2.round(3)}  tau {t2 * 1000:.0f} ms "
          f"(frozen was {gains.round(3)}, {tau * 1000:.0f} ms)")
    recal = {"gains": g2.round(4).tolist(), "tau_s": round(t2, 4)}
    for name, d in (("random_heldout", d_rand), ("mppi_fixed", d_fixed)):
        prior = sm.analytic_prior(d["actions"], g2, t2, sm.start_vel(d))
        pred = kin.integrate_batch(d["pose0"], prior, CONTROL_DT)
        truth = sm.integrate_all(kin, d["pose0"], d["vels"])
        e = np.linalg.norm(pred[:, H075, :2] - truth[:, H075, :2], axis=1) * 1000.0
        recal[f"{name}_mm_at_0.75s"] = round(float(np.median(e)), 3)
        print(f"  recalibrated on {name:16s} {recal[f'{name}_mm_at_0.75s']:6.2f} mm")
    recal["sweep"] = sweep(env, kin, AnalyticPredictor(kin, g2, t2, "recalibrated"),
                           args.chunks, args.trials)
    report["control_recalibrated"] = recal

    env.close()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n  {ASSETS / 'metrics.json'}")
    return report


def _round_report(kin, env, model, predictor, gains, tau, d_rand, d_fixed, d_on,
                  args, n_train: int, n_dagger: int, ckpt: str) -> dict:
    print(f"  -- evaluating [{ckpt}] --")
    out = {
        "checkpoint": ckpt,
        "n_train_windows": int(n_train),
        "n_dagger_windows": int(n_dagger),
        "dagger_fraction": round(n_dagger / max(n_train, 1), 4),
        "random_heldout": err075(kin, d_rand, gains, tau, model, "random held-out"),
        "mppi_fixed": err075(kin, d_fixed, gains, tau, model, "MPPI (fixed round-0 set)"),
        "mppi_onpolicy": err075(kin, d_on, gains, tau, model, "MPPI (on-policy)"),
    }
    for key in ("random_heldout", "mppi_fixed", "mppi_onpolicy"):
        r = out[key]
        print(f"    {r['split']:26s} calibrated {r['mm_at_0.75s']['calibrated']:6.2f} mm   "
              f"learned {r['mm_at_0.75s']['learned']:6.2f} mm   "
              f"advantage {r['learned_advantage_pct']:5.1f}%")
    out["sweep"] = sweep(env, kin, predictor, args.chunks, args.trials)
    return out


# ── Plots ───────────────────────────────────────────────────────────────────

def plots(report: dict, diag: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = sorted(int(k) for k in report["rounds"])
    R = [report["rounds"][str(r)] for r in rounds]
    ASSETS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # (1) 0.75 s open-loop error: does DAgger close the distribution gap?
    for key, style, lab in (("random_heldout", "o-", "learned, random actions"),
                            ("mppi_fixed", "s-", "learned, MPPI actions (fixed set)"),
                            ("mppi_onpolicy", "^-", "learned, MPPI actions (on-policy)")):
        ax[0].plot(rounds, [r[key]["mm_at_0.75s"]["learned"] for r in R], style,
                   lw=1.9, label=lab)
    ax[0].axhline(R[0]["mppi_fixed"]["mm_at_0.75s"]["calibrated"], ls="--", c="0.35", lw=1.2,
                  label="calibrated analytic, MPPI actions")
    ax[0].axhline(R[0]["random_heldout"]["mm_at_0.75s"]["calibrated"], ls=":", c="0.55", lw=1.2,
                  label="calibrated analytic, random actions")
    ax[0].set_ylabel("median 0.75 s open-loop pose error (mm)")
    ax[0].set_title("Prediction accuracy", fontsize=10)

    # (2) task success at the two replan intervals where the model can matter
    for chunk, style in ((8, "o-"), (15, "s-")):
        ax[1].plot(rounds, [r["sweep"][f"chunk{chunk}"]["success_rate"] for r in R], style,
                   lw=1.9, label=f"learned, chunk {chunk} ({chunk * 50} ms)")
    ref = report.get("reference", {})
    for cond, c, ls in (("mppi-calibrated", "#BA7517", "--"), ("mppi-naive", "#D85A30", ":")):
        if cond in ref:
            ax[1].axhline(ref[cond]["chunk15"]["success_rate"], ls=ls, c=c, lw=1.2,
                          label=f"{cond}, chunk 15")
    if diag and "d2_oracle" in diag:
        ax[1].axhline(diag["d2_oracle"]["chunk15"]["success_rate"], ls="-", c="#1D9E75", lw=1.4,
                      label="ORACLE (true slip model), chunk 15")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_ylabel("success rate")
    ax[1].set_title("Task success", fontsize=10)

    # (3) the mechanism: plan-vs-actual divergence
    for chunk, style in ((8, "o-"), (15, "s-")):
        ax[2].plot(rounds, [r["sweep"][f"chunk{chunk}"]["mean_plan_divergence_mm"] for r in R],
                   style, lw=1.9, label=f"learned, chunk {chunk}")
    if "mppi-calibrated" in ref:
        ax[2].axhline(ref["mppi-calibrated"]["chunk15"]["mean_plan_divergence_mm"],
                      ls="--", c="#BA7517", lw=1.2, label="calibrated, chunk 15")
    if diag and "d2_oracle" in diag:
        ax[2].axhline(diag["d2_oracle"]["chunk15"]["mean_plan_divergence_mm"],
                      ls="-", c="#1D9E75", lw=1.4, label="oracle, chunk 15")
    ax[2].set_ylabel("plan-vs-actual divergence (mm)")
    ax[2].set_title("Closed-loop model error", fontsize=10)

    for a in ax:
        a.set_xlabel("DAgger round")
        a.set_xticks(rounds)
        a.grid(alpha=0.25)
        a.legend(fontsize=7.5)
    fig.suptitle("M5 — DAgger on the learned world model: accuracy, success, and divergence",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "dagger_rounds.png", dpi=130)
    plt.close(fig)

    # The money plot: success against model accuracy, on one axis.
    #
    # x is each condition's OWN on-policy 0.75 s open-loop error, taken from the
    # corrected plan-vs-actual divergence of its own sweep - not a single error
    # measured on one condition's windows and reused for all of them.
    if diag and "d2_oracle" in diag:
        pts = []
        if "mppi-naive" in ref:
            pts.append(("naive", ref["mppi-naive"], "#D85A30"))
        if "mppi-calibrated" in ref:
            pts.append(("calibrated", ref["mppi-calibrated"], "#BA7517"))
        if "control_recalibrated" in report:
            pts.append(("recalibrated", report["control_recalibrated"]["sweep"], "#E0A33E"))
        for r, rep in zip(rounds, R):
            pts.append((f"DAgger r{r}", rep["sweep"], "#378ADD"))
        pts.append(("ORACLE", diag["d2_oracle"], "#1D9E75"))

        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        for chunk, marker, ls, dy in ((8, "o", "-", 12), (15, "s", "--", -15)):
            xs = [p[1][f"chunk{chunk}"]["mean_plan_divergence_mm"] for p in pts]
            ys = [p[1][f"chunk{chunk}"]["success_rate"] for p in pts]
            order = np.argsort(xs)[::-1]
            ax.plot(np.array(xs)[order], np.array(ys)[order], ls, c="0.6", lw=1.1, zorder=1)
            ax.scatter(xs, ys, s=78, marker=marker, zorder=3,
                       c=[p[2] for p in pts], edgecolors="white", linewidths=0.8,
                       label=f"chunk {chunk} ({chunk * 50} ms open loop)")
            # Stagger alternate labels so the tightly-clustered DAgger rounds
            # stay readable; they all sit within half a millimetre of each other,
            # which is the point of the figure.
            for i, ((lab, _, _), x, y) in enumerate(zip(pts, xs, ys)):
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(0, dy if i % 2 == 0 else dy * 1.9),
                            ha="center", fontsize=7.2, color="0.25")
        ax.set_xscale("log")
        ax.set_xlabel("on-policy 0.75 s open-loop pose error (mm, log scale) — "
                      "corrected plan-vs-actual divergence")
        ax.set_ylabel("gate-traversal success rate")
        ax.set_ylim(0.38, 1.06)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("Model accuracy buys task success only while it is comparable to the\n"
                     "task's spatial tolerance. Below ~4 mm into an 18-53 mm gate, it is free.",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(ASSETS / "accuracy_vs_success.png", dpi=130)
        plt.close(fig)
        print(f"  {ASSETS / 'accuracy_vs_success.png'}")

    # Action distributions: the mechanism M4 blamed, measured.
    stats = list(diag["action_stats"]) if diag else []
    stats += [r["action_stats"] for r in R if "action_stats" in r]
    if stats:
        fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.3))
        short = ["random walk" if "random" in s["label"] else
                 "MPPI " + s["label"].split("round-")[-1].replace(" model", "")
                 for s in stats]
        x = np.arange(len(stats))
        getters = {
            "mean_speed_m_s": lambda s: s["mean_speed_m_s"],
            "mean_abs_omega": lambda s: s["mean_abs"][2],
            "wheel_saturated_frac": lambda s: s["wheel_saturated_frac"],
            "jerk": lambda s: float(np.linalg.norm(s["mean_abs_jerk"][:2])),
        }
        for a, (key, title, fmt) in zip(ax, (
            ("mean_speed_m_s", "mean commanded translation speed (m/s)", "%.3f"),
            ("mean_abs_omega", "mean |commanded omega| (rad/s)", "%.3f"),
            ("wheel_saturated_frac", "ticks on the wheel-speed limit", "%.2f"),
            ("jerk", "mean |tick-to-tick action change| (m/s)", "%.4f"),
        )):
            vals = [getters[key](s) for s in stats]
            colours = ["#BA7517"] + ["#378ADD"] * (len(stats) - 1)
            bars = a.bar(x, vals, color=colours[:len(stats)])
            a.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
            a.set_xticks(x)
            a.set_xticklabels(short, fontsize=8, rotation=30, ha="right")
            a.set_ylim(0, max(vals) * 1.18)
            a.set_title(title, fontsize=9.5)
            a.grid(axis="y", alpha=0.25)
        fig.suptitle("The planner's actions are not wilder than the training distribution - they are "
                     "narrower,\nfaster forward, barely rotating and far smoother. And they do not "
                     "move as the model improves.", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(ASSETS / "action_distribution.png", dpi=130)
        plt.close(fig)

    print(f"  {ASSETS / 'dagger_rounds.png'}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="M5: DAgger on the learned world model")
    ap.add_argument("--stage", default="all", choices=["diag", "loop", "plots", "all"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=150, help="planner episodes per round")
    ap.add_argument("--trials", type=int, default=40, help="success-sweep trials per cell")
    ap.add_argument("--chunks", type=int, nargs="+", default=[8, 15])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--recollect", action="store_true")
    args = ap.parse_args()

    diag = report = None
    if args.stage in ("diag", "all"):
        diag = stage_diag(args)
    if args.stage in ("loop", "all"):
        report = stage_loop(args)
    if args.stage == "plots":
        diag = json.loads((ASSETS / "metrics-diag.json").read_text())
        report = json.loads((ASSETS / "metrics.json").read_text())
    if report is not None:
        if diag is None and (ASSETS / "metrics-diag.json").exists():
            diag = json.loads((ASSETS / "metrics-diag.json").read_text())
        plots(report, diag)


if __name__ == "__main__":
    main()
