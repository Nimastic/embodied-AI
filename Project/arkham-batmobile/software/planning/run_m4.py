"""
M4 evaluation: does planning inside a learned world model beat planning inside
an analytic one, and does either beat not planning at all?

Four conditions, same task instances, same seeds, same cost function:

    reactive            P controller at the goal, no model
    mppi-naive          MPPI over uncalibrated analytic FK
    mppi-calibrated     MPPI over calibrated analytic FK
    mppi-learned        MPPI over the M2 residual GRU

Also logs planned-vs-actual divergence every replan, which is the model
exploitation check: a planner that loves its model more than reality shows up
as a growing gap between where the plan said it would be and where it is.

Usage:
    python run_m4.py                      # 30 trials per condition, hard floor
    python run_m4.py --trials 10 --video
    python run_m4.py --surface mat        # transfer: model trained on hard
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.mecanum import MecanumKinematics  # noqa: E402
from planning.gate_task import GateTask  # noqa: E402
from planning.mppi import AnalyticPredictor, LearnedPredictor, MPPI, reactive_action  # noqa: E402
from sim.mecanum_env import CONTROL_DT, MecanumEnv, clamp_action  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "state_model-residual-h1.pt"
ASSETS = ROOT.parent / "experiments" / "M4-world-model-planning" / "assets"

MAX_TICKS = 80           # 4 s. A generous limit lets the planner creep through
                         # at near-zero speed, which hides model error entirely.
                         # Forcing commitment is what makes accuracy matter.
N_EXECUTE = 3            # actions executed per replan (150 ms chunk)
HISTORY = 1


def run_trial(env, task, condition, predictor, kin, seed, n_execute, record=False):
    obs = env.reset(pose=task.start)
    env.data.mocap_pos[0] = [*task.post_l, 0.10]
    env.data.mocap_pos[1] = [*task.post_r, 0.10]

    planner = None
    if condition != "reactive":
        planner = MPPI(predictor, task.cost, kin, seed=seed)
        planner.reset()

    hist_buf = [_feat(obs)]
    traj, frames, divergences = [obs["pose_gt"].copy()], [], []
    collided = False
    plan_times = []
    plan = None
    t = 0

    while t < MAX_TICKS:
        if condition == "reactive":
            actions = [reactive_action(obs["pose_gt"], task.goal, kin)]
        else:
            t0 = time.perf_counter()
            hist = np.asarray(hist_buf[-max(HISTORY, 1):], dtype=np.float32)
            plan = planner.plan(obs["pose_gt"], hist)
            plan_times.append(time.perf_counter() - t0)
            actions = plan[:n_execute]
            # Keep the whole predicted path; it can only be indexed correctly
            # after we know how many actions actually ran (see below).
            predicted_path = planner.last_diagnostics["predicted_path"]

        n_done = 0
        for a in actions:
            obs = env.step(clamp_action(a))
            hist_buf.append(_feat(obs))
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
            # Index by what ACTUALLY executed, not by the chunk length. This loop
            # breaks early when the goal is reached, so on the last replan of
            # nearly every successful episode fewer actions run than were
            # planned. Indexing at len(actions)-1 then scored a 15-step
            # prediction against a 3-step outcome and reported a predictor with
            # 0.27 mm of true error at 21 mm.
            #
            # This is the SECOND time this comparison has been mismatched. The
            # first version compared the full 0.75 s horizon against a 0.15 s
            # outcome; fixing that left this subtler case behind. Any
            # predicted-vs-actual comparison needs both sides pinned to the same
            # number of elapsed ticks, and "the number we intended to execute"
            # is not that number.
            planner.shift(n_done)
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


def _feat(obs) -> np.ndarray:
    return np.concatenate(
        [obs["action_cmd"], obs["body_vel_avg"], obs["wheel_meas"], obs["imu_gyro"],
         [obs["batt_v"]]]
    ).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="M4: planning inside a learned world model")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--surface", default="hard", choices=["hard", "mat"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 3, 8, 15],
                    help="actions executed per replan; the replan-interval sweep")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    kin = MecanumKinematics()
    if not CKPT.exists():
        raise SystemExit(f"no checkpoint at {CKPT} - train it first:\n"
                         f"  python models/state_model.py --epochs 25 --stride 2 --history 1")
    learned = LearnedPredictor(kin, CKPT)
    predictors = {
        "mppi-naive": AnalyticPredictor(kin, name="naive"),
        "mppi-calibrated": AnalyticPredictor(kin, learned.gains, learned.tau, "calibrated"),
        "mppi-learned": learned,
    }
    print(f"model trained on hard, evaluating on '{args.surface}'")
    print(f"calibration: gains {np.round(learned.gains, 3)}  tau {learned.tau * 1000:.0f} ms")
    print(f"{args.trials} trials per cell, gate clearance 38-78 mm per side\n")

    env = MecanumEnv(surface=args.surface, seed=args.seed)
    report, all_traj = {}, {}

    # Reactive is model-free and replans every tick by construction, so it is a
    # single row rather than part of the sweep.
    results = []
    for i in range(args.trials):
        task = GateTask(np.random.default_rng(args.seed + i))
        r = run_trial(env, task, "reactive", None, kin, args.seed + i, 1)
        all_traj.setdefault("reactive", []).append((task, r["traj"], r["success"]))
        r.pop("traj"); r.pop("frames")
        results.append(r)
    report["reactive"] = _summarise(results, args.trials)
    _print_row("reactive", "-", report["reactive"])

    for chunk in args.chunks:
        for cond, predictor in predictors.items():
            results, first_frames = [], []
            for i in range(args.trials):
                task = GateTask(np.random.default_rng(args.seed + i))
                want = args.video and i == 0 and chunk == 8 and cond == "mppi-learned"
                r = run_trial(env, task, cond, predictor, kin, args.seed + i, chunk, record=want)
                if want:
                    first_frames = r["frames"]
                if chunk == 8:
                    all_traj.setdefault(cond, []).append((task, r["traj"], r["success"]))
                r.pop("traj"); r.pop("frames")
                results.append(r)
            key = f"{cond}@chunk{chunk}"
            report[key] = _summarise(results, args.trials)
            _print_row(cond, chunk, report[key])
            if first_frames:
                from sim.collect import write_mp4
                ASSETS.mkdir(parents=True, exist_ok=True)
                write_mp4(np.stack(first_frames), ASSETS / "mppi-learned.mp4",
                          int(1 / CONTROL_DT))
        print()

    env.close()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / f"metrics-{args.surface}.json").write_text(
        json.dumps({"surface": args.surface, "trials": args.trials,
                    "chunks": args.chunks, "conditions": report}, indent=2))
    _plot_sweep(report, args.chunks, args.surface)
    _plot_traj(all_traj, report, args.surface)
    print(f"  {ASSETS / f'metrics-{args.surface}.json'}")


def _summarise(results, n_trials) -> dict:
    succ = np.array([r["success"] for r in results])
    coll = np.array([r["collided"] for r in results])
    times = np.array([r["time_s"] for r in results if r["success"]])
    dmin = np.array([r["min_post_dist_mm"] for r in results])
    div = [r["mean_plan_divergence_mm"] for r in results if r["mean_plan_divergence_mm"]]
    pms = [r["plan_ms"] for r in results if r["plan_ms"]]
    return {
        "success_rate": round(float(succ.mean()), 3),
        "collision_rate": round(float(coll.mean()), 3),
        "median_time_s": round(float(np.median(times)), 2) if len(times) else None,
        "median_min_post_dist_mm": round(float(np.median(dmin)), 1),
        "mean_plan_divergence_mm": round(float(np.mean(div)), 2) if div else None,
        "median_plan_ms": round(float(np.median(pms)), 1) if pms else None,
        "n_trials": n_trials,
    }


def _print_row(cond, chunk, r) -> None:
    print(f"  {cond:17s} chunk {str(chunk):>3s}  success {r['success_rate']:5.1%}   "
          f"collisions {r['collision_rate']:5.1%}   time {str(r['median_time_s']):>5s}s   "
          f"plan {str(r['median_plan_ms']):>6s}ms   divergence "
          f"{str(r['mean_plan_divergence_mm']):>6s}mm")


def _plot_sweep(report, chunks, surface) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ["mppi-naive", "mppi-calibrated", "mppi-learned"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.array(chunks) * CONTROL_DT * 1000

    for cond in conds:
        axes[0].plot(x, [report[f"{cond}@chunk{c}"]["success_rate"] for c in chunks],
                     "o-", lw=1.9, label=cond)
        axes[1].plot(x, [report[f"{cond}@chunk{c}"]["mean_plan_divergence_mm"] for c in chunks],
                     "o-", lw=1.9, label=cond)
    axes[0].axhline(report["reactive"]["success_rate"], ls="--", c="0.4", lw=1.2,
                    label="reactive (no model)")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("plan-vs-actual divergence (mm)")
    for ax in axes:
        ax.set_xlabel("open-loop chunk executed per replan (ms)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"M4 — model quality matters only when you cannot replan often ({surface} floor)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / f"replan_sweep-{surface}.png", dpi=130)
    plt.close(fig)


def _plot_traj(all_traj, report, surface) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(all_traj)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.9 * len(conds), 4.3), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        key = cond if cond == "reactive" else f"{cond}@chunk8"
        for task, traj, ok in all_traj[cond][:14]:
            ax.plot(traj[:, 0], traj[:, 1], lw=1.1, alpha=0.75,
                    color="#1D9E75" if ok else "#D85A30")
            ax.plot(*task.post_l, "o", ms=4, color="#BA7517")
            ax.plot(*task.post_r, "o", ms=4, color="#BA7517")
            ax.plot(*task.goal[:2], "x", ms=5, color="#378ADD")
        ax.set_title(f"{cond}\nsuccess {report[key]['success_rate']:.0%}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Gate traversal, 400 ms chunks, {surface} floor — "
                 f"green = success, orange = failure", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / f"trajectories-{surface}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
