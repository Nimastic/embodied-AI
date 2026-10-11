"""
Phase 0 dataset collection: write episodes in the REAL dataset schema.

The schema is defined once in software/README.md and must not drift between sim
and hardware - every model, plot and experiment depends on it. When the robot
exists, robot/logger.py writes these same columns and the training code does not
change.

Modes:
    random   temporally correlated random actions. The DEFAULT for world-model
             training: covers the action space, including combinations no
             sensible controller would ever choose.
    expert   drive to randomly placed goal poses with a P controller. Narrow
             action coverage, useful as a comparison and for M4 task data.
    teleop   drive it yourself in the viewer (WASD / QE / space).

Usage:
    python collect.py --minutes 10                        # random, hard floor
    python collect.py --minutes 10 --surface mat
    python collect.py --mode expert --minutes 5
    python collect.py --mode teleop
    python collect.py --minutes 2 --no-video              # fast, state only
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.mecanum_env import (
    ACTION_LIMITS,
    clamp_action,
    CONTROL_DT,
    CONTROL_HZ,
    MODEL_IMAGE_SIZE,
    MecanumEnv,
    smoothed_random_actions,
    wall_avoiding_action,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sim"


# ── Policies ────────────────────────────────────────────────────────────────

def expert_action(obs: dict, goal: np.ndarray) -> np.ndarray:
    """
    P controller to a goal pose, in the body frame. Deliberately simple: this is
    a data-collection policy, not a contribution.
    """
    x, y, th = obs["pose_gt"]
    dx, dy = goal[0] - x, goal[1] - y
    c, s = np.cos(th), np.sin(th)
    err_b = np.array([dx * c + dy * s, -dx * s + dy * c])
    dth = (goal[2] - th + np.pi) % (2 * np.pi) - np.pi
    # Must go through clamp_action, same as the random sampler: a P controller
    # far from its goal happily asks for all three axes at once, which no wheel
    # budget can deliver.
    return clamp_action([1.6 * err_b[0], 1.6 * err_b[1], 2.0 * dth])


# ── Episode runner ──────────────────────────────────────────────────────────

def run_episode(
    env: MecanumEnv,
    episode_id: str,
    n_ticks: int,
    mode: str,
    rng: np.random.Generator,
    record_video: bool,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    obs = env.reset(randomise_gate=(mode == "expert"))
    plan = smoothed_random_actions(n_ticks, rng) if mode == "random" else None
    goal = _sample_goal(rng)

    rows, frames = [], []
    for i in range(n_ticks):
        if mode == "random":
            action = wall_avoiding_action(obs, plan[i])
        else:
            action = expert_action(obs, goal)
            if np.linalg.norm(obs["pose_gt"][:2] - goal[:2]) < 0.06:
                goal = _sample_goal(rng)

        if record_video:
            frames.append(env.render("front", MODEL_IMAGE_SIZE))
        frame_idx = len(frames) - 1 if record_video else -1

        obs = env.step(action)
        rows.append(_row(episode_id, obs, frame_idx))

        if obs["out_of_bounds"]:
            obs = env.reset()

    return pd.DataFrame(rows), frames


def _sample_goal(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-np.pi, np.pi)])


def _row(episode_id: str, obs: dict, frame_idx: int) -> dict:
    """One row per control tick, matching the schema in software/README.md."""
    return {
        "episode_id": episode_id,
        "tick": obs["tick"],
        "t_mono": obs["t_mono"],
        "action_cmd": obs["action_cmd"].astype(np.float32),
        "wheel_cmd": obs["wheel_cmd"].astype(np.float32),
        "wheel_meas": obs["wheel_meas"].astype(np.float32),
        "encoder_counts": obs["encoder_counts"].astype(np.int32),
        "imu_gyro": obs["imu_gyro"].astype(np.float32),
        "imu_accel": obs["imu_accel"].astype(np.float32),
        "pose_gt": obs["pose_gt"].astype(np.float32),
        "pose_gt_valid": obs["pose_gt_valid"],
        "pose_odom": obs["pose_odom"].astype(np.float32),
        "body_vel_gt": obs["body_vel_gt"].astype(np.float32),
        "body_vel_avg": obs["body_vel_avg"].astype(np.float32),
        "scrub": np.float32(obs["scrub"]),
        "batt_v": np.float32(obs["batt_v"]),
        "saturated": obs["saturated"],
        "frame_idx": np.int32(frame_idx),
        "t_frame_mono": obs["t_mono"],      # sim renders synchronously; real rig will not
        "surface": obs["surface"],
        "shell": obs["shell"],
    }


# ── Teleop ──────────────────────────────────────────────────────────────────

def run_teleop(env: MecanumEnv, episode_id: str, record_video: bool):
    """
    Drive it yourself. Uses the passive viewer's key_callback, so no extra
    dependency - but note it fires on key press/repeat rather than on hold, so
    the controls accumulate velocity rather than tracking a held key.

        W / S   forward / back        A / D   strafe left / right
        Q / E   rotate CCW / CW       SPACE   stop
    """
    import mujoco.viewer

    cmd = np.zeros(3)
    step = np.array([0.12, 0.12, 0.5])

    def on_key(keycode: int) -> None:
        nonlocal cmd
        ch = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
        if ch == "W":
            cmd[0] += step[0]
        elif ch == "S":
            cmd[0] -= step[0]
        elif ch == "A":
            cmd[1] += step[1]
        elif ch == "D":
            cmd[1] -= step[1]
        elif ch == "Q":
            cmd[2] += step[2]
        elif ch == "E":
            cmd[2] -= step[2]
        elif ch == " ":
            cmd[:] = 0.0
        cmd[:] = clamp_action(cmd)

    obs = env.reset()
    rows, frames = [], []
    print(run_teleop.__doc__)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            if record_video:
                frames.append(env.render("front", MODEL_IMAGE_SIZE))
            obs = env.step(cmd)
            rows.append(_row(episode_id, obs, len(frames) - 1 if record_video else -1))
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - t0)))
    return pd.DataFrame(rows), frames


# ── Output ──────────────────────────────────────────────────────────────────

def write_mp4(frames: np.ndarray, path: Path, fps: int) -> bool:
    """
    Human-viewable mp4. Best effort: mediapy, then OpenCV. Never load-bearing -
    the lossless .npy is the dataset, this is for looking at.
    """
    try:
        import mediapy

        mediapy.write_video(str(path), frames, fps=fps)
        return True
    except Exception:
        pass
    try:
        import cv2

        h, w = frames.shape[1:3]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            return False
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return True
    except Exception:
        return False


def save_episode(
    df: pd.DataFrame,
    frames: list[np.ndarray],
    out_dir: Path,
    episode_id: str,
    manifest: dict,
    also_mp4: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{episode_id}.parquet", index=False)

    if frames:
        # Frames go to a LOSSLESS uint8 .npy, not an mp4.
        #
        # Two reasons. (1) h.264 artifacts land directly in the model's
        # observation, and at 64x64 the blocking is a meaningful fraction of the
        # image - you do not want your world model learning the codec. (2) .npy
        # memory-maps, so training reads frames without decoding anything.
        #
        # The real robot WILL need compression (a Pi cannot write raw frames at
        # 30 Hz for an hour), so that difference is itself worth an experiment
        # later: how much does compression cost prediction accuracy?
        stack = np.stack(frames)
        np.save(out_dir / f"{episode_id}.frames.npy", stack)
        if also_mp4 and not write_mp4(stack, out_dir / f"{episode_id}.mp4", int(CONTROL_HZ)):
            print("    (mp4 writer unavailable; .npy written)")

    manifest = dict(manifest)
    manifest.update(
        {
            "episode_id": episode_id,
            "n_ticks": int(len(df)),
            "duration_s": round(len(df) * CONTROL_DT, 2),
            "n_frames": len(frames),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    (out_dir / f"{episode_id}.manifest.json").write_text(json.dumps(manifest, indent=2))


def _git_commit() -> str:
    """Pin the dataset to the code that produced it. Reproducibility is cheap here."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 0 sim episodes")
    parser.add_argument("--mode", default="random", choices=["random", "expert", "teleop"])
    parser.add_argument("--surface", default="hard", choices=["hard", "mat"])
    parser.add_argument("--minutes", type=float, default=10.0, help="total data to collect")
    parser.add_argument("--episode-s", type=float, default=60.0, help="seconds per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="state only; much faster")
    parser.add_argument("--mp4", action="store_true", help="also write a viewable mp4")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (DATA_DIR / args.surface / args.mode)
    record_video = not args.no_video
    rng = np.random.default_rng(args.seed)
    env = MecanumEnv(surface=args.surface, seed=args.seed)

    base_manifest = {
        "mode": args.mode,
        "surface": args.surface,
        "control_hz": CONTROL_HZ,
        "action_limits": ACTION_LIMITS.tolist(),
        "image_size": MODEL_IMAGE_SIZE,
        "geometry": {
            "wheel_radius": env.kin.geom.wheel_radius,
            "half_wheelbase": env.kin.geom.half_wheelbase,
            "half_track": env.kin.geom.half_track,
        },
        "slip_params": env.slip.params.name,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "source": "sim",
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.mode == "teleop":
        episode_id = f"{stamp}-teleop-000"
        df, frames = run_teleop(env, episode_id, record_video)
        if len(df):
            save_episode(df, frames, out_dir, episode_id, base_manifest, args.mp4)
            print(f"  {episode_id}: {len(df)} ticks -> {out_dir}")
        env.close()
        return

    n_episodes = max(1, int(round(args.minutes * 60.0 / args.episode_s)))
    ticks = int(args.episode_s * CONTROL_HZ)
    print(f"collecting {n_episodes} x {args.episode_s:.0f}s = {args.minutes:.1f} min  "
          f"[{args.mode}, {args.surface}, video={record_video}]")

    t0 = time.perf_counter()
    total_rows = 0
    for ep in range(n_episodes):
        episode_id = f"{stamp}-{args.mode}-{ep:03d}"
        df, frames = run_episode(env, episode_id, ticks, args.mode, rng, record_video)
        save_episode(df, frames, out_dir, episode_id, base_manifest, args.mp4)
        total_rows += len(df)
        print(f"  {episode_id}: {len(df)} ticks, {len(frames)} frames")
    env.close()

    wall = time.perf_counter() - t0
    sim_s = total_rows * CONTROL_DT
    print(f"\n{total_rows} ticks ({sim_s/60:.1f} min of robot time) in {wall:.1f}s wall "
          f"= {sim_s/wall:.0f}x real time")
    print(f"written to {out_dir}")

    # A real-robot sanity check that is free to run here and essential there:
    # mean |scrub| should be small, and odometry should over-read.
    all_df = pd.concat(
        [pd.read_parquet(p) for p in sorted(out_dir.glob(f"{stamp}-*.parquet"))],
        ignore_index=True,
    )
    scrub = np.abs(all_df["scrub"].to_numpy()).mean()
    speed = np.linalg.norm(np.stack(all_df["body_vel_gt"].to_numpy())[:, :2], axis=1)
    print(f"\nsanity: mean |scrub| = {scrub:.3f} rad/s   "
          f"mean speed = {speed.mean():.3f} m/s   "
          f"saturated ticks = {all_df['saturated'].mean():.1%}")


if __name__ == "__main__":
    main()
