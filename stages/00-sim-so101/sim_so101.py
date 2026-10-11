"""
Stage 00 — Simulated SO-101: observation → action → physics → next observation

A simple reaching task: move the gripper to a target position using
proportional control in Cartesian space, resolved through MuJoCo's
built-in position actuators.

Usage:
    python sim_so101.py              # Run headless, save video + metrics
    python sim_so101.py --viewer     # Launch interactive MuJoCo viewer
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_XML = ROOT / "mujoco_menagerie" / "trs_so_arm100" / "scene.xml"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ── Task configuration ─────────────────────────────────────────────────────
TARGET_POS = np.array([0.10, -0.10, 0.12])   # xyz target for gripper
REACHING_THRESHOLD = 0.015                    # 15 mm = success
MAX_STEPS = 500                               # sim steps
CTRL_DT = 0.02                                # 50 Hz control
RENDER_W, RENDER_H = 640, 480


def get_ee_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the Cartesian position of the gripper (Fixed_Jaw body)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    return data.xpos[body_id].copy()


def get_ee_site_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Try to get an end-effector site, fall back to body position."""
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        return data.site_xpos[site_id].copy()
    except Exception:
        return get_ee_pos(model, data)


def build_scene_with_target(target_pos: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the SO-ARM100 scene and add a visual target sphere + table."""
    scene_xml = MODEL_XML.read_text()

    # Inject a target ball and a small table into the worldbody
    injection = f"""
    <!-- target marker -->
    <body name="target" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}">
      <geom type="sphere" size="0.012" rgba="1 0.2 0.2 0.7" contype="0" conaffinity="0"/>
    </body>
    <!-- small table surface -->
    <body name="table_surface" pos="0.12 0 0.005">
      <geom type="box" size="0.15 0.15 0.005" rgba="0.6 0.5 0.4 1"/>
    </body>
    <!-- cube to push/reach -->
    <body name="cube" pos="{target_pos[0]} {target_pos[1]} {target_pos[2] + 0.015}">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" rgba="0.2 0.6 1.0 1" mass="0.05"
            friction="1 0.005 0.0001"/>
    </body>
"""
    # Insert before closing </worldbody>
    scene_xml = scene_xml.replace("</worldbody>", injection + "\n  </worldbody>")

    # MuJoCo needs the model directory to resolve <include> and mesh paths
    model_dir = str(MODEL_XML.parent)
    model = mujoco.MjModel.from_xml_string(scene_xml, assets=_load_assets(model_dir))
    model.opt.timestep = 0.002  # 500 Hz physics
    data = mujoco.MjData(model)
    return model, data


def _load_assets(model_dir: str) -> dict[str, bytes]:
    """Load all files referenced by the MJCF as a flat dict for from_xml_string."""
    assets = {}
    model_path = Path(model_dir)

    # Load the included XML
    arm_xml_path = model_path / "so_arm100.xml"
    if arm_xml_path.exists():
        assets["so_arm100.xml"] = arm_xml_path.read_bytes()

    # Load all mesh files
    mesh_dir = model_path / "assets"
    if mesh_dir.exists():
        for f in mesh_dir.iterdir():
            if f.is_file():
                assets[f.name] = f.read_bytes()

    return assets


def simple_ik_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    gain: float = 8.0,
    damping: float = 0.01,
) -> np.ndarray:
    """
    Resolved-rate IK: compute joint velocity direction from Jacobian,
    then convert to position target deltas for the position actuators.

    This is the simplest "close the loop" controller: pure Cartesian
    proportional control through the Jacobian pseudoinverse.
    """
    ee_pos = get_ee_pos(model, data)
    error = target - ee_pos

    # Get the translational Jacobian for the Fixed_Jaw body
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    nv = model.nv
    jacp = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jacp, None, body_id)

    # Use only the first 5 joints (not the jaw/gripper)
    J = jacp[:, :5]

    # Damped least squares pseudoinverse
    JtJ = J.T @ J + damping * np.eye(5)
    dq = np.linalg.solve(JtJ, J.T @ error) * gain

    # Current joint positions + delta → new position targets
    ctrl = np.zeros(model.nu)
    for i in range(5):
        ctrl[i] = data.qpos[i] + dq[i] * CTRL_DT
        # Clamp to joint limits
        jnt_id = model.actuator_trnid[i, 0]
        ctrl[i] = np.clip(ctrl[i], model.jnt_range[jnt_id, 0], model.jnt_range[jnt_id, 1])

    # Keep gripper open
    ctrl[5] = 0.5

    return ctrl


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    render_frames: bool = True,
) -> dict:
    """
    Run one reaching episode.

    Returns a dict with metrics:
      - success: bool
      - steps: int
      - final_distance: float
      - min_distance: float
      - trajectory: list of (ee_pos, joint_pos, distance) per step
      - frames: list of RGB arrays (if render_frames)
    """
    # Reset to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    renderer = None
    if render_frames:
        renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)

    trajectory = []
    frames = []
    min_dist = float("inf")
    steps_per_ctrl = int(CTRL_DT / model.opt.timestep)

    for step in range(MAX_STEPS):
        # ── Observe ─────────────────────────────────────────────────────
        ee_pos = get_ee_pos(model, data)
        joint_pos = data.qpos[:6].copy()
        dist = np.linalg.norm(ee_pos - target)
        min_dist = min(min_dist, dist)

        trajectory.append({
            "step": step,
            "ee_pos": ee_pos.tolist(),
            "joint_pos": joint_pos.tolist(),
            "distance": float(dist),
        })

        # ── Render ──────────────────────────────────────────────────────
        if renderer is not None and step % 2 == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

        # ── Check success ───────────────────────────────────────────────
        if dist < REACHING_THRESHOLD:
            # Hold position for a few more frames for the video
            for _ in range(20):
                mujoco.mj_step(model, data)
                if renderer is not None:
                    renderer.update_scene(data)
                    frames.append(renderer.render().copy())
            break

        # ── Act ─────────────────────────────────────────────────────────
        ctrl = simple_ik_controller(model, data, target)
        data.ctrl[:] = ctrl

        # ── Physics step ────────────────────────────────────────────────
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)

    final_dist = np.linalg.norm(get_ee_pos(model, data) - target)
    success = final_dist < REACHING_THRESHOLD

    if renderer is not None:
        renderer.close()

    return {
        "success": success,
        "steps": len(trajectory),
        "final_distance": float(final_dist),
        "min_distance": float(min_dist),
        "trajectory": trajectory,
        "frames": frames,
    }


def save_video(frames: list[np.ndarray], path: Path, fps: int = 25):
    """Save frames as MP4 video using mediapy or fallback to raw frames."""
    try:
        import mediapy as media
        media.write_video(str(path), frames, fps=fps)
        print(f"  Video saved: {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    except ImportError:
        # Fallback: save as individual PNGs
        frame_dir = path.parent / "frames"
        frame_dir.mkdir(exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(frame_dir / f"frame_{i:04d}.png")
        print(f"  Frames saved to {frame_dir}/ ({len(frames)} frames)")


def save_trajectory_plot(trajectory: list[dict], target: np.ndarray, path: Path):
    """Plot distance-to-target over time and end-effector path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [t["step"] for t in trajectory]
    dists = [t["distance"] for t in trajectory]
    ee_xs = [t["ee_pos"][0] for t in trajectory]
    ee_ys = [t["ee_pos"][1] for t in trajectory]
    ee_zs = [t["ee_pos"][2] for t in trajectory]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distance over time
    ax = axes[0]
    ax.plot(steps, [d * 1000 for d in dists], "b-", linewidth=1.5, label="distance")
    ax.axhline(REACHING_THRESHOLD * 1000, color="r", linestyle="--", alpha=0.7, label=f"threshold ({REACHING_THRESHOLD*1000:.0f} mm)")
    ax.set_xlabel("Control step")
    ax.set_ylabel("Distance to target (mm)")
    ax.set_title("Reaching convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3D trajectory
    ax2 = fig.add_subplot(122, projection="3d")
    axes[1].set_visible(False)
    ax2.plot(ee_xs, ee_ys, ee_zs, "b-", linewidth=1.5, alpha=0.8, label="EE path")
    ax2.scatter(*target, color="red", s=100, marker="*", label="target")
    ax2.scatter(ee_xs[0], ee_ys[0], ee_zs[0], color="green", s=60, marker="o", label="start")
    ax2.scatter(ee_xs[-1], ee_ys[-1], ee_zs[-1], color="blue", s=60, marker="s", label="end")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title("End-effector trajectory")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def run_evaluation(model, data, n_targets: int = 10) -> list[dict]:
    """Run multiple reaching episodes with random targets to get a success rate."""
    rng = np.random.default_rng(42)
    results = []

    for i in range(n_targets):
        # Random reachable target in front of the arm
        t = np.array([
            rng.uniform(0.05, 0.18),   # x: forward
            rng.uniform(-0.15, 0.15),  # y: left-right
            rng.uniform(0.05, 0.20),   # z: height
        ])

        # Update target body position
        target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        model.body_pos[target_body] = t

        result = run_episode(model, data, t, render_frames=False)
        results.append({
            "target": t.tolist(),
            "success": bool(result["success"]),
            "final_distance": float(result["final_distance"]),
            "min_distance": float(result["min_distance"]),
            "steps": int(result["steps"]),
        })
        status = "OK" if result["success"] else "MISS"
        print(f"  Target {i+1:2d}: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]  "
              f"dist={result['final_distance']*1000:.1f}mm  {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 00: Simulated SO-101 reaching task")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive viewer")
    parser.add_argument("--eval", type=int, default=10, help="Number of random targets for evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 00: Simulated SO-101 — Reaching Task")
    print("=" * 60)

    # ── Build scene ─────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_XML}")
    model, data = build_scene_with_target(TARGET_POS)

    joint_names = [model.jnt(i).name for i in range(model.njnt)]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    print(f"  Joints ({model.njnt}): {joint_names}")
    print(f"  Actuators ({model.nu}): {actuator_names}")
    print(f"  Timestep: {model.opt.timestep}s, Control dt: {CTRL_DT}s")

    if args.viewer:
        print("\nLaunching interactive viewer...")
        print("  Press ESC to exit, SPACE to pause")
        import mujoco.viewer
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                ctrl = simple_ik_controller(model, data, TARGET_POS)
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
        return

    # ── Primary episode (with video) ────────────────────────────────────
    print(f"\n--- Primary reaching episode ---")
    print(f"  Target: {TARGET_POS}")

    result = run_episode(model, data, TARGET_POS, render_frames=True)

    status = "SUCCESS" if result["success"] else "FAILED"
    print(f"\n  Result: {status}")
    print(f"  Steps: {result['steps']}")
    print(f"  Final distance: {result['final_distance']*1000:.1f} mm")
    print(f"  Min distance:   {result['min_distance']*1000:.1f} mm")

    # Save video
    if result["frames"]:
        save_video(result["frames"], ASSETS_DIR / "reaching_episode.mp4")

    # Save trajectory plot
    save_trajectory_plot(result["trajectory"], TARGET_POS, ASSETS_DIR / "trajectory_plot.png")

    # ── Multi-target evaluation ─────────────────────────────────────────
    print(f"\n--- Evaluation: {args.eval} random targets ---")
    eval_results = run_evaluation(model, data, n_targets=args.eval)

    successes = sum(1 for r in eval_results if r["success"])
    success_rate = successes / len(eval_results)
    avg_dist = np.mean([r["final_distance"] for r in eval_results])
    avg_min = np.mean([r["min_distance"] for r in eval_results])

    print(f"\n  Success rate: {successes}/{len(eval_results)} ({success_rate*100:.0f}%)")
    print(f"  Avg final distance: {avg_dist*1000:.1f} mm")
    print(f"  Avg min distance:   {avg_min*1000:.1f} mm")

    # ── Save metrics ────────────────────────────────────────────────────
    metrics = {
        "primary_episode": {
            "target": TARGET_POS.tolist(),
            "success": bool(result["success"]),
            "steps": int(result["steps"]),
            "final_distance_mm": round(float(result["final_distance"]) * 1000, 1),
            "min_distance_mm": round(float(result["min_distance"]) * 1000, 1),
        },
        "evaluation": {
            "n_targets": len(eval_results),
            "success_rate": round(float(success_rate), 2),
            "successes": int(successes),
            "avg_final_distance_mm": round(float(avg_dist) * 1000, 1),
            "avg_min_distance_mm": round(float(avg_min) * 1000, 1),
            "results": eval_results,
        },
        "config": {
            "reaching_threshold_mm": REACHING_THRESHOLD * 1000,
            "max_steps": MAX_STEPS,
            "control_dt": CTRL_DT,
            "physics_timestep": model.opt.timestep,
            "controller": "damped_least_squares_ik",
            "gain": 8.0,
            "damping": 0.01,
        },
    }

    metrics_path = ASSETS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("Stage 00 complete. Artifacts in assets/")
    print("=" * 60)


if __name__ == "__main__":
    main()
