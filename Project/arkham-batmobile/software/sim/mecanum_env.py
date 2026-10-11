"""
Phase 0 mecanum base environment.

Mirrors the REAL robot's interface exactly, so the same training and planning
code runs against sim or hardware with only the dataset path changing:

    action  = (vx, vy, omega)   body frame, 20 Hz
    obs     = pose, velocity, wheel speeds, encoder counts, IMU, battery, frame

Architecture (see mecanum_base.xml for why the wheels are visual only):

    action -> slip_model -> achieved body velocity -> rotate to world frame
           -> MuJoCo velocity actuators (first-order lag) -> physics @ 500 Hz

Two layers of imperfection, each standing in for a different real effect:
    slip_model      wheel/floor interface  (encoders cannot see it)
    actuator lag    motor + PID bandwidth  (encoders can see it)

Usage:
    python mecanum_env.py                  # verify: lag, baseline error, artifacts
    python mecanum_env.py --viewer         # interactive viewer, random actions
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.mecanum import MecanumGeometry, MecanumKinematics, body_to_world  # noqa: E402
from sim.slip_model import SlipModel  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_XML = ROOT / "mecanum_base.xml"
ASSETS_DIR = ROOT.parents[1] / "experiments" / "P0-sim-baseline" / "assets"

# ── Interface constants: these must match the real robot ────────────────────
CONTROL_HZ = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ
ACTION_LIMITS = np.array([0.50, 0.50, 2.00])     # vx m/s, vy m/s, omega rad/s
WHEEL_SPEED_MAX = 14.0                           # rad/s; must match SlipModel
#
# NOTE: ACTION_LIMITS is a BOX but the achievable set is a POLYTOPE - the three
# axes share one wheel-speed budget. Commands are passed through
# kin.clamp_achievable() before use so sampled actions are realisable by
# construction; otherwise most ticks saturate and the dataset fills up with
# commands the robot never actually executed.
MODEL_IMAGE_SIZE = 64                            # what the world model sees
ENCODER_COUNTS_PER_RAD = 210.0                   # 1320 CPR output shaft / 2pi
ARENA_HALF = 0.92                                # keep the hull clear of walls


class MecanumEnv:
    """Velocity-controlled planar mecanum base in MuJoCo."""

    def __init__(
        self,
        surface: str = "hard",
        seed: int = 0,
        image_size: int = MODEL_IMAGE_SIZE,
        geometry: MecanumGeometry | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        self.kin = MecanumKinematics(geometry)
        self.slip = SlipModel(surface=surface, kinematics=self.kin, seed=seed)
        self.surface = surface
        self.rng = np.random.default_rng(seed)

        self.image_size = image_size
        self._renderer: mujoco.Renderer | None = None

        self.n_substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        assert np.isclose(self.n_substeps * self.model.opt.timestep, CONTROL_DT)

        self._jid = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("slide_x", "slide_y", "hinge_z")
        }
        self._gate_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gate_l", "gate_r")
        ]
        self._apply_surface_appearance(surface)
        self.tick = 0
        self.encoder_counts = np.zeros(4)
        self._vel_avg = np.zeros(3)
        self.odom_pose = np.zeros(3)

    def _apply_surface_appearance(self, surface: str) -> None:
        """
        Make the floor LOOK like the surface it physically is.

        M2b measured the surface to be only ~69% identifiable from
        proprioception even with 4 s of history, so a camera may be the only
        practical route to knowing what you are driving on. That is testable
        only if appearance and friction are coupled, as they are in reality.
        """
        mat_name = {"hard": "floor_hard", "mat": "floor_mat_surf"}[surface]
        mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.model.geom_matid[floor_id] = mat_id

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def reset(self, pose: np.ndarray | None = None, randomise_gate: bool = False) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        if pose is None:
            pose = np.array(
                [
                    self.rng.uniform(-0.55, 0.55),
                    self.rng.uniform(-0.55, 0.55),
                    self.rng.uniform(-np.pi, np.pi),
                ]
            )
        pose = np.asarray(pose, dtype=float)
        self.data.qpos[:] = pose
        self.data.qvel[:] = 0.0

        if randomise_gate:
            gx = self.rng.uniform(-0.3, 0.5)
            gy = self.rng.uniform(-0.3, 0.3)
            half = self.rng.uniform(0.17, 0.26)
            self.data.mocap_pos[0] = [gx, gy + half, 0.10]
            self.data.mocap_pos[1] = [gx, gy - half, 0.10]

        self.slip.reset()
        self.tick = 0
        self.encoder_counts = np.zeros(4)
        self._vel_avg = np.zeros(3)
        self.odom_pose = pose.copy()          # odometry starts correct, then drifts
        mujoco.mj_forward(self.model, self.data)
        return self._observe(np.zeros(3), self.slip.step(np.zeros(3), CONTROL_DT))

    # ── Core loop ───────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> dict:
        """
        One 20 Hz control tick. Returns the observation AFTER the step, which is
        the convention the dataset schema and every predictor assumes.
        """
        action = np.clip(np.asarray(action, dtype=float), -ACTION_LIMITS, ACTION_LIMITS)

        # Wheel/floor layer: what the wheels do, and what the body actually gets.
        slip_out = self.slip.step(action, CONTROL_DT)

        # Body frame -> world frame. The slide joints live in the world frame,
        # and so does the overhead camera.
        pose_before = np.array([float(self.data.qpos[0]), float(self.data.qpos[1]),
                                float(self.data.qpos[self._jid["hinge_z"]])])
        theta = pose_before[2]
        self.data.ctrl[:] = body_to_world(slip_out["body_vel"], theta)

        # Motor/PID layer: MuJoCo's velocity actuators add first-order lag.
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._vel_avg = self._average_body_vel(pose_before)
        self.tick += 1
        self.encoder_counts += slip_out["wheel_meas"] * CONTROL_DT * ENCODER_COUNTS_PER_RAD
        self.odom_pose = self.kin.integrate_pose(
            self.odom_pose, slip_out["odom_vel"], CONTROL_DT
        )
        return self._observe(action, slip_out)

    def _average_body_vel(self, pose_before: np.ndarray) -> np.ndarray:
        """
        The body velocity that, integrated by kin.integrate_pose for one tick,
        lands exactly on the new pose. Exact inverse of the midpoint integrator.

        This is the right learning target, and the right thing to call "measured
        velocity". The instantaneous end-of-tick velocity is NOT: the robot
        accelerates within the tick, so integrating instantaneous samples at
        20 Hz accumulates 8.9 mm of pure discretisation error over 1 s - two
        thirds of the analytic baseline, and nothing to do with world modelling.

        It is also what real hardware reports. A wheel encoder gives you counts
        ACCUMULATED over the interval, i.e. an average, never an instantaneous
        rate.
        """
        x0, y0, th0 = pose_before
        x1 = float(self.data.qpos[0])
        y1 = float(self.data.qpos[1])
        th1 = float(self.data.qpos[self._jid["hinge_z"]])

        om = _wrap(th1 - th0) / CONTROL_DT
        th_mid = th0 + 0.5 * om * CONTROL_DT
        c, s = np.cos(th_mid), np.sin(th_mid)
        dx, dy = x1 - x0, y1 - y0
        return np.array([(dx * c + dy * s) / CONTROL_DT, (-dx * s + dy * c) / CONTROL_DT, om])

    def _observe(self, action: np.ndarray, slip_out: dict) -> dict:
        qpos, qvel = self.data.qpos, self.data.qvel
        theta = float(qpos[self._jid["hinge_z"]])

        # True body-frame velocity: rotate the world-frame qvel back into the body.
        c, s = np.cos(theta), np.sin(theta)
        vx_w, vy_w = float(qvel[0]), float(qvel[1])
        body_vel_true = np.array([vx_w * c + vy_w * s, -vx_w * s + vy_w * c, float(qvel[2])])

        return {
            "tick": self.tick,
            "t_mono": self.tick * CONTROL_DT,
            "action_cmd": action.copy(),
            "wheel_cmd": slip_out["wheel_cmd"].copy(),
            "wheel_meas": slip_out["wheel_meas"].copy(),
            "encoder_counts": self.encoder_counts.copy(),
            "imu_gyro": self._sensor("imu_gyro").copy(),
            "imu_accel": self._sensor("imu_acc").copy(),
            "pose_gt": np.array([float(qpos[0]), float(qpos[1]), theta]),
            "pose_gt_valid": True,            # the sim's fiducial is never occluded
            "pose_odom": self.odom_pose.copy(),
            "body_vel_gt": body_vel_true,          # instantaneous, end of tick
            "body_vel_avg": self._vel_avg.copy(),  # tick average; the M2 target
            "scrub": self.kin.scrub(slip_out["wheel_meas"]),
            "batt_v": slip_out["batt_v"],
            "saturated": slip_out["saturated"],
            "surface": self.surface,
            "shell": False,                   # M6 condition; sim has no shell yet
            "out_of_bounds": bool(np.max(np.abs(qpos[:2])) > ARENA_HALF),
        }

    def _sensor(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
        return self.data.sensordata[adr : adr + dim]

    # ── Rendering ───────────────────────────────────────────────────────────

    def render(self, camera: str = "front", size: int | tuple[int, int] | None = None) -> np.ndarray:
        h, w = (
            (size, size) if isinstance(size, int)
            else size if size is not None
            else (self.image_size, self.image_size)
        )
        if self._renderer is None or self._renderer.height != h or self._renderer.width != w:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _wrap(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


# ── Action sampling ─────────────────────────────────────────────────────────

_kin = MecanumKinematics()   # module-level, for action clamping only


def clamp_action(action: np.ndarray) -> np.ndarray:
    """
    The single place an action becomes legal. Box-clips to ACTION_LIMITS, then
    scales into the achievable polytope. Every producer of actions - the random
    sampler, the expert, teleop, and later the MPPI planner - must go through
    this, or the dataset records commands the robot never executed.
    """
    boxed = np.clip(np.asarray(action, dtype=float), -ACTION_LIMITS, ACTION_LIMITS)
    return _kin.clamp_achievable(boxed, WHEEL_SPEED_MAX)

def smoothed_random_actions(
    n: int, rng: np.random.Generator, correlation: float = 0.88
) -> np.ndarray:
    """
    Temporally correlated random actions (an OU walk), clipped to the limits.

    Why not white noise: white noise at 20 Hz averages to nothing, the robot
    jitters in place, and the dataset contains no sustained motion for a model
    to learn multi-step dynamics from. Why not goal-directed expert data: it
    covers only the narrow slice of action space the controller happens to
    choose. For *dynamics* learning, action diversity beats task success - the
    model has to know what happens when you strafe hard into a turn, and no
    sensible expert ever does that.
    """
    actions = np.zeros((n, 3))
    a = np.zeros(3)
    for i in range(n):
        a = correlation * a + (1.0 - correlation) * rng.normal(0.0, 1.0, size=3) * 3.0
        actions[i] = clamp_action(a * ACTION_LIMITS)
    return actions


def wall_avoiding_action(obs: dict, base_action: np.ndarray) -> np.ndarray:
    """
    Nudge away from the arena edge. Keeps episodes long without biasing the
    action distribution much - pinning the robot against a wall produces ticks
    where the commanded action has no effect, which teaches a world model that
    actions do not matter.
    """
    x, y, theta = obs["pose_gt"]
    margin = 0.62
    push_w = np.array([-x if abs(x) > margin else 0.0, -y if abs(y) > margin else 0.0])
    if not push_w.any():
        return base_action
    c, s = np.cos(theta), np.sin(theta)
    push_b = np.array([push_w[0] * c + push_w[1] * s, -push_w[0] * s + push_w[1] * c])
    out = base_action.copy()
    out[:2] = 0.45 * out[:2] + 1.6 * push_b
    return clamp_action(out)


# ── Verification ────────────────────────────────────────────────────────────

# Step-response start poses. The robot must NOT reach a wall inside the
# measurement window, or the steady state is measured against a wall and the
# whole characterisation is nonsense. (It was, the first time.)
_LAG_PROBES = (
    ("forward", np.array([0.40, 0.0, 0.0]), 0, np.array([-0.80, 0.0, 0.0])),
    ("strafe", np.array([0.0, 0.40, 0.0]), 1, np.array([0.0, -0.80, 0.0])),
    ("rotate", np.array([0.0, 0.0, 1.50]), 2, np.zeros(3)),
)


def measure_actuator_tau() -> dict:
    """
    Isolate the ACTUATOR time constant: hold ctrl constant and step physics one
    timestep at a time, bypassing the control loop and the slip model entirely.

    Measured at the 2 ms physics rate, so tau is resolved properly instead of
    being quantised to the 50 ms control tick. Verifies the kv values in the XML
    empirically rather than trusting tau = mass / kv.
    """
    out = {}
    for mode, idx, target in (("translate", 0, 0.40), ("rotate", 2, 1.50)):
        env = MecanumEnv(seed=0)
        env.reset(pose=np.zeros(3))
        env.data.ctrl[:] = 0.0
        env.data.ctrl[idx] = target
        trace, n = [], 400                       # 400 * 2 ms = 0.8 s
        for _ in range(n):
            mujoco.mj_step(env.model, env.data)
            trace.append(float(env.data.qvel[idx]))
        env.close()
        trace = np.array(trace)
        final = float(np.mean(trace[-40:]))
        # Linear interpolation of the 63.2% crossing, for sub-timestep resolution
        k = int(np.argmax(trace >= 0.632 * final))
        if 0 < k < n:
            lo, hi = trace[k - 1], trace[k]
            frac = (0.632 * final - lo) / max(hi - lo, 1e-12)
            tau = (k - 1 + frac) * env.model.opt.timestep
        else:
            tau = float("nan")
        out[mode] = {"tau_ms": round(float(tau) * 1000.0, 1), "steady_state": round(final, 4)}
    return out


def measure_delivered_fraction(surface: str = "hard") -> dict:
    """
    Contact slip at steady state: of the velocity commanded, what fraction does
    the body actually achieve? Measured through the full control loop, away from
    walls, and truncated the moment the robot approaches one.
    """
    out = {}
    for mode, action, idx, start in _LAG_PROBES:
        env = MecanumEnv(surface=surface, seed=0)
        obs = env.reset(pose=start)
        trace = []
        for _ in range(40):                      # 2.0 s
            obs = env.step(action)
            if obs["out_of_bounds"]:
                break
            trace.append(obs["body_vel_gt"][idx])
        env.close()
        trace = np.array(trace)
        # Steady state from the last quarter of whatever clean window we got
        tail = trace[max(len(trace) * 3 // 4, 1):]
        final = float(np.mean(tail))
        out[mode] = {
            "commanded": float(action[idx]),
            "achieved": round(final, 4),
            "delivered_fraction": round(final / float(action[idx]), 4),
            "clean_ticks": int(len(trace)),
        }
    return out


# ── The analytic baseline ladder ───────────────────────────────────────────

def collect_windows(
    surface: str, n_episodes: int, horizon_ticks: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gather (start_pose, action_sequence, true_end_pose) windows for evaluating
    multi-step predictors. Windows that run into a wall are discarded: a tick
    where the commanded action has no effect is not a dynamics sample, it is a
    contact event, and mixing the two flatters every predictor equally.
    """
    rng = np.random.default_rng(seed)
    starts, seqs, ends = [], [], []

    for ep in range(n_episodes):
        env = MecanumEnv(surface=surface, seed=seed + ep)
        obs = env.reset()
        plan = smoothed_random_actions(600, rng)

        i = 0
        while i < len(plan) - horizon_ticks:
            start_pose = obs["pose_gt"].copy()
            applied, hit_wall = [], False
            for k in range(horizon_ticks):
                a = wall_avoiding_action(obs, plan[i + k])
                obs = env.step(a)
                applied.append(a)
                if obs["out_of_bounds"]:
                    hit_wall = True
                    break
            if not hit_wall and len(applied) == horizon_ticks:
                starts.append(start_pose)
                seqs.append(np.array(applied))
                ends.append(obs["pose_gt"].copy())
            else:
                obs = env.reset()
            i += horizon_ticks
        env.close()

    return np.array(starts), np.array(seqs), np.array(ends)


def first_order_lag(
    seq: np.ndarray, tau: float, dt: float = CONTROL_DT, v0: np.ndarray | None = None
) -> np.ndarray:
    """
    Low-pass an action sequence to model finite motor/PID bandwidth.

    A velocity command is not achieved instantly: the loop needs roughly tau to
    get there. Over a 1 s horizon with changing actions this matters more than
    steady-state slip does, which is why the calibrated baseline fits tau as
    well as the per-mode gains.

    `v0` is the measured velocity the robot is ALREADY travelling at when the
    window opens. Passing it matters enormously and is easy to forget: leaving
    the filter at zero asserts the robot starts from rest every time, which is
    false for all but a handful of windows and cripples the baseline. The
    learned model is handed the same measurement as an input, so withholding it
    here would be rigging the comparison.
    """
    if tau <= 1e-6:
        return seq
    alpha = 1.0 - np.exp(-dt / tau)
    out = np.empty_like(seq)
    v = np.zeros(seq.shape[1]) if v0 is None else np.asarray(v0, dtype=seq.dtype).copy()
    for i, target in enumerate(seq):
        v = v + alpha * (target - v)
        out[i] = v
    return out


def predict_calibrated(
    kin: MecanumKinematics, p0: np.ndarray, seq: np.ndarray, gains: np.ndarray, tau: float
) -> np.ndarray:
    """Analytic model plus the M1 calibration: per-mode gains and actuator lag."""
    return kin.predict(p0, first_order_lag(seq * gains, tau), CONTROL_DT)[-1]


def fit_calibration(
    starts: np.ndarray, seqs: np.ndarray, ends: np.ndarray, kin: MecanumKinematics
) -> tuple[np.ndarray, float]:
    """
    M1's calibration step: fit three per-mode efficiencies (forward, strafe,
    rotate) plus one actuator time constant, so the analytic model stops
    systematically over-predicting.

    One scalar per orthogonal mode - exactly the decomposition that section 5 of
    the kinematics note earns. Coordinate descent, because pose error is
    nonlinear in these once heading is involved.

    This is the baseline that matters. Beating uncalibrated FK proves nothing:
    fitting four scalars is an afternoon's work, and M2 has to do better than an
    afternoon to justify existing.
    """
    gains, tau = np.ones(3), 0.05

    def mean_err(g: np.ndarray, t: float) -> float:
        return float(
            np.mean([
                np.linalg.norm(predict_calibrated(kin, p0, seq, g, t)[:2] - p1[:2])
                for p0, seq, p1 in zip(starts, seqs, ends)
            ])
        )

    for _ in range(4):
        for m in range(3):
            best, best_err = gains[m], mean_err(gains, tau)
            for cand in np.arange(0.40, 1.101, 0.01):
                trial = gains.copy()
                trial[m] = cand
                err = mean_err(trial, tau)
                if err < best_err:
                    best, best_err = cand, err
            gains[m] = best
        best_tau, best_err = tau, mean_err(gains, tau)
        for cand in np.arange(0.0, 0.401, 0.01):
            err = mean_err(gains, cand)
            if err < best_err:
                best_tau, best_err = cand, err
        tau = best_tau

    if np.any(gains <= 0.405) or np.any(gains >= 1.095) or tau >= 0.395:
        print(f"  WARNING: calibration hit a search bound (gains={gains.round(3)}, "
              f"tau={tau:.3f}) - widen the range before trusting this baseline")
    return gains, float(tau)


def analytic_baseline(
    surface: str = "hard", n_episodes: int = 8, horizon_ticks: int = 20, seed: int = 7
) -> dict:
    """
    The sim version of M1: how wrong is the analytic model over a 1.0 s horizon?

    Two predictors, both given the true future action sequence:
        naive        integrate the commanded actions as if achieved exactly
        calibrated   per-mode efficiency + actuator lag, fitted on TRAIN only

    The calibrated number is the one that matters. A learned model that only
    beats `naive` has beaten a strawman.
    """
    kin = MecanumKinematics()
    starts, seqs, ends = collect_windows(surface, n_episodes, horizon_ticks, seed)

    # Split by position in the stream, not by random window: adjacent windows
    # share state and leak across a random split.
    cut = int(len(starts) * 0.6)
    train, test = slice(0, cut), slice(cut, len(starts))
    gains, tau = fit_calibration(starts[train], seqs[train], ends[train], kin)

    summary = {}
    for name, g, t in (("naive", np.ones(3), 0.0), ("calibrated", gains, tau)):
        errs = np.array(
            [
                np.linalg.norm(predict_calibrated(kin, p0, seq, g, t)[:2] - p1[:2])
                for p0, seq, p1 in zip(starts[test], seqs[test], ends[test])
            ]
        ) * 1000.0
        summary[name] = {
            "mean_mm": round(float(errs.mean()), 1),
            "median_mm": round(float(np.median(errs)), 1),
            "p95_mm": round(float(np.percentile(errs, 95)), 1),
        }
    return {
        "surface": surface,
        "horizon_s": horizon_ticks * CONTROL_DT,
        "n_train": int(cut),
        "n_test": int(len(starts) - cut),
        "fitted_mode_gains": {
            "forward": round(float(gains[0]), 3),
            "strafe": round(float(gains[1]), 3),
            "rotate": round(float(gains[2]), 3),
        },
        "fitted_tau_s": round(tau, 3),
        "predictors": summary,
    }


def _save_artifacts(tau: dict, delivered: dict, baselines: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Camera views: the front view has to be informative enough for M3, and the
    # overhead view is the fiducial-pose stand-in.
    env = MecanumEnv(seed=3)
    env.reset(pose=np.array([-0.4, -0.3, 0.6]))
    for _ in range(10):
        env.step(np.array([0.3, 0.1, 0.4]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (cam, size, title) in zip(
        axes,
        [("front", 64, "front camera, 64x64 (what the model sees)"),
         ("front", 320, "front camera, 320x320"),
         ("overhead", 320, "overhead (fiducial ground truth)")],
    ):
        ax.imshow(env.render(cam, size))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "camera_views.png", dpi=130)
    plt.close(fig)
    env.close()

    # Step responses through the full control loop, started away from walls.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (mode, action, idx, start) in zip(axes, _LAG_PROBES):
        env = MecanumEnv(seed=0)
        obs = env.reset(pose=start)
        trace = []
        for _ in range(40):
            obs = env.step(action)
            if obs["out_of_bounds"]:
                break
            trace.append(obs["body_vel_gt"][idx])
        env.close()
        t = np.arange(len(trace)) * CONTROL_DT
        ax.axhline(action[idx], ls="--", c="0.4", lw=1, label="commanded")
        ax.plot(t, trace, lw=1.8, label="achieved")
        ax.set_title(f"{mode}: {delivered[mode]['delivered_fraction']:.0%} delivered", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m/s" if idx < 2 else "rad/s")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "step_response.png", dpi=130)
    plt.close(fig)

    # The baseline ladder: naive vs calibrated analytic, per surface.
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    width, surfaces = 0.35, [b["surface"] for b in baselines]
    xs = np.arange(len(surfaces))
    for i, pred in enumerate(["naive", "calibrated"]):
        med = [b["predictors"][pred]["median_mm"] for b in baselines]
        p95 = [b["predictors"][pred]["p95_mm"] for b in baselines]
        bars = ax.bar(xs + i * width, med, width, label=f"{pred} (median)")
        ax.errorbar(xs + i * width, med, yerr=[[0] * len(med),
                    [p - m for p, m in zip(p95, med)]], fmt="none", ecolor="0.3",
                    capsize=4, lw=1)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(xs + width / 2)
    ax.set_xticklabels([f"{s} floor" for s in surfaces])
    ax.set_ylabel("1.0 s pose error (mm)")
    ax.set_title("Analytic baseline — calibrated is the number M2 must beat\n"
                 "(bars: median, whisker to p95)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "analytic_baseline.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 mecanum base: verify and characterise")
    parser.add_argument("--viewer", action="store_true", help="launch the interactive viewer")
    parser.add_argument("--surface", default="hard", choices=["hard", "mat"])
    parser.add_argument("--no-artifacts", action="store_true", help="skip plots")
    args = parser.parse_args()

    if args.viewer:
        import mujoco.viewer

        env = MecanumEnv(surface=args.surface, seed=0)
        obs = env.reset(randomise_gate=True)
        rng = np.random.default_rng(0)
        actions = smoothed_random_actions(100_000, rng)
        print("viewer: random exploration. close the window to stop.")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            i = 0
            while viewer.is_running():
                t0 = time.perf_counter()
                obs = env.step(wall_avoiding_action(obs, actions[i % len(actions)]))
                viewer.sync()
                i += 1
                time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - t0)))
        env.close()
        return

    print("-- Throughput ---------------------------------------------------")
    env = MecanumEnv(surface=args.surface, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    acts = smoothed_random_actions(2000, rng)
    t0 = time.perf_counter()
    for a in acts:
        env.step(a)
    dt_phys = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(200):
        env.render("front", MODEL_IMAGE_SIZE)
    dt_rend = time.perf_counter() - t0
    env.close()
    print(f"  physics only   {len(acts)/dt_phys:8.0f} ticks/s  "
          f"({len(acts)/dt_phys/CONTROL_HZ:.0f}x real time)")
    print(f"  64x64 render   {200/dt_rend:8.0f} frames/s")

    print("\n-- Actuator time constant (physics rate, slip bypassed) ---------")
    tau = measure_actuator_tau()
    for mode, r in tau.items():
        print(f"  {mode:10s} tau = {r['tau_ms']:5.1f} ms   steady state {r['steady_state']}")

    print("\n-- Contact slip: fraction of commanded velocity delivered -------")
    delivered = measure_delivered_fraction(args.surface)
    print(f"  {'mode':9s} {'commanded':>10s} {'achieved':>9s} {'delivered':>10s} {'ticks':>7s}")
    for mode, r in delivered.items():
        print(f"  {mode:9s} {r['commanded']:10.2f} {r['achieved']:9.3f} "
              f"{r['delivered_fraction']:9.1%} {r['clean_ticks']:7d}")

    print("\n-- Analytic baseline: 1.0 s pose prediction error ----------------")
    baselines = [analytic_baseline(s) for s in ("hard", "mat")]
    for b in baselines:
        g = b["fitted_mode_gains"]
        print(f"  surface {b['surface']}  "
              f"(train {b['n_train']} / test {b['n_test']} windows)")
        print(f"    fitted        forward {g['forward']}  strafe {g['strafe']}  "
              f"rotate {g['rotate']}  tau {b['fitted_tau_s']*1000:.0f} ms")
        for pred, r in b["predictors"].items():
            print(f"    {pred:11s} median {r['median_mm']:6.1f} mm   "
                  f"mean {r['mean_mm']:6.1f} mm   p95 {r['p95_mm']:6.1f} mm")

    if not args.no_artifacts:
        print("\n-- Artifacts -----------------------------------------------------")
        _save_artifacts(tau, delivered, baselines)
        metrics = {
            "control_hz": CONTROL_HZ,
            "action_limits": ACTION_LIMITS.tolist(),
            "image_size": MODEL_IMAGE_SIZE,
            "throughput_ticks_per_s": round(len(acts) / dt_phys, 1),
            "render_fps_64px": round(200 / dt_rend, 1),
            "actuator_tau": tau,
            "delivered_fraction": delivered,
            "analytic_baseline": baselines,
        }
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        (ASSETS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        for name in ("camera_views.png", "step_response.png", "analytic_baseline.png",
                     "metrics.json"):
            print(f"  {ASSETS_DIR / name}")


if __name__ == "__main__":
    main()
