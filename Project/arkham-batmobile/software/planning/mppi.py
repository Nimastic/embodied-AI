"""
M4: Model Predictive Path Integral control over a learned world model.

Sample K action sequences, roll every one through a predictor, score them,
take a softmax-weighted average, execute the first few actions, replan.

Williams et al., Information Theoretic MPC (papers/foundations/
williams-2017-information-theoretic-mpc.pdf) is the reference and is directly
implementable; this is that algorithm with the per-step cost specialised to the
gate task in gate_task.py.

The point of M4 is not that MPPI works - it does, it is 40 years old - but that
the SAME planner against different predictors isolates how much the learned
model is worth when it has to act rather than merely predict:

    reactive            no model at all, P controller straight at the goal
    MPPI + naive        analytic FK, uncalibrated
    MPPI + calibrated   analytic FK with fitted per-mode gains and lag
    MPPI + learned      the M2 residual GRU

Every predictor implements rollout(pose0, hist, action_seqs) -> poses, so the
condition is one constructor argument.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.mecanum import MecanumKinematics  # noqa: E402
from models.state_model import StatePredictor  # noqa: E402
from sim.mecanum_env import ACTION_LIMITS, CONTROL_DT, WHEEL_SPEED_MAX, first_order_lag  # noqa: E402


# ── Predictors ──────────────────────────────────────────────────────────────

class AnalyticPredictor:
    """Calibrated (or naive, with gains=1 and tau=0) analytic forward model."""

    def __init__(self, kin: MecanumKinematics, gains=None, tau: float = 0.0, name="analytic"):
        self.kin = kin
        self.gains = np.ones(3) if gains is None else np.asarray(gains, dtype=float)
        self.tau = float(tau)
        self.name = name

    def rollout(self, pose0: np.ndarray, hist: np.ndarray, actions: np.ndarray) -> np.ndarray:
        v0 = hist[-1, 3:6]
        scaled = actions * self.gains
        vels = np.stack([first_order_lag(seq, self.tau, v0=v0) for seq in scaled])
        return self.kin.integrate_batch(np.repeat(pose0[None], len(actions), 0), vels, CONTROL_DT)


class LearnedPredictor:
    """The M2 residual GRU, batched over MPPI samples."""

    def __init__(self, kin: MecanumKinematics, ckpt_path: Path, name="learned"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = StatePredictor(ckpt["feat_dim"], ckpt["hidden"], ckpt["variant"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.kin = kin
        self.gains = np.asarray(ckpt["gains"], dtype=float)
        self.tau = float(ckpt["tau"])
        self.history = int(ckpt["history"])
        self.variant = ckpt["variant"]
        self.name = name

    @torch.no_grad()
    def rollout(self, pose0: np.ndarray, hist: np.ndarray, actions: np.ndarray) -> np.ndarray:
        k = len(actions)
        v0 = hist[-1, 3:6]
        prior = np.stack(
            [first_order_lag(seq, self.tau, v0=v0) for seq in actions * self.gains]
        ).astype(np.float32)

        h = torch.from_numpy(hist[-self.history:][None].astype(np.float32)).expand(k, -1, -1)
        vels = self.model(
            h.contiguous(), torch.from_numpy(actions.astype(np.float32)), torch.from_numpy(prior)
        ).numpy()
        return self.kin.integrate_batch(np.repeat(pose0[None], k, 0), vels, CONTROL_DT)


# ── Planner ─────────────────────────────────────────────────────────────────

class MPPI:
    """
    Receding-horizon MPPI.

    Parameters that matter and why:

    horizon     Chosen from the M2 error-vs-horizon curve, not guessed. Planning
                further than the model stays accurate just optimises fiction.
    n_samples   More samples find better plans AND find more model exploits.
    temperature Low makes it greedy on the model's opinion; high keeps it near
                the previous plan. Directly trades plan quality against trust.
    action cost Penalising magnitude and jerk is not aesthetic. It keeps samples
                inside the action distribution the model was trained on, which
                is the main defence against the planner exploiting the model.
    """

    def __init__(
        self,
        predictor,
        cost_fn,
        kin: MecanumKinematics,
        horizon: int = 15,
        n_samples: int = 384,
        temperature: float = 0.12,
        noise_sigma=(0.18, 0.18, 0.70),
        w_action: float = 0.04,
        w_jerk: float = 0.25,
        seed: int = 0,
    ):
        self.predictor = predictor
        self.cost_fn = cost_fn
        self.kin = kin
        self.H = horizon
        self.K = n_samples
        self.temp = temperature
        self.sigma = np.asarray(noise_sigma, dtype=float)
        self.w_action = w_action
        self.w_jerk = w_jerk
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((horizon, 3))
        self.last_diagnostics: dict = {}

    def reset(self) -> None:
        self.mean = np.zeros((self.H, 3))

    def plan(self, pose0: np.ndarray, hist: np.ndarray) -> np.ndarray:
        # Sample around the previous plan, with smoothed noise so candidates are
        # physically plausible rather than per-tick jitter the robot cannot track.
        noise = self.rng.normal(0.0, 1.0, size=(self.K, self.H, 3)) * self.sigma
        noise = np.cumsum(noise, axis=1) / np.sqrt(np.arange(1, self.H + 1))[None, :, None]
        seqs = self.mean[None] + noise

        # Clamp every candidate into the achievable set. Planning over actions
        # the robot cannot execute produces plans that evaporate on contact.
        seqs = np.clip(seqs, -ACTION_LIMITS, ACTION_LIMITS)
        flat = seqs.reshape(-1, 3)
        peak = np.max(np.abs(flat @ self.kin.J.T) / self.kin.geom.wheel_radius, axis=1)
        scale = np.minimum(1.0, WHEEL_SPEED_MAX / np.maximum(peak, 1e-9))
        seqs = (flat * scale[:, None]).reshape(self.K, self.H, 3)

        poses = self.predictor.rollout(pose0, hist, seqs)

        cost = self.cost_fn(poses, seqs)
        cost = cost + self.w_action * np.sum(seqs**2, axis=(1, 2))
        cost = cost + self.w_jerk * np.sum(np.diff(seqs, axis=1) ** 2, axis=(1, 2))

        weights = np.exp(-(cost - cost.min()) / self.temp)
        weights /= weights.sum() + 1e-12
        self.mean = np.einsum("k,khi->hi", weights, seqs)

        self.last_diagnostics = {
            "cost_best": float(cost.min()),
            "cost_mean": float(cost.mean()),
            "effective_samples": float(1.0 / np.sum(weights**2)),
            "predicted_path": self.predictor.rollout(pose0, hist, self.mean[None])[0].copy(),
        }
        return self.mean.copy()

    def shift(self, n_executed: int) -> None:
        """
        Warm-start the next solve by shifting the plan forward and repeating
        the last action into the freed tail.

        When the whole horizon was executed open-loop there is nothing left to
        shift, so hold the final action instead of indexing off the end.
        """
        n = min(n_executed, self.H)
        if n >= self.H:
            self.mean[:] = self.mean[-1]
            return
        self.mean = np.roll(self.mean, -n, axis=0)
        self.mean[-n:] = self.mean[-n - 1]


# ── Reactive control condition ──────────────────────────────────────────────

def reactive_action(pose: np.ndarray, goal: np.ndarray, kin: MecanumKinematics) -> np.ndarray:
    """
    P controller straight at the goal, in the body frame. No model, no horizon,
    no awareness of the gate posts. This is the "do you need a model at all?"
    condition, and on an open run it is hard to beat.
    """
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    err = np.array([dx * c + dy * s, -dx * s + dy * c])
    dth = (goal[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi
    a = np.clip(np.array([1.8 * err[0], 1.8 * err[1], 2.2 * dth]), -ACTION_LIMITS, ACTION_LIMITS)
    return kin.clamp_achievable(a, WHEEL_SPEED_MAX)
