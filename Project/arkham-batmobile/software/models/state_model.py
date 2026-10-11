"""
M2: action-conditioned state predictor.

    (state_t, a_{t+1:t+H})  ->  body velocities  ->  pose at t+H

The claim being tested is narrow and falsifiable: a learned model can predict
what the CALIBRATED analytic model cannot. Beating uncalibrated forward
kinematics proves nothing - per-mode gains plus a lag constant is four scalars
and an afternoon.

Three predictors share one interface so the comparison is a config change:

    naive        integrate the commanded actions as if achieved exactly
    calibrated   per-mode efficiency + first-order actuator lag, fitted
    learned      GRU over the action sequence, conditioned on current state

and the learned model comes in two variants:

    residual     predicts a correction ON TOP of the calibrated prediction
    direct       predicts the velocity from scratch

The residual/direct ablation is the point of MILESTONES.md M2: learning a
correction to a nearly-right model should be a smaller, better-conditioned
problem than learning the dynamics outright.

Usage:
    python state_model.py                 # train on hard, eval hard + mat
    python state_model.py --epochs 40
    python state_model.py --variant direct
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.mecanum import MecanumKinematics  # noqa: E402
from sim.mecanum_env import CONTROL_DT, first_order_lag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sim"
ASSETS = ROOT.parent / "experiments" / "M2-state-predictor" / "assets"
CKPT_DIR = ROOT / "checkpoints"
RUN_TAG = ""

HORIZON = 20                       # 20 ticks @ 20 Hz = 1.0 s
STATE_KEYS = ("body_vel_avg", "wheel_meas", "imu_gyro")   # + batt_v scalar
POSE_JUMP_LIMIT = 0.10             # m; larger means collect.py reset mid-episode


# ── Data ────────────────────────────────────────────────────────────────────

def load_episodes(surface: str) -> list[dict]:
    """One dict per episode file, with arrays already stacked."""
    out = []
    for path in sorted((DATA / surface / "random").glob("*.parquet")):
        df = pd.read_parquet(path)
        out.append(
            {
                "path": path.name,
                "pose": np.stack(df["pose_gt"].to_numpy()).astype(np.float64),
                "vel": np.stack(df["body_vel_avg"].to_numpy()).astype(np.float32),
                "action": np.stack(df["action_cmd"].to_numpy()).astype(np.float32),
                "wheel": np.stack(df["wheel_meas"].to_numpy()).astype(np.float32),
                "gyro": np.stack(df["imu_gyro"].to_numpy()).astype(np.float32),
                "batt": df["batt_v"].to_numpy().astype(np.float32),
            }
        )
    if not out:
        raise FileNotFoundError(
            f"no episodes in {DATA / surface / 'random'} - run "
            f"`python sim/collect.py --minutes 30 --surface {surface} --no-video` first"
        )
    return out


def build_windows(
    episodes: list[dict], stride: int = 1, history: int = 1
) -> dict[str, np.ndarray]:
    """
    Slice episodes into (history_{t-K+1:t}, actions_{t+1:t+H}, vels_{t+1:t+H},
    pose_t, pose_{t+H}) windows.

    ALIGNMENT: vel[t] is the average velocity over the tick ENDING at pose[t].
    So predicting forward from pose[t] uses actions and velocities from t+1
    onward. Off by one here trains a model to predict the past - it reaches a
    lovely loss and is worthless. See software/README.md.

    HISTORY: each step carries [action, achieved vel, wheel speeds, gyro, batt]
    = 14 features. The action/achieved-velocity PAIR is the point. The gap
    between what was commanded and what actually happened is the only
    observable signature of how slippery the floor is, and one tick of it is
    nearly useless because the actuator lag has not played out yet. K ticks of
    it should be enough to identify the surface. That is the M2b hypothesis.

    collect.py re-centres the robot at the arena edge, putting a pose
    discontinuity mid-file. Windows spanning one - in the history or in the
    prediction span - are dropped.
    """
    Hst, A, V, P0, P1 = [], [], [], [], []
    dropped = 0

    for ep in episodes:
        n = len(ep["pose"])
        jump = np.r_[False, np.linalg.norm(np.diff(ep["pose"][:, :2], axis=0), axis=1)
                     > POSE_JUMP_LIMIT]
        feats = np.concatenate(
            [ep["action"], ep["vel"], ep["wheel"], ep["gyro"], ep["batt"][:, None]], axis=1
        ).astype(np.float32)

        for t in range(history - 1, n - HORIZON - 1, stride):
            fwd = slice(t + 1, t + 1 + HORIZON)
            past = slice(t - history + 1, t + 1)
            if jump[fwd].any() or jump[past].any():
                dropped += 1
                continue
            Hst.append(feats[past])
            A.append(ep["action"][fwd])
            V.append(ep["vel"][fwd])
            P0.append(ep["pose"][t])
            P1.append(ep["pose"][t + HORIZON])

    return {
        "hist": np.asarray(Hst, dtype=np.float32),       # (N, K, 14)
        "actions": np.asarray(A, dtype=np.float32),
        "vels": np.asarray(V, dtype=np.float32),
        "pose0": np.asarray(P0, dtype=np.float64),
        "poseH": np.asarray(P1, dtype=np.float64),
        "dropped": dropped,
    }


def start_vel(data: dict) -> np.ndarray:
    """Measured velocity at window start — the last history step's vel slot."""
    return data["hist"][:, -1, 3:6]


# ── Model ───────────────────────────────────────────────────────────────────

class StatePredictor(nn.Module):
    """
    GRU rolled over the action sequence, initial hidden state from the current
    proprioceptive state. Emits a body velocity per step.

    Small on purpose: ~50k parameters, CPU-trainable in a couple of minutes. If
    this cannot beat four fitted scalars, more capacity is not the answer.
    """

    def __init__(self, feat_dim: int, hidden: int = 96, variant: str = "residual"):
        super().__init__()
        self.variant = variant
        # Encoder is a GRU over the observed history, so K=1 and K=20 differ
        # only in how much history they are fed - same architecture, same
        # parameter count. Otherwise the ablation would confound "more history"
        # with "different model".
        self.encoder = nn.GRU(input_size=feat_dim, hidden_size=hidden, batch_first=True)
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        # Start as a no-op correction so the residual variant begins exactly at
        # the calibrated baseline and can only improve from there.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, hist: torch.Tensor, actions: torch.Tensor,
                prior: torch.Tensor) -> torch.Tensor:
        _, h0 = self.encoder(hist)
        out, _ = self.gru(actions, h0.contiguous())
        delta = self.head(out)
        return prior + delta if self.variant == "residual" else delta


# ── Prediction helpers ──────────────────────────────────────────────────────

def analytic_prior(
    actions: np.ndarray, gains: np.ndarray, tau: float, v0: np.ndarray | None = None
) -> np.ndarray:
    """
    Calibrated analytic velocity for each window; the residual model's base.

    v0 is the measured velocity at the start of each window. The learned model
    receives it as an input, so the baseline gets it too - otherwise the
    comparison is rigged. (Omitting it inflated the learned model's apparent
    margin on hard floor from 42% to 71%.)
    """
    scaled = actions * gains.astype(np.float32)
    if v0 is None:
        return np.stack([first_order_lag(w, tau) for w in scaled]).astype(np.float32)
    return np.stack([first_order_lag(w, tau, v0=s0) for w, s0 in zip(scaled, v0)]).astype(
        np.float32)


def integrate_all(kin: MecanumKinematics, pose0: np.ndarray, vels: np.ndarray) -> np.ndarray:
    """Pose after EVERY step, (N, H, 3). One vectorised pass, all horizons."""
    return kin.integrate_batch(pose0, vels, CONTROL_DT)


def fit_calibration_fast(
    kin: MecanumKinematics, pose0: np.ndarray, actions: np.ndarray, poseH: np.ndarray,
    vel0: np.ndarray, max_windows: int = 4000, seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Same four-parameter calibration as sim/mecanum_env.fit_calibration, but
    batched - the scalar version integrates one window at a time, which is fine
    for 139 windows and takes hours for 9,000.

    Subsampled as well: fitting four scalars does not need every window, and the
    estimate is stable long before 4,000.
    """
    rng = np.random.default_rng(seed)
    if len(pose0) > max_windows:
        idx = rng.choice(len(pose0), max_windows, replace=False)
        pose0, actions, poseH, vel0 = pose0[idx], actions[idx], poseH[idx], vel0[idx]

    def mean_err(g: np.ndarray, t: float) -> float:
        vels = np.stack([first_order_lag(w, t, v0=s0) for w, s0 in zip(actions * g, vel0)])
        pred = kin.integrate_batch(pose0, vels, CONTROL_DT)[:, -1]
        return float(np.mean(np.linalg.norm(pred[:, :2] - poseH[:, :2], axis=1)))

    gains, tau = np.ones(3), 0.05
    for _ in range(4):
        for m in range(3):
            best, best_err = gains[m], mean_err(gains, tau)
            for cand in np.arange(0.40, 1.101, 0.01):
                trial = gains.copy()
                trial[m] = cand
                if (err := mean_err(trial, tau)) < best_err:
                    best, best_err = cand, err
            gains[m] = best
        best_tau, best_err = tau, mean_err(gains, tau)
        for cand in np.arange(0.0, 0.401, 0.01):
            if (err := mean_err(gains, cand)) < best_err:
                best_tau, best_err = cand, err
        tau = best_tau

    if np.any(gains <= 0.405) or np.any(gains >= 1.095) or tau >= 0.395:
        print(f"  WARNING: calibration hit a search bound (gains={gains.round(3)}, tau={tau:.3f})")
    return gains, float(tau)


def pose_error_mm(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1) * 1000.0


def path_length_m(pose0: np.ndarray, poseH: np.ndarray) -> np.ndarray:
    """Straight-line displacement, used to normalise error across surfaces."""
    return np.linalg.norm(poseH[:, :2] - pose0[:, :2], axis=1)


# ── Train / evaluate ────────────────────────────────────────────────────────

def train(
    model: StatePredictor, data: dict, prior: np.ndarray, epochs: int, seed: int
) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    hist = torch.from_numpy(data["hist"])
    actions = torch.from_numpy(data["actions"])
    vels = torch.from_numpy(data["vels"])
    prior_t = torch.from_numpy(prior)

    n, bs = len(hist), 256
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(hist[idx], actions[idx], prior_t[idx])
            loss = loss_fn(pred, vels[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        history.append(total / n)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}  train vel MSE {history[-1]:.6f}")
    return history


@torch.no_grad()
def predict_learned(model: StatePredictor, data: dict, prior: np.ndarray) -> np.ndarray:
    model.eval()
    return model(
        torch.from_numpy(data["hist"]),
        torch.from_numpy(data["actions"]),
        torch.from_numpy(prior),
    ).numpy()


def evaluate(kin, data, gains, tau, model, label) -> dict:
    """All three predictors on one window set, plus error-vs-horizon."""
    v0 = start_vel(data)
    prior = analytic_prior(data["actions"], gains, tau, v0)
    naive = analytic_prior(data["actions"], np.ones(3, dtype=np.float32), 0.0)
    learned = predict_learned(model, data, prior)

    # Ground truth at EVERY horizon, by exact integration of the logged average
    # velocities (which reproduces pose_gt to 0.000 mm - see software/README.md).
    truth = integrate_all(kin, data["pose0"], data["vels"])
    dist = path_length_m(data["pose0"], data["poseH"])

    results, curves = {}, {}
    for name, vels in (("naive", naive), ("calibrated", prior), ("learned", learned)):
        pred = integrate_all(kin, data["pose0"], vels)
        err_h = np.linalg.norm(pred[..., :2] - truth[..., :2], axis=2) * 1000.0  # (N, H)
        err = err_h[:, -1]
        results[name] = {
            "median_mm": round(float(np.median(err)), 2),
            "mean_mm": round(float(err.mean()), 2),
            "p95_mm": round(float(np.percentile(err, 95)), 2),
            # Normalised: absolute mm is confounded with how far the robot went,
            # which is exactly what differs between surfaces.
            "median_pct_of_path": round(
                float(np.median(err / 1000.0 / np.maximum(dist, 1e-6)) * 100), 2),
        }
        curves[name] = np.median(err_h, axis=0).round(3).tolist()

    return {"split": label, "n_windows": int(len(dist)),
            "mean_path_m": round(float(dist.mean()), 3),
            "predictors": results, "error_vs_horizon_mm": curves}


def main() -> None:
    ap = argparse.ArgumentParser(description="M2: action-conditioned state predictor")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--variant", default="residual", choices=["residual", "direct", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--history", type=int, default=1,
                    help="ticks of observed history fed to the encoder (M2b)")
    args = ap.parse_args()

    global RUN_TAG
    RUN_TAG = f"-h{args.history}"
    report_history = args.history

    kin = MecanumKinematics()

    print("-- Data ---------------------------------------------------------")
    hard = load_episodes("hard")
    mat = load_episodes("mat")
    n_test = max(4, len(hard) // 5)
    train_eps, test_eps = hard[:-n_test], hard[-n_test:]
    print(f"  hard: {len(train_eps)} train episodes, {len(test_eps)} held-out")
    print(f"  mat:  {len(mat)} episodes, fully held out (transfer check)")

    d_train = build_windows(train_eps, args.stride, args.history)
    d_test = build_windows(test_eps, args.stride, args.history)
    d_mat = build_windows(mat, args.stride, args.history)
    for name, d in (("train", d_train), ("test", d_test), ("mat", d_mat)):
        print(f"  {name:5s} {len(d['hist']):6d} windows "
              f"({d['dropped']} dropped at arena resets)")

    print("\n-- Calibrating the analytic baseline on TRAIN only --------------")
    gains, tau = fit_calibration_fast(kin, d_train["pose0"], d_train["actions"],
                                      d_train["poseH"], start_vel(d_train))
    print(f"  gains forward {gains[0]:.3f}  strafe {gains[1]:.3f}  rotate {gains[2]:.3f}"
          f"   tau {tau * 1000:.0f} ms")

    variants = ["residual", "direct"] if args.variant == "both" else [args.variant]
    report = {"horizon_s": HORIZON * CONTROL_DT,
              "history_ticks": report_history,
              "calibration": {"gains": gains.round(4).tolist(), "tau_s": round(tau, 4)},
              "variants": {}}

    for variant in variants:
        print(f"\n-- Training [{variant}] -----------------------------------------")
        model = StatePredictor(d_train["hist"].shape[2], variant=variant)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {n_params:,} parameters")
        t0 = time.perf_counter()
        train(model, d_train,
              analytic_prior(d_train["actions"], gains, tau, start_vel(d_train)),
              args.epochs, args.seed)
        print(f"  trained in {time.perf_counter() - t0:.1f}s")

        evals = {
            "hard_heldout": evaluate(kin, d_test, gains, tau, model, "hard (held out)"),
            "mat_transfer": evaluate(kin, d_mat, gains, tau, model, "mat (never seen)"),
        }
        report["variants"][variant] = {"n_params": n_params, "evals": evals}

        # Checkpoint everything the planner needs to reconstruct this predictor:
        # weights, architecture, and the calibration the residual sits on top of.
        ckpt = CKPT_DIR / f"state_model-{variant}-h{args.history}.pt"
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feat_dim": d_train["hist"].shape[2],
                "hidden": 96,
                "variant": variant,
                "history": args.history,
                "horizon": HORIZON,
                "gains": gains.tolist(),
                "tau": tau,
                "trained_on": "hard",
            },
            ckpt,
        )
        print(f"  checkpoint -> {ckpt}")

        for key, ev in evals.items():
            print(f"\n  {ev['split']}  (n={ev['n_windows']}, mean path "
                  f"{ev['mean_path_m']:.3f} m)")
            print(f"    {'predictor':12s} {'median':>9s} {'mean':>8s} {'p95':>8s} {'% of path':>10s}")
            for pname, r in ev["predictors"].items():
                print(f"    {pname:12s} {r['median_mm']:8.2f}mm {r['mean_mm']:7.2f}mm "
                      f"{r['p95_mm']:7.2f}mm {r['median_pct_of_path']:9.2f}%")
            cal = ev["predictors"]["calibrated"]["median_mm"]
            lrn = ev["predictors"]["learned"]["median_mm"]
            verdict = "BEATS" if lrn < cal else "LOSES TO"
            print(f"    -> learned {verdict} calibrated by "
                  f"{abs(cal - lrn) / cal * 100:.1f}%  (S3 target: 30% better)")

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / f"metrics{RUN_TAG}.json").write_text(json.dumps(report, indent=2))
    _plot(report)
    print(f"\n  {ASSETS / 'metrics.json'}")
    print(f"  {ASSETS / 'error_vs_horizon.png'}")


def _plot(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(report["variants"])
    fig, axes = plt.subplots(len(variants), 2, figsize=(11, 3.9 * len(variants)), squeeze=False)
    t = np.arange(1, HORIZON + 1) * CONTROL_DT

    for r, variant in enumerate(variants):
        for c, (key, title) in enumerate(
            (("hard_heldout", "hard floor, held-out episodes"),
             ("mat_transfer", "mat, never seen in training"))
        ):
            ax = axes[r][c]
            curves = report["variants"][variant]["evals"][key]["error_vs_horizon_mm"]
            for name, style in (("naive", ":"), ("calibrated", "--"), ("learned", "-")):
                ax.plot(t, curves[name], style, lw=1.9, label=name)
            ax.set_title(f"{variant} — {title}", fontsize=10)
            ax.set_xlabel("prediction horizon (s)")
            ax.set_ylabel("median pose error (mm)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / f"error_vs_horizon{RUN_TAG}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
