"""
M3a part 2: does a VISION-DERIVED surface estimate actually fix M2's transfer
failure?

Part 1 established that the floor is ~100% identifiable from one 64x64 frame.
That is necessary but not sufficient: knowing the surface only helps if the
dynamics model can use the label, and if a model trained across both surfaces
does not already handle the range without being told (M2b's route 2).

So this is a three-way ablation with everything except the surface input held
fixed - same architecture, same parameter count, same training windows, same
analytic prior underneath:

    both, no surface input       domain randomisation: the honest control
    both + CNN-predicted surface the thing being tested
    both + TRUE surface          oracle upper bound

plus two reference points that are NOT matched and are labelled as such:

    hard-only learned model      the M2 predictor, reproduced on this dataset
    calibrated analytic          pooled fit, and a per-surface oracle fit

And then the question part 1 cannot answer on its own: HOW ACCURATE does the
surface estimate have to be? The label fed to the oracle model is corrupted at
a controlled rate, which puts M2b's 69%-from-proprioception number on the same
axis as the CNN's and says directly whether rejecting proprioception was right.

Usage:
    python surface_cnn.py --collect      # once, shared with part 1
    python surface_cnn.py                # trains the CNN this file loads
    python vision_conditioned.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.mecanum import MecanumKinematics  # noqa: E402
from sim.mecanum_env import CONTROL_DT  # noqa: E402
from models.state_model import (  # noqa: E402
    HORIZON,
    POSE_JUMP_LIMIT,
    StatePredictor,
    analytic_prior,
    build_windows,
    fit_calibration_fast,
    integrate_all,
    path_length_m,
    predict_learned,
    start_vel,
    train as train_state,
)
from models.surface_cnn import (  # noqa: E402
    ASSETS,
    CKPT_DIR,
    SURFACES,
    SURFACE_IDX,
    SurfaceCNN,
    cnn_logits,
    load_split,
    to_view,
)

M2B_PROPRIO_ACC = 0.690


# ── Windows, with the frame index carried along ─────────────────────────────

def build_windows_indexed(episodes: list[dict], stride: int = 1,
                          history: int = 1) -> dict[str, np.ndarray]:
    """
    Exactly state_model.build_windows, plus (episode index, start tick) per
    window so the frame at the window's START POSE can be attached.

    The logic is duplicated rather than imported because build_windows does not
    return indices - so `check_matches_state_model` below asserts the two agree
    element for element. If that assertion ever fires, this file is wrong and
    every number in it is meaningless.

    ALIGNMENT, restated because it is the one thing that silently ruins
    everything: vel[t] is the average velocity over the tick ENDING at pose[t].
    Predicting forward from pose[t] therefore consumes t+1 onward. The frame,
    collected by surface_cnn.collect, is rendered AFTER the step, so frame[t]
    is the view from pose[t] - the pose the prediction starts at.
    """
    H, A, V, P0, P1, EP, T, Y = [], [], [], [], [], [], [], []

    for ei, ep in enumerate(episodes):
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
                continue
            H.append(feats[past])
            A.append(ep["action"][fwd])
            V.append(ep["vel"][fwd])
            P0.append(ep["pose"][t])
            P1.append(ep["pose"][t + HORIZON])
            EP.append(ei)
            T.append(t)
            Y.append(ep["label"])

    return {"hist": np.asarray(H, dtype=np.float32),
            "actions": np.asarray(A, dtype=np.float32),
            "vels": np.asarray(V, dtype=np.float32),
            "pose0": np.asarray(P0, dtype=np.float64),
            "poseH": np.asarray(P1, dtype=np.float64),
            "ep": np.asarray(EP, dtype=np.int64),
            "t": np.asarray(T, dtype=np.int64),
            "label": np.asarray(Y, dtype=np.int64)}


def check_matches_state_model(episodes: list[dict], stride: int, history: int) -> None:
    ref = build_windows(episodes, stride, history)
    mine = build_windows_indexed(episodes, stride, history)
    for k in ("hist", "actions", "vels", "pose0", "poseH"):
        assert ref[k].shape == mine[k].shape, (k, ref[k].shape, mine[k].shape)
        assert np.array_equal(ref[k], mine[k]), k


def subset(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


# ── The surface input ───────────────────────────────────────────────────────

def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(SURFACES), dtype=np.float32)[labels]


def with_surface(hist: np.ndarray, surf: np.ndarray) -> np.ndarray:
    """
    Append the surface code to every history step.

    All four learned predictors get a 16-dimensional feature vector, so they
    have IDENTICAL architecture and parameter count; the ones that are not
    supposed to know the surface get a constant 0.5 in those two slots rather
    than having the slots removed. Otherwise "knows the surface" would be
    confounded with "is a slightly bigger model".
    """
    k = hist.shape[1]
    return np.concatenate([hist, np.repeat(surf[:, None, :], k, axis=1)], axis=2)


def uninformed(n: int) -> np.ndarray:
    """The surface slots, carrying no information: a constant, not a label."""
    return np.full((n, len(SURFACES)), 0.5, dtype=np.float32)


@torch.no_grad()
def cnn_surface_per_window(d: dict, episodes: list[dict], model: SurfaceCNN,
                           view: str) -> tuple[np.ndarray, float]:
    """
    Run the part-1 CNN on the SINGLE frame at each window's start pose.

    No temporal voting and no smoothing: this is the hardest honest version of
    "supply the CNN's prediction", and it is what a controller would have
    available at the instant it has to plan.
    """
    preds = np.empty(len(d["ep"]), dtype=np.int64)
    for ei, ep in enumerate(episodes):
        sel = np.flatnonzero(d["ep"] == ei)
        if not len(sel):
            continue
        x = to_view(ep["frames"][d["t"][sel]], view)
        preds[sel] = cnn_logits(model, x).argmax(1)
    return preds, float((preds == d["label"]).mean())


def corrupt(labels: np.ndarray, accuracy: float, seed: int) -> np.ndarray:
    """A surface estimator of exactly the given accuracy, flipping at random."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(labels)) > accuracy
    return np.where(flip, 1 - labels, labels)


# ── Evaluation ──────────────────────────────────────────────────────────────

def errors_mm(kin, d: dict, vels: np.ndarray) -> np.ndarray:
    """1.0 s pose error per window, against exact integration of the logged
    average velocities (which reproduces pose_gt to 0.000 mm)."""
    pred = integrate_all(kin, d["pose0"], vels)[:, -1]
    truth = integrate_all(kin, d["pose0"], d["vels"])[:, -1]
    return np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1) * 1000.0


def summarise(err: np.ndarray, d: dict) -> dict:
    dist = path_length_m(d["pose0"], d["poseH"])
    out = {}
    for name, m in [("all", np.ones(len(err), bool))] + [
            (s, d["label"] == SURFACE_IDX[s]) for s in SURFACES]:
        e, p = err[m], dist[m]
        out[name] = {"median_mm": round(float(np.median(e)), 2),
                     "mean_mm": round(float(e.mean()), 2),
                     "p95_mm": round(float(np.percentile(e, 95)), 2),
                     "median_pct_of_path": round(
                         float(np.median(e / 1000.0 / np.maximum(p, 1e-6)) * 100), 2),
                     "n": int(m.sum())}
    return out


def horizon_curve(kin, d: dict, vels: np.ndarray, surface: str) -> list[float]:
    m = d["label"] == SURFACE_IDX[surface]
    pred = integrate_all(kin, d["pose0"][m], vels[m])
    truth = integrate_all(kin, d["pose0"][m], d["vels"][m])
    e = np.linalg.norm(pred[..., :2] - truth[..., :2], axis=2) * 1000.0
    return np.median(e, axis=0).round(3).tolist()


# ── Driver ──────────────────────────────────────────────────────────────────

def run(epochs: int, stride_train: int, stride_eval: int, seed: int) -> dict:
    kin = MecanumKinematics()

    print("-- Data ---------------------------------------------------------")
    tr_eps = load_split("train", with_frames=False)
    te_eps = load_split("test", with_frames=True)
    check_matches_state_model(tr_eps[:2], stride_train, 1)
    print("  window builder matches state_model.build_windows exactly")

    d_tr = build_windows_indexed(tr_eps, stride_train, 1)
    d_te = build_windows_indexed(te_eps, stride_eval, 1)
    tr_hard = subset(d_tr, d_tr["label"] == SURFACE_IDX["hard"])
    tr_mat = subset(d_tr, d_tr["label"] == SURFACE_IDX["mat"])
    print(f"  train {len(d_tr['hist']):6d} windows "
          f"({len(tr_hard['hist'])} hard + {len(tr_mat['hist'])} mat)")
    print(f"  test  {len(d_te['hist']):6d} windows, held-out episodes, both surfaces")

    print("\n-- Calibrating the analytic baselines on TRAIN only -------------")
    cal = {}
    for name, dd in (("pooled", d_tr), ("hard", tr_hard), ("mat", tr_mat)):
        g, t = fit_calibration_fast(kin, dd["pose0"], dd["actions"], dd["poseH"],
                                    start_vel(dd), seed=seed)
        cal[name] = (g, t)
        print(f"  {name:7s} gains fwd {g[0]:.3f} strafe {g[1]:.3f} rot {g[2]:.3f}"
              f"  tau {t*1000:3.0f} ms")

    v0_te = start_vel(d_te)
    prior_pool = analytic_prior(d_te["actions"], cal["pooled"][0], cal["pooled"][1], v0_te)
    prior_hardcal = analytic_prior(d_te["actions"], cal["hard"][0], cal["hard"][1], v0_te)

    # Analytic with a perfect surface oracle: each window gets its own surface's fit.
    prior_oracle = np.empty_like(prior_pool)
    for s in SURFACES:
        m = d_te["label"] == SURFACE_IDX[s]
        prior_oracle[m] = analytic_prior(d_te["actions"][m], cal[s][0], cal[s][1], v0_te[m])

    print("\n-- The surface classifier at test time --------------------------")
    ck = torch.load(CKPT_DIR / "surface-cnn-rgb.pt", weights_only=False)
    cnn = SurfaceCNN(in_ch=ck["in_ch"], width=ck["width"], n_classes=ck["n_classes"])
    cnn.load_state_dict(ck["state_dict"])
    cnn_pred, cnn_acc = cnn_surface_per_window(d_te, te_eps, cnn, ck["view"])
    print(f"  single-frame CNN on the {len(d_te['ep'])} window start frames: "
          f"{cnn_acc:.2%} accurate")

    # ── Learned predictors ──────────────────────────────────────────────────
    results, curves = {}, {}
    models = {}

    def fit(tag: str, dd: dict, surf_feat: np.ndarray, prior_cal: tuple) -> StatePredictor:
        g, t = prior_cal
        hist = with_surface(dd["hist"], surf_feat)
        prior = analytic_prior(dd["actions"], g, t, start_vel(dd))
        model = StatePredictor(hist.shape[2], variant="residual")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n-- Training [{tag}] ({n_params:,} params, "
              f"{len(hist)} windows) ---")
        t0 = time.perf_counter()
        train_state(model, {"hist": hist, "actions": dd["actions"], "vels": dd["vels"]},
                    prior, epochs, seed)
        print(f"  trained in {time.perf_counter() - t0:.1f}s")
        models[tag] = model
        return model

    m_hard = fit("hard-only", tr_hard, uninformed(len(tr_hard["hist"])), cal["hard"])
    m_both = fit("both, no surface", d_tr, uninformed(len(d_tr["hist"])), cal["pooled"])
    m_cond = fit("both + surface", d_tr, one_hot(d_tr["label"]), cal["pooled"])

    uninf_te = uninformed(len(d_te["hist"]))
    preds = {
        "naive analytic": analytic_prior(d_te["actions"],
                                         np.ones(3, dtype=np.float32), 0.0),
        "calibrated, pooled fit": prior_pool,
        "calibrated, true-surface fit": prior_oracle,
        "learned, hard only (M2)": predict_learned(
            m_hard, {"hist": with_surface(d_te["hist"], uninf_te),
                     "actions": d_te["actions"]}, prior_hardcal),
        "learned, both surfaces": predict_learned(
            m_both, {"hist": with_surface(d_te["hist"], uninf_te),
                     "actions": d_te["actions"]}, prior_pool),
        "learned, both + CNN surface": predict_learned(
            m_cond, {"hist": with_surface(d_te["hist"], one_hot(cnn_pred)),
                     "actions": d_te["actions"]}, prior_pool),
        "learned, both + true surface": predict_learned(
            m_cond, {"hist": with_surface(d_te["hist"], one_hot(d_te["label"])),
                     "actions": d_te["actions"]}, prior_pool),
    }

    print("\n-- 1.0 s pose error on held-out episodes ------------------------")
    print(f"  {'predictor':30s} {'hard':>10s} {'mat':>10s} {'both':>10s}")
    for name, vels in preds.items():
        err = errors_mm(kin, d_te, vels)
        results[name] = summarise(err, d_te)
        curves[name] = {s: horizon_curve(kin, d_te, vels, s) for s in SURFACES}
        r = results[name]
        print(f"  {name:30s} {r['hard']['median_mm']:8.2f}mm "
              f"{r['mat']['median_mm']:8.2f}mm {r['all']['median_mm']:8.2f}mm")

    # ── How accurate does the surface estimate have to be? ──────────────────
    # The MEAN is reported alongside the median on purpose. A randomly wrong
    # label makes ~half the windows badly wrong and leaves the other half
    # perfect, and the median of that mixture is much kinder than the mixture
    # is - so a median-only sweep would overstate how tolerant the model is of
    # a poor surface estimate. The p95 is the planner-relevant number.
    print("\n-- Sweeping surface-estimate accuracy ---------------------------")
    print(f"  {'estimate':>9s} {'median':>9s} {'mean':>9s} {'p95':>9s}"
          f" {'med hard':>9s} {'med mat':>9s}")
    sweep = []
    for acc in (0.50, 0.60, M2B_PROPRIO_ACC, 0.80, 0.90, 0.95, 0.99, 1.00):
        per_seed = []
        for s in range(3):
            lab = corrupt(d_te["label"], acc, seed=seed + 7 * s + int(acc * 1000))
            v = predict_learned(m_cond,
                                {"hist": with_surface(d_te["hist"], one_hot(lab)),
                                 "actions": d_te["actions"]}, prior_pool)
            e = errors_mm(kin, d_te, v)
            per_seed.append([float(np.median(e)), float(e.mean()),
                             float(np.percentile(e, 95)),
                             float(np.median(e[d_te["label"] == 0])),
                             float(np.median(e[d_te["label"] == 1]))])
        m = np.mean(per_seed, axis=0)
        sweep.append({"estimate_accuracy": round(acc, 3),
                      "median_mm_all": round(float(m[0]), 2),
                      "mean_mm_all": round(float(m[1]), 2),
                      "p95_mm_all": round(float(m[2]), 2),
                      "median_mm_hard": round(float(m[3]), 2),
                      "median_mm_mat": round(float(m[4]), 2)})
        print(f"  {acc:8.0%} {m[0]:8.2f}mm {m[1]:8.2f}mm {m[2]:8.2f}mm "
              f"{m[3]:8.2f}mm {m[4]:8.2f}mm")
    results_sweep = sweep

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, key in (("both + surface", "surface-cond-both.pt"),
                     ("both, no surface", "surface-cond-domainrand.pt"),
                     ("hard-only", "surface-cond-hardonly.pt")):
        torch.save({"state_dict": models[tag].state_dict(),
                    "feat_dim": d_tr["hist"].shape[2] + len(SURFACES),
                    "hidden": 96, "variant": "residual", "history": 1,
                    "horizon": HORIZON,
                    "gains": cal["pooled" if tag != "hard-only" else "hard"][0].tolist(),
                    "tau": cal["pooled" if tag != "hard-only" else "hard"][1],
                    "surface_input": tag == "both + surface",
                    "trained_on": "hard" if tag == "hard-only" else "hard+mat"},
                   CKPT_DIR / key)

    return {"horizon_s": HORIZON * CONTROL_DT,
            "stride_train": stride_train, "stride_eval": stride_eval,
            "epochs": epochs, "seed": seed,
            "n_windows": {"train": int(len(d_tr["hist"])),
                          "train_hard": int(len(tr_hard["hist"])),
                          "train_mat": int(len(tr_mat["hist"])),
                          "test": int(len(d_te["hist"]))},
            "calibration": {k: {"gains": v[0].round(4).tolist(),
                                "tau_s": round(v[1], 4)} for k, v in cal.items()},
            "cnn_surface_accuracy_on_window_frames": round(cnn_acc, 4),
            "predictors": results,
            "error_vs_horizon_mm": curves,
            "estimate_accuracy_sweep": results_sweep,
            "m2b_proprio_reference": M2B_PROPRIO_ACC}


# ── Plots ───────────────────────────────────────────────────────────────────

def plots(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    res = report["predictors"]
    names = list(res)

    # 1. The headline bars, per surface.
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    xs = np.arange(len(names))
    for i, s in enumerate(SURFACES):
        bars = ax.bar(xs + i * 0.38, [res[n][s]["median_mm"] for n in names], 0.38,
                      label=f"{s} (held out)")
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax.set_xticks(xs + 0.19)
    ax.set_xticklabels([n.replace(", ", ",\n") for n in names], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median 1.0 s pose error (mm, log scale)")
    ax.set_title("Does a vision-derived surface estimate fix the transfer gap?\n"
                 "The last three bars differ ONLY in the two surface input "
                 "dimensions.", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "conditioned_error.png", dpi=130)
    plt.close(fig)

    # 2. Error vs horizon, per surface.
    t = np.arange(1, HORIZON + 1) * CONTROL_DT
    show = ["calibrated, pooled fit", "learned, hard only (M2)",
            "learned, both surfaces", "learned, both + CNN surface",
            "learned, both + true surface"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, s in zip(axes, SURFACES):
        for n in show:
            ax.plot(t, report["error_vs_horizon_mm"][n][s], lw=1.8, label=n)
        ax.set_title(f"{s} floor, held-out episodes", fontsize=10)
        ax.set_xlabel("prediction horizon (s)")
        ax.set_ylabel("median pose error (mm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "conditioned_error_vs_horizon.png", dpi=130)
    plt.close(fig)

    # 3. How good does the estimate have to be?
    sw = report["estimate_accuracy_sweep"]
    a = [r["estimate_accuracy"] * 100 for r in sw]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for ax, (key, dr_key, lab) in zip(axes, [
            ("median_mm_all", "median_mm", "median"),
            ("mean_mm_all", "mean_mm", "mean"),
            ("p95_mm_all", "p95_mm", "p95")]):
        ax.plot(a, [r[key] for r in sw], "o-", lw=1.9, ms=4, c="C0",
                label=f"{lab} 1.0 s error")
        ax.axhline(res["learned, both surfaces"]["all"][dr_key], ls="--", c="0.35",
                   lw=1.5, label="domain randomisation (no surface input)")
        ax.axhline(res["calibrated, pooled fit"]["all"][dr_key], ls="-.", c="0.6",
                   lw=1.3, label="calibrated analytic, pooled fit")
        ax.axvline(report["m2b_proprio_reference"] * 100, ls=":", c="crimson", lw=1.6)
        ax.axvline(report["cnn_surface_accuracy_on_window_frames"] * 100, ls=":",
                   c="seagreen", lw=1.6)
        top = ax.get_ylim()[1]
        ax.text(report["m2b_proprio_reference"] * 100 + 1, top * 0.86,
                "M2b ceiling\n(4 s proprioception)", fontsize=7.5, color="crimson")
        ax.text(99.5, top * 0.55, "CNN", fontsize=8, color="seagreen", ha="right")
        ax.set_xlabel("accuracy of the surface estimate fed to the model (%)")
        ax.set_ylabel(f"{lab} 1.0 s pose error (mm)")
        ax.set_title(f"{lab} error vs quality of the surface estimate", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
    fig.suptitle("How right does the surface estimate have to be?  The median is far "
                 "kinder than the mean, and\nin the tail a mediocre estimate is WORSE "
                 "than not conditioning at all (p95 crosses the dashed line).",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "estimate_accuracy_sweep.png", dpi=130)
    plt.close(fig)

    for n in ("conditioned_error.png", "conditioned_error_vs_horizon.png",
              "estimate_accuracy_sweep.png"):
        print(f"  {ASSETS / n}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M3a part 2: vision-conditioned dynamics")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--stride-train", type=int, default=1)
    ap.add_argument("--stride-eval", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures from metrics_part2.json without retraining")
    args = ap.parse_args()

    out = ASSETS / "metrics_part2.json"
    if args.plot_only:
        # software/README.md: plots are regenerated from metrics.json, never
        # hand-edited. This flag is what makes that true in practice.
        plots(json.loads(out.read_text()))
        return

    report = run(args.epochs, args.stride_train, args.stride_eval, args.seed)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  {out}")

    # software/README.md: one metrics.json per experiment folder. This file runs
    # last, so it merges part 1's into the combined one the README cites.
    part1 = ASSETS / "metrics_part1.json"
    combined = {"experiment": "M3a-surface-from-pixels",
                "part1_surface_from_pixels":
                    json.loads(part1.read_text()) if part1.exists() else None,
                "part2_vision_conditioned_dynamics": report}
    (ASSETS / "metrics.json").write_text(json.dumps(combined, indent=2))
    print(f"  {ASSETS / 'metrics.json'}")
    plots(report)


if __name__ == "__main__":
    main()
