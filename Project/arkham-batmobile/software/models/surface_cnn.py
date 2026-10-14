"""
M3a part 1: is the floor surface identifiable from a single camera frame?

M2b measured the surface to be only ~69% identifiable from FOUR SECONDS of
proprioceptive history, and concluded that vision was the leading remaining
candidate because "carpet and tile are trivially distinguishable by
appearance". That claim was never tested. This file tests it, and then tries
hard to break the result.

The experiment is deliberately adversarial towards its own conclusion. A 64x64
simulated floor differs between surfaces in flat colour and in checker pitch,
so a CNN that separates them may be learning nothing more than "brown vs grey"
- a cartoon of the real problem. So every classifier here is run against four
inputs of decreasing generosity:

    rgb                 what the model would actually see
    rgb + jitter        brightness/colour shifted at test time only
    gray_std            per-frame standardised greyscale: chroma AND global
                        brightness removed, so only spatial structure survives
    floor_std           bottom half of the frame (almost all floor), same
                        standardisation

and against a one-line MEAN PIXEL VALUE threshold, which is the honest
null hypothesis for "the CNN found something interesting".

Usage:
    python surface_cnn.py --collect            # render the paired dataset
    python surface_cnn.py                      # train + evaluate + plots
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.mecanum_env import (  # noqa: E402
    MODEL_IMAGE_SIZE,
    MecanumEnv,
    smoothed_random_actions,
    wall_avoiding_action,
)

ROOT = Path(__file__).resolve().parents[1]
VISION_DATA = ROOT / "data" / "vision"
CKPT_DIR = ROOT / "checkpoints"
ASSETS = ROOT.parent / "experiments" / "M3a-surface-from-pixels" / "assets"

SURFACES = ("hard", "mat")
SURFACE_IDX = {s: i for i, s in enumerate(SURFACES)}

# Episode-level split. NEVER split by frame: adjacent frames at 20 Hz are
# nearly identical, so a random frame split puts near-duplicates on both sides
# and reports ~99% for a classifier that has learnt nothing.
N_EPISODES = 30
EP_TICKS = 400                      # 20 s per episode
SPLIT = {"train": range(0, 20), "val": range(20, 25), "test": range(25, 30)}

# M2b's best proprioceptive number, for the headline comparison.
M2B_PROPRIO_ACC = 0.690
M2B_PROPRIO_WINDOW_S = 4.0


# ── Collection ──────────────────────────────────────────────────────────────

def collect(n_episodes: int = N_EPISODES, ticks: int = EP_TICKS,
            image_size: int = MODEL_IMAGE_SIZE) -> None:
    """
    Render a paired (frames, proprioception, pose) dataset on both surfaces.

    TWO DELIBERATE CHOICES, both about not faking the result:

    1. The start pose and the commanded action plan for episode k are drawn
       from a per-episode RNG seeded IDENTICALLY on both surfaces. If `hard`
       episodes started in different parts of the arena from `mat` episodes,
       a classifier could key on which landmark is in view rather than on the
       floor, and part 1 would be measuring the collection script. Trajectories
       still diverge, because the dynamics differ - that is unavoidable and is
       noted as a caveat - but the initial conditions do not.

    2. The frame is rendered AFTER env.step(), so frame[t] is the view from
       pose_gt[t]. sim/collect.py renders before the step, which puts frame[t]
       at pose_gt[t-1]. Rendering after removes an off-by-one from part 2,
       where the surface estimate has to come from the frame at the window's
       start pose.
    """
    for surface in SURFACES:
        out_dir = VISION_DATA / surface
        out_dir.mkdir(parents=True, exist_ok=True)
        env = MecanumEnv(surface=surface, seed=0, image_size=image_size)
        t0 = time.perf_counter()

        for ep in range(n_episodes):
            # Identical on both surfaces by construction.
            rng = np.random.default_rng(10_000 + ep)
            pose0 = np.array([rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55),
                              rng.uniform(-np.pi, np.pi)])
            plan = smoothed_random_actions(ticks, rng)

            obs = env.reset(pose=pose0)
            frames, rows = [], []
            for i in range(ticks):
                obs = env.step(wall_avoiding_action(obs, plan[i]))
                frames.append(env.render("front", image_size))
                rows.append(obs)
                if obs["out_of_bounds"]:
                    obs = env.reset()

            np.save(out_dir / f"ep{ep:03d}.frames.npy", np.stack(frames))
            np.savez(
                out_dir / f"ep{ep:03d}.npz",
                action=np.stack([r["action_cmd"] for r in rows]).astype(np.float32),
                vel=np.stack([r["body_vel_avg"] for r in rows]).astype(np.float32),
                wheel=np.stack([r["wheel_meas"] for r in rows]).astype(np.float32),
                gyro=np.stack([r["imu_gyro"] for r in rows]).astype(np.float32),
                batt=np.array([r["batt_v"] for r in rows], dtype=np.float32),
                pose=np.stack([r["pose_gt"] for r in rows]).astype(np.float64),
            )
        env.close()
        print(f"  {surface}: {n_episodes} x {ticks} ticks in "
              f"{time.perf_counter() - t0:.1f}s -> {out_dir}")


def load_split(split: str, with_frames: bool = True) -> list[dict]:
    """One dict per episode. Episode-level, so splits never share a frame."""
    out = []
    for surface in SURFACES:
        for ep in SPLIT[split]:
            d = VISION_DATA / surface
            npz = d / f"ep{ep:03d}.npz"
            if not npz.exists():
                raise FileNotFoundError(
                    f"{npz} missing - run `python models/surface_cnn.py --collect` first")
            z = np.load(npz)
            rec = {k: z[k] for k in z.files}
            rec.update({"surface": surface, "label": SURFACE_IDX[surface],
                        "episode": f"{surface}/ep{ep:03d}"})
            if with_frames:
                rec["frames"] = np.load(d / f"ep{ep:03d}.frames.npy")
            out.append(rec)
    return out


def stack_frames(eps: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frames uint8 NHWC, labels, episode index per frame)."""
    f = np.concatenate([e["frames"] for e in eps])
    y = np.concatenate([np.full(len(e["frames"]), e["label"], dtype=np.int64) for e in eps])
    g = np.concatenate([np.full(len(e["frames"]), i, dtype=np.int64)
                        for i, e in enumerate(eps)])
    return f, y, g


# ── Input views: the same frames, with progressively less to go on ──────────

def jitter(frames: np.ndarray, seed: int, bright=(0.6, 1.4), chroma=(0.80, 1.20),
           per_frame: bool = True) -> np.ndarray:
    """
    Photometric nuisance: a global brightness gain plus an independent
    per-channel gain. This is the cheapest stand-in for the thing that actually
    varies on a real robot - the light in the room and the camera's auto white
    balance - and it is exactly what a classifier keyed on absolute colour
    cannot survive.
    """
    rng = np.random.default_rng(seed)
    n = len(frames) if per_frame else 1
    b = rng.uniform(*bright, size=(n, 1, 1, 1))
    c = rng.uniform(*chroma, size=(n, 1, 1, 3))
    out = frames.astype(np.float32) * b * c
    return np.clip(out, 0, 255).astype(np.uint8)


def to_view(frames: np.ndarray, view: str) -> np.ndarray:
    """
    uint8 NHWC -> float32 NCHW, normalised according to `view`.

        rgb        /255, dataset-independent. Absolute colour survives.
        gray_std   luminance, then PER-FRAME zero mean / unit variance.
                   Removes chroma and any global brightness or contrast gain,
                   so the only thing left is spatial structure - here, the
                   checker pitch (10 repeats on hard vs 26 on mat).
        floor_std  gray_std on the bottom half of the frame, which the 15-deg
                   downward camera tilt fills entirely with floor. Tests
                   whether the cue is the floor or the rest of the scene.
        sky_std    gray_std on the TOP QUARTER, which is black sky and wall and
                   contains no floor at all. The negative control: `mat`
                   episodes keep the robot slightly nearer the arena centre
                   (mean |pos| 0.27 m vs 0.31 m) because it slips more, so the
                   scene behind the floor is weakly surface-correlated through
                   the trajectory. If sky_std classifies well, part of any
                   other number here is that confound rather than the floor.
    """
    x = frames.astype(np.float32)
    if view == "rgb":
        return np.transpose(x / 255.0, (0, 3, 1, 2))
    if view == "floor_std":
        x = x[:, x.shape[1] // 2:, :, :]
    elif view == "sky_std":
        x = x[:, : x.shape[1] // 4, :, :]
    g = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)      # (N, H, W)
    m = g.mean(axis=(1, 2), keepdims=True)
    s = g.std(axis=(1, 2), keepdims=True)
    return ((g - m) / np.maximum(s, 1e-6))[:, None, :, :]


VIEWS = {
    "rgb":          dict(view="rgb", test_jitter=False, train_jitter=False),
    "rgb+jitter":   dict(view="rgb", test_jitter=True,  train_jitter=False),
    "rgb+aug":      dict(view="rgb", test_jitter=True,  train_jitter=True),
    "gray_std":     dict(view="gray_std", test_jitter=True, train_jitter=False),
    "floor_std":    dict(view="floor_std", test_jitter=True, train_jitter=False),
    "sky_std":      dict(view="sky_std", test_jitter=True, train_jitter=False),
}


# ── The null hypotheses: one global scalar, computed from the CNN's own input ──

def _laplacian_energy(g: np.ndarray) -> np.ndarray:
    """
    Mean |Laplacian| of a (N, H, W) image, divided by its own standard
    deviation so the statistic measures how much of the image's variance sits
    at high spatial frequency, not how bright or contrasty it is.

    This is the texture-scale analogue of "mean pixel value": the two floors
    use the same checker builtin at 10 and 26 repeats, so a single
    high-frequency energy number should separate them. Including it keeps the
    standardised-greyscale CNN honest - if this scalar matches it, that CNN is
    also reading one global statistic, just a less obvious one.
    """
    lap = (4.0 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
           - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
    return np.abs(lap).mean(axis=(1, 2)) / np.maximum(g.std(axis=(1, 2)), 1e-6)


def scalar_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Every one-line classifier, computed from exactly the tensor the CNN is
    handed, so the comparison is about the model and not about the input.
    """
    g = x.mean(axis=1)                                  # (N, H, W)
    feats = {"mean_pixel": x.mean(axis=(1, 2, 3)),
             "hf_energy": _laplacian_energy(g)}
    if x.shape[1] == 3:                                 # chroma only exists in RGB
        feats["chroma_r_minus_b"] = (x[:, 0] - x[:, 2]).mean(axis=(1, 2))
    return feats


def fit_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """
    Best single threshold on a scalar feature, fitted on TRAIN.

    Returns (threshold, polarity) where polarity +1 means "above the threshold
    is class 1". Both polarities are searched because "mat is brighter" and
    "mat is darker" are the same one-parameter rule and nothing should hinge
    on which way round the surfaces happen to be.
    """
    cands = np.quantile(x, np.linspace(0.0, 1.0, 513))
    best = (float(cands[0]), 1, 0.0)
    for t in cands:
        for pol in (1, -1):
            acc = threshold_accuracy(x, y, float(t), pol)
            if acc > best[2]:
                best = (float(t), pol, acc)
    return best[0], best[1]


def threshold_accuracy(x: np.ndarray, y: np.ndarray, t: float, pol: int) -> float:
    pred = ((x > t) if pol > 0 else (x <= t)).astype(np.int64)
    return float((pred == y).mean())


# ── The CNN ─────────────────────────────────────────────────────────────────

class SurfaceCNN(nn.Module):
    """
    ~6k parameters, three strided convs and a global average pool. Small on
    purpose: if the floor is identifiable at all it is identifiable by a
    texture statistic, and a bigger network would only make it harder to argue
    about what was learnt. Global pooling because the cue should be a property
    of the surface, not of where it appears in the frame.
    """

    def __init__(self, in_ch: int = 3, width: int = 8, n_classes: int = 2):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(4 * w, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_cnn(x: np.ndarray, y: np.ndarray, epochs: int = 8, seed: int = 0,
              bs: int = 128, lr: float = 3e-3) -> SurfaceCNN:
    torch.manual_seed(seed)
    model = SurfaceCNN(in_ch=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    n = len(xt)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(n).split(bs):
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def cnn_logits(model: SurfaceCNN, x: np.ndarray, bs: int = 512) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + bs])).numpy()
                           for i in range(0, len(x), bs)])


# ── How many frames does a confident call need? ─────────────────────────────

def vote_accuracy(logits: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  k: int) -> float:
    """
    Accuracy when the mean logit over k CONSECUTIVE frames decides, with the
    window required to stay inside one episode.
    """
    correct, total = 0, 0
    for g in np.unique(groups):
        sel = np.flatnonzero(groups == g)
        lg, ly = logits[sel], y[sel]
        if len(sel) < k:
            continue
        # Non-overlapping windows, so each frame is counted once.
        m = (len(sel) // k) * k
        avg = lg[:m].reshape(-1, k, lg.shape[1]).mean(axis=1)
        tgt = ly[:m].reshape(-1, k)[:, 0]
        correct += int((avg.argmax(1) == tgt).sum())
        total += len(tgt)
    return correct / max(total, 1)


# ── Driver ──────────────────────────────────────────────────────────────────

def run(epochs: int, seed: int) -> dict:
    print("-- Data ---------------------------------------------------------")
    tr_eps, va_eps, te_eps = (load_split(s) for s in ("train", "val", "test"))
    f_tr, y_tr, _ = stack_frames(tr_eps)
    f_va, y_va, _ = stack_frames(va_eps)
    f_te, y_te, g_te = stack_frames(te_eps)
    print(f"  train {len(f_tr):6d} frames / {len(tr_eps)} episodes")
    print(f"  val   {len(f_va):6d} frames / {len(va_eps)} episodes")
    print(f"  test  {len(f_te):6d} frames / {len(te_eps)} episodes")

    report: dict = {
        "n_frames": {"train": int(len(f_tr)), "val": int(len(f_va)), "test": int(len(f_te))},
        "n_episodes_per_split": {k: len(v) * len(SURFACES) for k, v in SPLIT.items()},
        "image_size": int(f_tr.shape[1]),
        "m2b_proprio_reference": {"accuracy": M2B_PROPRIO_ACC,
                                  "window_s": M2B_PROPRIO_WINDOW_S},
    }

    # How different are the two floors, really? State it as a number first.
    print("\n-- Mean frame RGB (the whole reason to be suspicious) -----------")
    appearance = {}
    for s in SURFACES:
        sel = y_te == SURFACE_IDX[s]
        mean_rgb = f_te[sel].reshape(-1, 3).mean(axis=0)
        gray = (f_te[sel].astype(np.float32) @ np.array([0.299, 0.587, 0.114],
                                                        dtype=np.float32))
        appearance[s] = {
            "mean_rgb": mean_rgb.round(1).tolist(),
            "mean_gray": round(float(gray.mean()), 1),
            "mean_r_minus_b": round(float(mean_rgb[0] - mean_rgb[2]), 1),
        }
        print(f"  {s:5s} mean RGB {mean_rgb.round(1)}  mean grey "
              f"{appearance[s]['mean_gray']:5.1f}  R-B "
              f"{appearance[s]['mean_r_minus_b']:+6.1f}")
    print("  note: the two floors are nearly IDENTICAL in luminance and differ "
          "strongly in chroma,\n        so plain 'mean pixel value' is a weak "
          "null hypothesis here and R-B is the right one.")
    report["appearance"] = appearance

    # Jittered copies, fixed seeds so every classifier sees the same nuisance.
    j_tr = jitter(f_tr, seed=101)
    j_te = jitter(f_te, seed=202)

    print("\n-- Classifiers ---------------------------------------------------")
    print(f"  {'condition':12s} {'mean px':>8s} {'R-B':>8s} {'hi-freq':>8s} "
          f"{'| CNN':>9s}")
    rows = []
    for name, cfg in VIEWS.items():
        view = cfg["view"]
        tr_src = j_tr if cfg["train_jitter"] else f_tr
        te_src = j_te if cfg["test_jitter"] else f_te
        x_tr, x_te = to_view(tr_src, view), to_view(te_src, view)

        # (1) the null hypotheses: single global scalars, threshold fitted on
        #     TRAIN, applied unchanged to TEST.
        ft_tr, ft_te = scalar_features(x_tr), scalar_features(x_te)
        scalars = {}
        for fname, v_tr in ft_tr.items():
            thr, pol = fit_threshold(v_tr, y_tr)
            scalars[fname] = round(threshold_accuracy(ft_te[fname], y_te, thr, pol), 4)

        # (2) the CNN, handed exactly the same tensor
        t0 = time.perf_counter()
        model = train_cnn(x_tr, y_tr, epochs=epochs, seed=seed)
        lg = cnn_logits(model, x_te)
        acc_cnn = float((lg.argmax(1) == y_te).mean())
        secs = time.perf_counter() - t0

        votes = {str(k): round(vote_accuracy(lg, y_te, g_te, k), 4)
                 for k in (1, 2, 5, 10, 20, 40, 80)}

        rows.append({"condition": name, "view": view,
                     "train_jitter": cfg["train_jitter"],
                     "test_jitter": cfg["test_jitter"],
                     "scalar_baselines": scalars,
                     "best_scalar_acc": round(max(scalars.values()), 4),
                     "best_scalar": max(scalars, key=scalars.get),
                     "cnn_acc": round(acc_cnn, 4),
                     "cnn_vote_acc": votes,
                     "train_s": round(secs, 1)})
        print(f"  {name:12s} {scalars['mean_pixel']:7.1%} "
              f"{scalars.get('chroma_r_minus_b', float('nan')):7.1%} "
              f"{scalars['hf_energy']:7.1%}  |{acc_cnn:7.1%}   ({secs:.0f}s)")

        if name == "rgb":
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "in_ch": x_tr.shape[1],
                        "width": 8, "view": view, "n_classes": len(SURFACES),
                        "surfaces": list(SURFACES), "test_acc": acc_cnn},
                       CKPT_DIR / "surface-cnn-rgb.pt")
        if name == "gray_std":
            torch.save({"state_dict": model.state_dict(), "in_ch": x_tr.shape[1],
                        "width": 8, "view": view, "n_classes": len(SURFACES),
                        "surfaces": list(SURFACES), "test_acc": acc_cnn},
                       CKPT_DIR / "surface-cnn-graystd.pt")
    report["classifiers"] = rows

    # How few TRAINING frames are enough? Subsample within the train episodes.
    # Fixed gradient-step budget at every size, so the curve measures data and
    # not how long each run happened to train; three seeds, because at 4 frames
    # per class a single run is mostly luck.
    print("\n-- Training-set size sweep (gray_std: the hardest honest view) ---")
    x_tr_g, x_te_g = to_view(f_tr, "gray_std"), to_view(j_te, "gray_std")
    steps = 400
    sweep = []
    for n_per_class in (1, 4, 16, 64, 256, 1024):
        accs = []
        for s in range(3):
            rng = np.random.default_rng(1000 * s + n_per_class)
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y_tr == c), n_per_class, replace=False)
                for c in range(len(SURFACES))])
            bs = min(64, len(idx))
            ep = max(1, int(round(steps / max(1, len(idx) / bs))))
            m = train_cnn(x_tr_g[idx], y_tr[idx], epochs=ep, seed=seed + s, bs=bs)
            accs.append(float((cnn_logits(m, x_te_g).argmax(1) == y_te).mean()))
        sweep.append({"frames_per_class": n_per_class,
                      "cnn_acc_mean": round(float(np.mean(accs)), 4),
                      "cnn_acc_min": round(float(np.min(accs)), 4),
                      "cnn_acc_max": round(float(np.max(accs)), 4)})
        print(f"  {n_per_class:5d} frames/class -> {np.mean(accs):6.1%} "
              f"(3 seeds: {np.min(accs):.1%} to {np.max(accs):.1%})")
    report["train_size_sweep"] = sweep

    return report, (f_te, j_te, y_te)


# ── Plots ───────────────────────────────────────────────────────────────────

def plots(report: dict, data) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f_te, j_te, y_te = data
    ASSETS.mkdir(parents=True, exist_ok=True)

    # 1. What the input views actually look like.
    pick = [int(np.flatnonzero(y_te == c)[300]) for c in range(len(SURFACES))]
    views = [("rgb", f_te, "RGB (as rendered)"),
             ("rgb", j_te, "RGB + photometric jitter"),
             ("gray_std", j_te, "per-frame standardised grey"),
             ("floor_std", j_te, "floor crop, standardised"),
             ("sky_std", j_te, "sky/wall crop (control)")]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for r, (c, p) in enumerate(zip(range(len(SURFACES)), pick)):
        for col, (v, src, title) in enumerate(views):
            img = to_view(src[p:p + 1], v)[0]
            ax = axes[r][col]
            if img.shape[0] == 3:
                ax.imshow(np.transpose(img, (1, 2, 0)))
            else:
                ax.imshow(img[0], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(SURFACES[c], fontsize=10)
    fig.suptitle("The inputs every classifier is run against", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "input_views.png", dpi=130)
    plt.close(fig)

    # 2. The single-scalar classifiers, as distributions. This is the plot that
    #    shows what the clean RGB result really is: R-B separates the surfaces
    #    completely on its own, and the moment the light changes it stops.
    panels = [(f_te, "rgb", "chroma_r_minus_b", "mean R - mean B, as rendered"),
              (j_te, "rgb", "chroma_r_minus_b", "mean R - mean B, + jitter"),
              (j_te, "gray_std", "mean_pixel", "mean pixel, standardised grey"),
              (j_te, "floor_std", "hf_energy", "high-frequency energy, floor crop")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    for ax, (src, v, feat, title) in zip(axes, panels):
        s = scalar_features(to_view(src, v))[feat]
        bins = np.linspace(float(s.min()), float(s.max()) + 1e-9, 70)
        for c, surf in enumerate(SURFACES):
            ax.hist(s[y_te == c], bins=bins, alpha=0.6, label=surf)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("frames")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("One scalar at a time: which cheap statistic already separates the floors",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "scalar_histograms.png", dpi=130)
    plt.close(fig)

    # 3. Accuracy: CNN vs the best one-line baseline, per input view, vs M2b.
    rows = report["classifiers"]
    names = [r["condition"] for r in rows]
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for i, (key, lab) in enumerate(
            [("best_scalar_acc", "best single global scalar"),
             ("cnn_acc", "CNN, single frame")]):
        bars = ax.bar(xs + i * 0.38, [r[key] * 100 for r in rows], 0.38, label=lab)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.axhline(M2B_PROPRIO_ACC * 100, ls="--", c="crimson", lw=1.4,
               label=f"M2b: {M2B_PROPRIO_ACC:.0%} from {M2B_PROPRIO_WINDOW_S:.0f}s "
                     "of proprioception")
    ax.axhline(50, ls=":", c="0.4", lw=1.2, label="chance")
    ax.set_xticks(xs + 0.19)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(40, 108)
    ax.set_ylabel("surface-classification accuracy (%)")
    ax.set_title("Surface from one frame, held-out episodes.  sky_std is the "
                 "no-floor control.\nWhere the two bars are equal, the CNN "
                 "found nothing a one-liner did not.", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "classifier_accuracy.png", dpi=130)
    plt.close(fig)

    # 4. How few frames: training frames, and frames per decision.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    sw = report["train_size_sweep"]
    n = [r["frames_per_class"] for r in sw]
    axes[0].semilogx(n, [r["cnn_acc_mean"] * 100 for r in sw], "o-", lw=1.8)
    axes[0].fill_between(n, [r["cnn_acc_min"] * 100 for r in sw],
                         [r["cnn_acc_max"] * 100 for r in sw], alpha=0.2)
    axes[0].axhline(M2B_PROPRIO_ACC * 100, ls="--", c="crimson", lw=1.4,
                    label="M2b, 4 s of proprioception")
    axes[0].axhline(50, ls=":", c="0.4", lw=1.2)
    axes[0].set_xlabel("training frames per surface")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].set_title("how much labelled data the cue needs\n"
                      "(standardised grey; band = 3 seeds)", fontsize=9.5)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    for r in rows:
        ks = sorted(int(k) for k in r["cnn_vote_acc"])
        axes[1].semilogx(ks, [r["cnn_vote_acc"][str(k)] * 100 for k in ks],
                         "o-", lw=1.6, ms=4, label=r["condition"])
    axes[1].axhline(M2B_PROPRIO_ACC * 100, ls="--", c="crimson", lw=1.4)
    axes[1].set_xlabel("frames averaged per decision")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("frames needed per decision\n(red: M2b's 80-tick proprioceptive result)",
                      fontsize=9.5)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "frames_needed.png", dpi=130)
    plt.close(fig)

    for name in ("input_views.png", "scalar_histograms.png",
                 "classifier_accuracy.png", "frames_needed.png"):
        print(f"  {ASSETS / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M3a part 1: surface from pixels")
    ap.add_argument("--collect", action="store_true", help="render the paired dataset")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--ticks", type=int, default=EP_TICKS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.collect:
        print(f"collecting {args.episodes} x {args.ticks} ticks per surface, with frames")
        collect(args.episodes, args.ticks)
        return

    report, data = run(args.epochs, args.seed)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "metrics_part1.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  {out}")
    plots(report, data)


if __name__ == "__main__":
    main()
