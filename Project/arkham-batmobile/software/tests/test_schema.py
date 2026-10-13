"""
Dataset schema completeness.

Protects against
----------------
BUG 8  Schema drift. software/README.md fixes the dataset schema up front
       because "the schema is the one decision that is expensive to change
       later - every trained model, every experiment and every plot depends on
       it", and the same columns must come out of the sim collector and, later,
       out of robot/logger.py.

       A missing or silently-retyped column does not crash anything. A model
       trains on whatever is there, reports a loss, and the gap only shows up
       as an unexplainable result months later. `pose_gt_valid` is the sharpest
       example the README calls out: drop it and occluded-marker rows become
       zeros, which look like a stationary robot to a training loop.

Also pins the two non-negotiable schema rules that are checkable from a
collected episode alone: ONE CLOCK (monotonic tick, monotonic `t_mono` derived
from it) and RAW ENCODER COUNTS kept alongside derived velocities.

The documented schema is transcribed below from software/README.md. If that
table changes, this list is the thing to change with it.
"""

import json

import numpy as np

from _harness import expected_failure, main, sim_episode

# column -> (python/numpy element kind, element shape)  from software/README.md
SCHEMA = {
    "episode_id":     ("str",     ()),
    "tick":           ("int",     ()),
    "t_mono":         ("float64", ()),
    "action_cmd":     ("float32", (3,)),
    "wheel_cmd":      ("float32", (4,)),
    "wheel_meas":     ("float32", (4,)),
    "encoder_counts": ("int32",   (4,)),
    "imu_gyro":       ("float32", (3,)),
    "imu_accel":      ("float32", (3,)),
    "pose_gt":        ("float32", (3,)),
    "pose_gt_valid":  ("bool",    ()),
    "pose_odom":      ("float32", (3,)),
    "body_vel_avg":   ("float32", (3,)),
    "frame_idx":      ("int32",   ()),
    "t_frame_mono":   ("float64", ()),
    "batt_v":         ("float32", ()),
    "cpu_temp":       ("float32", ()),
    "surface":        ("str",     ()),
    "shell":          ("bool",    ()),
}

# Documented in software/README.md but NOT written by software/sim/collect.py.
# See tests/README.md, "Live findings". Keep this list empty-or-explained: any
# OTHER missing column fails test_every_documented_column_is_present.
KNOWN_MISSING = {"cpu_temp"}


def _ep():
    return sim_episode(n_ticks=120, surface="hard", seed=11)


def _check_kind(column, value, kind, shape):
    if shape:
        assert isinstance(value, np.ndarray), f"{column}: expected an array, got {type(value)}"
        assert value.shape == shape, f"{column}: shape {value.shape}, expected {shape}"
        assert value.dtype == np.dtype(kind), f"{column}: dtype {value.dtype}, expected {kind}"
        return
    if kind == "str":
        assert isinstance(value, str), f"{column}: expected str, got {type(value)}"
    elif kind == "bool":
        assert isinstance(value, (bool, np.bool_)), f"{column}: expected bool, got {type(value)}"
    elif kind == "int":
        assert np.issubdtype(np.asarray(value).dtype, np.integer), f"{column}: {type(value)}"
    else:
        assert np.asarray(value).dtype == np.dtype(kind), (
            f"{column}: dtype {np.asarray(value).dtype}, expected {kind}"
        )


# ── BUG 8 ───────────────────────────────────────────────────────────────────

def test_every_documented_column_is_present():
    df = _ep().df
    missing = sorted(set(SCHEMA) - set(df.columns) - KNOWN_MISSING)
    assert not missing, (
        f"collected episode is missing documented column(s) {missing}. "
        f"software/README.md is the schema contract and training code will "
        f"quietly train without them."
    )


# Was an expected_failure when this suite was written: collect.py._row() never
# wrote cpu_temp despite the schema documenting it, so sim and robot schemas
# were already diverging. Fixed 2026-10-13 - sim writes NaN, which is the honest
# value for "this machine has no such sensor" and is loud if anything trains on
# it. Marker removed - this now guards the fix.
def test_cpu_temp_column_exists():
    assert "cpu_temp" in _ep().df.columns


def test_documented_columns_have_the_documented_shape_and_dtype():
    df = _ep().df
    for column, (kind, shape) in SCHEMA.items():
        if column not in df.columns:
            continue
        for row in (0, len(df) // 2, len(df) - 1):
            _check_kind(column, df[column].iloc[row], kind, shape)


def test_no_undocumented_column_is_mistaken_for_a_documented_one():
    """
    Extra columns are fine - the sim logs `body_vel_gt`, `scrub` and
    `saturated` on purpose, and they are documented in sim/README.md. This test
    exists so that adding a column named like a schema column (a typo, a
    rename-in-progress) is noticed rather than absorbed.
    """
    extra = sorted(set(_ep().df.columns) - set(SCHEMA))
    assert extra == ["body_vel_gt", "saturated", "scrub"], (
        f"unexpected extra columns {extra}; if these are intentional, add them "
        f"here and to the schema table in software/README.md"
    )


def test_pose_gt_valid_is_present_and_boolean():
    """
    The README: "Keep pose_gt_valid. Occluded-marker rows silently become zeros
    otherwise, and zeros look like a stationary robot to a training loop."
    """
    df = _ep().df
    assert df["pose_gt_valid"].dtype == np.dtype("bool"), df["pose_gt_valid"].dtype
    assert bool(df["pose_gt_valid"].all()), "the sim fiducial is never occluded"


# ── Schema rules that are checkable from one episode ────────────────────────

def test_one_clock_and_a_monotonic_tick():
    """
    "Monotonic within episode, 20 Hz" and "every timestamp from one clock".
    `collect.run_episode` calls `env.reset()` when the robot leaves the arena,
    and `reset()` zeroes `env.tick` - so if a wall contact ever happens inside
    an episode, `tick` and `t_mono` both restart mid-file. This test is what
    would catch that.
    """
    df = _ep().df
    tick = df["tick"].to_numpy()
    assert np.all(np.diff(tick) == 1), (
        f"tick is not monotonic by 1: {np.flatnonzero(np.diff(tick) != 1)[:5]}"
    )
    t = df["t_mono"].to_numpy()
    assert np.all(np.diff(t) > 0), "t_mono is not strictly increasing"
    assert np.allclose(t, tick * 0.05), "t_mono must be tick / CONTROL_HZ"


def test_raw_encoder_counts_are_kept_and_accumulate():
    """
    "Keep raw encoder counts as well as derived velocities." Counts are what
    lets every velocity-estimation choice be revisited without recollecting,
    and roller ripple is only predictable from integrated wheel angle.
    """
    df = _ep().df
    counts = np.stack(df["encoder_counts"].to_numpy()).astype(np.int64)
    assert counts.dtype.kind == "i"
    moved = np.abs(counts[-1] - counts[0])
    assert np.all(moved > 0), f"encoder counts did not move on all four wheels: {moved}"


def test_episode_id_is_constant_and_matches_the_manifest():
    ep = _ep()
    assert set(ep.df["episode_id"]) == {ep.episode_id}
    manifest = json.loads(ep.manifest_path.read_text())
    assert manifest["episode_id"] == ep.episode_id
    assert manifest["n_ticks"] == len(ep.df)


def test_frame_index_is_minus_one_when_no_video_was_recorded():
    df = _ep().df
    assert np.all(np.stack(df["frame_idx"].to_numpy()) == -1)


def test_frame_index_points_into_the_saved_frame_stack():
    """
    The row carries an index into the episode's frames; no images in the
    Parquet. An index that does not line up with the stack silently pairs each
    action with somebody else's picture.
    """
    ep = sim_episode(n_ticks=40, surface="hard", seed=11, video=True)
    idx = np.stack(ep.df["frame_idx"].to_numpy())
    assert np.all(np.diff(idx) == 1) and idx[0] == 0, idx[:5]
    frames = np.load(ep.frames_path, mmap_mode="r")
    assert frames.shape[0] == len(ep.df) == idx[-1] + 1, frames.shape
    assert frames.dtype == np.uint8, frames.dtype
    assert frames.shape[1] == frames.shape[2] == 64, frames.shape


if __name__ == "__main__":
    raise SystemExit(main(globals()))
