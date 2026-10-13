"""
Shared bootstrap for the regression suite.

Does three jobs and nothing else:

1. Puts `software/` and `software/tests/` on `sys.path`, so every test module can
   `from kinematics.mecanum import ...` exactly the way the source modules do.
2. Provides `skip()` and `expected_failure()` that work BOTH under pytest and
   under the dependency-free runner in `run_all.py`.
3. Builds the small shared sim artifacts (one short episode, one set of
   prediction windows) once per process and caches them, so the whole suite
   stays well inside its time budget.

No test logic lives here.
"""

import atexit
import importlib
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SOFTWARE_DIR = TESTS_DIR.parent
PROJECT_DIR = SOFTWARE_DIR.parent

for _p in (str(TESTS_DIR), str(SOFTWARE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── skip / expected-failure, portable across pytest and the plain runner ─────

try:
    import pytest as _pytest
except Exception:                                      # pragma: no cover
    _pytest = None


if _pytest is not None:
    SKIP_EXC: type[BaseException] = _pytest.skip.Exception

    def skip(message: str) -> None:
        """Skip the current test with a message that says what was missing."""
        _pytest.skip(message)

    def expected_failure(reason: str):
        """
        Mark a test that documents a bug which is still LIVE in code this suite
        does not own.

        Under pytest this is `xfail(strict=True)`: the suite stays green while
        the bug exists, and turns RED the moment the test starts passing, which
        is the reminder to delete the marker. The plain runner does the same.
        """
        def deco(fn):
            marked = _pytest.mark.xfail(strict=True, reason=reason)(fn)
            marked._xfail_reason = reason
            return marked
        return deco

else:                                                  # pragma: no cover
    class SkipTest(Exception):
        pass

    SKIP_EXC = SkipTest

    def skip(message: str) -> None:
        raise SkipTest(message)

    def expected_failure(reason: str):
        def deco(fn):
            fn._xfail_reason = reason
            return fn
        return deco


def import_or_skip(dotted: str, note: str = ""):
    """
    Import a module or skip the test with the real reason.

    Used for anything that needs MuJoCo, torch, or `software/models/` — a test
    that cannot run must say so loudly rather than quietly passing.
    """
    try:
        return importlib.import_module(dotted)
    except Exception as exc:
        skip(f"cannot import {dotted}: {type(exc).__name__}: {exc}{note}")


# ── Shared sim artifacts ─────────────────────────────────────────────────────

_TMP: dict[str, Path] = {}
_CACHE: dict[tuple, object] = {}


def _tmpdir() -> Path:
    if "dir" not in _TMP:
        path = Path(tempfile.mkdtemp(prefix="batmobile-tests-"))
        _TMP["dir"] = path
        atexit.register(shutil.rmtree, path, True)
    return _TMP["dir"]


class Episode:
    """A collected episode, read back from Parquet the way training reads it."""

    def __init__(self, df, directory: Path, episode_id: str, n_frames: int):
        self.df = df
        self.dir = directory
        self.episode_id = episode_id
        self.n_frames = n_frames

    @property
    def frames_path(self) -> Path:
        return self.dir / f"{self.episode_id}.frames.npy"

    @property
    def manifest_path(self) -> Path:
        return self.dir / f"{self.episode_id}.manifest.json"


def sim_episode(n_ticks: int = 300, surface: str = "hard", seed: int = 11,
                mode: str = "random", video: bool = False) -> Episode:
    """
    Collect a short episode with the REAL collector and read it back through
    Parquet, so these tests exercise the writer and its dtypes rather than a
    hand-built DataFrame.

    Cached per (args) for the process. ~0.1 s for 300 state-only ticks.
    """
    key = ("episode", n_ticks, surface, seed, mode, video)
    if key in _CACHE:
        return _CACHE[key]                                 # type: ignore[return-value]

    import_or_skip("mujoco", " (the Phase 0 sim needs it)")
    pd = import_or_skip("pandas")
    env_mod = import_or_skip("sim.mecanum_env")
    collect = import_or_skip("sim.collect")
    np = import_or_skip("numpy")

    episode_id = f"regression-{surface}-{seed}-{n_ticks}{'-vid' if video else ''}"
    out_dir = _tmpdir() / episode_id
    env = env_mod.MecanumEnv(surface=surface, seed=seed)
    try:
        if video:
            try:
                env.render("front", env_mod.MODEL_IMAGE_SIZE)
            except Exception as exc:
                skip(f"offscreen rendering unavailable: {type(exc).__name__}: {exc}")
        df, frames = collect.run_episode(
            env, episode_id, n_ticks, mode, np.random.default_rng(seed + 1), video
        )
    finally:
        env.close()

    collect.save_episode(df, frames, out_dir, episode_id, {"source": "regression-suite"})
    back = pd.read_parquet(out_dir / f"{episode_id}.parquet")
    ep = Episode(back, out_dir, episode_id, len(frames))
    _CACHE[key] = ep
    return ep


def prediction_windows(surface: str = "hard", n_ticks: int = 1200,
                       horizon: int = 20, seed: int = 5) -> list:
    """
    (start_pose, measured_start_velocity, action_sequence, true_end_pose) windows.

    This is `mecanum_env.collect_windows` plus the one thing it does not return:
    the velocity the robot is ALREADY travelling at when the window opens. That
    omission is the M2 baseline-fairness bug, so a fair-baseline test has to
    gather its own windows.

    Cached per (args). ~0.2 s for 1200 ticks.
    """
    key = ("windows", surface, n_ticks, horizon, seed)
    if key in _CACHE:
        return _CACHE[key]                                 # type: ignore[return-value]

    import_or_skip("mujoco", " (the Phase 0 sim needs it)")
    np = import_or_skip("numpy")
    env_mod = import_or_skip("sim.mecanum_env")

    env = env_mod.MecanumEnv(surface=surface, seed=seed)
    windows = []
    try:
        obs = env.reset(pose=np.zeros(3))
        plan = env_mod.smoothed_random_actions(n_ticks, np.random.default_rng(seed + 1))
        i = 0
        while i < len(plan) - horizon:
            start_pose = obs["pose_gt"].copy()
            start_vel = obs["body_vel_avg"].copy()
            applied, hit_wall = [], False
            for k in range(horizon):
                action = env_mod.wall_avoiding_action(obs, plan[i + k])
                obs = env.step(action)
                applied.append(action)
                if obs["out_of_bounds"]:
                    hit_wall = True
                    break
            if not hit_wall and len(applied) == horizon:
                windows.append(
                    (start_pose, start_vel, np.array(applied), obs["pose_gt"].copy())
                )
            else:
                obs = env.reset()
            i += horizon
    finally:
        env.close()

    _CACHE[key] = windows
    return windows


def reset_boundaries(poses, max_step_m: float = 0.1) -> set[int]:
    """
    Indices j where pose[j] -> pose[j+1] is not a physical step.

    `collect.run_episode` calls `env.reset()` when the robot leaves the arena,
    which teleports it. Any window spanning that index is not a dynamics
    sample and must be excluded, exactly as `collect_windows` excludes
    wall-contact windows.
    """
    import numpy as np

    jumps = np.linalg.norm(np.diff(np.asarray(poses)[:, :2], axis=0), axis=1)
    return set(int(j) for j in np.flatnonzero(jumps > max_step_m))


# ── Dependency-free runner (also used by run_all.py) ─────────────────────────

def run_module_tests(namespace: dict, label: str = "") -> tuple[int, int, int, int]:
    """
    Run every `test_*` callable in `namespace`, in definition order.

    Returns (passed, failed, skipped, xfailed). Honours `expected_failure`:
    a marked test that fails counts as xfail, and one that PASSES counts as a
    failure, because a fixed bug should force the marker to be removed.
    """
    tests = [(n, f) for n, f in namespace.items()
             if n.startswith("test_") and callable(f)]
    passed = failed = skipped = xfailed = 0

    if label:
        print(f"\n{label}")
    for name, fn in tests:
        reason = getattr(fn, "_xfail_reason", None)
        t0 = time.perf_counter()
        try:
            fn()
        except SKIP_EXC as exc:
            skipped += 1
            print(f"  SKIP {name}  ({exc})")
            continue
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            dt = time.perf_counter() - t0
            if reason:
                xfailed += 1
                print(f"  XFAIL {name}  [{dt*1000:.0f} ms]  known bug: {reason}")
            else:
                failed += 1
                print(f"  FAIL {name}  [{dt*1000:.0f} ms]")
                print("    " + traceback.format_exc().replace("\n", "\n    ").rstrip())
            continue
        dt = time.perf_counter() - t0
        if reason:
            failed += 1
            print(f"  FAIL {name}  [{dt*1000:.0f} ms]  XPASS: the bug this test "
                  f"documents appears to be FIXED - remove the expected_failure "
                  f"marker.\n    (reason was: {reason})")
        else:
            passed += 1
            print(f"  ok   {name}  [{dt*1000:.0f} ms]")
    return passed, failed, skipped, xfailed


def main(namespace: dict) -> int:
    """`if __name__ == '__main__': raise SystemExit(main(globals()))` in each module."""
    label = namespace.get("__file__", "tests")
    passed, failed, skipped, xfailed = run_module_tests(namespace, Path(str(label)).name)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped, {xfailed} xfailed")
    return 1 if failed else 0
