"""
Dependency-free test runner.

pytest is the primary entry point:

    python -m pytest software/tests -q

This file exists so the suite is still useful on a machine without pytest -
most relevantly the Raspberry Pi, where these same invariants will need
checking against real logged episodes:

    python software/tests/run_all.py
    python -m software.tests.run_all          # from Project/arkham-batmobile
    python software/tests/run_all.py test_kinematics test_schema

It understands the two things the tests need from a runner: `skip` (a missing
dependency must say so, not pass quietly) and `expected_failure` (a test that
documents a live bug counts as expected, and FAILS if it starts passing).
"""

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import TESTS_DIR, run_module_tests  # noqa: E402


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    wanted = {a.removesuffix(".py") for a in argv}
    paths = [p for p in sorted(TESTS_DIR.glob("test_*.py"))
             if not wanted or p.stem in wanted]
    if not paths:
        print(f"no test modules matched {sorted(wanted)}")
        return 1

    totals = [0, 0, 0, 0]
    t0 = time.perf_counter()
    for path in paths:
        module = _load(path)
        counts = run_module_tests(vars(module), path.name)
        totals = [t + c for t, c in zip(totals, counts)]

    passed, failed, skipped, xfailed = totals
    print(f"\n{'=' * 64}")
    print(f"{passed} passed, {failed} failed, {skipped} skipped, {xfailed} xfailed "
          f"in {time.perf_counter() - t0:.1f}s")
    if failed:
        print("FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
