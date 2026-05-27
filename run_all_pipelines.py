"""Run run_pipeline.py for every CBT loop variant, simplest first."""

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Ordered from least to most complex: minimal → +STN → +STN+SC.
VARIANTS = [
    "cbt_loop_noSCnoSTN",
    "cbt_loop_noSC",
    "cbt_loop",
]


def _run(variant):
    cwd = ROOT / variant
    print(f"\n{'#' * 70}\n# pipeline: {variant}\n{'#' * 70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "run_pipeline.py"],
        cwd=cwd,
        check=False,
    )
    print(f"# {variant} exited with code {result.returncode} in {time.time() - t0:.1f}s")
    return result.returncode


def main():
    failures = []
    for v in VARIANTS:
        code = _run(v)
        if code != 0:
            failures.append((v, code))
    if failures:
        print("\nFAILURES:")
        for v, code in failures:
            print(f"  {v}: exit {code}")
        sys.exit(1)
    print("\nAll variants completed successfully.")


if __name__ == "__main__":
    main()
