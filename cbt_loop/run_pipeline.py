"""Run the full pipeline: Pavlovian pretraining, self-timed fine-tuning, then testing."""

import time

import train_pavlovian
import train_from_pavlovian
import train_hybrid
import test_hybrid
import train_from_hybrid
import testing_script


def _run(name, fn):
    print(f"\n{'=' * 60}\n[pipeline] starting: {name}\n{'=' * 60}")
    t0 = time.time()
    fn()
    print(f"[pipeline] finished {name} in {time.time() - t0:.1f}s")


def main():
    _run("train_pavlovian", train_pavlovian.main)
    _run("train_hybrid", train_hybrid.main)
    _run("test_hybrid", test_hybrid.main)
    _run("train_from_hybrid", train_from_hybrid.main)
    _run("testing_script", testing_script.main)


if __name__ == "__main__":
    main()
