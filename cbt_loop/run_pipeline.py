"""Run the full pipeline: Pavlovian pretraining, self-timed fine-tuning, then testing."""

import time

import cbt_loop.train_pavlovian as tp
import cbt_loop.train_from_pavlovian as tfp
import cbt_loop.train_hybrid as th
import cbt_loop.test_hybrid as test_hybrid
import cbt_loop.train_from_hybrid as train_from_hybrid
import cbt_loop.testing_script as testing_script


def _run(name, fn):
    print(f"\n{'=' * 60}\n[pipeline] starting: {name}\n{'=' * 60}")
    t0 = time.time()
    fn()
    print(f"[pipeline] finished {name} in {time.time() - t0:.1f}s")


def main():
    _run("train_pavlovian", tp.main)
    _run("train_hybrid", th.main)
    _run("test_hybrid", test_hybrid.main)
    _run("train_from_hybrid", train_from_hybrid.main)
    _run("testing_script", testing_script.main)


if __name__ == "__main__":
    main()
