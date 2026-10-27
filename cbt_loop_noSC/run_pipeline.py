"""Run the full pipeline: Pavlovian pretraining, hybrid shaping, self-timed
fine-tuning, then testing (with an evaluation/plot step after each training stage)."""
# training/ and testing/ hold the scripts imported below (moved there 2026-10-27).
import sys as _sys, pathlib as _pl
_FAMILY_ROOT = _pl.Path(__file__).resolve().parent
for _d in (_FAMILY_ROOT / "testing", _FAMILY_ROOT / "training", _FAMILY_ROOT):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import time

import train_pavlovian
import test_pavlovian
import train_hybrid
import test_hybrid
import train_from_hybrid
import train_from_pavlovian
import testing_script


def _run(name, fn):
    print(f"\n{'=' * 60}\n[pipeline] starting: {name}\n{'=' * 60}")
    t0 = time.time()
    fn()
    print(f"[pipeline] finished {name} in {time.time() - t0:.1f}s")


def main():
    _run("train_pavlovian", train_pavlovian.main)
    _run("test_pavlovian", test_pavlovian.main)
    _run("train_hybrid", train_hybrid.main)  # temporarily disabled — already trained (params_hybrid.pkl)
    _run("test_hybrid", test_hybrid.main)
    _run("train_from_hybrid", train_from_hybrid.main)
    _run("testing_script", testing_script.main)


if __name__ == "__main__":
    main()
