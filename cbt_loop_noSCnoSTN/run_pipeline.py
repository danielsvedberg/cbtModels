"""Run the full pipeline: Pavlovian pretraining, self-timed fine-tuning, then testing."""

import time

import train_pavlovian
import train_from_pavlovian
import testing_script
import test_pavlovian
import training_script
import train_hybrid

def _run(name, fn):
    print(f"\n{'=' * 60}\n[pipeline] starting: {name}\n{'=' * 60}")
    t0 = time.time()
    fn()
    print(f"[pipeline] finished {name} in {time.time() - t0:.1f}s")


def main():
    _run("train_pavlovian", train_pavlovian.main)
    _run('test_pavlovian', test_pavlovian.main)
    _run("train_from_pavlovian", train_from_pavlovian.main)
    #_run("train_hybrid", train_hybrid.main)
    _run("training_script", training_script.main)
    _run("testing_script", testing_script.main)


if __name__ == "__main__":
    main()
 