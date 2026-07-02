"""Continue iterating: one more training run from params_shaped.pkl, then plot.

Convenience wrapper that runs train_from_pkl (continues training the existing
params_shaped.pkl bundle for TRAINING_CONFIG["num_iters"] on the final
self-timed task, saving back to params_shaped.pkl) and then testing_script
(regenerates all evaluation plots in plots/). Run it repeatedly to keep
iterating on the same model.

    python -u train_and_test.py
"""

import time

import train_from_pkl
import testing_script


def _run(name, fn):
    print(f"\n{'=' * 60}\n[train_and_test] starting: {name}\n{'=' * 60}")
    t0 = time.time()
    fn()
    print(f"[train_and_test] finished {name} in {time.time() - t0:.1f}s")


def main():
    _run("train_from_pkl", train_from_pkl.main)
    _run("testing_script", testing_script.main)


if __name__ == "__main__":
    main()
