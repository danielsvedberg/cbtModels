"""TEST: train_hybrid_desat -- train_hybrid learns from scratch on the DE-SATURATED loop.

WHY
---
Before the wrapper-aware logit-init (tests/loop_desaturation), exc=sigmoid collapsed every
weight to ~0.5 and railed the cortico-thalamic loop at ~0.99; a saturated cortex zeros the
task gradient, so train_hybrid was stuck near the old ~0.07 ceiling. This plots a fresh
train_hybrid run (--init scratch, objective_mode=log_reward, 10k iters) on the de-saturated
model, parsed from train_desat_full.log.

FINDING
-------
It LEARNS. reward/success climbs 0 -> 0.61 (and still rising steeply at 10k: 0.45->0.54->0.61
over the last 400 iters), after a long ~0.05 plateau (steps 4.8k-8k) then a sharp breakthrough
once the timing is found. silence/tail collapse to ~0 early (output leaves the dead floor);
norm_resp_time rises as real responses form. Overturns the old hybrid ceiling: de-saturating
the loop (not more penalty tuning) was the missing piece. Diagnostic of the trained model:
D1 alive 0.95 in the response window, SNr low (gate open), D2 suppressed -- clean D1-branch
opponency. Still climbing at 10k, so more iters should push higher.
"""
import re
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FAMILY_DIR = HERE.parents[1]
LOG = FAMILY_DIR / "train_desat_full.log"

PAT = re.compile(r"step (\d+), loss: ([\d.]+), reward: ([\d.]+), log_reward: (-?[\d.]+), "
                 r"success: ([\d.]+), entropy: ([\d.]+), .*norm_resp_time: ([\d.]+), "
                 r"silence: ([\d.]+), tail: ([\d.]+)")


def main():
    rows = []
    for line in LOG.read_text().splitlines():
        m = PAT.search(line)
        if m:
            rows.append([float(x) for x in m.groups()])
    a = np.array(rows)
    step, loss, reward, logr, succ, ent, nrt, sil, tail = a.T

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(step, reward, lw=2, color="green")
    ax[0].axhline(0.07, color="gray", ls=":", label="old ~0.07 ceiling")
    ax[0].set_title("reward / success"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("reward")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(step, loss, lw=1.6, color="crimson")
    ax[1].set_yscale("log"); ax[1].set_title("loss (log scale)")
    ax[1].set_xlabel("iter"); ax[1].grid(alpha=0.3)

    ax[2].plot(step, sil, lw=1.4, label="silence")
    ax[2].plot(step, tail, lw=1.4, label="tail")
    ax[2].plot(step, nrt, lw=1.4, label="norm_resp_time")
    ax[2].set_title("output activation"); ax[2].set_xlabel("iter")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle(f"train_hybrid on the de-saturated loop: reward 0 -> {reward[-1]:.2f} "
                 f"(scratch, log_reward, {int(step[-1])} iters)", y=1.02)
    fig.tight_layout()
    out = HERE / "train_hybrid_desat.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"final reward={reward[-1]:.3f} success={succ[-1]:.3f} at iter {int(step[-1])}")
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
