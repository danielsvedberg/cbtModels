"""Demonstrate WHY the old PKA setup could not integrate (the bug fixed in
commit 5daa4c4).

The PKA trace is meant to be a leaky integrator whose slow leak (tau_pka_fall =
1440 → ~998-step half-life) makes it the model's interval-timing substrate. The
legacy code applied nln() = sigmoid(4(x-0.5)) to the CARRIED STATE every step:

    pka = (1 - 1/tau_fall) * pka + (1/tau_rise) * prod
    pka = nln(pka)                          # legacy: squash the state itself

Re-squashing the state each step multiplies the effective per-step retention by
nln'(pka), so the leak becomes (1 - 1/tau_fall)*nln'(pka) instead of
(1 - 1/tau_fall). In the model's actual operating regime the integrator is
supposed to RAMP well past 1 (toward ~12), where the sigmoid nln saturates and
nln'(pka) → 0. There the effective retention collapses to ~0: the trace is pinned
at the nln ceiling (~1), carries no ramp, and any perturbation dies in a few
steps — so there is no interval timer. The fix (pka_integrator=True) keeps the
state unsquashed and squashes only where it is USED (the bg_nln gate).

Reproduces cbt_loop/pka_integration_failure.png.

Run:  python cbt_loop/demo_pka_integration_failure.py
"""

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "plots" / "pka_integration_failure.png"

TAU_FALL, TAU_RISE = 1440.0, 10.0     # tau_pka_fall / tau_pka_rise (config)
RETAIN = 1.0 - 1.0 / TAU_FALL         # intended per-step retention


def nln(x):
    return 1.0 / (1.0 + np.exp(-4.0 * (x - 0.5)))            # self_timed_movement_task.nln


def nln_prime(x):
    s = nln(x)
    return 4.0 * s * (1.0 - s)                               # d/dx sigmoid(4(x-0.5))


def simulate(squash_state, T, prod, pulse=None):
    """Leaky-integrate production; squash_state=True reproduces the legacy bug.

    If pulse is None, production is applied continuously from t=50 (a ramp, the
    real operating regime). If pulse=(a,b), production is applied only on [a,b).
    """
    pka = np.zeros(T)
    pka[0] = 0.3                                             # resting init (pka_d10)
    for t in range(1, T):
        if pulse is None:
            drive = prod if t >= 50 else 0.0
        else:
            drive = prod if pulse[0] <= t < pulse[1] else 0.0
        p = RETAIN * pka[t - 1] + (1.0 / TAU_RISE) * drive
        pka[t] = nln(p) if squash_state else p
    return pka


def half_life_from_peak(trace):
    t_peak = int(np.argmax(trace))
    peak, floor = trace[t_peak], trace[-1]
    half = 0.5 * (peak - floor) + floor
    below = np.where(trace[t_peak:] <= half)[0]
    return (int(below[0]) if len(below) else None), t_peak


def main():
    # --- operating regime: continuous drive that ramps the true integrator to ~12 ---
    T = 900
    prod = 0.2
    fixed = simulate(squash_state=False, T=T, prod=prod)
    legacy = simulate(squash_state=True,  T=T, prod=prod)

    # --- persistence: charge briefly into saturation, then remove drive ---
    Tp = 600
    fixed_p = simulate(squash_state=False, T=Tp, prod=1.0, pulse=(50, 120))
    legacy_p = simulate(squash_state=True,  T=Tp, prod=1.0, pulse=(50, 120))
    hl_fixed, _ = half_life_from_peak(fixed_p)
    hl_legacy, _ = half_life_from_peak(legacy_p)

    print(f"RAMP   fixed:  {fixed.min():.2f} -> {fixed.max():.2f}  (monotone timer)")
    print(f"RAMP   legacy: {legacy.min():.2f} -> {legacy.max():.2f}  "
          f"(pinned at nln ceiling, flat -> no timing signal)")
    print(f"DECAY  fixed half-life  ~{hl_fixed} steps "
          f"(intended ln(2)*tau_fall = {int(np.log(2)*TAU_FALL)})")
    print(f"DECAY  legacy half-life ~{hl_legacy} steps")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: the ramp — fixed integrates into a clean timer; legacy saturates flat
    ax0 = ax[0]
    ax0.plot(fixed, lw=2.5, color="#2a6fdb", label="FIXED (pka_integrator=True)")
    ax0.plot(legacy, lw=2.5, color="#d1495b", label="LEGACY (nln on state)")
    ax0.axhline(1.0, ls="--", color="grey", alpha=.6)
    ax0.text(T*0.98, 1.02, "nln ceiling ≈ 1", ha="right", va="bottom",
             fontsize=8, color="grey")
    ax0.set_title("1. Operating regime (continuous post-cue drive):\n"
                  "fixed RAMPS to a timer; legacy pins at the ceiling", fontsize=11)
    ax0.set_xlabel("time step"); ax0.set_ylabel("PKA state")
    ax0.legend(fontsize=9); ax0.grid(alpha=.25)

    # Panel 2: WHY — effective per-step retention (1-1/tau)*nln'(x) vs the state
    ax1 = ax[1]
    xs = np.linspace(0, 4, 500)
    eff = RETAIN * nln_prime(xs)
    ax1.plot(xs, eff, lw=2.5, color="#d1495b", label="legacy: (1−1/τ)·nln'(x)")
    ax1.axhline(RETAIN, ls="--", color="#2a6fdb", lw=2,
                label="fixed: (1−1/τ) = 0.9993")
    # implied half-life annotation at a saturated operating point
    x_op = 2.0
    r = RETAIN * nln_prime(x_op)
    hl = np.log(0.5) / np.log(r) if 0 < r < 1 else np.inf
    ax1.plot([x_op], [r], "o", color="black")
    ax1.annotate(f"at x={x_op}: retention {r:.3f}\n→ half-life ≈ {hl:.1f} steps",
                 (x_op, r), xytext=(x_op+0.3, r+0.35), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax1.set_title("2. Why: re-squashing multiplies retention by nln'(x),\n"
                  "which collapses to 0 once PKA saturates", fontsize=11)
    ax1.set_xlabel("PKA state  x"); ax1.set_ylabel("effective per-step retention")
    ax1.set_ylim(0, 1.05); ax1.legend(fontsize=9); ax1.grid(alpha=.25)

    # Panel 3: persistence after a brief charge into saturation
    ax2 = ax[2]
    ax2.plot(fixed_p, lw=2.5, color="#2a6fdb",
             label=f"FIXED — half-life ~{hl_fixed} steps")
    ax2.plot(legacy_p, lw=2.5, color="#d1495b",
             label=f"LEGACY — half-life ~{hl_legacy} steps")
    ax2.axvspan(50, 120, color="grey", alpha=.15, label="charge pulse")
    ax2.set_title("3. Persistence after a brief charge:\n"
                  "fixed holds across the delay; legacy does not", fontsize=11)
    ax2.set_xlabel("time step"); ax2.set_ylabel("PKA state")
    ax2.legend(fontsize=9); ax2.grid(alpha=.25)

    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
