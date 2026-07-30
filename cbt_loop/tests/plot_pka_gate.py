"""Illustrate the PKA soft-threshold gate and how it feeds the striatal
nonlinearity (bg_nln). Reproduces cbt_loop/pka_gate.png.

The function forms are copied faithfully from cbt_loop/cbt_rnn.py (the gate at
lines ~597-600) and self_timed_movement_task.py (bg_nln). Config values are the
canonical cbt_loop values from config_script.py (pka_gate_min/max/slope,
pka_thresh init, tau_pka_fall/rise).

Run:  python cbt_loop/plot_pka_gate.py
"""

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "plots" / "pka_gate.png"

# --- canonical cbt_loop config values (config_script.py) ---
GMIN, GMAX, SLOPE = 0.05, 0.95, 1.0   # pka_gate_min / _max / _slope
TAU_FALL, TAU_RISE = 1440.0, 10.0     # tau_pka_fall / tau_pka_rise


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def gate(pka, thresh):
    """Soft-threshold readout of the PKA integrator (cbt_rnn.py:597)."""
    return GMIN + (GMAX - GMIN) * sig(SLOPE * (pka - thresh))


def bg_nln(x, b):
    """Striatal I/O curve with PKA-shifted rheobase (self_timed_movement_task.py)."""
    c = 3.0 / (1.0 - b)
    d = (1.0 / 6.0) * ((1.0 - b) / b)
    return sig(c * (x - d))


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: the soft-threshold gate g(pka), thresh = 0.5 vs 4.0 (default init)
    pka = np.linspace(0, 12, 500)
    axA = ax[0, 0]
    axA.plot(pka, gate(pka, 0.5), lw=2.5, color="#2a6fdb", label="pka_thresh = 0.5")
    axA.plot(pka, gate(pka, 4.0), lw=2.5, color="#d1495b",
             label="pka_thresh = 4.0 (default init)")
    for t, c in [(0.5, "#2a6fdb"), (4.0, "#d1495b")]:
        axA.axvline(t, ls=":", color=c, alpha=.6)
    axA.axhline(GMIN, ls="--", color="grey", alpha=.7)
    axA.axhline(GMAX, ls="--", color="grey", alpha=.7)
    axA.text(11.6, GMIN + .01, "gate_min 0.05", ha="right", va="bottom",
             fontsize=8, color="grey")
    axA.text(11.6, GMAX - .01, "gate_max 0.95", ha="right", va="top",
             fontsize=8, color="grey")
    axA.set_title("A. The gate:  pka_gate = 0.05 + 0.90·σ(1·(pka − thresh))",
                  fontsize=11)
    axA.set_xlabel("PKA integrator state  (pka_d1)")
    axA.set_ylabel("gate value  b  → into bg_nln")
    axA.set_ylim(-0.02, 1.0)
    axA.legend(fontsize=9)
    axA.grid(alpha=.25)

    # Panel B: bg_nln striatal I/O for several gate values b (incl. b=0.5)
    axB = ax[0, 1]
    xin = np.linspace(-0.5, 3, 500)
    for b, c in [(0.05, "#7a7a7a"), (0.3, "#e8a33d"), (0.5, "#2a6fdb"),
                 (0.7, "#4c9a5a"), (0.95, "#d1495b")]:
        axB.plot(xin, bg_nln(xin, b), lw=2, color=c, label=f"b = {b}")
    axB.set_title("B. What the gate does downstream: bg_nln(x, b)\n"
                  "higher b → lower half-max d, steeper c → more excitable",
                  fontsize=11)
    axB.set_xlabel("striatal input current  x_d1  (pre-nonlinearity)")
    axB.set_ylabel("D1 firing rate  bg_nln(x, b)")
    axB.legend(fontsize=9, title="gate value b")
    axB.grid(alpha=.25)

    # Panel C: c(b) and d(b) — the rheobase shift, showing gate is bounded away from 0/1
    axC = ax[1, 0]
    b = np.linspace(0.02, 0.98, 500)
    c_ = 3.0 / (1.0 - b)
    d_ = (1.0 / 6.0) * ((1.0 - b) / b)
    axC.plot(b, d_, lw=2.5, color="#2a6fdb", label="d(b): half-max (rheobase)")
    axC2 = axC.twinx()
    axC2.plot(b, c_, lw=2.5, color="#d1495b", label="c(b): slope/gain")
    axC.axvspan(GMIN, GMAX, color="green", alpha=.07)
    axC.axvline(GMIN, ls="--", color="grey")
    axC.axvline(GMAX, ls="--", color="grey")
    axC.text(0.5, d_.max() * 0.9,
             "gate is clamped to\n[0.05, 0.95]  (green band)\n→ c, d stay finite",
             ha="center", fontsize=9, color="darkgreen")
    axC.set_title("C. Why the gate output is bounded (0.05–0.95):\n"
                  "b→1 blows up slope c; b→0 blows up half-max d", fontsize=11)
    axC.set_xlabel("gate value  b")
    axC.set_ylabel("d  (half-max)", color="#2a6fdb")
    axC2.set_ylabel("c  (slope)", color="#d1495b")
    axC.set_ylim(0, 6)
    axC2.set_ylim(0, 60)
    axC.grid(alpha=.25)

    # Panel D: interval-timer illustration — integrator ramp crosses thresh -> gate opens
    axD = ax[1, 1]
    T = 700
    drive = np.zeros(T)
    drive[50:] = 0.02          # constant post-cue production (illustrative)
    pk = np.zeros(T)
    pk[0] = 0.3
    for t in range(1, T):
        pk[t] = (1 - 1 / TAU_FALL) * pk[t - 1] + (1 / TAU_RISE) * drive[t]
    axD.plot(pk, lw=2.5, color="black", label="PKA integrator (ramps ~0.3→…)")
    for t, c in [(0.5, "#2a6fdb"), (4.0, "#d1495b")]:
        axD.axhline(t, ls=":", color=c, alpha=.7)
        cross = np.argmax(pk > t) if np.any(pk > t) else None
        if cross:
            axD.axvline(cross, ls="--", color=c, alpha=.5)
    axDt = axD.twinx()
    axDt.plot(gate(pk, 0.5), color="#2a6fdb", lw=2, alpha=.8, label="gate (thresh 0.5)")
    axDt.plot(gate(pk, 4.0), color="#d1495b", lw=2, alpha=.8, label="gate (thresh 4.0)")
    axDt.set_ylabel("gate value b")
    axDt.set_ylim(0, 1)
    axD.set_title("D. Interval timer: threshold = WHEN the gate opens\n"
                  "(lower thresh → earlier opening → shorter learned interval)",
                  fontsize=11)
    axD.set_xlabel("time step")
    axD.set_ylabel("PKA integrator")
    h1, l1 = axD.get_legend_handles_labels()
    h2, l2 = axDt.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    axD.grid(alpha=.25)

    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
