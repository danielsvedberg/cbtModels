"""TEST: init_state_fix -- remove the saturated START by lowering the PKA init 0.5 -> 0.3.

WHY
---
With the loop de-saturated (tests/loop_desaturation), the remaining striatal saturation is
purely the PKA operating point. bg_nln is steep in pka: D1/D2 ~= 0.5 at pka~0.30, but ~0.95
at pka=0.5 (the saturating regime). The noSCnoSTN init started pka at 0.5, and because pka
RISES fast but FALLS slowly (tau_pka_fall >> tau_pka_rise), starting above the production
level makes pka LINGER high -> a saturated D1/D2 spike over the first few hundred steps.

FIX (config_script noSCnoSTN extra_init): start pka at 0.3 -- below the production level, in
the in-band bg_nln regime -- so pka rises INTO band with no saturated transient. The D1/D2
mutual exclusivity at steady state is accepted opponency (direct/indirect go/no-go); a low-m_a2
config then settles into the D1 branch. This test overlays pka_0=0.5 (before) vs 0.3 (after)
for the search's best gain config on the hybrid trial.

FINDING
-------
pka_0=0.5 -> D2 spikes to ~0.83 mid-trial (saturated transient); pka_0=0.3 removes it (D1/D2
start ~0.36, in band) and lifts the fraction of the trial in [0.15,0.85] (D2 0.47 -> 0.58).
Values <=0.35 are identical (fast pka rise overrides the exact init), so 0.3 is a safe choice.
"""
import sys
import pathlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FAMILY_DIR = HERE.parents[1]
_root = next(p for p in HERE.parents if (p / "config_script.py").exists())
for _p in (str(FAMILY_DIR), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cbt_rnn as cbtl
import train_hybrid

A = list(cbtl.STATE_AREA_ORDER)
BEST = dict(m_d1=0.705, m_d2=0.980, m_a1=0.226, m_a2=0.020, g_da_release=0.020, g_ado_release=0.355)


def logit(e):
    e = np.clip(e, 1e-3, 1 - 1e-3)
    return float(np.log(e / (1 - e)))


def run(pka0, inp):
    B, T, _ = inp.shape
    p, c = cbtl.init_params(jr.PRNGKey(0), n_input=inp.shape[-1])
    p = dict(p)
    for k, e in BEST.items():
        p[k] = jnp.array(logit(e))
    p["pka_d10"] = jnp.full_like(jnp.asarray(p["pka_d10"]), pka0)
    p["pka_d20"] = jnp.full_like(jnp.asarray(p["pka_d20"]), pka0)
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    _, xs = cbtl.batched_rnn(p, c, inp, jnp.zeros((B, T, nd1 + nd2)), jr.split(jr.PRNGKey(0), B))
    tc = {}
    for nm in ("D1", "D2", "pkaD1", "pkaD2"):
        x = np.asarray(xs[A.index(nm)]); x = x.mean(-1) if x.ndim > 2 else x
        tc[nm] = x.mean(0)
    return tc


def main():
    inp = train_hybrid._build_hybrid_batch()[0][:16]
    before = run(0.5, inp)   # old arbitrary value
    after = run(0.3, inp)    # new config default
    inb = lambda x: float(((x > 0.15) & (x < 0.85)).mean())

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for a, keys, title in [(ax[0], ("D1", "D2"), "striatal rate"),
                           (ax[1], ("pkaD1", "pkaD2"), "PKA (bg_nln excitability b)")]:
        for nm in keys:
            ln, = a.plot(after[nm], lw=1.9, label=f"{nm}  pka0=0.3")
            a.plot(before[nm], lw=1.3, ls="--", color=ln.get_color(), alpha=0.6,
                   label=f"{nm}  pka0=0.5")
        a.axhspan(0.15, 0.85, color="green", alpha=0.07)
        a.set_title(title); a.set_xlabel("t"); a.set_ylim(-0.02, 1.02)
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("init_state_fix: PKA start 0.5 (dashed) vs 0.3 (solid) -- removes the saturated "
                 "start spike", y=1.0)
    fig.tight_layout()
    out = HERE / "init_state_fix.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"frac in [0.15,0.85]:  D1 {inb(before['D1']):.2f}->{inb(after['D1']):.2f}   "
          f"D2 {inb(before['D2']):.2f}->{inb(after['D2']):.2f}")
    print(f"D2 mid-trial peak:    {before['D2'].max():.2f} (pka0=0.5) -> {after['D2'].max():.2f} (pka0=0.3)")
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
