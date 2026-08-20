"""Clean isolated sweep of the PKA gains m_d1/m_d2/m_a1/m_a2.

For each gain: sweep its value while the other three are held at 1.0. Every gain is
HOMOGENEOUS (same value for all neurons in its population -- no per-neuron exponential,
no _wi scaling: the swept number IS the gain). pka_d1/pka_d2 initialised at 0.5. Runs to
the pka steady state and records DA/adenosine/pkaD1/pkaD2/D1/D2 + stability.
"""
import sys, pathlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cbt_rnn as cbtl
_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
A = list(cbtl.STATE_AREA_ORDER)
T = 4000
WIN = 500
GRID = [0.0, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]


def measure(p, c):
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    ys, xs = cbtl.batched_rnn(p, c, jnp.zeros((1, T, 1)), jnp.zeros((1, T, nd1 + nd2)),
                              jr.split(jr.PRNGKey(0), 1))
    def m(name):
        x = np.asarray(xs[A.index(name)])[0]; x = x.mean(-1) if x.ndim > 1 else x
        return float(x[-WIN:].mean()), float(x[-WIN:].std())
    r = {}
    for a in ("DA", "Adenosine", "pkaD1", "pkaD2", "D1", "D2"):
        r[a], r[a + "_std"] = m(a)
    return r


def main():
    p0, c = cbtl.init_params(jr.PRNGKey(0), n_input=1)
    c = dict(c)
    c["snc_pacer_min"] = 0.1; c["snc_pacer_max"] = 0.1   # PIN the SNc pacemaker to 0.1
    p0 = {k: np.asarray(v) for k, v in p0.items()}
    nd1 = p0["J_d1"].shape[0]; nd2 = p0["J_d2"].shape[0]
    gains = ["m_d1", "m_d2", "m_a1", "m_a2"]
    sizes = {"m_d1": nd1, "m_d2": nd2, "m_a1": nd1, "m_a2": nd2}
    print(f"da_pka_gain={c['da_pka_gain']}, pka_max={c['pka_max']}, "
          f"pka_init=[{c['pka_init_floor']},{c['pka_init_cap']}]; all held gains=1.0, pka0=0.5")

    results = {}
    for g in gains:
        results[g] = []
        print(f"\n=== sweep {g} (others=1.0, homogeneous) ===")
        print(" value | pkaD1 pkaD2 |  D1    D2   | x_da x_ado | std(pkaD1,D2)")
        for v in GRID:
            p = dict(p0)
            for gg in gains:
                p[gg] = np.ones(sizes[gg]) * (v if gg == g else 1.0)
            p["pka_d10"] = np.ones(nd1) * 0.5
            p["pka_d20"] = np.ones(nd2) * 0.5
            r = measure({k: jnp.asarray(val) for k, val in p.items()}, c)
            results[g].append((v, r))
            print(f" {v:5.2f} | {r['pkaD1']:.3f} {r['pkaD2']:.3f} | {r['D1']:.3f} {r['D2']:.3f} "
                  f"| {r['DA']:.3f} {r['Adenosine']:.3f} | {r['pkaD1_std']:.4f},{r['pkaD2_std']:.4f}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for i, g in enumerate(gains):
        a = ax[i // 2][i % 2]
        vs = [v for v, _ in results[g]]
        a.plot(vs, [r["pkaD1"] for _, r in results[g]], "o-", color="C3", label="pkaD1")
        a.plot(vs, [r["pkaD2"] for _, r in results[g]], "o-", color="C0", label="pkaD2")
        a.plot(vs, [r["D1"] for _, r in results[g]], "s--", color="C3", alpha=0.5, label="D1 rate")
        a.plot(vs, [r["D2"] for _, r in results[g]], "s--", color="C0", alpha=0.5, label="D2 rate")
        a.axhline(0.5, color="grey", ls=":", lw=1); a.axvline(1.0, color="k", ls=":", lw=0.6)
        a.set_xlabel(f"{g} value (others = 1.0)"); a.set_ylabel("steady value")
        a.set_ylim(-0.02, 1.02); a.set_title(f"sweep {g}"); a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=8)
    fig.suptitle("noSCnoSTN PKA gains: homogeneous, others=1.0, pka0=0.5, SNc pacemaker PINNED=0.1", y=1.0)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / "pka_gains_sweep_sncpin.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
