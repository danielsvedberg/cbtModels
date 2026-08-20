"""Sweep the four _wi PKA-gain scales (m_d1, m_d2, m_a1, m_a2) and map their effect on
the DA / adenosine / PKA steady state and its stability.

Each point runs the model to the pka steady state (long run -- tau_pka_fall=900, so a
1000-step trial never equilibrates) at rest, and records:
  x_da, x_ado (concentrations), pkaD1, pkaD2 (excitability), D1, D2 (rates),
  and a stability flag = std of pkaD1/pkaD2 over the last window (≈0 => converged).
One panel per gain; others held at their _wi default.
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
T = 6000          # >> tau_pka_fall=900 so pka reaches steady state
WIN = 500         # window for steady value + stability std


def measure(p, c):
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    ys, xs = cbtl.batched_rnn(p, c, jnp.zeros((1, T, 1)), jnp.zeros((1, T, nd1 + nd2)),
                              jr.split(jr.PRNGKey(0), 1))
    def tr(name):
        return np.asarray(xs[A.index(name)])[0]  # (T, n) or (T,)
    def m(name):
        x = tr(name); x = x.mean(-1) if x.ndim > 1 else x
        return float(x[-WIN:].mean()), float(x[-WIN:].std())
    out = {}
    for a in ("DA", "Adenosine", "pkaD1", "pkaD2", "D1", "D2"):
        out[a], out[a + "_std"] = m(a)
    return out


def main():
    p0, c = cbtl.init_params(jr.PRNGKey(0), n_input=1)
    p0 = {k: np.asarray(v) for k, v in p0.items()}
    factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    gains = ["m_d1", "m_d2", "m_a1", "m_a2"]

    results = {g: [] for g in gains}
    for g in gains:
        print(f"\n=== sweep {g} (x default) ===")
        print(" factor | pkaD1 pkaD2 |  D1    D2   | x_da x_ado | pka_std(D1,D2)")
        for f in factors:
            p = dict(p0); p[g] = p0[g] * f
            p = {k: jnp.asarray(v) for k, v in p.items()}
            r = measure(p, c)
            results[g].append((f, r))
            print(f"  {f:4.2f}  | {r['pkaD1']:.3f} {r['pkaD2']:.3f} | "
                  f"{r['D1']:.3f} {r['D2']:.3f} | {r['DA']:.3f} {r['Adenosine']:.3f} | "
                  f"{r['pkaD1_std']:.4f},{r['pkaD2_std']:.4f}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for i, g in enumerate(gains):
        a = ax[i // 2][i % 2]
        fs = [f for f, _ in results[g]]
        a.plot(fs, [r["pkaD1"] for _, r in results[g]], "o-", color="C3", label="pkaD1")
        a.plot(fs, [r["pkaD2"] for _, r in results[g]], "o-", color="C0", label="pkaD2")
        a.plot(fs, [r["D1"] for _, r in results[g]], "s--", color="C3", alpha=0.5, label="D1 rate")
        a.plot(fs, [r["D2"] for _, r in results[g]], "s--", color="C0", alpha=0.5, label="D2 rate")
        a.axhline(0.5, color="grey", ls=":", lw=1)
        a.set_xscale("log"); a.set_xlabel(f"{g} scale (x default)"); a.set_ylabel("steady value")
        a.set_ylim(-0.02, 1.02); a.set_title(f"Effect of {g}"); a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=8)
    fig.suptitle("noSCnoSTN: _wi PKA-gain scales vs DA/adenosine/PKA steady state (rest)", y=1.0)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / "wi_pka_sweep.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
