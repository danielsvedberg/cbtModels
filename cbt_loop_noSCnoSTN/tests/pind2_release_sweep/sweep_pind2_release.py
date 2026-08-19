"""TEST: pka_d2 pinned to 0.5 -> sweep m_d1 / g_da_release / g_ado_release (x default),
12 seeds, measure DA / adenosine / pkaD1 / D1 steady state.

WHAT & WHY
----------
With the adenosine tweak (`ado_release = mean_spn`), pinning SNc low sent the network
into a D2-dominant lock (D1 dead). This test pins D2's EXCITABILITY at pka_d2=0.5 (so D2
isn't a runaway PKA state) and asks: can any of the three DA/adenosine RELEASE knobs
rescue the D1 pathway? Each param is swept as a multiplier of its default; 12 random-seed
inits give mean +/- std bands (robustness of the relationship).

FINDING
-------
- pka_d2=0.5 does NOT save D1: D2's *firing rate* still saturates (input-driven), so
  mean_spn ~ 0.5 -> x_ado ~ 0.33, and the adenosine brake buries D1 (pkaD1 ~ 0.02).
- m_d1 x0.25..x4: NO effect (pkaD1 flat at 0.02) -- more D1 DA gain can't beat the flood.
- g_da_release x0.25..x4: almost none -- even x4 (x_da=0.26) only lifts pkaD1 to ~0.03.
- g_ado_release is the ONLY lever: at 0 (x_ado=0) pkaD1 jumps to 0.46 (D1 rate 0.88,
  ALIVE); but even 0.25x (x_ado=0.11) kills it again (pkaD1=0.02). A brittle, threshold
  relationship -- the SPN->adenosine feedback dominates D1's balance almost entirely.
- std ~ 0 across seeds except at the transition points -> the dead/alive states are
  robust attractors, seed-independent.

Conclusion: the SPN-derived adenosine flood (not the DA side) is what suppresses D1, and
it's near-binary. Rescuing D1 needs the adenosine RELEASE gain cut, not more dopamine.
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
FAMILY_DIR = HERE.parents[1]                                   # cbt_loop_noSCnoSTN/
_root = next(p for p in HERE.parents if (p / "config_script.py").exists())
for _p in (str(FAMILY_DIR), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cbt_rnn as cbtl

A = list(cbtl.STATE_AREA_ORDER)
T = 3000
WIN = 200
FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = list(range(12))
PARAMS = ["m_d1", "g_da_release", "g_ado_release"]
METRICS = ["pkaD1", "D1", "D2", "DA", "Adenosine"]


def run_one(seed, param, factor):
    p, c = cbtl.init_params(jr.PRNGKey(seed), n_input=1)
    c = dict(c); c["pin_pka_d2"] = 0.5                          # pin D2 excitability
    p = dict(p)
    if param == "m_d1":
        p["m_d1"] = jnp.asarray(p["m_d1"]) * factor
    elif param == "g_da_release":
        c["da_release"] = c["da_release"] * factor
    elif param == "g_ado_release":
        c["ado_release"] = c["ado_release"] * factor
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    ys, xs = cbtl.batched_rnn(p, c, jnp.zeros((1, T, 1)), jnp.zeros((1, T, nd1 + nd2)),
                              jr.split(jr.PRNGKey(0), 1))
    out = {}
    for a in METRICS:
        x = np.asarray(xs[A.index(a)])[0]; x = x.mean(-1) if x.ndim > 1 else x
        out[a] = float(x[-WIN:].mean())
    return out


def main():
    results = {pp: {m: np.full((len(FACTORS), len(SEEDS)), np.nan) for m in METRICS} for pp in PARAMS}
    for pp in PARAMS:
        print(f"\n=== sweep {pp} (pka_d2 pinned 0.5, {len(SEEDS)} seeds) ===")
        print(" factor | pkaD1(mean±std)   D1            x_da    x_ado")
        for fi, f in enumerate(FACTORS):
            for si, s in enumerate(SEEDS):
                r = run_one(s, pp, f)
                for m in METRICS:
                    results[pp][m][fi, si] = r[m]
            pk = results[pp]["pkaD1"][fi]; d1 = results[pp]["D1"][fi]
            da = results[pp]["DA"][fi]; ado = results[pp]["Adenosine"][fi]
            print(f"  {f:4.2f}  | {pk.mean():.3f}±{pk.std():.3f}   "
                  f"{d1.mean():.3f}±{d1.std():.3f}   {da.mean():.3f}   {ado.mean():.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    colors = {"pkaD1": "C3", "D1": "C1", "DA": "C2", "Adenosine": "C0"}
    labels = {"pkaD1": "pkaD1", "D1": "D1 rate", "DA": "x_da", "Adenosine": "x_ado"}
    for i, pp in enumerate(PARAMS):
        a = ax[i]
        for m in ["pkaD1", "D1", "DA", "Adenosine"]:
            arr = results[pp][m]; mu = arr.mean(1); sd = arr.std(1)
            a.plot(FACTORS, mu, "o-", color=colors[m], label=labels[m])
            a.fill_between(FACTORS, mu - sd, mu + sd, color=colors[m], alpha=0.18)
        a.axhline(0.5, color="grey", ls=":", lw=1); a.axvline(1.0, color="k", ls=":", lw=0.6)
        a.set_xlabel(f"{pp}  (x default)"); a.set_title(f"sweep {pp}")
        a.set_ylim(-0.02, 1.05); a.grid(alpha=0.3)
        if i == 0:
            a.set_ylabel("steady value"); a.legend(fontsize=9)
    fig.suptitle(f"pka_d2 pinned 0.5: m_d1 / DA-release / adenosine-release vs pkaD1 "
                 f"(mean±std over {len(SEEDS)} seeds)", y=1.02)
    fig.tight_layout()
    out = HERE / "pind2_release_sweep.png"          # plot saved next to this script
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
