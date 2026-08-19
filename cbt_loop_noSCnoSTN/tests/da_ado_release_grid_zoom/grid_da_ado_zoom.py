"""TEST: g_da_release x g_ado_release grids ZOOMED to [0, 0.5] (0.1 steps), BOTH pathways.

WHAT & WHY
----------
The full-range [0,2] grids (../da_ado_release_grid{,_d2}) showed a near-binary cliff: D1
alive only at g_ado~0, D2 saturated for any g_ado>=0.25. Those grids jumped 0 -> 0.25, so
the cliff location was unresolved. This zooms both grids into g_da,g_ado in [0, 0.5] at
0.1 resolution to pin down WHERE the direct/indirect switch happens.

Two grids, 10 seeds each:
  - D1: pin pka_d2=0.5, measure pkaD1 / D1 rate.
  - D2: pin pka_d1=0.5, measure pkaD2 / D2 rate.
x = g_ado_release, y = g_da_release (both == multiplier since defaults are 1.0).

FINDING
-------
The switch is a near-perfect STEP at g_ado in (0, 0.1] -- 0.1 resolution still can't
resolve a gradient; it's essentially binary at g_ado ~ 0. Perfectly complementary:
  D1: pkaD1 alive ONLY in the g_ado=0 column (0.22 -> 0.37 as g_da 0.1 -> 0.5), dead
      (0.02) at every g_ado >= 0.1. g_da grades D1 there but can't push it past ~0.4.
  D2: dead (0.04) only at g_ado=0, then jumps to 0.79-0.81 at g_ado=0.1 and grades up to
      0.94 by g_ado=0.5. g_da (DA brake) is nearly inert (0.81 -> 0.79 across g_da).
So adenosine release is an almost-ideal SWITCH between direct (D1) and indirect (D2): any
nonzero release flips the network to D2-dominant. Once past the switch, adenosine LEVEL
grades D2; dopamine only matters in the adenosine-free (g_ado=0) column. The switching
threshold is ~0 -- the SPN->adenosine feedback needs to be shut off almost entirely to
keep D1 viable.
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

A = list(cbtl.STATE_AREA_ORDER)
T = 3000
WIN = 200
G_DA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
G_ADO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = list(range(10))


def run_grid(pin_key, pka_name, rate_name):
    """Grid of (mean pka, mean rate) over G_DA x G_ADO, averaged over seeds."""
    pk = np.zeros((len(G_DA), len(G_ADO), len(SEEDS)))
    rt = np.zeros_like(pk)
    for i, gda in enumerate(G_DA):
        for j, gado in enumerate(G_ADO):
            for k, s in enumerate(SEEDS):
                p, c = cbtl.init_params(jr.PRNGKey(s), n_input=1)
                c = dict(c); c[pin_key] = 0.5
                c["da_release"] = c["da_release"] * gda
                c["ado_release"] = c["ado_release"] * gado
                nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
                _, xs = cbtl.batched_rnn(p, c, jnp.zeros((1, T, 1)),
                                         jnp.zeros((1, T, nd1 + nd2)), jr.split(jr.PRNGKey(0), 1))
                def m(name):
                    x = np.asarray(xs[A.index(name)])[0]; x = x.mean(-1) if x.ndim > 1 else x
                    return float(x[-WIN:].mean())
                pk[i, j, k] = m(pka_name); rt[i, j, k] = m(rate_name)
        print(f"  {pka_name} g_da={gda}: " + " ".join(f"{pk[i, j].mean():.2f}" for j in range(len(G_ADO))))
    return pk.mean(-1), rt.mean(-1)


def main():
    print("=== D1 grid (pin pka_d2=0.5) ===")
    pkd1, rd1 = run_grid("pin_pka_d2", "pkaD1", "D1")
    print("=== D2 grid (pin pka_d1=0.5) ===")
    pkd2, rd2 = run_grid("pin_pka_d1", "pkaD2", "D2")

    ext = [G_ADO[0], G_ADO[-1], G_DA[0], G_DA[-1]]
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    panels = [(ax[0][0], pkd1, "pkaD1  (D1 grid, pin pka_d2)", "viridis", 0.6),
              (ax[0][1], rd1, "D1 rate", "magma", 1.0),
              (ax[1][0], pkd2, "pkaD2  (D2 grid, pin pka_d1)", "viridis", 0.6),
              (ax[1][1], rd2, "D2 rate", "magma", 1.0)]
    for a, data, title, cmap, vmax in panels:
        im = a.imshow(data, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for i in range(len(G_DA)):
            for j in range(len(G_ADO)):
                a.text(G_ADO[j], G_DA[i], f"{data[i, j]:.2f}", ha="center", va="center",
                       color="w", fontsize=8)
        a.set_xlabel("g_ado_release"); a.set_ylabel("g_da_release"); a.set_title(title)
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("DA/adenosine release grid ZOOM [0,0.5]: D1 (pka_d2=0.5) vs D2 (pka_d1=0.5), "
                 f"mean over {len(SEEDS)} seeds", y=1.0)
    fig.tight_layout()
    out = HERE / "da_ado_release_grid_zoom.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
