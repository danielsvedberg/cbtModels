"""TEST: 2D grid over g_da_release x g_ado_release in [0, 2], pka_d2 pinned 0.5, m_d1=1x.

WHAT & WHY
----------
The 1D sweep (../pind2_release_sweep) showed D1 is alive only when adenosine release is
~0, and that DOPAMINE release (g_da_release) at 1x could not rescue it. This grid asks
the follow-up: is there a DA:adenosine RELEASE ratio that keeps D1 alive at nonzero
adenosine? i.e. can more dopamine release buy back the D1 pathway as adenosine rises?

Setup: pka_d2 pinned 0.5 (D2 excitability fixed, so the D2 PKA runaway is removed but its
firing rate -- hence mean_spn -> adenosine -- is not); m_d1 held at 1x default. Sweep
both release gains 0..2 on a grid; 10 random-seed inits per cell -> mean pkaD1 / D1 rate.

Reads the heatmaps: x = g_ado_release, y = g_da_release; brighter = higher pkaD1 / D1.
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
G_DA = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]     # y axis
G_ADO = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]    # x axis
SEEDS = list(range(10))


def run_one(seed, g_da, g_ado):
    p, c = cbtl.init_params(jr.PRNGKey(seed), n_input=1)
    c = dict(c)
    c["pin_pka_d2"] = 0.5
    c["da_release"] = c["da_release"] * g_da      # default 1.0 -> value == g_da
    c["ado_release"] = c["ado_release"] * g_ado
    # m_d1 left at 1x default (unchanged)
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    ys, xs = cbtl.batched_rnn(p, c, jnp.zeros((1, T, 1)), jnp.zeros((1, T, nd1 + nd2)),
                              jr.split(jr.PRNGKey(0), 1))
    def m(name):
        x = np.asarray(xs[A.index(name)])[0]; x = x.mean(-1) if x.ndim > 1 else x
        return float(x[-WIN:].mean())
    return m("pkaD1"), m("D1")


def main():
    pk = np.zeros((len(G_DA), len(G_ADO), len(SEEDS)))
    d1 = np.zeros_like(pk)
    for i, gda in enumerate(G_DA):
        for j, gado in enumerate(G_ADO):
            for k, s in enumerate(SEEDS):
                pk[i, j, k], d1[i, j, k] = run_one(s, gda, gado)
        print(f"g_da={gda}: pkaD1 row (mean over seeds) = "
              + " ".join(f"{pk[i, j].mean():.2f}" for j in range(len(G_ADO))))
    pk_m = pk.mean(-1); d1_m = d1.mean(-1); pk_sd = pk.std(-1)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ext = [G_ADO[0], G_ADO[-1], G_DA[0], G_DA[-1]]
    for a, data, title, cmap in [
            (ax[0], pk_m, "pkaD1 (mean over seeds)", "viridis"),
            (ax[1], d1_m, "D1 firing rate (mean)", "magma"),
            (ax[2], pk_sd, "pkaD1 std across seeds", "cividis")]:
        im = a.imshow(data, origin="lower", extent=ext, aspect="auto", cmap=cmap,
                      vmin=0, vmax=(0.6 if data is not pk_sd else None))
        # annotate cells
        for i in range(len(G_DA)):
            for j in range(len(G_ADO)):
                a.text(G_ADO[j], G_DA[i], f"{data[i, j]:.2f}", ha="center", va="center",
                       color="w", fontsize=7)
        a.set_xlabel("g_ado_release"); a.set_ylabel("g_da_release"); a.set_title(title)
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("pka_d2=0.5, m_d1=1x: DA-release x adenosine-release grid "
                 f"(mean over {len(SEEDS)} seeds)", y=1.02)
    fig.tight_layout()
    out = HERE / "da_ado_release_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
