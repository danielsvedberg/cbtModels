"""TEST: 2D grid over g_da_release x g_ado_release in [0, 2], pka_d1 PINNED 0.5, D2 side.

WHAT & WHY
----------
Symmetric counterpart to ../da_ado_release_grid (which pinned pka_d2 and watched D1). Here
we pin pka_d1=0.5 (D1 excitability fixed) and watch the D2 pathway as the two release
gains vary. D2's PKA is DRIVEN by adenosine (m_a2 * x_ado) and BRAKED by dopamine
(m_d2 * x_da) -- the mirror of D1 -- so we expect D2 to prefer the HIGH g_ado / LOW g_da
corner, opposite to where D1 lived. Question: is D2 as brittle/near-binary as D1 was, or
more graded? All gains at default; 10 random-seed inits per cell -> mean pkaD2 / D2 rate.

FINDING
-------
D2 is the near-perfect MIRROR of D1, and adenosine dominates both:
- D2 is DEAD (~0.03) only at g_ado=0 (no adenosine drive -> D2 PKA has no + input), and
  SATURATED (~0.90-0.97) for ANY g_ado >= 0.25.
- g_da (the DA brake on D2) barely matters: at g_ado=0.25, pkaD2 only drops 0.90 -> 0.88
  as g_da goes 0 -> 2. Dopamine cannot suppress D2 once adenosine is present.
- If it even saturates HARDER than D1 activates: at g_ado=0.25 pkaD2 is already 0.90,
  whereas D1 at g_ado=0 needed g_da=2 to reach only 0.59.
Combined with the D1 grid: adenosine is the switch for BOTH pathways -- low adenosine ->
D1 on / D2 off; any adenosine -> D2 on / D1 off. Dopamine (g_da) only fine-tunes the
pathway in the adenosine-free (g_ado=0) column. This is the DA-vs-A2A/A1 opponent
balance, and the SPN-derived adenosine feedback pins it in the high-adenosine / D2-
dominant state unless g_ado_release is driven to ~0.
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
    c["pin_pka_d1"] = 0.5                          # pin D1 excitability (watch D2)
    c["da_release"] = c["da_release"] * g_da
    c["ado_release"] = c["ado_release"] * g_ado
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    ys, xs = cbtl.batched_rnn(p, c, jnp.zeros((1, T, 1)), jnp.zeros((1, T, nd1 + nd2)),
                              jr.split(jr.PRNGKey(0), 1))
    def m(name):
        x = np.asarray(xs[A.index(name)])[0]; x = x.mean(-1) if x.ndim > 1 else x
        return float(x[-WIN:].mean())
    return m("pkaD2"), m("D2")


def main():
    pk = np.zeros((len(G_DA), len(G_ADO), len(SEEDS)))
    d2 = np.zeros_like(pk)
    for i, gda in enumerate(G_DA):
        for j, gado in enumerate(G_ADO):
            for k, s in enumerate(SEEDS):
                pk[i, j, k], d2[i, j, k] = run_one(s, gda, gado)
        print(f"g_da={gda}: pkaD2 row (mean over seeds) = "
              + " ".join(f"{pk[i, j].mean():.2f}" for j in range(len(G_ADO))))
    pk_m = pk.mean(-1); d2_m = d2.mean(-1); pk_sd = pk.std(-1)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ext = [G_ADO[0], G_ADO[-1], G_DA[0], G_DA[-1]]
    for a, data, title, cmap, vmax in [
            (ax[0], pk_m, "pkaD2 (mean over seeds)", "viridis", 0.6),
            (ax[1], d2_m, "D2 firing rate (mean)", "magma", 1.0),
            (ax[2], pk_sd, "pkaD2 std across seeds", "cividis", None)]:
        im = a.imshow(data, origin="lower", extent=ext, aspect="auto", cmap=cmap,
                      vmin=0, vmax=vmax)
        for i in range(len(G_DA)):
            for j in range(len(G_ADO)):
                a.text(G_ADO[j], G_DA[i], f"{data[i, j]:.2f}", ha="center", va="center",
                       color="w", fontsize=7)
        a.set_xlabel("g_ado_release"); a.set_ylabel("g_da_release"); a.set_title(title)
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("pka_d1=0.5: DA-release x adenosine-release grid, D2 pathway "
                 f"(mean over {len(SEEDS)} seeds)", y=1.02)
    fig.tight_layout()
    out = HERE / "da_ado_release_grid_d2.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
