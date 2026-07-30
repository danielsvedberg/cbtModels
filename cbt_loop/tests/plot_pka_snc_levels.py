"""PKA time-trajectories under sustained dopamine (SNc) levels, gate-free cbt_loop.

Like Panel B of plot_pka_snc_relationship.py (pka vs time), but instead of a single
SNc=0 condition it holds SNc constant at each of 10 levels from min to max and shows
how D1 / D2 PKA evolve from the resting init (0.5) toward that level's equilibrium.
Top = D1 (dopamine excites → higher SNc drives PKA up), bottom = D2 (dopamine brakes
→ higher SNc drives PKA down). Curves are color-coded by SNc level.

Uses the model's real effective gains (m_* = |param| + floor; k_a sigmoid-bounded)
and the mass-action steady-state update (pka_max=1).

Run:  python cbt_loop/tests/plot_pka_snc_levels.py
"""
import sys, pathlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cbt_loop")); sys.path.insert(0, str(ROOT))
import cbt_rnn as cbtl
import config_script as C
import self_timed_movement_task as stmt  # noqa

OUT = pathlib.Path(__file__).resolve().parent / "plots" / "pka_snc_levels.png"
cfg = C.for_family("cbt_loop")
params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=2)

g = config["da_pka_gain"]; tf = config["tau_pka_fall"]; tr_ = config["tau_pka_rise"]
pmax = config["pka_max"]
PKA0 = C.init_state_for("cbt_loop")["pka_d10"]  # resting init (0.5)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def cf(x, f):                            # ciel_floor(exc(.)) = |x| + floor
    return f + np.abs(x) * (1 - f)


m_d1 = cf(np.asarray(params["m_d1"]), config["m_floor"])
m_d2 = cf(np.asarray(params["m_d2"]), config["m_floor"])
m_a1 = cf(np.asarray(params["m_a1"]), config["m_floor_a1"])
m_a2 = cf(np.asarray(params["m_a2"]), config["m_floor_a2"])
k_a = config["k_a_floor"] + sig(float(np.asarray(params["k_a"]))) * (config["k_a_cap"] - config["k_a_floor"])


def evolve(prod_raw, T):
    """Population-mean PKA(t) under constant production, from resting init."""
    pka = np.full(prod_raw.shape, PKA0)
    out = np.zeros(T)
    for t in range(T):
        out[t] = pka.mean()
        pka = (1 - 1 / tf) * pka + (1 / tr_) * prod_raw * np.maximum(1 - pka / pmax, 0)
    return out


# resting SNc (for annotation)
T0 = cfg.TASK_CONFIG["t_total"]
null = cbtl.match_input_channels(jnp.zeros((1, T0, 2), np.float32), params)
c0 = dict(config); c0["noise_std"] = 0.0
nd1 = params["J_d1"].shape[0]; nd2 = params["J_d2"].shape[0]
_, xs = cbtl.batched_rnn(params, c0, null, jnp.zeros((1, T0, nd1 + nd2)), jr.split(jr.PRNGKey(0), 1))
snc_rest = float(np.asarray(xs[cbtl.STATE_AREA_ORDER.index("SNc")][0]).mean(-1)[300:].mean())
print(f"resting mean_snc ~ {snc_rest:.3f}")

T = 3000
levels = np.linspace(0.0, 0.9, 10)                 # min -> max dopamine
cmap = cm.viridis
norm = Normalize(vmin=levels.min(), vmax=levels.max())

fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
for snc in levels:
    col = cmap(norm(snc))
    prod1 = np.maximum(g * m_d1 * snc - m_a1 * k_a, 0.0)
    prod2 = np.maximum(m_a2 * k_a - g * m_d2 * snc, 0.0)
    ax[0].plot(evolve(prod1, T), color=col, lw=2)
    ax[1].plot(evolve(prod2, T), color=col, lw=2)

for a, ttl in ((ax[0], "D1: dopamine EXCITES (higher SNc → higher PKA)"),
               (ax[1], "D2: dopamine BRAKES (higher SNc → lower PKA)")):
    a.axhline(PKA0, ls="--", color="grey", alpha=.5)
    a.text(T * 0.99, PKA0, " init 0.5", ha="right", va="bottom", fontsize=8, color="grey")
    a.set_ylabel("pka (= bg_nln b)"); a.set_ylim(0, 1); a.grid(alpha=.25)
    a.set_title(ttl)
ax[1].set_xlabel("time step (SNc held constant)")

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("sustained mean_snc (dopamine level)")
cb.ax.axhline(snc_rest, color="red", lw=1.5)
cb.ax.text(1.5, snc_rest, " rest", color="red", va="center", fontsize=8, transform=cb.ax.get_yaxis_transform())

fig.suptitle("PKA trajectories under 10 sustained dopamine levels (from resting init 0.5)", fontsize=12)
plt.savefig(OUT, dpi=110, bbox_inches="tight"); print("saved", OUT)
