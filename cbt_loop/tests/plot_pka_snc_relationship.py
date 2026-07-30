"""Relationship between SNc (dopamine) activity and D1/D2 PKA, from the real
cbt_loop parameters/equations (gate-free architecture: PKA is bg_nln's b directly).

SNc feeds only PRODUCTION, with opponent signs:
  prod_d1 = ReLU(da_pka_gain*m_d1*mean_snc - m_a1*k_a)   # DA drives D1 up
  prod_d2 = ReLU(m_a2*k_a - da_pka_gain*m_d2*mean_snc)   # DA brakes D2
each throttled by mass-action (1 - pka/pka_max), pka_max=1. Effective gains use the
model's transforms (m_* = |param| + floor; k_a sigmoid-bounded).

Panel A: steady-state pka_d1 (rises) and pka_d2 (falls) vs mean_snc.
Panel B: SNc silenced at t=0 -> pka_d1 decays toward 0 (leak, DA drive gone) while
         pka_d2 RISES (unopposed adenosine) -- the opponent effect.

Run:  python cbt_loop/tests/plot_pka_snc_relationship.py
"""
import sys, pathlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cbt_loop")); sys.path.insert(0, str(ROOT))
import cbt_rnn as cbtl
import config_script as C
import self_timed_movement_task as stmt  # noqa

OUT = pathlib.Path(__file__).resolve().parent / "plots" / "pka_snc_relationship.png"
cfg = C.for_family("cbt_loop")
params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=2)

g = config["da_pka_gain"]; tf = config["tau_pka_fall"]; tr_ = config["tau_pka_rise"]
pmax = config["pka_max"]; L = 1.0 / tf


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def cf(x, f):                            # replicate ciel_floor(exc(.)) = |x|+floor
    return f + np.abs(x) * (1 - f)


m_d1 = cf(np.asarray(params["m_d1"]), config["m_floor"])
m_d2 = cf(np.asarray(params["m_d2"]), config["m_floor"])
m_a1 = cf(np.asarray(params["m_a1"]), config["m_floor_a1"])
m_a2 = cf(np.asarray(params["m_a2"]), config["m_floor_a2"])
k_a = config["k_a_floor"] + sig(float(np.asarray(params["k_a"]))) * (config["k_a_cap"] - config["k_a_floor"])


def prod_d1(snc):
    return np.maximum(g * m_d1 * snc - m_a1 * k_a, 0.0)


def prod_d2(snc):
    return np.maximum(m_a2 * k_a - g * m_d2 * snc, 0.0)


def pka_ss(prod):                        # mass-action steady state
    P = prod / tr_
    return P * pmax / (L * pmax + P)


# measure resting mean_snc from a real uncued forward run
T = cfg.TASK_CONFIG["t_total"]
null = cbtl.match_input_channels(jnp.zeros((1, T, 2), np.float32), params)
c0 = dict(config); c0["noise_std"] = 0.0
nd1 = params["J_d1"].shape[0]; nd2 = params["J_d2"].shape[0]
_, xs = cbtl.batched_rnn(params, c0, null, jnp.zeros((1, T, nd1 + nd2)), jr.split(jr.PRNGKey(0), 1))
snc_rest = float(np.asarray(xs[cbtl.STATE_AREA_ORDER.index("SNc")][0]).mean(-1)[300:].mean())
print(f"resting mean_snc ~ {snc_rest:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: steady-state pka vs mean_snc (opponent)
snc = np.linspace(0, 1, 400)
p1 = np.array([pka_ss(prod_d1(s)).mean() for s in snc])
p2 = np.array([pka_ss(prod_d2(s)).mean() for s in snc])
a = ax[0]
a.plot(snc, p1, lw=2.5, color="#2a6fdb", label="pka_d1 (DA excites → rises)")
a.plot(snc, p2, lw=2.5, color="#d1495b", label="pka_d2 (DA brakes → falls)")
a.axhline(0.5, ls="--", color="grey", alpha=.5); a.text(0.02, 0.5, " b=0.5 (bg_nln≈nln)", fontsize=8, color="grey", va="bottom")
a.axvline(snc_rest, ls="--", color="green", alpha=.6); a.text(snc_rest, 0.02, " resting SNc", color="green", fontsize=8)
a.set_title("A. Opponent SNc→PKA (steady state)\nboth rest ~0.5 at resting SNc")
a.set_xlabel("mean_snc (dopamine)"); a.set_ylabel("pka (= bg_nln b)")
a.set_ylim(0, 1); a.legend(fontsize=8.5); a.grid(alpha=.25)

# Panel B: SNc silenced at t=0 -> D1 decays, D2 rises
Td = 3000
pk1 = pka_ss(prod_d1(snc_rest)).copy()
pk2 = pka_ss(prod_d2(snc_rest)).copy()
pr1 = np.maximum(-m_a1 * k_a, 0.0)       # snc=0 → D1 production off
pr2 = np.maximum(m_a2 * k_a, 0.0)        # snc=0 → D2 adenosine unopposed
tr1 = np.zeros(Td); tr2 = np.zeros(Td)
for t in range(Td):
    tr1[t] = pk1.mean(); tr2[t] = pk2.mean()
    pk1 = (1 - 1 / tf) * pk1 + (1 / tr_) * pr1 * np.maximum(1 - pk1 / pmax, 0)
    pk2 = (1 - 1 / tf) * pk2 + (1 / tr_) * pr2 * np.maximum(1 - pk2 / pmax, 0)
a = ax[1]
a.plot(tr1, lw=2.5, color="#2a6fdb", label="pka_d1 → decays (DA drive gone)")
a.plot(tr2, lw=2.5, color="#d1495b", label="pka_d2 → rises (adenosine unopposed)")
a.axhline(0.5, ls="--", color="grey", alpha=.5)
a.set_title("B. SNc silenced at t=0: opponent drift\n(slow, τ ≈ τ_pka_fall)")
a.set_xlabel("time step (SNc = 0)"); a.set_ylabel("pka (= bg_nln b)")
a.set_ylim(0, 1); a.legend(fontsize=8.5); a.grid(alpha=.25)

plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
