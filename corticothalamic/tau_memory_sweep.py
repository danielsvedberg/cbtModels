"""Effective memory timescale vs membrane tau, for linear-sum vs per-term-tanh
synapses, in the Dale corticothalamic testbed. Answers: can raising the membrane
tau counteract the memory loss that tanh-wrapping the synapses causes?

Both modes are balanced_init'd to rho_lin=1.0 at each tau, so the comparison is
purely leak-timescale + synaptic nonlinearity. "effective tau" = fitted decay
constant (steps) of a perturbation around the resting fixed point.

Run:  python corticothalamic/tau_memory_sweep.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp, jax.random as jr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corticothalamic")); sys.path.insert(0, ROOT)
import corticothalamic_rnn as ct
import config_script as C

TAUS = [10, 20, 40, 80, 160]
SEEDS = 5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "tau_memory_sweep.png")


def rest(p, conf, T=800):
    _, (xc, xt) = ct.corticothalamic_rnn(p, dict(conf, noise_std=0.0), jnp.zeros((T, 1)))
    return np.concatenate([np.asarray(xc)[-1], np.asarray(xt)[-1]])


def eff_tau(p, conf, eps=1e-3, T=800):
    nc = conf["x_ctx0"].shape[0]; nt = conf["x_t0"].shape[0]; xr = rest(p, conf)
    c = dict(conf); c["noise_std"] = 0.0
    inp = jnp.zeros((1, T, 1)); stim = jnp.zeros((1, T, nc + nt)); keys = jr.split(jr.PRNGKey(0), 1)
    ca = dict(c); ca["x_ctx0"] = jnp.asarray(xr[:nc]); ca["x_t0"] = jnp.asarray(xr[nc:])
    cb = dict(ca); cb["x_ctx0"] = jnp.asarray(xr[:nc]).at[0].add(eps)
    _, (xa, _) = ct.batched_rnn(p, ca, inp, stim, keys)
    _, (xb, _) = ct.batched_rnn(p, cb, inp, stim, keys)
    d = np.linalg.norm(np.asarray(xa)[0] - np.asarray(xb)[0], axis=-1) / eps
    pk = int(np.argmax(d)); tail = d[pk:pk + 400]; tail = tail[tail > 1e-6]
    return (-1.0 / np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0]) if len(tail) > 15 else np.nan


res = {"none": [], "tanh": []}
for mode in ("none", "tanh"):
    for tau in TAUS:
        C.CORTICOTHALAMIC_RUNTIME_CONFIG["tau_ctx"] = float(tau)
        C.CORTICOTHALAMIC_RUNTIME_CONFIG["tau_t"] = float(tau)
        C.CORTICOTHALAMIC_RNN_CONFIG["syn_nln"] = mode
        taus = []
        for s in range(SEEDS):
            p, conf = ct.init_params(jr.PRNGKey(s), n_input=1); conf["syn_nln"] = mode
            taus.append(eff_tau(p, conf))
        res[mode].append(np.nanmedian(taus))
        print(f"syn={mode:>4} membrane_tau={tau:>4}  effective_tau={res[mode][-1]:6.1f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(TAUS, res["none"], "o-", color="#2a6fdb", lw=2, label="linear sum (syn=none)")
ax.plot(TAUS, res["tanh"], "s-", color="#d1495b", lw=2, label="per-term tanh (syn=tanh)")
ax.plot(TAUS, TAUS, "k:", alpha=.5, label="effective = membrane τ")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("membrane τ (tau_ctx = tau_t)"); ax.set_ylabel("effective memory τ (steps)")
ax.set_title("Corticothalamic: raising membrane τ recovers tanh's memory loss\n"
             "(balanced_init rho_lin=1.0 at each τ)")
ax.legend(); ax.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
