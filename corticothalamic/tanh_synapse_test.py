"""Impact of wrapping each synaptic term in tanh() on stability/decay, in the Dale
corticothalamic testbed. Both modes are balanced_init'd to rho_lin=1.0, so any
difference is purely the per-synapse nonlinearity.

For syn_nln in {none (linear sum), tanh (per-term saturating)}:
  - resting rates (mean over pools)
  - empirical perturbation decay around the rest fixed point: kick unit 0 by eps,
    measure |dx|(t), report half-life and fitted tau  (= effective memory timescale)
  - cortex cue-response decay tau

Run:  python corticothalamic/tanh_synapse_test.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp, jax.random as jr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corticothalamic")); sys.path.insert(0, ROOT)
import corticothalamic_rnn as ct
import config_script as C

SEEDS = 6


def rest(p, conf, T=600):
    ys, (xc, xt) = ct.corticothalamic_rnn(p, dict(conf, noise_std=0.0), jnp.zeros((T, 1)))
    return np.concatenate([np.asarray(xc)[-1], np.asarray(xt)[-1]])


def perturb_tau(p, conf, eps=1e-3, T=450):
    nc = conf["x_ctx0"].shape[0]; nt = conf["x_t0"].shape[0]
    xr = rest(p, conf)
    c = dict(conf); c["noise_std"] = 0.0
    inp = jnp.zeros((1, T, 1)); stim = jnp.zeros((1, T, nc + nt)); keys = jr.split(jr.PRNGKey(0), 1)
    ca = dict(c); ca["x_ctx0"] = jnp.asarray(xr[:nc]); ca["x_t0"] = jnp.asarray(xr[nc:])
    cb = dict(ca); cb["x_ctx0"] = jnp.asarray(xr[:nc]).at[0].add(eps)
    _, (xa, _) = ct.batched_rnn(p, ca, inp, stim, keys)
    _, (xb, _) = ct.batched_rnn(p, cb, inp, stim, keys)
    d = np.linalg.norm(np.asarray(xa)[0] - np.asarray(xb)[0], axis=-1) / eps
    pk = int(np.argmax(d)); peak = d[pk]
    below = np.where(d[pk:] < peak / 2)[0]; half = int(below[0]) if len(below) else None
    tail = d[pk:pk + 200]; tail = tail[tail > 1e-6]
    tau = (-1.0 / np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0]) if len(tail) > 10 else np.nan
    return peak, half, tau


def cue_tau(p, conf, T=1000, t0=200):
    c = dict(conf); c["noise_std"] = 0.0
    inp = np.zeros((1, T, 1), np.float32); inp[0, t0:t0 + 10, 0] = 1.0
    nc = conf["x_ctx0"].shape[0]; nt = conf["x_t0"].shape[0]; stim = jnp.zeros((1, T, nc + nt)); keys = jr.split(jr.PRNGKey(0), 1)
    _, (xa, _) = ct.batched_rnn(p, c, jnp.asarray(inp), stim, keys)
    _, (xb, _) = ct.batched_rnn(p, c, jnp.zeros((1, T, 1)), stim, keys)
    dc = np.abs(np.asarray(xa)[0].mean(-1) - np.asarray(xb)[0].mean(-1))
    pk = t0 + int(np.argmax(dc[t0:t0 + 150])); peak = dc[pk]
    tail = dc[pk:pk + 200]; tail = tail[tail > 1e-6]
    tau = (-1.0 / np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0]) if len(tail) > 10 else np.nan
    return peak, tau


print("corticothalamic: per-synapse tanh() vs linear-sum (both balanced_init rho_lin=1.0)")
print(f"{'syn_nln':>8} {'rest rate':>10} {'perturb peak':>13} {'half-life':>10} {'perturb tau':>12} {'cue peak':>9} {'cue tau':>8}")
for mode in ("none", "tanh"):
    C.CORTICOTHALAMIC_RNN_CONFIG["syn_nln"] = mode
    rr, pp, hh, pt, cp, cta = [], [], [], [], [], []
    for s in range(SEEDS):
        p, conf = ct.init_params(jr.PRNGKey(s), n_input=1)
        conf["syn_nln"] = mode
        rr.append(rest(p, conf).mean())
        peak, half, tau = perturb_tau(p, conf); pp.append(peak); hh.append(half or np.nan); pt.append(tau)
        cpk, ct_ = cue_tau(p, conf); cp.append(cpk); cta.append(ct_)
    fin = lambda a: np.nanmedian([x for x in a if np.isfinite(x)]) if any(np.isfinite(x) for x in a) else np.nan
    print(f"{mode:>8} {np.mean(rr):>10.3f} {np.mean(pp):>13.3f} {fin(hh):>10.0f} {fin(pt):>12.1f} {np.mean(cp):>9.3f} {fin(cta):>8.1f}")
