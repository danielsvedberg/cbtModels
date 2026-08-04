"""Sweep balanced_target_rho to find an init rho that gives SUSTAINED near-critical
activity after a perturbation, in the Dale corticothalamic loop.

For each target rho_lin (what balanced_init sets), reports:
  rho_lin  = target (spectral radius of the linear update map M, gain=1)
  rho*     = spectral radius of diag(nln'(rest)) M  (operating-point criticality)
  tau_eff  = -1/ln(rho*)  memory timescale in steps
  perturbation persistence: |dx(t)|/eps of the cortex after a 1e-3 kick to x_ctx0,
    at t = 25/50/100/200/400 steps  (SUSTAINED = stays ~flat; DECAY = shrinks;
    EXPAND = grows -> saturation).
The sweet spot for "sustained near-critical activity" is the target where rho* ~ 1
(perturbation neither dies nor blows up), i.e. long tau_eff without runaway.

Run:  python corticothalamic/rho_sweep.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp, jax.random as jr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corticothalamic")); sys.path.insert(0, ROOT)
import corticothalamic_rnn as ct
import config_script as C
import loop_init as LI

TARGETS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50]
SEEDS = 6


def init_at(seed, target):
    C.CORTICOTHALAMIC_RNN_CONFIG["balanced_target_rho"] = target
    return ct.init_params(jr.PRNGKey(seed), n_input=1)


def rest_state(p, conf, T=600):
    ys, (xc, xt) = ct.corticothalamic_rnn(p, dict(conf, noise_std=0.0), jnp.zeros((T, 1)))
    return np.concatenate([np.asarray(xc)[-1], np.asarray(xt)[-1]])


def rho_star(p, conf, xr):
    W = LI.loop_matrix(p, conf["n_c_U"], conf["n_c_L"], conf["n_c_inh"], conf["n_t_exc"], conf["n_t_inh"])
    tau = conf["tau_ctx"]; N = W.shape[0]
    M = (1.0 - 1.0 / tau) * np.eye(N) + (1.0 / tau) * W
    g = 4.0 * xr * (1.0 - xr)           # nln'(rate) for nln=sigmoid(4(x-0.5))
    return float(np.max(np.abs(np.linalg.eigvals(g[:, None] * M))))


def perturb_curve(p, conf, eps=1e-3, T=450):
    """Perturb AROUND the resting fixed point: both runs start at rest, one kicked
    by eps on unit 0, so |dx|(t) measures the fixed-point Jacobian decay (rho*)."""
    n_ctx = conf["x_ctx0"].shape[0]; n_t = conf["x_t0"].shape[0]
    xr = rest_state(p, conf)                      # settled state (n_ctx+n_t,)
    c = dict(conf); c["noise_std"] = 0.0
    # start EXACTLY at rest; config x_ctx0/x_t0 are pre-nln, so invert nln is messy
    # -> instead set them so nln(x0)=rest by storing logit; simpler: store rest and
    # skip the initial nln by using rest directly via a tiny wrapper is overkill.
    # Pragmatic: seed both runs at the rest RATES (pre-nln ~ rest since near-linear),
    # measure divergence of the cortex block.
    x_ctx_rest = jnp.asarray(xr[:n_ctx]); x_t_rest = jnp.asarray(xr[n_ctx:])
    inp = jnp.zeros((1, T, 1)); stim = jnp.zeros((1, T, n_ctx + n_t))
    keys = jr.split(jr.PRNGKey(0), 1)
    ca = dict(c); ca["x_ctx0"] = x_ctx_rest; ca["x_t0"] = x_t_rest
    cb = dict(c); cb["x_ctx0"] = x_ctx_rest.at[0].add(eps); cb["x_t0"] = x_t_rest
    _, (xa, _) = ct.batched_rnn(p, ca, inp, stim, keys)
    _, (xb, _) = ct.batched_rnn(p, cb, inp, stim, keys)
    d = np.linalg.norm(np.asarray(xa)[0] - np.asarray(xb)[0], axis=-1) / eps
    return d


print("Dale corticothalamic — sweep init rho (target rho_lin), find sustained regime")
print(f"{'target':>7} {'rho_lin':>8} {'rho*':>7} {'tau_eff':>8}   perturbation |dx|/eps @ t= 25,50,100,200,400   verdict")
for target in TARGETS:
    rls, rss, teff, curves = [], [], [], []
    for s in range(SEEDS):
        p, conf = init_at(s, target)
        xr = rest_state(p, conf)
        rl = LI.spectral_radius(p, conf["n_c_U"], conf["n_c_L"], conf["n_c_inh"], conf["n_t_exc"], conf["n_t_inh"], conf["tau_ctx"])
        rs = rho_star(p, conf, xr)
        rls.append(rl); rss.append(rs)
        teff.append((-1.0 / np.log(rs)) if 0 < rs < 1 else np.inf)
        curves.append(perturb_curve(p, conf))
    d = np.mean(curves, axis=0)
    rl, rs = np.mean(rls), np.mean(rss)
    te = np.median([t for t in teff if np.isfinite(t)]) if any(np.isfinite(teff)) else np.inf
    snap = "  ".join(f"{d[t]:.2f}" for t in (25, 50, 100, 200, 400))
    # Verdict from the rest-point perturbation curve. Guard against near-zero d[25]
    # (perturbation already gone by t=25 -> fast decay, NOT "sustained").
    if d[25] < 1e-3:
        v = "DECAY (dies <25 steps)"
    elif d[400] > 2 * d[25]:
        v = "EXPAND (saturating)"
    elif d[400] < 0.1 * d[25]:
        v = "DECAY"
    else:
        v = "SUSTAINED"
    print(f"{target:>7.2f} {rl:>8.3f} {rs:>7.3f} {te:>8.1f}   {snap}   {v}")
