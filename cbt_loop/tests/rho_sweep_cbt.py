"""Apply the corticothalamic 'sustained near-critical' lesson to cbt_loop: sweep
balanced_target_rho and find where rho* (operating-point criticality of the
cortico-thalamic loop) peaks. Same method as corticothalamic/rho_sweep.py, but on
cbt_loop's real loop (30 cortex + 15 thalamus) with the full BG running underneath.

rho_lin = spectral radius of the loop update map M = (1-1/tau)I + (1/tau)W (gain=1)
rho*    = spectral radius of diag(nln'(rest)) M  (WITH the nln gain at rest)
Reports rho_lin, rho*, tau_eff, and an empirical cortex perturbation-decay curve.

Run:  python cbt_loop/tests/rho_sweep_cbt.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp, jax.random as jr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cbt_loop")); sys.path.insert(0, ROOT)
import cbt_rnn as cbtl
import config_script as C
import loop_init as LI

A = cbtl.STATE_AREA_ORDER
TARGETS = [1.00, 1.10, 1.25, 1.50, 1.76, 2.0]   # 1.76 = cbt_loop's raw (un-normalized) rho
SEEDS = 4


def init_at(seed, target):
    C.CBT_RNN_CONFIG["balanced_target_rho"] = target
    return cbtl.init_params(jr.PRNGKey(seed), n_input=2)


def loop_sizes(params):
    return (params["J_cU"].shape[0], params["J_cL"].shape[0], params["J_c_ii"].shape[0],
            params["J_t_ee"].shape[0], params["J_t_ii"].shape[0])


def rest_and_perturb(params, config, eps=1e-3, T=500):
    c = dict(config); c["noise_std"] = 0.0
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    inp = jnp.zeros((1, T, params["B_cue_cU"].shape[1])); stim = jnp.zeros((1, T, n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), 1)
    _, xa = cbtl.batched_rnn(params, c, inp, stim, keys)
    cortex = np.asarray(xa[A.index("Cortex")][0]); thal = np.asarray(xa[A.index("Thalamus")][0])
    rest = np.concatenate([cortex[-200:].mean(0), thal[-200:].mean(0)])
    # perturb the trainable cortex init state x_c0_U by eps, measure cortex |dx|
    p2 = dict(params); p2["x_c0_U"] = jnp.asarray(params["x_c0_U"]).at[0].add(eps)
    _, xb = cbtl.batched_rnn(p2, c, inp, stim, keys)
    d = np.linalg.norm(cortex - np.asarray(xb[A.index("Cortex")][0]), axis=-1) / eps
    return rest, d


print("cbt_loop cortico-thalamic loop — sweep balanced_target_rho for sustained regime")
print(f"{'target':>7} {'rho_lin':>8} {'rho*':>7} {'tau_eff':>8}   cortex |dx|/eps @ t=25,50,100,200,400   verdict")
tau = C.CBT_RUNTIME_CONFIG["tau_c"]
for target in TARGETS:
    rls, rss, teff, curves = [], [], [], []
    for s in range(SEEDS):
        p, conf = init_at(s, target)
        szs = loop_sizes(p)
        rl = LI.spectral_radius(p, *szs, tau)
        rest, d = rest_and_perturb(p, conf)
        W = LI.loop_matrix(p, *szs); N = W.shape[0]
        M = (1.0 - 1.0 / tau) * np.eye(N) + (1.0 / tau) * W
        g = 4.0 * rest * (1.0 - rest)
        rs = float(np.max(np.abs(np.linalg.eigvals(g[:, None] * M))))
        rls.append(rl); rss.append(rs)
        teff.append((-1.0 / np.log(rs)) if 0 < rs < 1 else np.inf)
        curves.append(d)
    d = np.mean(curves, axis=0); rl, rs = np.mean(rls), np.mean(rss)
    te = np.median([t for t in teff if np.isfinite(t)]) if any(np.isfinite(teff)) else np.inf
    snap = "  ".join(f"{d[t]:.2f}" for t in (25, 50, 100, 200, 400))
    v = ("EXPAND" if d[400] > 2 * d[25] else "DECAY (dies <25)" if d[25] < 1e-3
         else "DECAY" if d[400] < 0.1 * d[25] else "SUSTAINED")
    print(f"{target:>7.2f} {rl:>8.3f} {rs:>7.3f} {te:>8.1f}   {snap}   {v}")
