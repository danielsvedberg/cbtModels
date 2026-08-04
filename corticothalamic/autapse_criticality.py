"""How does the no-autapse rule (zero the diagonal of recurrent weight blocks)
affect criticality of the corticothalamic loop, and does pool size compensate?

The loop update map is M = (1-1/tau) I + (1/tau) W, with the 2-area coupling
    W = [[w_ctx_ctx, w_t_ctx],
         [w_ctx_t,   w_t_t  ]]
Criticality measures:
  rho_lin = max|eig(M)|                         (gain=1 structural upper bound)
  rho*    = max|eig((1-1/tau)I + (1/tau) diag(g) W))  at the nonlinear rest point,
            g = nln'(x_rest), nln=sigmoid(4(x-0.5))
no_autapse zeroes the diagonal of the WITHIN-area blocks (w_ctx_ctx, w_t_t) only
(cross-area blocks have no diagonal self-term to remove).

Run:  python corticothalamic/autapse_criticality.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp, jax.random as jr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corticothalamic")); sys.path.insert(0, ROOT)
import corticothalamic_rnn as ct
import config_script as C

cfg = C.for_family("corticothalamic")
rt = C.runtime_config_for("corticothalamic")
TAU = rt["tau_ctx"]


def loop_W(params, no_autapse=False):
    wcc = np.asarray(params["w_ctx_ctx"]).copy()
    wtt = np.asarray(params["w_t_t"]).copy()
    wtc = np.asarray(params["w_t_ctx"])   # ctx <- t   (n_ctx, n_t)
    wct = np.asarray(params["w_ctx_t"])   # t   <- ctx (n_t, n_ctx)
    if no_autapse:
        np.fill_diagonal(wcc, 0.0)
        np.fill_diagonal(wtt, 0.0)
    nc, nt = wcc.shape[0], wtt.shape[0]
    W = np.zeros((nc + nt, nc + nt))
    W[:nc, :nc] = wcc; W[:nc, nc:] = wtc
    W[nc:, :nc] = wct; W[nc:, nc:] = wtt
    return W


def rho_lin(W):
    N = W.shape[0]
    M = (1.0 - 1.0 / TAU) * np.eye(N) + (1.0 / TAU) * W
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def nlnp(x):  # nln'(x), nln=sigmoid(4(x-0.5))
    s = 1.0 / (1.0 + np.exp(-4.0 * (x - 0.5)))
    return 4.0 * s * (1.0 - s)


def rest_rho(params, config, no_autapse=False, T=600):
    # run the (noise-free) model to its resting state, then Jacobian rho*
    c = dict(config); c["noise_std"] = 0.0
    p = dict(params)
    if no_autapse:
        p = dict(params)
        p["w_ctx_ctx"] = jnp.asarray(np.asarray(params["w_ctx_ctx"]) * (1 - np.eye(np.asarray(params["w_ctx_ctx"]).shape[0])))
        p["w_t_t"] = jnp.asarray(np.asarray(params["w_t_t"]) * (1 - np.eye(np.asarray(params["w_t_t"]).shape[0])))
    T_in = jnp.zeros((1, T, config["x_ctx0"].shape[0] if False else 1))
    inp = jnp.zeros((1, T, p["w_in_ctx"].shape[1]))
    ys, (xc, xt) = ct.corticothalamic_rnn(p, c, inp[0])
    xr = np.concatenate([np.asarray(xc)[-1], np.asarray(xt)[-1]])
    W = loop_W(p, no_autapse=False)  # p already has diagonal zeroed if requested
    N = W.shape[0]
    g = nlnp(xr)
    Jstar = (1.0 - 1.0 / TAU) * np.eye(N) + (1.0 / TAU) * (g[:, None] * W)
    return float(np.max(np.abs(np.linalg.eigvals(Jstar)))), xr.mean()


def sweep(n, seeds=12):
    """mean rho_lin (autapse on/off) over `seeds` fresh inits at pool size n."""
    on, off = [], []
    for s in range(seeds):
        # override pool sizes via a patched config
        rc = dict(C.rnn_config_for("corticothalamic")); rc["n_ctx"] = n; rc["n_t"] = n
        import types
        # init_params reads rnn_config_for(_FAMILY); monkeypatch via config override
        p, conf = _init_with_sizes(s, n)
        on.append(rho_lin(loop_W(p, no_autapse=False)))
        off.append(rho_lin(loop_W(p, no_autapse=True)))
    return np.mean(on), np.std(on), np.mean(off), np.std(off)


def _init_with_sizes(seed, n):
    # build params directly at pool size n (mirror init_params, free-sign)
    rt_ = rt
    rec, cross, insc, outsc = rt_["rec_scale"], rt_["cross_scale"], rt_["in_scale"], rt_["out_scale"]
    k = jr.split(jr.PRNGKey(seed), 6)
    p = {
        "w_ctx_ctx": rec * jr.normal(k[0], (n, n)),
        "w_t_t": rec * jr.normal(k[1], (n, n)),
        "w_ctx_t": cross * jr.normal(k[2], (n, n)),
        "w_t_ctx": cross * jr.normal(k[3], (n, n)),
        "w_in_ctx": insc * jr.normal(k[4], (n, 1)),
        "w_out_t": outsc * jr.normal(k[5], (1, n)),
        "b_ctx": jnp.zeros((n,)), "b_t": jnp.zeros((n,)), "b_out": jnp.zeros((1,)),
        "x_ctx0": jnp.ones((n,)) * rt_["x_init"], "x_t0": jnp.ones((n,)) * rt_["x_init"],
    }
    conf = dict(rt_); conf["x_ctx0"] = p["x_ctx0"]; conf["x_t0"] = p["x_t0"]
    return p, conf


print(f"corticothalamic loop criticality  (tau={TAU}, rec_scale={rt['rec_scale']}, nln=sigmoid(4(x-0.5)))")
print(f"{'N/area':>7} {'rho_lin(autapse)':>18} {'rho_lin(NO-autapse)':>20} {'delta':>8}")
for n in (10, 20, 30, 40, 60, 100, 150):
    mon, son, moff, soff = sweep(n)
    print(f"{n:>7} {mon:>10.3f}±{son:.3f}   {moff:>12.3f}±{soff:.3f}   {moff-mon:>+7.3f}")

# rho* at the default N=30 (nonlinear rest point), autapse on vs off
p0, c0 = ct.init_params(jr.PRNGKey(0), n_input=1)
rs_on, xm_on = rest_rho(p0, c0, no_autapse=False)
rs_off, xm_off = rest_rho(p0, c0, no_autapse=True)
print(f"\nAt default N=30 (nonlinear rest point):")
print(f"  autapse ON : rho_lin={rho_lin(loop_W(p0,False)):.3f}  rho*={rs_on:.3f}  (mean rest rate {xm_on:.3f})")
print(f"  NO-autapse : rho_lin={rho_lin(loop_W(p0,True)):.3f}  rho*={rs_off:.3f}  (mean rest rate {xm_off:.3f})")
