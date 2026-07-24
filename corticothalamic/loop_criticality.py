"""Stability / criticality of the REAL thalamocortical loop of a CBT family,
as built by the current init_params (not the synthetic loop in
stability_analysis.py). Run per family:

    python corticothalamic/loop_criticality.py cbt_loop
    python corticothalamic/loop_criticality.py cbt_loop_noSCnoSTN

What it measures. Each cortical/thalamic area updates as
    x <- nln( (1-1/tau) x + (1/tau) (W x + external) ),   nln(z)=sigmoid(4(z-0.5))
so the recurrent update map is  M = (1-1/tau) I + (1/tau) W,  and stability of the
resting fixed point is governed by the JACOBIAN at that point,
    J* = diag(g) M ,   g_i = nln'(pre_i) = 4 r_i (1 - r_i)   (r_i = resting rate).
We report:
  * rho_lin  = spectral radius of M (the "gain=1" upper bound), and
  * rho*     = spectral radius of J* at the network's real resting rates,
    which is what actually decides sub/critical/super-critical,
plus an empirical perturbation-growth test in the FULL nonlinear model.
"""
import os
import sys

import numpy as np
import jax.numpy as jnp
import jax.random as jr

FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, FAM))   # family's cbt_rnn
sys.path.insert(0, ROOT)                        # config_script / stmt

import cbt_rnn as cbtl          # noqa: E402
import config_script            # noqa: E402

cfg = config_script.for_family(FAM)
AREAS = cbtl.STATE_AREA_ORDER
exc = lambda w: np.abs(np.asarray(w))
inh = lambda w: -np.abs(np.asarray(w))

# Thalamocortical recurrent edges (post <- pre, sign), state order cU,cL,cI,tE,tI.
EDGES = [
    ("cU", "cU", "J_cU", +1), ("cU", "cL", "B_cL_cU", +1), ("cU", "cI", "J_ci_cU", -1), ("cU", "tE", "B_t_cU", +1),
    ("cL", "cL", "J_cL", +1), ("cL", "cU", "B_cU_cL", +1), ("cL", "cI", "J_ci_cL", -1),
    ("cI", "cU", "J_cU_ci", +1), ("cI", "cL", "J_cL_ci", +1), ("cI", "cI", "J_c_ii", -1), ("cI", "tE", "B_t_c_inh", +1),
    ("tE", "tE", "J_t_ee", +1), ("tE", "tI", "J_t_ei", -1), ("tE", "cU", "B_cU_t_exc", +1),
    ("tI", "tE", "J_t_ie", +1), ("tI", "tI", "J_t_ii", -1), ("tI", "cU", "B_cU_t_inh", +1),
]


def build_W(params):
    """Dense signed recurrent matrix of the cortico-thalamic loop from params."""
    sizes = [("cU", params["J_cU"].shape[0]), ("cL", params["J_cL"].shape[0]),
             ("cI", params["J_c_ii"].shape[0]), ("tE", params["J_t_ee"].shape[0]),
             ("tI", params["J_t_ii"].shape[0])]
    idx, off = {}, 0
    for nm, sz in sizes:
        idx[nm] = slice(off, off + sz); off += sz
    W = np.zeros((off, off))
    for post, pre, key, sign in EDGES:
        blk = (exc if sign > 0 else inh)(params[key])
        W[idx[post], idx[pre]] = blk
    return W, idx, off, dict(sizes)


def rest_rates(params, config):
    """Per-unit resting rates of cortex(cU,cL,cI) and thalamus(tE,tI), noise off."""
    c = dict(config); c["noise_std"] = 0.0
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    T = 1000
    inp = jnp.zeros((1, T, params["B_cue_cU"].shape[1]))
    stim = jnp.zeros((1, T, n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), 1)
    _, xs = cbtl.batched_rnn(params, c, inp, stim, keys)
    cortex = np.asarray(xs[AREAS.index("Cortex")][0, -200:]).mean(0)   # (n_cU+n_cL+n_cI,)
    thal = np.asarray(xs[AREAS.index("Thalamus")][0, -200:]).mean(0)   # (n_tE+n_tI,)
    return np.concatenate([cortex, thal])


def perturb_growth(params, config, eps=1e-3, T=400):
    """Empirical: perturb x_c0_U by eps, measure cortex |dx| over time (full model)."""
    c = dict(config); c["noise_std"] = 0.0
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    inp = jnp.zeros((1, T, params["B_cue_cU"].shape[1]))
    stim = jnp.zeros((1, T, n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), 1)
    p2 = dict(params); p2["x_c0_U"] = jnp.asarray(params["x_c0_U"]) + eps
    _, xa = cbtl.batched_rnn(params, c, inp, stim, keys)
    _, xb = cbtl.batched_rnn(p2, c, inp, stim, keys)
    ca = np.asarray(xa[AREAS.index("Cortex")][0]); cb = np.asarray(xb[AREAS.index("Cortex")][0])
    return np.linalg.norm(ca - cb, axis=-1) / eps


def main():
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=1)
    tau = config["tau_c"]
    W, idx, N, sizes = build_W(params)
    M = (1.0 - 1.0 / tau) * np.eye(N) + (1.0 / tau) * W
    rho_lin = float(np.max(np.abs(np.linalg.eigvals(M))))

    r = rest_rates(params, config)          # resting rates, order cU,cL,cI,tE,tI
    g = 4.0 * r * (1.0 - r)                  # nln'(pre) = 4 s (1-s), s = rate
    Jstar = g[:, None] * M
    rho_star = float(np.max(np.abs(np.linalg.eigvals(Jstar))))

    print("=" * 72)
    print(f"THALAMOCORTICAL LOOP CRITICALITY — family: {FAM}")
    print(f"  sizes {sizes} | N={N} | tau_c={tau} | nln=sigmoid(4(x-0.5))")
    print("=" * 72)
    # resting activity per area
    off = 0
    print("resting rates (noise off):")
    for nm, sz in sizes.items():
        seg = r[off:off + sz]; off += sz
        state = ("DEAD" if seg.mean() < 1e-3 else "SATURATED" if (seg > 0.9).mean() > 0.5
                 else "alive")
        print(f"  {nm:3} mean={seg.mean():.3f} (min {seg.min():.3f}, max {seg.max():.3f})  {state}")
    print(f"  mean nln'-gain g over loop = {g.mean():.3f}  (max possible 1.0 at r=0.5)")

    tau_eff = (-1.0 / np.log(rho_star)) if 0 < rho_star < 1 else float("inf")
    print(f"\nrho_lin (recurrent update map M, gain=1)          = {rho_lin:.3f}")
    print(f"rho*    (Jacobian diag(g)M at resting fixed point)= {rho_star:.3f}"
          f"   -> memory timescale tau_eff = {tau_eff:.1f} steps")
    verdict = ("SUPER-CRITICAL (resting FP unstable; expect ramping/limit-cycle/runaway)"
               if rho_star > 1.02 else
               "CRITICAL / near-critical (rho* ~ 1; long memory, marginal)"
               if rho_star >= 0.95 else
               "SUB-CRITICAL (resting FP strongly stable; perturbations decay fast)")
    print(f"VERDICT: {verdict}")

    if "x_c0_U" in params:
        d = perturb_growth(params, config)
        print("\nempirical cortex |dx|/eps after a 1e-3 perturbation of x_c0_U:")
        for t in (0, 5, 10, 25, 50, 100, 200, 399):
            print(f"  t={t:4d}  {d[t]:.3e}")
        if d[-1] > 2 * d[0]:
            print("  -> EXPANDING (super-critical / unstable)")
        elif (d < d[0] / 2).any():
            half = int(np.argmax(d < d[0] / 2))
            print(f"  -> CONTRACTING; halves in ~{half} steps (matches tau_eff above)")
        else:
            print("  -> ~marginal (slow decay)")
    else:
        print("\n(empirical perturbation test skipped: this family's initial states "
              "are fixed, not trainable params — rho*/tau_eff above are the measure.)")


if __name__ == "__main__":
    main()
