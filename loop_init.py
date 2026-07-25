"""Shared cortico-thalamic loop initialization for the CBT families.

All three CBT families (cbt_loop, cbt_loop_noSC, cbt_loop_noSCnoSTN) have the
identical 5-population loop (cU, cL, cI, tE, tI) built from the same 17 weight
blocks, so the loop init lives here rather than being triplicated.

WHY: with the raw Dale init the loop is strongly super-critical -- the update map
    M = (1 - 1/tau) I + (1/tau) W
has rho(M) ~ 1.76. Activity therefore grows until the nonlinearity
nln(x)=sigmoid(4(x-0.5)) saturates, and the resulting collapse of the local gain
(nln' -> 0 as r -> 1) is the only thing that stabilizes it. Saturated cortex
(r ~ 0.99, gain ~ 0.2) neither transmits the cue forward nor the gradient
backward, so the task gradient vanishes (see corticothalamic/loop_gradient.py).

FIX: rescale the 17 loop blocks by a single factor so rho(M) equals a chosen
target (~1.0). Measured effect: cortex leaves saturation (r ~ 0.3-0.6, i.e. near
nln's maximum-gain point) and the cue->output gradient rises ~1000x.

`target_rho` is a CONFIG CONSTANT applied ONCE here, at init. It is not a
trainable parameter: it never enters the params dict, the optimizer never sees
it, and there is no gradient with respect to it. It sets the starting dynamical
regime; training then moves the weights freely and the realized rho drifts.

NOTE on history: an earlier version of this routine also applied a per-row E/I
balance (rescaling each cell's inhibition to exactly cancel its excitation). That
was written for the old rectifying nln = max(0, tanh) and drove the loop to a
DEAD fixed point (zero net drive + a rectifier => x*=0 is the only attractor).
It is deliberately not used here; plain spectral normalization is what
de-saturates under the current sigmoid nln.
"""
import numpy as np
import jax.numpy as jnp

# The 17 recurrent blocks of the loop, as (post, pre, param key, sign).
# State order: cU, cL, cI, tE, tI. exc -> |w|, inh -> -|w|.
LOOP_EDGES = (
    ("cU", "cU", "J_cU", +1), ("cU", "cL", "B_cL_cU", +1),
    ("cU", "cI", "J_ci_cU", -1), ("cU", "tE", "B_t_cU", +1),
    ("cL", "cL", "J_cL", +1), ("cL", "cU", "B_cU_cL", +1),
    ("cL", "cI", "J_ci_cL", -1),
    ("cI", "cU", "J_cU_ci", +1), ("cI", "cL", "J_cL_ci", +1),
    ("cI", "cI", "J_c_ii", -1), ("cI", "tE", "B_t_c_inh", +1),
    ("tE", "tE", "J_t_ee", +1), ("tE", "tI", "J_t_ei", -1),
    ("tE", "cU", "B_cU_t_exc", +1),
    ("tI", "tE", "J_t_ie", +1), ("tI", "tI", "J_t_ii", -1),
    ("tI", "cU", "B_cU_t_inh", +1),
)
LOOP_BLOCKS = tuple(e[2] for e in LOOP_EDGES)


def loop_matrix(params, n_cU, n_cL, n_cI, n_tE, n_tI):
    """Dense signed recurrent matrix W of the loop (state order cU,cL,cI,tE,tI)."""
    sizes = (("cU", n_cU), ("cL", n_cL), ("cI", n_cI), ("tE", n_tE), ("tI", n_tI))
    idx, off = {}, 0
    for name, sz in sizes:
        idx[name] = slice(off, off + sz)
        off += sz
    W = np.zeros((off, off))
    for post, pre, key, sign in LOOP_EDGES:
        W[idx[post], idx[pre]] = sign * np.abs(np.asarray(params[key]))
    return W


def spectral_radius(params, n_cU, n_cL, n_cI, n_tE, n_tI, tau):
    """rho of the loop's update map M = (1-1/tau) I + (1/tau) W."""
    lam = np.linalg.eigvals(loop_matrix(params, n_cU, n_cL, n_cI, n_tE, n_tI))
    return float(np.max(np.abs((1.0 - 1.0 / tau) + lam / tau)))


def normalize_loop(params, n_cU, n_cL, n_cI, n_tE, n_tI, tau, target_rho):
    """Scale the 17 loop blocks by one factor so rho(M) == target_rho.

    Returns (new_params, rho_before, rho_after). Only loop blocks are touched;
    every other projection (cue, BG, readout) is left alone.
    """
    lam = np.linalg.eigvals(loop_matrix(params, n_cU, n_cL, n_cI, n_tE, n_tI))
    shift = 1.0 - 1.0 / tau
    rho_of = lambda s: float(np.max(np.abs(shift + (s / tau) * lam)))
    rho_before = rho_of(1.0)

    # Bisect the global scale s; rho_of is monotone increasing in s.
    lo, hi = 1e-6, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rho_of(mid) < target_rho:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)

    p = dict(params)
    for key in LOOP_BLOCKS:
        p[key] = jnp.asarray(p[key]) * s
    return p, rho_before, rho_of(s)
