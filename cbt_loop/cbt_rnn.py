import math
import numpy as np
from pathlib import Path
import sys

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.nn import sigmoid, tanh, softplus

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt
import config_script as _rootcfg
_FAMILY = Path(__file__).resolve().parent.name


def exc(w):
    return stmt.exc(w)

def ciel_floor(x, c, f):
    # scale x (in [0, 1]) onto [f, c]: x=0 -> f (floor), x=1 -> c (ceiling)
    return f + x * (c - f)




def inh(w):
    return stmt.inh(w)


def nln(x):
    return stmt.nln(x)


def bg_nln(x, b):
    return stmt.bg_nln(x, b)


def match_input_channels(inputs, params):
    """Zero-pad cue input channels to match the model's cue→cortex width.

    Tasks that emit fewer cue channels than the model expects (e.g. a 1-channel
    STMT cue fed to a model trained with an extra Pavlovian-cue channel) are
    padded with zeros — the missing cues are simply treated as absent.
    """
    n_expected = params["B_cue_cU"].shape[1]
    n_have = inputs.shape[-1]
    if n_have >= n_expected:
        return inputs
    pad = jnp.zeros(inputs.shape[:-1] + (n_expected - n_have,), dtype=inputs.dtype)
    return jnp.concatenate([inputs, pad], axis=-1)


# Canonical state tuple order used across CBT loop code.
STATE_VAR_ORDER = (
    "x_c", "x_d1", "x_d2", "x_snc", "x_gpe", "x_stn", "x_snr", "x_sc", "x_t", "pka_d1", "pka_d2", "x_med",
)
STATE_AREA_ORDER = (
    "Cortex", "D1", "D2", "SNc", "GPe", "STN", "SNr", "SC", "Thalamus", "pkaD1", "pkaD2", "Medulla",
)


# Thalamocortical recurrent blocks (all 17), used by the balanced-init option.
_TC_EXC_BLOCKS = ["J_cU", "J_cL", "B_cU_cL", "B_cL_cU", "J_cU_ci", "J_cL_ci",
                  "J_t_ee", "J_t_ie", "B_t_cU", "B_t_c_inh", "B_cU_t_exc", "B_cU_t_inh"]
_TC_INH_BLOCKS = ["J_ci_cU", "J_ci_cL", "J_c_ii", "J_t_ei", "J_t_ii"]


def _tc_recurrent_matrix(p, n_cU, n_cL, n_cI, n_tE, n_tI):
    """Dense signed recurrent matrix of the cortico-thalamic loop (state order
    cU, cL, cI, tE, tI) built from params (exc=|w|, inh=-|w|)."""
    sizes = [("cU", n_cU), ("cL", n_cL), ("cI", n_cI), ("tE", n_tE), ("tI", n_tI)]
    idx, off = {}, 0
    for nm, sz in sizes:
        idx[nm] = slice(off, off + sz); off += sz
    W = np.zeros((off, off))
    E = lambda k: np.abs(np.asarray(p[k]))
    I = lambda k: -np.abs(np.asarray(p[k]))
    W[idx["cU"], idx["cU"]] = E("J_cU");    W[idx["cU"], idx["cL"]] = E("B_cL_cU")
    W[idx["cU"], idx["cI"]] = I("J_ci_cU"); W[idx["cU"], idx["tE"]] = E("B_t_cU")
    W[idx["cL"], idx["cL"]] = E("J_cL");    W[idx["cL"], idx["cU"]] = E("B_cU_cL")
    W[idx["cL"], idx["cI"]] = I("J_ci_cL")
    W[idx["cI"], idx["cU"]] = E("J_cU_ci"); W[idx["cI"], idx["cL"]] = E("J_cL_ci")
    W[idx["cI"], idx["cI"]] = I("J_c_ii");  W[idx["cI"], idx["tE"]] = E("B_t_c_inh")
    W[idx["tE"], idx["tE"]] = E("J_t_ee");  W[idx["tE"], idx["tI"]] = I("J_t_ei")
    W[idx["tE"], idx["cU"]] = E("B_cU_t_exc")
    W[idx["tI"], idx["tE"]] = E("J_t_ie");  W[idx["tI"], idx["tI"]] = I("J_t_ii")
    W[idx["tI"], idx["cU"]] = E("B_cU_t_inh")
    return W


def _balanced_thalamocortical_init(params, n_cU, n_cL, n_cI, n_tE, n_tI,
                                   tau=10.0, target_rho=0.95, persistent_self_gain=None):
    """Put the Dale's-law cortico-thalamic loop into a near-critical regime:
      (1) matched 1/sqrt(n) scaling for the excitatory 1/n blocks (E->I, cross-area);
      (2) per-row E/I balance -- each cell's local inhibition is rescaled to cancel
          its total excitation (net recurrent drive -> 0, kills the mean-mode outlier);
      (3) spectral-radius normalization -- scale all loop weights so the update map
          J = (1-1/tau)I + (1/tau)W has spectral radius = target_rho.
      (4) persistent activity (if persistent_self_gain is not None): set the
          excitatory self-recurrence diagonal so each E unit is a leaky integrator
          with effective self-gain = persistent_self_gain, i.e. it HOLDS a cue-evoked
          bump across the delay. A positive unit feeding itself stays in nln's linear
          band, so this survives the rectification that collapses distributed modes.
    Only thalamocortical blocks are touched. Returns (params, realized_rho)."""
    p = dict(params)
    # (1) matched scaling: g/n -> g/sqrt(n) on the excitatory 1/n blocks.
    for k, pre_n in (("J_cU_ci", n_cU), ("J_cL_ci", n_cL), ("J_t_ie", n_tE),
                     ("B_t_cU", n_tE), ("B_t_c_inh", n_tE),
                     ("B_cU_t_exc", n_cU), ("B_cU_t_inh", n_cU)):
        p[k] = p[k] * math.sqrt(pre_n)
    # (2) per-row balance: local inhibition cancels total excitation, per cell.
    aE = lambda k: np.abs(np.asarray(p[k]))
    exc_tot = {
        "J_ci_cU": aE("J_cU").sum(1) + aE("B_cL_cU").sum(1) + aE("B_t_cU").sum(1),
        "J_ci_cL": aE("J_cL").sum(1) + aE("B_cU_cL").sum(1),
        "J_c_ii":  aE("J_cU_ci").sum(1) + aE("J_cL_ci").sum(1) + aE("B_t_c_inh").sum(1),
        "J_t_ei":  aE("J_t_ee").sum(1) + aE("B_cU_t_exc").sum(1),
        "J_t_ii":  aE("J_t_ie").sum(1) + aE("B_cU_t_inh").sum(1),
    }
    for k, Etot in exc_tot.items():
        Itot = aE(k).sum(1)
        f = np.where(Itot > 1e-9, Etot / np.maximum(Itot, 1e-9), 1.0)
        p[k] = p[k] * jnp.asarray(f)[:, None]
    # (3) spectral-radius normalization (bisection on a global scale s).
    lam = np.linalg.eigvals(_tc_recurrent_matrix(p, n_cU, n_cL, n_cI, n_tE, n_tI))
    shift = 1.0 - 1.0 / tau
    rho_J = lambda s: float(np.max(np.abs(shift + (s / tau) * lam)))
    lo, hi = 1e-4, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if rho_J(mid) < target_rho:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    for k in _TC_EXC_BLOCKS + _TC_INH_BLOCKS:
        p[k] = p[k] * s
    # (4) persistent-activity integrator diagonal on the excitatory self-blocks.
    if persistent_self_gain is not None:
        d_val = tau * persistent_self_gain - (tau - 1.0)   # |J_ii| for target self-gain
        for k, nk in (("J_cU", n_cU), ("J_cL", n_cL), ("J_t_ee", n_tE)):
            di = jnp.arange(nk)
            p[k] = jnp.asarray(p[k]).at[di, di].set(d_val)
    # realized spectral radius of the final loop
    lam_f = np.linalg.eigvals(_tc_recurrent_matrix(p, n_cU, n_cL, n_cI, n_tE, n_tI))
    return p, float(np.max(np.abs(shift + (1.0 / tau) * lam_f)))


def init_params(rng_key, n_input):
    """Initialize multiregion CBT loop params and runtime config.

    Cortex and thalamus follow Dale's law as fully separate populations:
    ``n_*_exc`` excitatory projection neurons and ``n_*_inh`` local inhibitory
    interneurons live in their own state vectors with their own connectivity
    blocks. Only the excitatory pool sends out-of-area projections.

    This family additionally includes the subthalamic nucleus (STN) and the
    superior colliculus (SC). STN is a glutamatergic relay (cortical hyperdirect
    pathway + GPe inhibition → GPe / SNr / SNc). SC has free-sign recurrence,
    receives cortical excitation and SNr inhibition, and drives thalamus and
    the medullary motor output.

    Every structural/runtime value comes from the central config (crashes on a
    missing key — no silent defaults). n_input is the only per-call argument
    (it is task-derived: 1 for single-cue, 2 for the dual/go-cue tasks).
    """
    _rc = _rootcfg.rnn_config_for(_FAMILY)
    _is = _rootcfg.init_state_for(_FAMILY)
    n_c_U = _rc["n_c_U"]
    n_c_L = _rc["n_c_L"]
    n_c_inh = _rc["n_c_inh"]
    n_d1 = _rc["n_d1"]
    n_d2 = _rc["n_d2"]
    n_snc = _rc["n_snc"]
    n_snr = _rc["n_snr"]
    n_gpe = _rc["n_gpe"]
    n_stn = _rc["n_stn"]
    n_sc = _rc["n_sc"]
    n_t_exc = _rc["n_t_exc"]
    n_t_inh = _rc["n_t_inh"]
    n_med = _rc["n_med"]
    n_output = _rc["n_output"]
    g_bg = _rc["g_bg"]
    g_nm = _rc["g_nm"]
    noise_std = _rc["noise_std"]
    balanced_init = _rc["balanced_init"]
    balanced_target_rho = _rc["balanced_target_rho"]
    persistent_self_gain = _rc["persistent_self_gain"]
    skeys = jr.split(rng_key, 60)

    # Fan-in scaling (adapted from the promising_version design):
    #  - Feed-forward sign-constrained projections (wrapped in exc()/inh() = |w|)
    #    sum same-signed weights against non-negative rates — nothing cancels, so
    #    the drive would grow as sqrt(fan_in) under 1/sqrt(n) and saturate nln
    #    downstream of a large population. They scale by 1/fan_in, keeping the mean
    #    drive O(1) in area size.
    #  - Recurrent EXCITATORY self-connection blocks (cortex J_cU/J_cL/B_cU_cL/
    #    B_cL_cU, thalamus J_t_ee) keep 1/sqrt(n): the E->E loop gain must stay
    #    >~1 to sustain persistent/ramping activity. 1/fan_in makes it subcritical
    #    (~0.8*g_bg), so the cortex decays to silence and the whole loop starves.
    #  - Free-sign pathways (J_sc) have zero-mean weights that cancel -> 1/sqrt(n).
    #  - Pacemaker vectors (P_snc/P_snr/P_gpe) are per-unit biases, not fan-in
    #    sums, so they keep 1/sqrt(n).
    params = {
        # Cortex E/I recurrence (4 blocks: post × pre, pre identity sets sign).
        # Cortex split into two excitatory PT-like populations — cU (upper) and
        # cL (lower) — plus a shared inhibitory pool c_inh (Economo et al. 2018).
        "J_cU": (g_bg / math.sqrt(n_c_U)) * jr.normal(skeys[2],  (n_c_U, n_c_U)),
        "J_cL": (g_bg / math.sqrt(n_c_L)) * jr.normal(skeys[52], (n_c_L, n_c_L)),
        "B_cU_cL": (g_bg / math.sqrt(n_c_U)) * jr.normal(skeys[53], (n_c_L, n_c_U)),  # cU -> cL
        "B_cL_cU": (g_bg / math.sqrt(n_c_L)) * jr.normal(skeys[54], (n_c_U, n_c_L)),  # cL -> cU
        "J_cU_ci": (g_bg / (n_c_U)) * jr.normal(skeys[43], (n_c_inh, n_c_U)),
        "J_cL_ci": (g_bg / (n_c_L)) * jr.normal(skeys[56], (n_c_inh, n_c_L)),
        "J_ci_cU": (g_bg / (n_c_inh)) * jr.normal(skeys[42], (n_c_U, n_c_inh)),
        "J_ci_cL": (g_bg / (n_c_inh)) * jr.normal(skeys[55], (n_c_L, n_c_inh)),
        "J_c_ii": (g_bg / (n_c_inh)) * jr.normal(skeys[44], (n_c_inh, n_c_inh)),
        "J_d1": (g_bg / (n_d1)) * jr.normal(skeys[0], (n_d1, n_d1)),
        "J_d2": (g_bg / (n_d2)) * jr.normal(skeys[8], (n_d2, n_d2)),
        # Pacemaker vectors (no within-area recurrence for SNc/SNr).
        "P_snc": (g_nm / math.sqrt(n_snc)) * jr.normal(skeys[7], (n_snc,)),
        "J_gpe": (g_bg / (n_gpe)) * jr.normal(skeys[19], (n_gpe, n_gpe)),
        "P_snr": (g_bg / math.sqrt(n_snr)) * jr.normal(skeys[18], (n_snr,)),
        "P_gpe": (g_bg / math.sqrt(n_gpe)) * jr.normal(skeys[34], (n_gpe,)),
        # Subthalamic nucleus.
        "J_stn": (g_bg / (n_stn)) * jr.normal(skeys[20], (n_stn, n_stn)),
        # Superior colliculus. J_sc is free-sign (unconstrained recurrence).
        "J_sc": (g_bg / math.sqrt(n_sc)) * jr.normal(skeys[39], (n_sc, n_sc)),
        # Thalamus E/I recurrence.
        "J_t_ee": (g_bg / math.sqrt(n_t_exc)) * jr.normal(skeys[5],  (n_t_exc, n_t_exc)),
        "J_t_ei": (g_bg / (n_t_inh)) * jr.normal(skeys[45], (n_t_exc, n_t_inh)),
        "J_t_ie": (g_bg / (n_t_exc)) * jr.normal(skeys[46], (n_t_inh, n_t_exc)),
        "J_t_ii": (g_bg / (n_t_inh)) * jr.normal(skeys[47], (n_t_inh, n_t_inh)),
        # Cue → cortex (excitatory pools only; the inhibitory pool receives no cue).
        "B_cue_cU": (1 / (n_input)) * jr.normal(skeys[3],  (n_c_U, n_input)),
        "B_cue_cL": (1 / (n_input)) * jr.normal(skeys[57], (n_c_L, n_input)),
        # Thalamus exc → cU (reciprocal); feedforward inhibition to c_inh.
        "B_t_cU": (g_bg / (n_t_exc)) * jr.normal(skeys[4],  (n_c_U, n_t_exc)),
        "B_t_c_inh": (g_bg / (n_t_exc)) * jr.normal(skeys[49], (n_c_inh, n_t_exc)),
        # cU → thalamus (both pools receive).
        "B_cU_t_exc": (1 / (n_c_U)) * jr.normal(skeys[29], (n_t_exc, n_c_U)),
        "B_cU_t_inh": (1 / (n_c_U)) * jr.normal(skeys[50], (n_t_inh, n_c_U)),
        # cU → striatum / GPe / SC; cL → SNc / medulla; both cU & cL → STN.
        "B_cU_d1": (g_bg / (n_c_U)) * jr.normal(skeys[1], (n_d1, n_c_U)),
        "B_cU_d2": (g_bg / (n_c_U)) * jr.normal(skeys[12], (n_d2, n_c_U)),
        "B_cU_gpe": (g_bg / (n_c_U)) * jr.normal(skeys[58], (n_gpe, n_c_U)),
        "B_cL_snc": (1 / (n_c_L)) * jr.normal(skeys[32], (n_snc, n_c_L)),
        "B_cU_stn": (1 / (n_c_U)) * jr.normal(skeys[26], (n_stn, n_c_U)),  # hyperdirect (cU)
        "B_cL_stn": (1 / (n_c_L)) * jr.normal(skeys[59], (n_stn, n_c_L)),  # hyperdirect (cL)
        "B_cU_sc": (1 / (n_c_U)) * jr.normal(skeys[35], (n_sc, n_c_U)),  # cU → SC (cU only)
        "B_d1_snc": (1 / (n_d1)) * jr.normal(skeys[17], (n_snc, n_d1)),
        "B_d2_snc": (1 / (n_d2)) * jr.normal(skeys[28], (n_snc, n_d2)),
        "B_d1_snr": (1 / (n_d1)) * jr.normal(skeys[22], (n_snr, n_d1)),
        "B_d2_gpe": (1 / (n_d2)) * jr.normal(skeys[24], (n_gpe, n_d2)),
        "B_gpe_snr": (1 / (n_gpe)) * jr.normal(skeys[40], (n_snr, n_gpe)),  # GPe → SNr (inh)
        "B_gpe_snc": (1 / (n_gpe)) * jr.normal(skeys[33], (n_snc, n_gpe)),  # GPe → SNc (inh)
        "B_gpe_stn": (1 / (n_gpe)) * jr.normal(skeys[25], (n_stn, n_gpe)),  # GPe → STN (inh)
        "B_stn_gpe": (1 / (n_stn)) * jr.normal(skeys[27], (n_gpe, n_stn)),  # STN → GPe (exc)
        "B_stn_snr": (1 / (n_stn)) * jr.normal(skeys[23], (n_snr, n_stn)),  # STN → SNr (exc)
        "B_stn_snc": (1 / (n_stn)) * jr.normal(skeys[9],  (n_snc, n_stn)),  # STN → SNc (exc)
        # SNr → thalamus (both pools receive).
        "B_snr_t_exc": (1 / (n_snr)) * jr.normal(skeys[6],  (n_t_exc, n_snr)),
        "B_snr_t_inh": (1 / (n_snr)) * jr.normal(skeys[51], (n_t_inh, n_snr)),
        # Superior colliculus afferents/efferents.
        "B_snr_sc": (1 / (n_snr)) * jr.normal(skeys[36], (n_sc, n_snr)),  # SNr → SC (inh)
        "B_sc_t_exc": (1 / (n_sc)) * jr.normal(skeys[37], (n_t_exc, n_sc)),  # SC → thalamus exc
        "B_sc_t_inh": (1 / (n_sc)) * jr.normal(skeys[52], (n_t_inh, n_sc)),  # SC → thalamus inh
        "B_sc_med": (1 / (n_sc)) * jr.normal(skeys[38], (n_med // 2, n_sc)),  # SC → medulla E units
        # Dopamine→PKA gains: per-neuron so each striatal neuron has its own
        # dopamine sensitivity (no shared population PKA).
        "m_d1": (1 / (n_snc)) * jr.normal(skeys[10], (n_d1,)),
        "m_d2": (1 / (n_snc)) * jr.normal(skeys[11], (n_d2,)),

        # Lateral inhibition between D1 and D2 populations.
        "B_d1_d2": (g_bg / (n_d1)) * jr.normal(skeys[16], (n_d2, n_d1)),  # D1 → D2 (inh)
        "B_d2_d1": (g_bg / (n_d2)) * jr.normal(skeys[31], (n_d1, n_d2)),  # D2 → D1 (inh)
        # Medullary area: two E/I pairs (E0,I0) and (E1,I1) coupled reciprocally.
        # Each 2×2 block: col 0 = from E (exc), col 1 = from I (inh).
        "J_med_w1": (g_bg / (2)) * jr.normal(skeys[13], (2, 2)),  # within pair 1
        "J_med_w2": (g_bg / (2)) * jr.normal(skeys[21], (2, 2)),  # within pair 2
        "J_med_x":  (g_bg / (2)) * jr.normal(skeys[30], (2, 2)),  # cross-pair
        "B_cL_med": (1 / (n_c_L)) * jr.normal(skeys[14], (n_med // 2, n_c_L)),  # cL → medulla E units only
        "B_snr_med": (1 / (n_snr)) * jr.normal(skeys[41], (n_med // 2, n_snr)),  # SNr → Medulla E units (inh)
        "C_med": (1 / (n_med // 2)) * jr.normal(skeys[15], (n_output, n_med // 2)),  # E units only
        #"rb": jnp.abs((1 / (n_med)) * jr.normal(skeys[16], (n_output,))),
        # Output readout gain/bias: y = sigmoid(out_gain * (c_med @ x_med_E) + out_bias).
        # out_bias = logit(0.25) gives a nonzero resting response prob (~0.25) so the
        # policy can explore from the start (instead of being floored at 0 by nln);
        # out_gain sets the readout's dynamic range. Both are trainable.
        "out_gain": jnp.array(4.0),
        "out_bias": jnp.array(-1.0986123),  # logit(0.25)
        # Trainable initial states (resting/baseline activity per area); the
        # starting values are declared centrally (config_script.CBT_INIT_STATE).
        "x_c0_U": jnp.ones((n_c_U,)) * _is["x_c0_U"],
        "x_c0_L": jnp.ones((n_c_L,)) * _is["x_c0_L"],
        "x_c0_inh": jnp.ones((n_c_inh,)) * _is["x_c0_inh"],
        "x_d10":  jnp.ones((n_d1,))  * _is["x_d10"],
        "x_d20":  jnp.ones((n_d2,))  * _is["x_d20"],
        "x_snc0": jnp.ones((n_snc,)) * _is["x_snc0"],
        "x_gpe0": jnp.ones((n_gpe,)) * _is["x_gpe0"],
        "x_stn0": jnp.ones((n_stn,)) * _is["x_stn0"],
        "x_snr0": jnp.ones((n_snr,)) * _is["x_snr0"],
        "x_sc0":  jnp.ones((n_sc,))  * _is["x_sc0"],
        "x_t0_exc": jnp.ones((n_t_exc,)) * _is["x_t0_exc"],
        "x_t0_inh": jnp.ones((n_t_inh,)) * _is["x_t0_inh"],
        "x_med0": jnp.ones((n_med,)) * _is["x_med0"],
        "pka_d10": jnp.ones((n_d1,)) * _is["pka_d10"],
        "pka_d20": jnp.ones((n_d2,)) * _is["pka_d20"],
        # Adenosine: one tunable tonic level k_a (scalar — will become a
        # dynamic state later) feeding per-SPN weights m_a1 / m_a2, mirroring
        # m_d1 / m_d2 for the broadcast DA gain.
        "k_a": jnp.array(1.0),
        "m_a1": jnp.ones((n_d1,)) * 0.05,  # A1R inhibitory drive on D1 PKA
        "m_a2": jnp.ones((n_d2,)) * 0.01,  # A2R excitatory drive on D2 PKA
    }

    # Biophysical runtime constants are declared centrally (config_script.
    # CBT_RUNTIME_CONFIG + this family's architecture-unique extras).
    config = dict(_rootcfg.runtime_config_for(_FAMILY))
    config.update({
        "n_c_U": n_c_U,
        "n_c_L": n_c_L,
        "n_c_inh": n_c_inh,
        "n_t_exc": n_t_exc,
        "n_t_inh": n_t_inh,
        "noise_std": noise_std,
    })

    if balanced_init:
        # Steps 1-4: balance the Dale's-law cortico-thalamic loop and set it
        # near-critical so it can hold/ramp a cue signal (see
        # ../corticothalamic/stability_analysis.py).
        config["tau_c"] = 10.0   # step 4: slower cortical integration
        config["tau_t"] = 10.0   # loop spans thalamus too; keep tau uniform
        # target_rho sets the loop's memory timescale tau_eff = -1/ln(rho). To hold
        # a cue across the self-timed delay (~t_cue+t_wait steps) the mode must
        # decay slowly: rho ~ exp(-1/delay). 0.95 (tau_eff~20) is far too fast;
        # ~0.997 matches a ~300-step delay. Tunable via balanced_target_rho.
        params, realized_rho = _balanced_thalamocortical_init(
            params, n_c_U, n_c_L, n_c_inh, n_t_exc, n_t_inh,
            tau=config["tau_c"], target_rho=balanced_target_rho,
            persistent_self_gain=persistent_self_gain,
        )
        pg = f", persistent self-gain={persistent_self_gain}" if persistent_self_gain else ""
        print(f"[balanced_init] thalamocortical loop rho(J) = {realized_rho:.3f} "
              f"(target {balanced_target_rho}); matched scaling + per-row E/I balance, "
              f"tau_c=tau_t=10{pg}.")

    return params, config


def multiregion_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    # Per-area unit counts (used by the fan-in normalizations below).
    n_c_U_   = params["J_cU"].shape[0]
    n_c_L_   = params["J_cL"].shape[0]
    n_c_inh_ = params["J_c_ii"].shape[0]
    n_d1_    = params["J_d1"].shape[0]
    n_d2_    = params["J_d2"].shape[0]
    n_snc_   = params["P_snc"].shape[0]
    n_gpe_   = params["J_gpe"].shape[0]
    n_stn_   = params["J_stn"].shape[0]
    n_sc_    = params["J_sc"].shape[0]
    n_snr_   = params["P_snr"].shape[0]
    n_t_exc_ = params["J_t_ee"].shape[0]
    n_t_inh_ = params["J_t_ii"].shape[0]
    n_med_   = params["J_med_w1"].shape[0] * 2

    # Trainable initial states come straight from params (crashes if absent — no
    # fallback); their starting values were set from config_script.CBT_INIT_STATE.
    x_c0_U   = jnp.asarray(params["x_c0_U"])
    x_c0_L   = jnp.asarray(params["x_c0_L"])
    x_c0_inh = jnp.asarray(params["x_c0_inh"])
    x_d10    = jnp.asarray(params["x_d10"])
    x_d20    = jnp.asarray(params["x_d20"])
    x_snc0   = jnp.asarray(params["x_snc0"])
    x_gpe0   = jnp.asarray(params["x_gpe0"])
    x_stn0   = jnp.asarray(params["x_stn0"])
    x_sc0    = jnp.asarray(params["x_sc0"])
    x_snr0   = jnp.asarray(params["x_snr0"])
    x_t0_exc = jnp.asarray(params["x_t0_exc"])
    x_t0_inh = jnp.asarray(params["x_t0_inh"])
    x_med0   = jnp.asarray(params["x_med0"])
    pka_d10  = jnp.asarray(params["pka_d10"])
    pka_d20  = jnp.asarray(params["pka_d20"])


    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    noise_std = jnp.asarray(config["noise_std"])
    x_c0_U = jnp.minimum(nln(x_c0_U + noise_std * jr.normal(init_key, x_c0_U.shape)), 0.5)
    x_c0_L = jnp.minimum(nln(x_c0_L + noise_std * jr.normal(init_key, x_c0_L.shape)), 0.5)
    x_c0_inh = jnp.minimum(nln(x_c0_inh + noise_std * jr.normal(init_key, x_c0_inh.shape)), 0.5)
    x_d10 = nln(x_d10 + noise_std * jr.normal(init_key, x_d10.shape))
    x_d20 = nln(x_d20 + noise_std * jr.normal(init_key, x_d20.shape))
    x_snc0 = nln(x_snc0 + noise_std * jr.normal(init_key, x_snc0.shape))
    x_gpe0 = nln(x_gpe0 + noise_std * jr.normal(init_key, x_gpe0.shape))
    x_stn0 = nln(x_stn0 + noise_std * jr.normal(init_key, x_stn0.shape))
    x_sc0 = nln(x_sc0 + noise_std * jr.normal(init_key, x_sc0.shape))
    x_snr0 = nln(x_snr0 + noise_std * jr.normal(init_key, x_snr0.shape))
    x_t0_exc = nln(x_t0_exc + noise_std * jr.normal(init_key, x_t0_exc.shape))
    x_t0_inh = nln(x_t0_inh + noise_std * jr.normal(init_key, x_t0_inh.shape))
    x_med0 = nln(x_med0 + noise_std * jr.normal(init_key, x_med0.shape))

    # Cortex blocks: cU/cL excitatory PT populations + shared inhibitory c_inh.
    j_cU = exc(params["J_cU"])        # cU → cU
    j_cL = exc(params["J_cL"])        # cL → cL
    b_cU_cL = exc(params["B_cU_cL"])  # cU → cL
    b_cL_cU = exc(params["B_cL_cU"])  # cL → cU
    j_cU_ci = exc(params["J_cU_ci"])  # cU → c_inh
    j_cL_ci = exc(params["J_cL_ci"])  # cL → c_inh
    j_ci_cU = inh(params["J_ci_cU"])  # c_inh → cU
    j_ci_cL = inh(params["J_ci_cL"])  # c_inh → cL
    j_c_ii = inh(params["J_c_ii"])    # c_inh → c_inh
    # Thalamus E/I recurrent blocks.
    j_t_ee = exc(params["J_t_ee"])  # T_exc → T_exc
    j_t_ei = inh(params["J_t_ei"])  # T_inh → T_exc
    j_t_ie = exc(params["J_t_ie"])  # T_exc → T_inh
    j_t_ii = inh(params["J_t_ii"])  # T_inh → T_inh

    j_d1 = inh(params["J_d1"])
    j_d2 = inh(params["J_d2"])
    j_gpe = inh(params["J_gpe"])
    j_stn = exc(params["J_stn"])
    j_sc  = params["J_sc"]  # free-sign recurrence

    p_snr = exc(params["P_snr"])
    p_snc = exc(params["P_snc"])
    p_gpe = exc(params.get("P_gpe", jnp.zeros(j_gpe.shape[0])))

    # Cue → cortex (excitatory pools only).
    b_cue_cU = exc(params["B_cue_cU"])
    b_cue_cL = exc(params["B_cue_cL"])
    # Thalamus exc → cU (reciprocal) and → c_inh (feedforward inhibition).
    b_t_cU = exc(params["B_t_cU"])
    b_t_c_inh = exc(params["B_t_c_inh"])
    # cU → thalamus (both pools).
    b_cU_t_exc = exc(params["B_cU_t_exc"])
    b_cU_t_inh = exc(params["B_cU_t_inh"])
    # cU → striatum / GPe / SC; cL → SNc; both → STN (hyperdirect).
    b_cU_d1 = exc(params["B_cU_d1"])+(0.1/n_c_U_)
    b_cU_d2 = exc(params["B_cU_d2"])+(0.1/n_c_U_)
    b_cU_gpe = exc(params["B_cU_gpe"])+(0.1/n_c_U_)
    b_cL_snc = exc(params["B_cL_snc"])+(0.1/n_c_L_)
    b_cU_stn = exc(params["B_cU_stn"])  # hyperdirect (cU)
    b_cL_stn = exc(params["B_cL_stn"])  # hyperdirect (cL)
    b_cU_sc = exc(params["B_cU_sc"])    # cU → SC (cU only)
    b_d1_snc = inh(params["B_d1_snc"])
    b_d2_snc = inh(params["B_d2_snc"])
    b_d1_snr = inh(params["B_d1_snr"])-(0.1/n_d1_)
    b_d2_gpe = inh(params["B_d2_gpe"])-(0.1/n_d2_)
    b_gpe_snr = inh(params["B_gpe_snr"])-(0.1/n_gpe_)
    b_gpe_snc = inh(params["B_gpe_snc"])
    b_gpe_stn = inh(params["B_gpe_stn"])
    b_stn_gpe = exc(params["B_stn_gpe"])
    b_stn_snr = exc(params["B_stn_snr"])
    b_stn_snc = exc(params["B_stn_snc"])
    # SNr → thalamus (both pools).
    b_snr_t_exc = inh(params["B_snr_t_exc"])-(0.1/n_snr_)
    b_snr_t_inh = inh(params["B_snr_t_inh"])
    # Superior colliculus: exc from cortex, inh from SNr, exc to thalamus and medulla.
    b_snr_sc = inh(params["B_snr_sc"])
    b_sc_t_exc = exc(params["B_sc_t_exc"])
    b_sc_t_inh = exc(params["B_sc_t_inh"])
    b_sc_med = exc(params["B_sc_med"])
    # Dopamine / adenosine→PKA gains. Floored exc keeps the per-SPN weights
    # ≥ m_floor with a live gradient (no dead zone) for both DA and tonic
    # adenosine drives; k_a is the (currently scalar) adenosine level shared
    # by all SPNs.
    m_floor = config["m_floor"]
    # Adenosine weights get their own floor so the tonic A1R/A2R drive can be
    # decoupled from the DA floor. The DA→D1 PKA term is structurally weak
    # (mean_snc is small), so a high shared floor pins A1R inhibition above it
    # and kills pka_d1. m_floor_a1 lets the A1R floor drop (revive D1) while
    # m_floor_a2 preserves the A2R drive that keeps pka_d2 alive.
    m_floor_a1 = config["m_floor_a1"]
    m_floor_a2 = config["m_floor_a2"]
    m_d1 = ciel_floor(exc(params["m_d1"]),1, m_floor)
    m_d2 = ciel_floor(exc(params["m_d2"]),1, m_floor)
    m_a1 = ciel_floor(exc(params["m_a1"]), 1, m_floor_a1)
    m_a2 = ciel_floor(exc(params["m_a2"]), 1, m_floor_a2)
    _zeros_d1_d2 = jnp.zeros((j_d2.shape[0], j_d1.shape[0]))
    _zeros_d2_d1 = jnp.zeros((j_d1.shape[0], j_d2.shape[0]))
    b_d1_d2 = inh(params.get("B_d1_d2", _zeros_d1_d2))  # D1 → D2 lateral inhibition
    b_d2_d1 = inh(params.get("B_d2_d1", _zeros_d2_d1))  # D2 → D1 lateral inhibition
    # Medullary E/I pairs. Each 2×2 block: col 0 from E (exc), col 1 from I (inh).
    def _med_block(raw):
        return jnp.concatenate([exc(raw[:, :1]), inh(raw[:, 1:])], axis=1)

    j_w1 = _med_block(params["J_med_w1"])  # within pair 1
    j_w2 = _med_block(params["J_med_w2"])  # within pair 2
    j_x  = _med_block(params["J_med_x"])   # cross-pair

    # Assemble 4×4 matrix in [E0, E1, I0, I1] order.
    # Pair 1 = (E0 idx 0, I0 idx 2); Pair 2 = (E1 idx 1, I1 idx 3).
    j_med = jnp.stack([
        jnp.stack([j_w1[0, 0], j_x[0, 0],  j_w1[0, 1], j_x[0, 1]]),   # E0
        jnp.stack([j_x[0, 0],  j_w2[0, 0], j_x[0, 1],  j_w2[0, 1]]),   # E1
        jnp.stack([j_w1[1, 0], j_x[1, 0],  j_w1[1, 1], j_x[1, 1]]),    # I0
        jnp.stack([j_x[1, 0],  j_w2[1, 0], j_x[1, 1],  j_w2[1, 1]]),   # I1
    ])
    b_cL_med = exc(params["B_cL_med"])  # shape (n_med//2, n_c_L): cL → medulla E units
    # SNr → Medulla E units: inhibitory with a minimum magnitude (floored exc,
    # negated) so each weight stays ≤ -snr_med_floor and the tonic gate persists.
    snr_med_floor = config["snr_med_floor"]
    b_snr_med = -(exc(params["B_snr_med"]) + snr_med_floor)  # shape (n_med//2, n_snr)
    c_med = exc(params["C_med"])  # shape (n_output, 2): reads from E units only
    # Readout gain/bias (fall back to constants for legacy bundles without them).
    out_gain = jnp.asarray(params["out_gain"])
    out_bias = jnp.asarray(params["out_bias"])
    #rb = params["rb"]


    tau_c = config["tau_c"]
    tau_d1 = config["tau_d1"]
    tau_d2 = config["tau_d2"]
    tau_t = config["tau_t"]
    tau_snr = config["tau_snr"]
    tau_gpe = config["tau_gpe"]
    tau_stn = config["tau_stn"]
    tau_sc = config["tau_sc"]
    tau_snc = config["tau_snc"]
    tau_pka_fall = config["tau_pka_fall"]
    tau_pka_rise = config["tau_pka_rise"]
    # Gain on the DA→PKA drive (tanh(da_pka_gain * m_d * mean_snc)). mean_snc is
    # small, so the raw DA term barely registers against tonic adenosine; a gain
    # >1 lets phasic DA actually drive D1 PKA (and inhibit D2 PKA) within range.
    da_pka_gain = config["da_pka_gain"]

    # Tonic adenosine level: a single tunable scalar shared by both SPN
    # populations. Kept explicit so it can later be promoted to a dynamic
    # state (e.g. activity-dependent A1R/A2R modulation) without touching the
    # m_a1 / m_a2 connection weights. Sigmoid bound keeps k_a ∈ [floor, cap]
    # with a live gradient across the full range.
    k_a_floor = config["k_a_floor"]
    k_a_cap = config["k_a_cap"]
    k_a = k_a_floor + sigmoid(jnp.asarray(params["k_a"])) * (k_a_cap - k_a_floor)
    snc_pacer_min = config["snc_pacer_min"]
    snc_pacer_max = config["snc_pacer_max"]
    snr_pacer_max = config["snr_pacer_max"]
    snr_pacer_min = config["snr_pacer_min"]
    gpe_pacer_max = config["gpe_pacer_max"]
    gpe_pacer_min = config["gpe_pacer_min"]
    stn_pacer_max = config["stn_pacer_max"]

    snc_pacer = snc_pacer_min + sigmoid(p_snc) * (snc_pacer_max - snc_pacer_min)
    snr_pacer = snr_pacer_min + sigmoid(p_snr) * (snr_pacer_max - snr_pacer_min)
    gpe_pacer = gpe_pacer_min + sigmoid(p_gpe) * (gpe_pacer_max - gpe_pacer_min)
    stn_pacer = stn_pacer_max * jnp.ones(j_stn.shape[0])

    tau_med = config["tau_med"]

    n_steps = inputs.shape[0]
    n_d1_cells = j_d1.shape[0]
    n_d2_cells = j_d2.shape[0]
    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_d1_cells + n_d2_cells))

    def _step(carry, inp_stim_rng):
        (x_d1, x_d2,
         x_c_U, x_c_L, x_c_inh,
         x_t_exc, x_t_inh,
         x_snr, x_sc, x_gpe, x_stn, x_snc, pka_d1, pka_d2, x_med) = carry
        u_t, stim_t, step_rng = inp_stim_rng
        stim_d1 = stim_t[:n_d1_cells]
        stim_d2 = stim_t[n_d1_cells:]

        # add noise
        (rng_d1, rng_d2,
         rng_c_U, rng_c_L, rng_c_inh,
         rng_t_exc, rng_t_inh,
         rng_snr, rng_sc, rng_gpe, rng_stn, rng_snc, rng_med) = jr.split(step_rng, 13)
        coef = noise_std / jnp.sqrt(2.0 * tau_c)
        x_d1 = x_d1 + coef * jr.normal(rng_d1, x_d1.shape)
        x_d2 = x_d2 + coef * jr.normal(rng_d2, x_d2.shape)
        x_c_U = x_c_U + coef * jr.normal(rng_c_U, x_c_U.shape)
        x_c_L = x_c_L + coef * jr.normal(rng_c_L, x_c_L.shape)
        x_c_inh = x_c_inh + coef * jr.normal(rng_c_inh, x_c_inh.shape)
        x_t_exc = x_t_exc + coef * jr.normal(rng_t_exc, x_t_exc.shape)
        x_t_inh = x_t_inh + coef * jr.normal(rng_t_inh, x_t_inh.shape)
        x_snr = x_snr + coef * jr.normal(rng_snr, x_snr.shape)
        x_sc = x_sc + coef * jr.normal(rng_sc, x_sc.shape)
        x_gpe = x_gpe + coef * jr.normal(rng_gpe, x_gpe.shape)
        x_stn = x_stn + coef * jr.normal(rng_stn, x_stn.shape)
        coef_snc = noise_std / jnp.sqrt(2.0 * tau_snc)
        x_snc = x_snc + coef_snc * jr.normal(rng_snc, x_snc.shape)
        x_med = x_med + coef * jr.normal(rng_med, x_med.shape)

        # cortex: cU/cL excitatory PT populations + shared inhibitory c_inh.
        # Snapshot every recurrent/cross-pool drive before overwriting any pool.
        cU_rec = j_cU @ x_c_U + b_cL_cU @ x_c_L + j_ci_cU @ x_c_inh
        cL_rec = j_cL @ x_c_L + b_cU_cL @ x_c_U + j_ci_cL @ x_c_inh
        ci_rec = j_cU_ci @ x_c_U + j_cL_ci @ x_c_L + j_c_ii @ x_c_inh

        # cU: reciprocal thalamic input + cue.
        x_c_U = (1.0 - 1.0 / tau_c) * x_c_U + (1.0 / tau_c) * cU_rec
        x_c_U = x_c_U + (1.0 / tau_c) * b_t_cU @ x_t_exc
        x_c_U = x_c_U + (1.0 / tau_c) * b_cue_cU @ u_t
        x_c_U = nln(x_c_U)

        # cL: cue only (no direct thalamic input).
        x_c_L = (1.0 - 1.0 / tau_c) * x_c_L + (1.0 / tau_c) * cL_rec
        x_c_L = x_c_L + (1.0 / tau_c) * b_cue_cL @ u_t
        x_c_L = nln(x_c_L)

        # c_inh: recurrence + thalamic feedforward inhibition (no cue drive).
        x_c_inh = (1.0 - 1.0 / tau_c) * x_c_inh + (1.0 / tau_c) * ci_rec
        x_c_inh = x_c_inh + (1.0 / tau_c) * b_t_c_inh @ x_t_exc
        x_c_inh = nln(x_c_inh)

        # thalamus: same pre-step snapshot trick. SC drive uses prior-step x_sc.
        t_rec_to_exc = j_t_ee @ x_t_exc + j_t_ei @ x_t_inh
        t_rec_to_inh = j_t_ie @ x_t_exc + j_t_ii @ x_t_inh

        x_t_exc = (1.0 - 1.0 / tau_t) * x_t_exc + (1.0 / tau_t) * t_rec_to_exc
        x_t_exc = x_t_exc + (1.0 / tau_t) * b_cU_t_exc @ x_c_U
        x_t_exc = x_t_exc + (1.0 / tau_t) * b_snr_t_exc @ x_snr
        x_t_exc = x_t_exc + (1.0 / tau_t) * b_sc_t_exc @ x_sc
        x_t_exc = nln(x_t_exc)

        x_t_inh = (1.0 - 1.0 / tau_t) * x_t_inh + (1.0 / tau_t) * t_rec_to_inh
        x_t_inh = x_t_inh + (1.0 / tau_t) * b_cU_t_inh @ x_c_U
        x_t_inh = x_t_inh + (1.0 / tau_t) * b_snr_t_inh @ x_snr
        x_t_inh = x_t_inh + (1.0 / tau_t) * b_sc_t_inh @ x_sc
        x_t_inh = nln(x_t_inh)

        x_snc = (1.0 - (1.0 / tau_snc)) * x_snc
        x_snc = x_snc + (1.0 / tau_snc) * snc_pacer
        x_snc = x_snc + (1.0 / tau_snc) * b_stn_snc @ x_stn
        x_snc = x_snc + (1.0 / tau_snc) * b_cL_snc @ x_c_L
        x_snc = x_snc + (1.0 / tau_snc) * b_d1_snc @ x_d1
        x_snc = x_snc + (1.0 / tau_snc) * b_d2_snc @ x_d2
        x_snc = x_snc + (1.0 / tau_snc) * b_gpe_snc @ x_gpe
        x_snc = nln(x_snc)
        # SNc is broadcast as a single scalar to every SPN; each SPN scales it
        # by its own per-neuron gain m_d1[i] / m_d2[i].
        mean_snc = jnp.mean(x_snc)

        # PKA dynamics (leaky saturating integrator):
        # exponential leak with tau_pka_fall, rectified DA-driven production
        # (receptor activation can't make negative cAMP), tanh-saturating output.
        # Asymmetric timescales emerge from the gain ratio tau_fall/tau_rise.
        #   D1: D1R (DA) activates PKA; A1R (tonic adenosine) inhibits.
        pka_d1 = (1.0 - 1.0 / tau_pka_fall) * pka_d1
        pka_d1 = pka_d1 + (1.0 / tau_pka_rise) * jnp.maximum(da_pka_gain * m_d1 * mean_snc - m_a1 * k_a, 0)
        pka_d1 = nln(pka_d1)

        #   D2: A2R (tonic adenosine) activates PKA; D2R (DA) inhibits.
        pka_d2 = (1.0 - 1.0 / tau_pka_fall) * pka_d2
        pka_d2 = pka_d2 + (1.0 / tau_pka_rise) * jnp.maximum(m_a2 * k_a - da_pka_gain * m_d2 * mean_snc, 0)
        pka_d2 = nln(pka_d2)

        # PKA shifts rheobase in bg_nln: higher PKA → lower threshold → more excitable.
        x_d1 = (1.0 - (1.0 / tau_d1)) * x_d1
        x_d1 = x_d1 + (1.0 / tau_d1) * j_d1 @ x_d1
        x_d1 = x_d1 + (1.0 / tau_d1) * b_cU_d1 @ x_c_U
        x_d1 = x_d1 + (1.0 / tau_d1) * b_d2_d1 @ x_d2
        x_d1 = x_d1 + (1.0 / tau_d1) * stim_d1
        x_d1 = bg_nln(x_d1, pka_d1)

        x_d2 = (1.0 - (1.0 / tau_d2)) * x_d2
        x_d2 = x_d2 + (1.0 / tau_d2) * j_d2 @ x_d2
        x_d2 = x_d2 + (1.0 / tau_d2) * b_cU_d2 @ x_c_U
        x_d2 = x_d2 + (1.0 / tau_d2) * b_d1_d2 @ x_d1
        x_d2 = x_d2 + (1.0 / tau_d2) * stim_d2
        x_d2 = bg_nln(x_d2, pka_d2)

        x_gpe = (1.0 - (1.0 / tau_gpe)) * x_gpe #+ (1.0 / tau_gpe) * (j_gpe @ x_gpe)
        x_gpe = x_gpe + (1.0 / tau_gpe) * gpe_pacer
        x_gpe = x_gpe + (1.0 / tau_gpe) * b_d2_gpe @ x_d2
        x_gpe = x_gpe + (1.0 / tau_gpe) * b_cU_gpe @ x_c_U  # cU → GPe (exc)
        x_gpe = x_gpe + (1.0 / tau_gpe) * b_stn_gpe @ x_stn
        x_gpe = nln(x_gpe)

        # subthalamic nucleus: cortical hyperdirect drive + GPe inhibition.
        x_stn = (1.0 - (1.0 / tau_stn)) * x_stn
        x_stn = x_stn + (1.0 / tau_stn) * stn_pacer
        x_stn = x_stn + (1.0 / tau_stn) * (j_stn @ x_stn)
        x_stn = x_stn + (1.0 / tau_stn) * b_cU_stn @ x_c_U  # hyperdirect (cU)
        x_stn = x_stn + (1.0 / tau_stn) * b_cL_stn @ x_c_L  # hyperdirect (cL)
        x_stn = x_stn + (1.0 / tau_stn) * b_gpe_stn @ x_gpe
        x_stn = nln(x_stn)

        x_snr = (1.0 - (1.0 / tau_snr)) * x_snr
        x_snr = x_snr + (1.0 / tau_snr) * snr_pacer
        x_snr = x_snr + (1.0 / tau_snr) * b_d1_snr @ x_d1
        x_snr = x_snr + (1.0 / tau_snr) * b_gpe_snr @ x_gpe
        x_snr = x_snr + (1.0 / tau_snr) * b_stn_snr @ x_stn
        x_snr = nln(x_snr)

        # superior colliculus: free-sign recurrence, exc from cortex, inh from SNr.
        x_sc = (1.0 - (1.0 / tau_sc)) * x_sc + (1.0 / tau_sc) * (j_sc @ x_sc)
        x_sc = x_sc + (1.0 / tau_sc) * b_cU_sc @ x_c_U
        x_sc = x_sc + (1.0 / tau_sc) * b_snr_sc @ x_snr
        x_sc = nln(x_sc)

        # medulla: two E/I pairs with reciprocal coupling; cortical (exc), SC (exc)
        # and inhibitory SNr drive all target the E units only
        x_med = (1.0 - (1.0 / tau_med)) * x_med
        x_med = x_med + (1.0 / tau_med) * j_med @ x_med
        x_med = x_med.at[:2].add((1.0 / tau_med) * b_snr_med @ x_snr) # SNr → Medulla E units only
        x_med = x_med.at[:2].add((1.0 / tau_med) * b_cL_med @ x_c_L)
        x_med = x_med.at[:2].add((1.0 / tau_med) * b_sc_med @ x_sc)
        x_med = nln(x_med)


        # Smooth (sigmoid) readout so a silent medulla still yields a nonzero,
        # differentiable baseline output. A hard-rectified readout (max(0, tanh))
        # gives an exactly-zero gradient once the network collapses to silence,
        # which traps log_reward training in a permanent no-output state.
        y_t = sigmoid(out_gain * (c_med @ x_med[:2]) + out_bias)  # readout from E units only

        # Pack the full cortex/thalamus state ([exc..., inh...]) into the output
        # so downstream analysis code (get_brain_area, slope, ratios) still sees
        # a single Cortex / Thalamus array.
        x_c = jnp.concatenate([x_c_U, x_c_L, x_c_inh])
        x_t = jnp.concatenate([x_t_exc, x_t_inh])

        new_carry = (x_d1, x_d2,
                     x_c_U, x_c_L, x_c_inh,
                     x_t_exc, x_t_inh,
                     x_snr, x_sc, x_gpe, x_stn, x_snc, pka_d1, pka_d2, x_med)
        out = (y_t, x_c, x_d1, x_d2, x_snc, x_gpe, x_stn, x_snr, x_sc, x_t, pka_d1, pka_d2, x_med)
        return new_carry, out

    step_keys = jr.split(step_key, n_steps)
    _, (ys, xc, xd1, xd2, xsnc, xgpe, xstn, xsnr, xsc, xt, pkad1, pkad2, xmed) = lax.scan(
        _step,
        (x_d10, x_d20,
         x_c0_U, x_c0_L, x_c0_inh,
         x_t0_exc, x_t0_inh,
         x_snr0, x_sc0, x_gpe0, x_stn0, x_snc0, pka_d10, pka_d20, x_med0),
        (inputs, opto_stimulation, step_keys),
    )
    return ys, (xc, xd1, xd2, xsnc, xgpe, xstn, xsnr, xsc, xt, pkad1, pkad2, xmed)


batched_rnn = vmap(multiregion_rnn, in_axes=(None, None, 0, 0, 0))


def rnn_func(params, config, batch_inputs, opto_stim, rng_keys):
    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    if opto_stim is None:
        batch_stim = jnp.zeros((batch_inputs.shape[0], batch_inputs.shape[1], n_d1 + n_d2))
    else:
        batch_stim = opto_stim
    ys, xs = batched_rnn(params, config, batch_inputs, batch_stim, rng_keys)
    # State tuple ends with (..., pkad1, pkad2, xmed); expose PKA traces plus the
    # GPe trajectory for loss shaping (PKA asymmetry + GPe activity floor). The
    # full state tuple ``xs`` is returned last so the loss can enforce an
    # activity floor on *every* area (dead-area / inactivity penalty), not just GPe.
    gpe = xs[STATE_AREA_ORDER.index("GPe")]
    return ys, xs[-3], xs[-2], gpe, xs


def evaluate(params, config, all_inputs, noise_std=None, n_seeds=8):
    all_ys = []
    all_xs = []
    all_as = []

    eval_config = dict(config)
    if noise_std is not None:
        eval_config["noise_std"] = noise_std

    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    batch_stim = jnp.zeros((all_inputs.shape[0], all_inputs.shape[1], n_d1 + n_d2))

    for seed in range(n_seeds):
        rng_key = jr.PRNGKey(seed)
        rng_key, action_key = jr.split(rng_key)
        batch_rng_keys = jr.split(rng_key, all_inputs.shape[0])
        ys, xs = batched_rnn(params, eval_config, all_inputs, batch_stim, batch_rng_keys)
        actions = jr.bernoulli(action_key, p=ys).astype(ys.dtype)
        all_ys.append(ys)
        all_xs.append(xs)
        all_as.append(actions)

    all_ys = jnp.stack(all_ys)
    all_xs = [jnp.stack(parts) for parts in zip(*all_xs)]
    all_as = jnp.stack(all_as)
    return all_ys, all_xs, all_as


def get_brain_area(brain_area, xs, zs=None):
    """Return the state array for ``brain_area``.

    ``zs`` is accepted for backward compatibility with older call sites that
    passed a separate neuromodulator-state tuple; the current model packs all
    states (including PKA excitability and Medulla) into ``xs``, so ``zs`` is
    ignored.
    """
    return xs[STATE_AREA_ORDER.index(brain_area)]


def get_response_times_opto(ys, cue_start=0, dt=0.01, threshold=0.5, exclude_nan=True, rng_key=None):
    """First-crossing response times for opto-format outputs.

    Args:
        ys: array of shape (n_seeds, T, 1) or (n_seeds, T).
        cue_start: timestep to start looking for a response.
        dt: timestep duration in seconds.
        threshold: output threshold counted as a response.
        exclude_nan: if True, drop trials with no response; otherwise keep NaN.
        rng_key: if provided, sample Bernoulli actions from ``ys`` (treated as
            probabilities) before thresholding. The network is a stochastic
            policy whose probability output rarely exceeds 0.5 deterministically,
            so thresholding the raw probabilities yields empty CDFs. If None,
            threshold raw values directly (legacy behavior).

    Returns:
        1-D array of response times (seconds) when exclude_nan, else (n_seeds,).
    """
    if ys.ndim == 3:
        ys = ys[..., 0]
    if rng_key is not None:
        signal = jr.bernoulli(rng_key, p=jnp.clip(ys, 0.0, 1.0)).astype(ys.dtype)
    else:
        signal = ys
    post = signal[:, cue_start:]
    crossed = post > threshold
    has_resp = jnp.any(crossed, axis=1)
    first_idx = jnp.argmax(crossed, axis=1)
    rts = jnp.where(has_resp, first_idx.astype(jnp.float32) * dt, jnp.nan)
    if exclude_nan:
        return rts[~jnp.isnan(rts)]
    return rts


def get_d1_d2_ratio(all_xs, t_start, t_end, avg_time=True, remove_outliers=False):
    """D1 minus D2 mean activity over a time window.

    Args:
        all_xs: state list/tuple of arrays shaped (n_seeds, n_conditions, T, N).
        t_start, t_end: time-window bounds (timesteps).
        avg_time: if True, average over the window -> (n_seeds, n_conditions).
        remove_outliers: if True, NaN out per-condition outliers (z > 3).

    Returns:
        (n_seeds, n_conditions[, win]) array of D1-D2 activity differences.
    """
    d1 = get_brain_area("D1", all_xs)
    d2 = get_brain_area("D2", all_xs)
    d1m = jnp.mean(d1[..., t_start:t_end, :], axis=-1)
    d2m = jnp.mean(d2[..., t_start:t_end, :], axis=-1)
    ratio = d1m - d2m
    if avg_time:
        ratio = jnp.mean(ratio, axis=-1)
    if remove_outliers:
        z = jnp.abs((ratio - jnp.nanmean(ratio)) / (jnp.nanstd(ratio) + 1e-8))
        ratio = jnp.where(z > 3, jnp.nan, ratio)
    return ratio


def get_slope(all_xs, t_start, t_end, avg_neurons=True, remove_outliers=False):
    """Least-squares ramp slope of cortical activity over a time window.

    Returns:
        (n_seeds, n_conditions) array of slopes (per unit time-step).
    """
    cortex = get_brain_area("Cortex", all_xs)  # (n_seeds, n_conditions, T, N)
    seg = cortex[..., t_start:t_end, :]
    if avg_neurons:
        seg = jnp.mean(seg, axis=-1)  # (n_seeds, n_conditions, win)
    else:
        seg = jnp.mean(seg, axis=-1)
    win = seg.shape[-1]
    t = jnp.arange(win, dtype=seg.dtype)
    t_centered = t - jnp.mean(t)
    y_centered = seg - jnp.mean(seg, axis=-1, keepdims=True)
    num = jnp.sum(t_centered * y_centered, axis=-1)
    den = jnp.sum(t_centered ** 2)
    slope = num / (den + 1e-8)
    if remove_outliers:
        z = jnp.abs((slope - jnp.nanmean(slope)) / (jnp.nanstd(slope) + 1e-8))
        slope = jnp.where(z > 3, jnp.nan, slope)
    return slope
