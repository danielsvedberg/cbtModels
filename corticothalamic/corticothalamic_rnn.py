from pathlib import Path
import math
import sys

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

# Allow importing from repository root when running scripts from corticothalamic/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt

# Sign constraints (Dale's law): exc(w) = |w| >= 0, inh(w) = -|w| <= 0.
exc = stmt.exc
inh = stmt.inh


def _no_autapse(m):
    """Zero the diagonal of a square within-population recurrence (no self-synapse)."""
    return m * (1.0 - jnp.eye(m.shape[0], dtype=m.dtype))


def init_params(rng_key, n_input):
    """Dale's-law corticothalamic loop.

    Cortex = two excitatory PT-like pools (cU, cL) + a shared inhibitory pool (cI);
    thalamus = excitatory (t_exc) + inhibitory (t_inh) pools. EVERY population's
    outgoing connections are sign-constrained (exc/inh) — this is the sign-constrained
    testbed replacing the old free-sign 2-node model. External drives (cue input,
    readout) are left free-sign (they are not population outputs).

    State is exposed as two concatenated vectors for interface compatibility:
      x_ctx = [cU | cL | cI]   (n_ctx = n_c_U + n_c_L + n_c_inh)
      x_t   = [t_exc | t_inh]  (n_t   = n_t_exc + n_t_inh)
    """
    import config_script as _rootcfg
    _rc = _rootcfg.CORTICOTHALAMIC_RNN_CONFIG
    _rt = _rootcfg.CORTICOTHALAMIC_RUNTIME_CONFIG
    nU, nL, nI = _rc["n_c_U"], _rc["n_c_L"], _rc["n_c_inh"]
    nTe, nTi = _rc["n_t_exc"], _rc["n_t_inh"]
    n_output = _rc["n_output"]
    noise_std = _rc["noise_std"]
    g = _rc["g"]
    in_scale = _rt["in_scale"]
    out_scale = _rt["out_scale"]
    k = jr.split(rng_key, 24)

    def _mag(key, shape):
        # exponential synaptic MAGNITUDE ~ Exp(1) (mean 1); with each block's 1/fan_in
        # prefactor this is Exp with mean 1/fan_in (rate = fan_in) -- the one-signed (Dale)
        # analog of He init (keeps a positive fan-in sum's mean drive O(1)). Dale sign is
        # applied by the +/- prefactor below. exc blocks -> +, inh blocks -> -.
        return jr.exponential(key, shape)

    params = {
        # --- cortex recurrence (Dale): exc = +, inh = - (signs match the forward's
        # exc/inh clip; magnitudes log-normal) ---
        "J_cU": (g / math.sqrt(nU)) * _mag(k[0], (nU, nU)),   # cU->cU exc
        "J_cL": (g / math.sqrt(nL)) * _mag(k[1], (nL, nL)),   # cL->cL exc
        "B_cU_cL": (g / math.sqrt(nU)) * _mag(k[2], (nL, nU)),  # cU->cL exc
        "B_cL_cU": (g / math.sqrt(nL)) * _mag(k[3], (nU, nL)),  # cL->cU exc
        "J_cU_ci": (g / nU) * _mag(k[4], (nI, nU)),          # cU->cI exc
        "J_cL_ci": (g / nL) * _mag(k[5], (nI, nL)),          # cL->cI exc
        "J_ci_cU": -(g / nI) * _mag(k[6], (nU, nI)),          # cI->cU inh
        "J_ci_cL": -(g / nI) * _mag(k[7], (nL, nI)),          # cI->cL inh
        "J_c_ii": -(g / nI) * _mag(k[8], (nI, nI)),           # cI->cI inh
        # --- thalamus recurrence (Dale) ---
        "J_t_ee": (g / math.sqrt(nTe)) * _mag(k[9], (nTe, nTe)),  # t_exc->t_exc exc
        "J_t_ei": -(g / nTi) * _mag(k[10], (nTe, nTi)),       # t_inh->t_exc inh
        "J_t_ie": (g / nTe) * _mag(k[11], (nTi, nTe)),       # t_exc->t_inh exc
        "J_t_ii": -(g / nTi) * _mag(k[12], (nTi, nTi)),       # t_inh->t_inh inh
        # --- cross-area (Dale) ---
        "B_t_cU": (g / nTe) * _mag(k[13], (nU, nTe)),        # t_exc->cU exc
        "B_t_c_inh": (g / nTe) * _mag(k[14], (nI, nTe)),     # t_exc->cI EXC (drives ffwd inhib); forward applies _exc
        "B_cU_t_exc": (g / nU) * _mag(k[15], (nTe, nU)),     # cU->t_exc exc
        "B_cU_t_inh": (g / nU) * _mag(k[16], (nTi, nU)),     # cU->t_inh exc
        # --- external drives (constrained positive): log-normal magnitude ---
        "B_cue_cU": in_scale * _mag(k[17], (nU, n_input)),
        "B_cue_cL": in_scale * _mag(k[18], (nL, n_input)),
        "w_out_t": out_scale * _mag(k[19], (n_output, nTe)),
        # --- biases ---
        "b_cU": jnp.zeros((nU,)), "b_cL": jnp.zeros((nL,)), "b_cI": jnp.zeros((nI,)),
        "b_t_exc": jnp.zeros((nTe,)), "b_t_inh": jnp.zeros((nTi,)),
        "b_out": jnp.zeros((n_output,)),
    }
    if _rc["balanced_init"]:
        # Zero the self-recurrence diagonals FIRST so normalization targets the
        # actual no-autapse loop (realized rho then hits target exactly, not ~0.014
        # below). Then spectral-normalize the 17 loop blocks to target_rho.
        import loop_init as _loop_init
        for key in ("J_cU", "J_cL", "J_c_ii", "J_t_ee", "J_t_ii"):
            m = jnp.asarray(params[key])
            params[key] = m * (1.0 - jnp.eye(m.shape[0], dtype=m.dtype))
        params, rho0, rho1 = _loop_init.normalize_loop(
            params, nU, nL, nI, nTe, nTi, _rt["tau_ctx"], _rc["balanced_target_rho"])
        print(f"[balanced_init] corticothalamic loop rho(M): {rho0:.3f} -> {rho1:.3f} "
              f"(target {_rc['balanced_target_rho']})")
    config = {
        "x_ctx0": jnp.ones((nU + nL + nI,)) * _rt["x_init"],
        "x_t0": jnp.ones((nTe + nTi,)) * _rt["x_init"],
        "n_c_U": nU, "n_c_L": nL, "n_c_inh": nI, "n_t_exc": nTe, "n_t_inh": nTi,
        "tau_ctx": _rt["tau_ctx"],
        "tau_t": _rt["tau_t"],
        "noise_std": noise_std,
    }
    return params, config


def corticothalamic_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    nln = stmt.nln
    nU, nL, nI = config["n_c_U"], config["n_c_L"], config["n_c_inh"]
    nTe, nTi = config["n_t_exc"], config["n_t_inh"]

    # Dale sign application. 'dale_abs' (default): exc=|w|, inh=-|w| (hard nonneg
    # magnitude; convex, so symmetric ES noise inflates rho). 'signed': exc=w, inh=-w
    # applied to an init-positive signed magnitude -- LINEAR, so ES perturbs symmetrically
    # and the rho-inflation bias vanishes (a near-zero synapse may transiently flip sign).
    if config.get("weight_mode", "dale_abs") == "signed":
        _exc = lambda w: w
        _inh = lambda w: -w
    else:
        _exc, _inh = exc, inh

    # Sign-constrained effective weights (no_autapse on square self-recurrences).
    j_cU = _no_autapse(_exc(params["J_cU"]));  j_cL = _no_autapse(_exc(params["J_cL"]))
    b_cU_cL = _exc(params["B_cU_cL"]);  b_cL_cU = _exc(params["B_cL_cU"])
    j_cU_ci = _exc(params["J_cU_ci"]);  j_cL_ci = _exc(params["J_cL_ci"])
    j_ci_cU = _inh(params["J_ci_cU"]);  j_ci_cL = _inh(params["J_ci_cL"])
    j_c_ii = _no_autapse(_inh(params["J_c_ii"]))
    j_t_ee = _no_autapse(_exc(params["J_t_ee"]));  j_t_ii = _no_autapse(_inh(params["J_t_ii"]))
    j_t_ei = _inh(params["J_t_ei"]);  j_t_ie = _exc(params["J_t_ie"])
    b_t_cU = _exc(params["B_t_cU"]);  b_t_c_inh = _exc(params["B_t_c_inh"])
    b_cU_t_exc = _exc(params["B_cU_t_exc"]);  b_cU_t_inh = _exc(params["B_cU_t_inh"])
    b_cue_cU = _exc(params["B_cue_cU"]);  b_cue_cL = _exc(params["B_cue_cL"])  # excitatory (constrained +)
    w_out_t = _exc(params["w_out_t"])
    b_cU_b, b_cL_b, b_cI_b = params["b_cU"], params["b_cL"], params["b_cI"]
    b_te_b, b_ti_b, b_out = params["b_t_exc"], params["b_t_inh"], params["b_out"]

    tau_c = config["tau_ctx"]; tau_t = config["tau_t"]
    noise_std = config.get("noise_std", 0.0)

    n_steps = inputs.shape[0]
    n_ctx = nU + nL + nI; n_t = nTe + nTi

    x_ctx0 = nln(config["x_ctx0"]); x_t0 = nln(config["x_t0"])
    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_ctx + n_t))
    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)
    if noise_std > 0:
        x_ctx0 = nln(x_ctx0 + noise_std * jr.normal(init_key, x_ctx0.shape))
        x_t0 = nln(x_t0 + noise_std * jr.normal(jr.split(init_key)[0], x_t0.shape))

    def _step(carry, inp_stim_rng):
        x_ctx, x_t = carry
        u_t, stim_t, step_rng = inp_stim_rng
        if noise_std > 0:
            cc = noise_std / jnp.sqrt(2.0 * tau_c); ct = noise_std / jnp.sqrt(2.0 * tau_t)
            rc, rt = jr.split(step_rng)
            x_ctx = x_ctx + cc * jr.normal(rc, x_ctx.shape)
            x_t = x_t + ct * jr.normal(rt, x_t.shape)

        cU, cL, cI = x_ctx[:nU], x_ctx[nU:nU + nL], x_ctx[nU + nL:]
        te, ti = x_t[:nTe], x_t[nTe:]
        s_cU, s_cL, s_cI = stim_t[:nU], stim_t[nU:nU + nL], stim_t[nU + nL:n_ctx]
        s_te, s_ti = stim_t[n_ctx:n_ctx + nTe], stim_t[n_ctx + nTe:]

        # cortex (snapshot recurrent drives before overwriting). Synaptic currents are
        # summed linearly (no per-term nonlinearity), matching the other families.
        cU_rec = j_cU @ cU + b_cL_cU @ cL + j_ci_cU @ cI
        cL_rec = j_cL @ cL + b_cU_cL @ cU + j_ci_cL @ cI
        cI_rec = j_cU_ci @ cU + j_cL_ci @ cL + j_c_ii @ cI

        cU_n = nln((1.0 - 1.0 / tau_c) * cU + (1.0 / tau_c) * (cU_rec + b_t_cU @ te + b_cue_cU @ u_t + b_cU_b + s_cU))
        cL_n = nln((1.0 - 1.0 / tau_c) * cL + (1.0 / tau_c) * (cL_rec + b_cue_cL @ u_t + b_cL_b + s_cL))
        cI_n = nln((1.0 - 1.0 / tau_c) * cI + (1.0 / tau_c) * (cI_rec + b_t_c_inh @ te + b_cI_b + s_cI))

        # thalamus
        te_rec = j_t_ee @ te + j_t_ei @ ti
        ti_rec = j_t_ie @ te + j_t_ii @ ti
        te_n = nln((1.0 - 1.0 / tau_t) * te + (1.0 / tau_t) * (te_rec + b_cU_t_exc @ cU + b_te_b + s_te))
        ti_n = nln((1.0 - 1.0 / tau_t) * ti + (1.0 / tau_t) * (ti_rec + b_cU_t_inh @ cU + b_ti_b + s_ti))

        y_t = nln(w_out_t @ te_n + b_out)
        x_ctx_n = jnp.concatenate([cU_n, cL_n, cI_n])
        x_t_n = jnp.concatenate([te_n, ti_n])
        return (x_ctx_n, x_t_n), (y_t, x_ctx_n, x_t_n)

    step_keys = jr.split(step_key, n_steps)
    _, (ys, x_ctx_hist, x_t_hist) = lax.scan(_step, (x_ctx0, x_t0), (inputs, opto_stimulation, step_keys))
    return ys, (x_ctx_hist, x_t_hist)


# RL-compatible wrapper used by stmt.reinforce_loss / stmt.fit_rnn_reinforce.
def rnn_func(params, config, batch_inputs, opto_stim, rng_keys):
    n_ctx = config["x_ctx0"].shape[0]
    n_t = config["x_t0"].shape[0]
    if opto_stim is None:
        batch_stim = jnp.zeros((batch_inputs.shape[0], batch_inputs.shape[1], n_ctx + n_t))
    else:
        batch_stim = opto_stim
    ys, (x_ctx, x_t) = batched_rnn(params, config, batch_inputs, batch_stim, rng_keys)
    xs = jnp.concatenate([x_ctx, x_t], axis=-1)
    return ys, xs, None


batched_rnn = vmap(corticothalamic_rnn, in_axes=(None, None, 0, 0, 0))


def evaluate(params, config, all_inputs, noise_std=None, n_seeds=8):
    all_ys, all_x_ctx, all_x_t, all_as = [], [], [], []
    eval_config = dict(config)
    if noise_std is not None:
        eval_config["noise_std"] = noise_std
    n_ctx = eval_config["x_ctx0"].shape[0]
    n_t = eval_config["x_t0"].shape[0]
    batch_stim = jnp.zeros((all_inputs.shape[0], all_inputs.shape[1], n_ctx + n_t))
    for seed in range(n_seeds):
        rng_key = jr.PRNGKey(seed)
        rng_key, action_key = jr.split(rng_key)
        batch_rng_keys = jr.split(rng_key, all_inputs.shape[0])
        ys, (x_ctx, x_t) = batched_rnn(params, eval_config, all_inputs, batch_stim, batch_rng_keys)
        actions = jr.bernoulli(action_key, p=ys).astype(ys.dtype)
        all_ys.append(ys); all_x_ctx.append(x_ctx); all_x_t.append(x_t); all_as.append(actions)
    return jnp.stack(all_ys), jnp.stack(all_x_ctx), jnp.stack(all_x_t), jnp.stack(all_as)
