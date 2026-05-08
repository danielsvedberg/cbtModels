import math
from pathlib import Path
import sys

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.nn import sigmoid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt


def exc(w):
    return stmt.exc(w)


def inh(w):
    return stmt.inh(w)


def nln(x):
    return stmt.nln(x)


def bg_nln(x):
    return stmt.bg_nln(x)


# Canonical state tuple order used across CBT loop code.
STATE_VAR_ORDER = (
    "x_c", "x_d1", "x_d2", "x_snc", "x_gpe", "x_stn", "x_snr", "x_t", "e_d1", "e_d2",
)
STATE_AREA_ORDER = (
    "Cortex", "D1", "D2", "SNc", "GPe", "STN", "SNr", "Thalamus", "eD1", "eD2",
)


def init_params(
    rng_key,
    n_c=16,
    n_d1=6,
    n_d2=6,
    n_snc=4,
    n_snr=6,
    n_gpe=6,
    n_stn=6,
    n_t=10,
    n_input=1,
    n_output=1,
    g_bg=0.5,
    g_nm=0.5,
    noise_std=0.01,
):
    """Initialize multiregion CBT loop params and runtime config."""
    skeys = jr.split(rng_key, 30)

    params = {
        "J_c": (g_bg / math.sqrt(n_c)) * jr.normal(skeys[2], (n_c, n_c)),
        "J_d1": (g_bg / math.sqrt(n_d1)) * jr.normal(skeys[0], (n_d1, n_d1)),
        "J_d2": (g_bg / math.sqrt(n_d2)) * jr.normal(skeys[8], (n_d2, n_d2)),
        # Pacemaker vectors (no within-area recurrence for SNc/SNr).
        "P_snc": (g_nm / math.sqrt(n_snc)) * jr.normal(skeys[7], (n_snc,)),
        "J_gpe": (g_bg / math.sqrt(n_gpe)) * jr.normal(skeys[19], (n_gpe, n_gpe)),
        "J_stn": (g_bg / math.sqrt(n_stn)) * jr.normal(skeys[20], (n_stn, n_stn)),
        "P_snr": (g_bg / math.sqrt(n_snr)) * jr.normal(skeys[18], (n_snr,)),
        "J_t": (g_bg / math.sqrt(n_t)) * jr.normal(skeys[5], (n_t, n_t)),
        "B_cue_c": (1 / math.sqrt(n_input)) * jr.normal(skeys[3], (n_c, n_input)),
        "B_t_c": (g_bg / math.sqrt(n_t)) * jr.normal(skeys[4], (n_c, n_t)),
        "B_c_t": (1 / math.sqrt(n_c)) * jr.normal(skeys[29], (n_t, n_c)),
        "B_c_d1": (g_bg / math.sqrt(n_c)) * jr.normal(skeys[1], (n_d1, n_c)),
        "B_c_d2": (g_bg / math.sqrt(n_c)) * jr.normal(skeys[1], (n_d2, n_c)),
        "B_stn_snc": (1 / math.sqrt(n_stn)) * jr.normal(skeys[9], (n_snc, n_stn)),
        "B_d1_snc": (1 / math.sqrt(n_d1)) * jr.normal(skeys[17], (n_snc, n_d1)),
        "B_d2_snc": (1 / math.sqrt(n_d2)) * jr.normal(skeys[17], (n_snc, n_d2)),
        "B_d1_snr": (1 / math.sqrt(n_d1)) * jr.normal(skeys[22], (n_snr, n_d1)),
        "B_d2_gpe": (1 / math.sqrt(n_d2)) * jr.normal(skeys[24], (n_gpe, n_d2)),
        "B_stn_gpe": (1 / math.sqrt(n_stn)) * jr.normal(skeys[27], (n_gpe, n_stn)),
        "B_gpe_stn": (1 / math.sqrt(n_gpe)) * jr.normal(skeys[25], (n_stn, n_gpe)),
        "B_c_stn": (1 / math.sqrt(n_c)) * jr.normal(skeys[26], (n_stn, n_c)),
        "B_stn_snr": (1 / math.sqrt(n_stn)) * jr.normal(skeys[23], (n_snr, n_stn)),
        "B_snr_t": (1 / math.sqrt(n_snr)) * jr.normal(skeys[6], (n_t, n_snr)),
        "m_d1": (1 / math.sqrt(n_snc)) * jr.normal(skeys[10], (1, n_snc)),
        "m_d2": (1 / math.sqrt(n_snc)) * jr.normal(skeys[11], (1, n_snc)),
        "C": (1 / math.sqrt(n_t)) * jr.normal(skeys[15], (n_output, n_t)),
        "rb": jnp.abs((1 / math.sqrt(n_t)) * jr.normal(skeys[16], (n_output,))),
    }

    config = {
        "x_c0": jnp.ones((n_c,)) * 0.1,
        "x_d10": jnp.ones((n_d1,)) * 0.1,
        "x_d20": jnp.ones((n_d2,)) * 0.1,
        "x_snc0": jnp.ones((n_snc,)) * 0.1,
        "x_gpe0": jnp.ones((n_gpe,)) * 0.1,
        "x_stn0": jnp.ones((n_stn,)) * 0.1,
        "x_snr0": jnp.ones((n_snr,)) * 0.1,
        "x_t0": jnp.ones((n_t,)) * 0.01,
        "e_d10": 0.1,
        "e_d20": 0.5,
        "tau_c": 10.0,
        "tau_d1": 10.0,
        "tau_d2": 10.0,
        "tau_t": 10.0,
        "tau_snr": 10.0,
        "tau_gpe": 10.0,
        "tau_stn": 10.0,
        "tau_snc": 40.0,
        "tau_d1_rise": 50.0,
        "tau_d1_fall": 2000.0,
        "tau_d2_rise": 50.0,
        "tau_d2_fall": 2000.0,
        "snc_pacer_max": 0.1,
        "stn_pacer_max": 0.1,
        "snr_pacer_max": 0.8,
        "noise_std": noise_std,
    }
    return params, config


def multiregion_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    x_c0 = config["x_c0"]
    x_d10 = config["x_d10"]
    x_d20 = config["x_d20"]
    x_snc0 = config["x_snc0"]
    x_gpe0 = config["x_gpe0"]
    x_stn0 = config["x_stn0"]
    x_snr0 = config["x_snr0"]
    x_t0 = config["x_t0"]
    e_d10 = jnp.asarray(config.get("e_d10", 0.1))
    e_d20 = jnp.asarray(config.get("e_d20", 0.6))

    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    noise_std = jnp.asarray(config.get("noise_std", 0.0))
    x_c0 = nln(x_c0 + noise_std * jr.normal(init_key, x_c0.shape))
    x_d10 = nln(x_d10 + noise_std * jr.normal(init_key, x_d10.shape))
    x_d20 = nln(x_d20 + noise_std * jr.normal(init_key, x_d20.shape))
    x_snc0 = nln(x_snc0 + noise_std * jr.normal(init_key, x_snc0.shape))
    x_gpe0 = nln(x_gpe0 + noise_std * jr.normal(init_key, x_gpe0.shape))
    x_stn0 = nln(x_stn0 + noise_std * jr.normal(init_key, x_stn0.shape))
    x_snr0 = nln(x_snr0 + noise_std * jr.normal(init_key, x_snr0.shape))
    x_t0 = nln(x_t0 + noise_std * jr.normal(init_key, x_t0.shape))

    j_c = params["J_c"]
    j_d1 = inh(params["J_d1"])
    j_d2 = inh(params["J_d2"])
    j_gpe = inh(params["J_gpe"])
    j_stn = exc(params["J_stn"])
    j_t = params["J_t"]

    p_snr = exc(params["P_snr"])
    p_snc = exc(params["P_snc"])

    b_cue_c = params["B_cue_c"]
    b_t_c = params["B_t_c"]
    b_c_t = params["B_c_t"]
    b_c_d1 = params["B_c_d1"]
    b_c_d2 = params["B_c_d2"]
    b_stn_snc = exc(params["B_stn_snc"])
    b_d1_snc = inh(params["B_d1_snc"])
    b_d2_snc = inh(params["B_d2_snc"])
    b_d1_snr = inh(params["B_d1_snr"])
    b_d2_gpe = inh(params["B_d2_gpe"])
    b_stn_gpe = exc(params["B_stn_gpe"])
    b_gpe_stn = inh(params["B_gpe_stn"])
    b_c_stn = params["B_c_stn"]
    b_stn_snr = exc(params["B_stn_snr"])
    b_snr_t = inh(params["B_snr_t"])
    m_d1 = params["m_d1"]
    m_d2 = params["m_d2"]
    c = params["C"]
    rb = params["rb"]


    tau_c = config.get("tau_c", 10.0)
    tau_d1 = config.get("tau_d1", tau_c)
    tau_d2 = config.get("tau_d2", tau_c)
    tau_t = config.get("tau_t", tau_c)
    tau_snr = config.get("tau_snr", tau_c)
    tau_gpe = config.get("tau_gpe", tau_c)
    tau_stn = config.get("tau_stn", tau_c)
    tau_snc = config.get("tau_snc", tau_c)
    tau_ed1_rise = config.get("tau_d1_rise", 40.0)
    tau_ed1_fall = config.get("tau_d1_fall", 20000.0)
    tau_ed2_rise = config.get("tau_d2_rise", 40.0)
    tau_ed2_fall = config.get("tau_d2_fall", 20000.0)
    snc_pacer_max = config.get("snc_pacer_max", 0.1)
    stn_pacer_max = config.get("stn_pacer_max", 0.1)
    snr_pacer_max = config.get("snr_pacer_max", 0.8)

    snc_pacer = sigmoid(p_snc) * snc_pacer_max
    stn_pacer = stn_pacer_max * jnp.ones(j_stn.shape[0])
    snr_pacer = sigmoid(p_snr) * snr_pacer_max

    n_steps = inputs.shape[0]
    n_d1_cells = j_d1.shape[0]
    n_d2_cells = j_d2.shape[0]
    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_d1_cells + n_d2_cells))

    def _step(carry, inp_stim_rng):
        x_d1, x_d2, x_c, x_t, x_snr, x_gpe, x_stn, x_snc, e_d1, e_d2 = carry
        u_t, stim_t, step_rng = inp_stim_rng
        stim_d1 = stim_t[:n_d1_cells]
        stim_d2 = stim_t[n_d1_cells:]

        # add noise
        coef = noise_std / jnp.sqrt(2.0 * tau_c)
        x_d1 = x_d1 + coef * jr.normal(step_rng, x_d1.shape)
        x_d2 = x_d2 + coef * jr.normal(step_rng, x_d2.shape)
        x_c = x_c + coef * jr.normal(step_rng, x_c.shape)
        x_t = x_t + coef * jr.normal(step_rng, x_t.shape)
        x_snr = x_snr + coef * jr.normal(step_rng, x_snr.shape)
        x_gpe = x_gpe + coef * jr.normal(step_rng, x_gpe.shape)
        x_stn = x_stn + coef * jr.normal(step_rng, x_stn.shape)
        coef_snc = noise_std / jnp.sqrt(2.0 * tau_snc)
        x_snc = x_snc + coef_snc * jr.normal(step_rng, x_snc.shape)

        # cortex
        x_c = (1.0 - (1.0 / tau_c)) * x_c + (1.0 / tau_c) * (j_c @ x_c)
        x_c = x_c + (1.0 / tau_c) * (b_t_c @ x_t)
        x_c = x_c + (1.0 / tau_c) * (b_cue_c @ u_t)
        x_c = nln(x_c)

        x_t = (1.0 - (1.0 / tau_t)) * x_t + (1.0 / tau_t) * (j_t @ x_t)
        x_t = x_t + (1.0 / tau_t) * (b_c_t @ x_c)
        x_t = x_t + (1.0 / tau_t) * (b_snr_t @ x_snr)
        x_t = nln(x_t)

        x_snc = (1.0 - (1.0 / tau_snc)) * x_snc
        x_snc = x_snc + (1.0 / tau_snc) * snc_pacer
        x_snc = x_snc + (1.0 / tau_snc) * (b_stn_snc @ x_stn)
        x_snc = x_snc + (1.0 / tau_snc) * (b_d1_snc @ x_d1)
        x_snc = x_snc + (1.0 / tau_snc) * (b_d2_snc @ x_d2)
        x_snc = nln(x_snc)

        g_d1 = jnp.ravel(m_d1 @ x_snc)[0]
        g_d2 = jnp.ravel(m_d2 @ x_snc)[0]
        e_d1 = nln((1.0 - (1.0 / tau_ed1_fall)) * e_d1 + (1.0 / tau_ed1_rise) * g_d1)
        e_d2 = nln((1.0 + (1.0 / tau_ed2_rise)) * e_d2 - (1.0 / tau_ed2_fall) * g_d2)

        x_d1 = (1.0 - (1.0 / tau_d1)) * x_d1
        x_d1 = x_d1 + (1.0 / tau_d1) * e_d1 * (b_c_d1 @ x_c)
        x_d1 = x_d1 + (1.0 / tau_d1) * (1.0 - e_d1) * (j_d1 @ x_d1)
        x_d1 = x_d1 + (1.0 / tau_d1) * stim_d1
        x_d1 = bg_nln(x_d1)

        x_d2 = (1.0 - (1.0 / tau_d2)) * x_d2
        x_d2 = x_d2 + (1.0 / tau_d2) * e_d2 * (b_c_d2 @ x_c)
        x_d2 = x_d2 + (1.0 / tau_d2) * (1.0 - e_d2) * (j_d2 @ x_d2)
        x_d2 = x_d2 + (1.0 / tau_d2) * stim_d2
        x_d2 = bg_nln(x_d2)

        x_gpe = (1.0 - (1.0 / tau_gpe)) * x_gpe + (1.0 / tau_gpe) * (j_gpe @ x_gpe)
        x_gpe = x_gpe + (1.0 / tau_gpe) * (b_d2_gpe @ x_d2)
        x_gpe = x_gpe + (1.0 / tau_gpe) * (b_stn_gpe @ x_stn)
        x_gpe = nln(x_gpe)

        x_stn = (1.0 - (1.0 / tau_stn)) * x_stn
        x_stn = x_stn + (1.0 / tau_stn) * stn_pacer
        x_stn = x_stn + (1.0 / tau_stn) * (j_stn @ x_stn)
        x_stn = x_stn + (1.0 / tau_stn) * (b_gpe_stn @ x_gpe)
        x_stn = x_stn + (1.0 / tau_stn) * (b_c_stn @ x_c)
        x_stn = nln(x_stn)

        x_snr = (1.0 - (1.0 / tau_snr)) * x_snr
        x_snr = x_snr + (1.0 / tau_snr) * snr_pacer
        x_snr = x_snr + (1.0 / tau_snr) * (b_d1_snr @ x_d1)
        x_snr = x_snr + (1.0 / tau_snr) * (b_stn_snr @ x_stn)
        x_snr = nln(x_snr)



        y_t = sigmoid((4*(c @ x_t)) - 0.5)# + rb)

        new_carry = (x_d1, x_d2, x_c, x_t, x_snr, x_gpe, x_stn, x_snc, e_d1, e_d2)
        out = (y_t, x_c, x_d1, x_d2, x_snc, x_gpe, x_stn, x_snr, x_t, e_d1, e_d2)
        return new_carry, out

    step_keys = jr.split(step_key, n_steps)
    _, (ys, xc, xd1, xd2, xsnc, xgpe, xstn, xsnr, xt, ed1, ed2) = lax.scan(
        _step,
        (x_d10, x_d20, x_c0, x_t0, x_snr0, x_gpe0, x_stn0, x_snc0, e_d10, e_d20),
        (inputs, opto_stimulation, step_keys),
    )
    return ys, (xc, xd1, xd2, xsnc, xgpe, xstn, xsnr, xt, ed1, ed2)


batched_rnn = vmap(multiregion_rnn, in_axes=(None, None, 0, 0, 0))


def rnn_func(params, config, batch_inputs, opto_stim, rng_keys):
    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    if opto_stim is None:
        batch_stim = jnp.zeros((batch_inputs.shape[0], batch_inputs.shape[1], n_d1 + n_d2))
    else:
        batch_stim = opto_stim
    ys, _ = batched_rnn(params, config, batch_inputs, batch_stim, rng_keys)
    return ys, None, None


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


def get_brain_area(brain_area, xs):
    return xs[STATE_AREA_ORDER.index(brain_area)]

