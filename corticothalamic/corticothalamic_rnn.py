from pathlib import Path
import sys

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

# Allow importing from repository root when running scripts from corticothalamic/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt


def init_params(
    rng_key,
    n_ctx=20,
    n_t=20,
    n_input=1,
    n_output=1,
    rec_scale=0.15,
    cross_scale=0.15,
    in_scale=0.25,
    out_scale=0.2,
    noise_std=0.01,
):
    k1, k2, k3, k4, k5, k6 = jr.split(rng_key, 6)
    params = {
        # Within-area recurrence
        "w_ctx_ctx": rec_scale * jr.normal(k1, (n_ctx, n_ctx)),
        "w_t_t": rec_scale * jr.normal(k2, (n_t, n_t)),
        # Cross-area projections (open sign)
        "w_ctx_t": cross_scale * jr.normal(k3, (n_t, n_ctx)),
        "w_t_ctx": cross_scale * jr.normal(k4, (n_ctx, n_t)),
        # Input/readout projections
        "w_in_ctx": in_scale * jr.normal(k5, (n_ctx, n_input)),
        "w_out_t": out_scale * jr.normal(k6, (n_output, n_t)),
        # Biases
        "b_ctx": jnp.zeros((n_ctx,)),
        "b_t": jnp.zeros((n_t,)),
        "b_out": jnp.zeros((n_output,)),
    }
    config = {
        "x_ctx0": jnp.ones((n_ctx,)) * 0.1,
        "x_t0": jnp.ones((n_t,)) * 0.1,
        "tau_ctx": 20.0,
        "tau_t": 20.0,
        "noise_std": noise_std,
    }
    return params, config


def corticothalamic_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    x_ctx0 = stmt.nln(config["x_ctx0"])
    x_t0 = stmt.nln(config["x_t0"])

    w_ctx_ctx = params["w_ctx_ctx"]
    w_t_t = params["w_t_t"]
    w_ctx_t = params["w_ctx_t"]
    w_t_ctx = params["w_t_ctx"]
    w_in_ctx = params["w_in_ctx"]
    w_out_t = params["w_out_t"]

    b_ctx = params["b_ctx"]
    b_t = params["b_t"]
    b_out = params["b_out"]

    tau_ctx = config.get("tau_ctx", 20.0)
    tau_t = config.get("tau_t", 20.0)
    noise_std = config.get("noise_std", 0.0)

    n_steps = inputs.shape[0]
    n_ctx = x_ctx0.shape[0]
    n_t = x_t0.shape[0]

    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_ctx + n_t))

    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    if noise_std > 0:
        x_ctx0 = stmt.nln(x_ctx0 + noise_std * jr.normal(init_key, x_ctx0.shape))
        x_t0 = stmt.nln(x_t0 + noise_std * jr.normal(init_key, x_t0.shape))

    def _step(carry, inp_stim_rng):
        x_ctx, x_t = carry
        u_t, stim_t, step_rng = inp_stim_rng

        stim_ctx = stim_t[:n_ctx]
        stim_t_area = stim_t[n_ctx:]

        if noise_std > 0:
            coef_ctx = noise_std / jnp.sqrt(2.0 * tau_ctx)
            coef_t = noise_std / jnp.sqrt(2.0 * tau_t)
            x_ctx = x_ctx + coef_ctx * jr.normal(step_rng, x_ctx.shape)
            x_t = x_t + coef_t * jr.normal(step_rng, x_t.shape)

        # Cortex receives cue input and thalamic feedback.
        x_ctx = (1.0 - 1.0 / tau_ctx) * x_ctx + (1.0 / tau_ctx) * (w_ctx_ctx @ x_ctx)
        x_ctx = x_ctx + (1.0 / tau_ctx) * (w_t_ctx @ x_t)
        x_ctx = x_ctx + (1.0 / tau_ctx) * (w_in_ctx @ u_t)
        x_ctx = x_ctx + (1.0 / tau_ctx) * b_ctx
        x_ctx = x_ctx + (1.0 / tau_ctx) * stim_ctx
        x_ctx = stmt.nln(x_ctx)

        # Thalamus receives cortical drive.
        x_t = (1.0 - 1.0 / tau_t) * x_t + (1.0 / tau_t) * (w_t_t @ x_t)
        x_t = x_t + (1.0 / tau_t) * (w_ctx_t @ x_ctx)
        x_t = x_t + (1.0 / tau_t) * b_t
        x_t = x_t + (1.0 / tau_t) * stim_t_area
        x_t = stmt.nln(x_t)

        y_t = stmt.nln(w_out_t @ x_t + b_out)

        new_carry = (x_ctx, x_t)
        out = (y_t, x_ctx, x_t)
        return new_carry, out

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
    all_ys = []
    all_x_ctx = []
    all_x_t = []
    all_as = []
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
        all_ys.append(ys)
        all_x_ctx.append(x_ctx)
        all_x_t.append(x_t)
        all_as.append(actions)

    return jnp.stack(all_ys), jnp.stack(all_x_ctx), jnp.stack(all_x_t), jnp.stack(all_as)
