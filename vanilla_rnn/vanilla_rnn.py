from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import vanilla_config as cfg
from jax import lax, vmap

# Allow importing from repository root when running scripts from vanilla_rnn/.
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt


def init_params(
    rng_key,
    n_hidden=64,
    n_input=1,
    n_output=1,
    rec_scale=0.15,
    in_scale=0.25,
    out_scale=0.5,
    noise_std=0.01,
):
    k1, k2, k3 = jr.split(rng_key, 3)
    params = {
        "w_rec": rec_scale * jr.normal(k1, (n_hidden, n_hidden)),
        "w_in": in_scale * jr.normal(k2, (n_hidden, n_input)),
        "w_out": out_scale * jr.normal(k3, (n_output, n_hidden)),
        "b_h": jnp.zeros((n_hidden,)),
        "b_out": jnp.zeros((n_output,)),
    }
    config = {
        "noise_std": noise_std,
        "x0": jnp.ones((n_hidden,))*0.1,
        "tau": 20.0,
    }
    return params, config


def vanilla_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    w_rec = params["w_rec"]
    w_in = params["w_in"]
    w_out = params["w_out"]
    b_h = params["b_h"]
    b_out = params["b_out"]
    tau = config.get("tau", 20.0)
    x0 = stmt.nln(config["x0"])
    noise_std = config["noise_std"]



    n_steps = inputs.shape[0]
    n_hidden = x0.shape[0]

    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_hidden))

    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    if noise_std > 0:
        x0 = x0 + noise_std * jr.normal(init_key, x0.shape)
        x0 = stmt.nln(x0)

    def _step(x, inp_stim_rng):
        u_t, stim_t, step_rng = inp_stim_rng

        if noise_std > 0:
            coef = noise_std / jnp.sqrt(2.0 * tau)
            x = x + coef * jr.normal(step_rng, x.shape)

        x = (1.0 - 1.0 / tau) * x + (1.0 / tau) * (w_rec @ x)
        x = x + (1.0 / tau) * (w_in @ u_t)
        x = x + (1.0 / tau) * b_h
        x = x + (1.0 / tau) * stim_t
        x = stmt.nln(x)

        y_t = stmt.nln(w_out @ x + b_out)
        return x, (y_t, x)

    step_keys = jr.split(step_key, n_steps)
    _, (ys, xs) = lax.scan(_step, x0, (inputs, opto_stimulation, step_keys))
    return ys, xs


batched_rnn = vmap(vanilla_rnn, in_axes=(None, None, 0, 0, 0))

def evaluate(params, config, all_inputs, noise_std=None, n_seeds=8):
    return stmt.evaluate(batched_rnn, params, config, all_inputs, noise_std=noise_std, n_seeds=n_seeds)
