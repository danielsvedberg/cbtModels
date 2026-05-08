import pickle as pkl

import jax.numpy as jnp
import jax.random as jr

import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt


def main():
    with cfg.params_path().open("rb") as f:
        bundle = pkl.load(f)
    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        params = bundle["params"]
        config = bundle["config"]
    else:
        params = bundle
        _, config = cbtl.init_params(
            jr.PRNGKey(0),
            n_c=params["J_c"].shape[0],
            n_d1=params["J_d1"].shape[0],
            n_d2=params["J_d2"].shape[0],
            n_snc=params["P_snc"].shape[0],
            n_snr=params["P_snr"].shape[0],
            n_gpe=params["J_gpe"].shape[0],
            n_stn=params["J_stn"].shape[0],
            n_t=params["J_t"].shape[0],
            n_input=1,
            n_output=1,
            noise_std=cfg.RNN_CONFIG["noise_std"],
        )

    # Single condition around opto cue for quick perturbation checks.
    starts = jnp.array([cfg.opto_tstart])
    inputs, _, _ = stmt.self_timed_movement_task(
        starts,
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )

    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    stim_temporal = (
        (jnp.arange(cfg.TASK_CONFIG["t_total"]) >= cfg.opto_start)
        & (jnp.arange(cfg.TASK_CONFIG["t_total"]) < cfg.opto_end)
    )[:, None]

    print("Running opto sweep...")
    for stim_vec, label, strength in zip(cfg.spatial_stim_list, cfg.stim_labels, cfg.stim_strengths):
        stim_profile = stim_temporal * stim_vec[None, :]
        batch_stim = jnp.repeat(stim_profile[None, :, :], cfg.n_opto_seeds, axis=0)
        batch_inputs = jnp.repeat(inputs, cfg.n_opto_seeds, axis=0)
        rng_keys = jr.split(jr.PRNGKey(0), cfg.n_opto_seeds)

        ys, _ = cbtl.batched_rnn(params, config, batch_inputs, batch_stim, rng_keys)
        actions = jr.bernoulli(jr.PRNGKey(1), p=ys).astype(ys.dtype)
        responded = jnp.any(actions[:, cfg.opto_tstart:, 0] > 0.5, axis=1)

        print(f"{label:8s} | strength={float(strength): .3f} | response_rate={float(jnp.mean(responded)):.3f}")


if __name__ == "__main__":
    main()
