import os
import pickle as pkl
from pathlib import Path
import sys
import jax.numpy as jnp
import jax.random as jr
import cbt_loop.cbt_rnn as cbtl
import cbt_loop.config_script as cfg
import self_timed_movement_task as stmt
import cbt_loop.plotting_functions as pf


def _build_test_inputs(task_cfg, starts):
    kwargs = {
        "T_start": starts,
        "T_cue": task_cfg["t_cue"],
        "T_wait": task_cfg["t_wait"],
        "T_movement": task_cfg["t_movement"],
        "T": task_cfg["t_total"],
    }
    mode = task_cfg.get("task_mode", "self_timed")
    if mode == "hybrid":
        return stmt.hybrid_stmt(**kwargs)
    if mode == "pavlovian":
        return stmt.pavlovian_stmt(**kwargs)
    return stmt.self_timed_movement_task(**kwargs)


def _response_times_from_actions(actions, starts, t_cue, threshold=0.5):
    n_seeds, n_conditions = actions.shape[:2]
    response_times = jnp.full((n_seeds, n_conditions), jnp.nan)
    for seed_idx in range(n_seeds):
        for cond_idx in range(n_conditions):
            cue_end = starts[cond_idx] + t_cue
            post = actions[seed_idx, cond_idx, cue_end:, 0]
            response_idx = jnp.argmax(post > threshold)
            if post[response_idx] > threshold:
                response_times = response_times.at[seed_idx, cond_idx].set(response_idx * 0.01)
    return response_times


def main():
    params_path = cfg.params_path()
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}. Run training_script.py first.")

    with params_path.open("rb") as f:
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

    starts = cfg.TEST_CONFIG["start_t"]
    inputs, targets, masks = _build_test_inputs(cfg.TASK_CONFIG, starts)

    all_ys, all_xs, all_actions = cbtl.evaluate(
        params,
        config,
        inputs,
        noise_std=cfg.TEST_CONFIG["noise_std"],
        n_seeds=cfg.TEST_CONFIG["n_seeds"],
    )

    pf.plot_cue_algn_activity(all_xs, ys=all_ys)

    # Reward proxy: any sampled first-response inside target window.
    target_2d = targets[..., 0]
    action_2d = all_actions[..., 0]
    in_target = target_2d[None, ...] > 0.5
    success = jnp.any((action_2d > 0.5) & in_target, axis=2)

    response_times = _response_times_from_actions(all_actions, starts, cfg.TASK_CONFIG["t_cue"])
    valid_rts = response_times[~jnp.isnan(response_times)]

    print("ys shape:", all_ys.shape)
    print("actions shape:", all_actions.shape)
    print("num state arrays:", len(all_xs))
    print("state[0] shape:", all_xs[0].shape)
    print("mean success:", float(jnp.mean(success)))
    if valid_rts.size > 0:
        print("mean response time (s):", float(jnp.mean(valid_rts)))
    else:
        print("mean response time (s): NaN (no responses)")


if __name__ == "__main__":
    main()
