import pickle as pkl

import optax
import jax.random as jr

import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt


def _build_task(task_cfg):
    kwargs = {
        "T_start": task_cfg["t_start"],
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


def main():
    params_path = cfg.params_path()
    print(f"Loading existing parameters from {params_path}...")
    try:
        with params_path.open("rb") as f:
            bundle = pkl.load(f)
    except FileNotFoundError:
        print("Error: params bundle not found. Please run training_script.py first.")
        raise SystemExit(1)

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

    inputs, targets, masks = _build_task(cfg.TASK_CONFIG)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )

    best_params, losses, rewards = stmt.fit_rnn_reinforce(
        cbtl.rnn_func,
        params,
        config,
        inputs,
        masks,
        optimizer,
        cfg.TRAINING_CONFIG["num_iters"],
        log_interval=cfg.TRAINING_CONFIG["log_interval"],
        seed=cfg.TRAINING_CONFIG["seed"],
        baseline_momentum=cfg.RL_CONFIG["baseline_momentum"],
        entropy_coef=cfg.RL_CONFIG["entropy_coef"],
        objective_mode=cfg.RL_CONFIG.get("objective_mode", "log_reward"),
        batch_targets=targets,
        brevity_coef=cfg.RL_CONFIG.get("brevity_coef", 0.0),
        silence_coef=cfg.RL_CONFIG.get("silence_coef", 0.0),
        tail_coef=cfg.RL_CONFIG.get("tail_coef", 0.0),
    )

    with params_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")
    print(f"Training complete. Continued for {cfg.TRAINING_CONFIG['num_iters']} iterations.")


if __name__ == "__main__":
    main()
