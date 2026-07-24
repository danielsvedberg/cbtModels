import pickle as pkl

import jax.random as jr
import optax

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSC')
import self_timed_movement_task as stmt


def _build_task(task_cfg):
    task_kwargs = {
        "T_start": task_cfg["t_start"],
        "T_cue": task_cfg["t_cue"],
        "T_wait": task_cfg["t_wait"],
        "T_movement": task_cfg["t_movement"],
        "T": task_cfg["t_total"],
    }
    mode = task_cfg.get("task_mode", "self_timed")
    if mode == "hybrid":
        return stmt.hybrid_stmt(**task_kwargs)
    if mode == "pavlovian":
        return stmt.pavlovian_stmt(**task_kwargs)
    return stmt.self_timed_movement_task(**task_kwargs)


def main():
    task_cfg = cfg.TASK_CONFIG
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG
    rnn_cfg = cfg.RNN_CONFIG

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
            n_c_U=params["J_cU"].shape[0],
            n_c_L=params["J_cL"].shape[0],
            n_c_inh=params["J_c_ii"].shape[0],
            n_d1=params["J_d1"].shape[0],
            n_d2=params["J_d2"].shape[0],
            n_snc=params["P_snc"].shape[0],
            n_snr=params["P_snr"].shape[0],
            n_gpe=params["J_gpe"].shape[0],
            n_stn=params["J_stn"].shape[0],
            n_t_exc=params["J_t_ee"].shape[0],
            n_t_inh=params["J_t_ii"].shape[0],
            n_med=params["J_med_w1"].shape[0] * 2,
            n_input=1,
            n_output=1,
            noise_std=rnn_cfg["noise_std"],
        )

    inputs, targets, masks = _build_task(task_cfg)
    # The loaded model may expect more cue channels than the task emits (e.g. a
    # 2-channel [timing, go] cue from the hybrid lineage vs. the 1-channel
    # self-timed cue). Zero-pad the missing channels — the final self-timed task
    # has no go cue, so channel 1 is simply held at zero.
    inputs = cbtl.match_input_channels(inputs, params)
    print(f"Task inputs (cue-channel matched): {tuple(inputs.shape)}")

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
        train_cfg["num_iters"],
        log_interval=train_cfg["log_interval"],
        seed=train_cfg["seed"],
        baseline_momentum=rl_cfg["baseline_momentum"],
        entropy_coef=rl_cfg["entropy_coef"],
        objective_mode=rl_cfg.get("objective_mode", "log_reward"),
        batch_targets=targets,
        brevity_coef=rl_cfg.get("brevity_coef", 0.0),
        silence_coef=rl_cfg.get("silence_coef", 0.0),
        tail_coef=rl_cfg.get("tail_coef", 0.0),
        dead_area_coef=rl_cfg.get("dead_area_coef", 0.0),
        dead_area_min=rl_cfg.get("dead_area_min", 0.0),
        dead_proj_coef=rl_cfg.get("dead_proj_coef", 0.0),
        dead_proj_floor=rl_cfg.get("dead_proj_floor", 0.1),
    )

    with params_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    print(f"Saved params to: {params_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
