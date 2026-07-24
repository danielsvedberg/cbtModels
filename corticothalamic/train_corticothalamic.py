import pickle as pkl

import jax.random as jr
import optax

import corticothalamic_rnn as ctrnn
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('corticothalamic')
import self_timed_movement_task as stmt


def main():
    task = cfg.TASK_CONFIG
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG

    task_kwargs = {
        "T_start": task["t_start"],
        "T_cue": task["t_cue"],
        "T_wait": task["t_wait"],
        "T_movement": task["t_movement"],
        "T": task["t_total"],
    }
    task_mode = task.get("task_mode", "self_timed")
    if task_mode == "hybrid":
        inputs, targets, masks = stmt.hybrid_stmt(**task_kwargs)
    elif task_mode == "pavlovian":
        inputs, targets, masks = stmt.pavlovian_stmt(**task_kwargs)
    else:
        inputs, targets, masks = stmt.self_timed_movement_task(**task_kwargs)

    params, config = ctrnn.init_params(jr.PRNGKey(train_cfg["seed"]), n_input=inputs.shape[-1])

    optimizer = optax.adam(cfg.OPTIM_CONFIG["learning_rate"])
    best_params, losses, rewards = stmt.fit_rnn_reinforce(
        ctrnn.rnn_func,
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
        objective_mode=rl_cfg["objective_mode"],
        batch_targets=targets,
        brevity_coef=rl_cfg["brevity_coef"],
        silence_coef=rl_cfg["silence_coef"],
        tail_coef=rl_cfg["tail_coef"],
    )

    out_path = cfg.params_path()
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    print(f"Saved params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()

