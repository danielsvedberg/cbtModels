import pickle as pkl

import jax.random as jr
import optax

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSCnoSTN')

import self_timed_movement_task as stmt


def _build_task(task_cfg):
    task_kwargs = {
        "T_start": task_cfg["t_start"],
        "T_cue": task_cfg["t_cue"],
        "T_wait": task_cfg["t_wait"],
        "T_movement": task_cfg["t_movement"],
        "T": task_cfg["t_total"],
    }
    mode = task_cfg["task_mode"]
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

    inputs, targets, masks = _build_task(task_cfg)

    params, config = cbtl.init_params(jr.PRNGKey(train_cfg["seed"]), n_input=inputs.shape[-1])

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
        objective_mode=rl_cfg["objective_mode"],
        batch_targets=targets,
        brevity_coef=rl_cfg["brevity_coef"],
        silence_coef=rl_cfg["silence_coef"],
        tail_coef=rl_cfg["tail_coef"],
        asym_coef=rl_cfg["asym_coef"],
        asym_margin=rl_cfg["asym_margin"],
        rest_pka_coef=rl_cfg["rest_pka_coef"],
        rest_pka_margin=rl_cfg["rest_pka_margin"],
        dead_area_coef=rl_cfg["dead_area_coef"],
        dead_area_min=rl_cfg["dead_area_min"],
        dead_proj_coef=rl_cfg["dead_proj_coef"],
        dead_proj_floor=rl_cfg["dead_proj_floor"],
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
