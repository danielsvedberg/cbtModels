"""Train the CBT loop on the Pavlovian conditioning task from scratch.

A single cue arrives at a random time within the trial and the network is
reinforced for responding immediately after it. Follows the mold of
training_script.py; trained parameters are saved to params_pavlovian.pkl.
"""

import pickle as pkl

import jax.random as jr
import optax

import cbt_rnn as cbtl
import config_script as cfg

import self_timed_movement_task as stmt


def _build_pavlovian_task(task_cfg):
    return stmt.pavlovian_task(
        T_start=task_cfg["t_start"],
        T_cue=task_cfg["t_cue"],
        T_response=task_cfg["t_response"],
        T=task_cfg["t_total"],
    )


def main():
    task_cfg = cfg.PAVLOVIAN_CONFIG
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG
    rnn_cfg = cfg.RNN_CONFIG

    inputs, targets, masks = _build_pavlovian_task(task_cfg)

    params, config = cbtl.init_params(
        jr.PRNGKey(train_cfg["seed"]),
        n_c=rnn_cfg["n_c"],
        n_d1=rnn_cfg["n_d1"],
        n_d2=rnn_cfg["n_d2"],
        n_snc=rnn_cfg["n_snc"],
        n_snr=rnn_cfg["n_snr"],
        n_gpe=rnn_cfg["n_gpe"],
        n_t=rnn_cfg["n_t"],
        n_input=inputs.shape[-1],
        n_output=1,
        g_bg=rnn_cfg["g_bg"],
        g_nm=rnn_cfg["g_nm"],
        noise_std=rnn_cfg["noise_std"],
    )

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
        500, #train_cfg["num_iters"], # number of iters
        log_interval=train_cfg["log_interval"],
        seed=train_cfg["seed"],
        baseline_momentum=rl_cfg["baseline_momentum"],
        entropy_coef=rl_cfg["entropy_coef"],
        objective_mode=rl_cfg.get("objective_mode", "log_reward"),
        batch_targets=targets,
        brevity_coef=rl_cfg.get("brevity_coef", 0.0),
        silence_coef=rl_cfg.get("silence_coef", 0.0),
        tail_coef=rl_cfg.get("tail_coef", 0.0),
        asym_coef=rl_cfg.get("asym_coef", 0.0),
        asym_margin=rl_cfg.get("asym_margin", 1.0),
        rest_pka_coef=rl_cfg.get("rest_pka_coef", 0.0),
        rest_pka_margin=rl_cfg.get("rest_pka_margin", 1.0),
    )

    out_path = cfg.pavlovian_params_path()
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    print(f"Saved Pavlovian params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
