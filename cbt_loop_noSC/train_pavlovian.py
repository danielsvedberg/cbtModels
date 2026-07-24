"""Train the CBT loop on the Pavlovian conditioning task from scratch.

A single cue arrives at a random time within the trial and the network is
reinforced for responding immediately after it. Follows the mold of
training_script.py; trained parameters are saved to params_pavlovian.pkl.
"""

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
        n_c_U=rnn_cfg["n_c_U"],
        n_c_L=rnn_cfg["n_c_L"],
        n_c_inh=rnn_cfg["n_c_inh"],
        n_d1=rnn_cfg["n_d1"],
        n_d2=rnn_cfg["n_d2"],
        n_snc=rnn_cfg["n_snc"],
        n_snr=rnn_cfg["n_snr"],
        n_gpe=rnn_cfg["n_gpe"],
        n_stn=rnn_cfg["n_stn"],
        n_t_exc=rnn_cfg["n_t_exc"],
        n_t_inh=rnn_cfg["n_t_inh"],
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

    mode = train_cfg.get("mode", "reinforce")
    if mode == "supervised":
        # Dense supervised regression onto the target trajectory. Structural
        # priors are pulled from RL_CONFIG (off unless set there).
        best_params, losses, rewards = stmt.fit_rnn_supervised(
            cbtl.rnn_func,
            params,
            config,
            inputs,
            masks,
            optimizer,
            train_cfg["num_iters"], # number of iters
            batch_targets=targets,
            log_interval=train_cfg["log_interval"],
            seed=train_cfg["seed"],
            loss_type=train_cfg.get("loss_type", "bce"),
            asym_coef=rl_cfg.get("asym_coef", 0.0),
            asym_margin=rl_cfg.get("asym_margin", 1.0),
            rest_pka_coef=rl_cfg.get("rest_pka_coef", 0.0),
            rest_pka_margin=rl_cfg.get("rest_pka_margin", 1.0),
            pathway_floor_coef=rl_cfg.get("pathway_floor_coef", 0.0),
            pathway_floor_min=rl_cfg.get("pathway_floor_min", 1.0),
            c_snc_floor_coef=rl_cfg.get("c_snc_floor_coef", 0.0),
            c_snc_floor_min=rl_cfg.get("c_snc_floor_min", 0.0),
            gpe_floor_coef=rl_cfg.get("gpe_floor_coef", 0.0),
            gpe_floor_min=rl_cfg.get("gpe_floor_min", 0.0),
            dead_area_coef=rl_cfg.get("dead_area_coef", 0.0),
            dead_area_min=rl_cfg.get("dead_area_min", 0.0),
            dead_proj_coef=rl_cfg.get("dead_proj_coef", 0.0),
            dead_proj_floor=rl_cfg.get("dead_proj_floor", 0.0),
        )
    else:
        best_params, losses, rewards = stmt.fit_rnn_reinforce(
            cbtl.rnn_func,
            params,
            config,
            inputs,
            masks,
            optimizer,
            train_cfg["num_iters"], # number of iters
            log_interval=train_cfg["log_interval"],
            seed=train_cfg["seed"],
            baseline_momentum=rl_cfg["baseline_momentum"],
            entropy_coef=rl_cfg["entropy_coef"],
            objective_mode=rl_cfg.get("objective_mode", "log_reward"),
            batch_targets=targets,
            brevity_coef=rl_cfg.get("brevity_coef", 1.0),
            silence_coef=rl_cfg.get("silence_coef", 1.0),
            tail_coef=rl_cfg.get("tail_coef", 1.0),
            # Structural penalties disabled during Pavlovian: with tanh() wrapping
            # every inter-area projection, weight-norm-based floors no longer map
            # to effective drive (saturates at ~1 regardless of norm) and only push
            # weights into the dead-gradient zone. Re-enabled in train_from_pavlovian.
            asym_coef=0.0,
            asym_margin=rl_cfg.get("asym_margin", 1.0),
            rest_pka_coef=0.0,
            rest_pka_margin=rl_cfg.get("rest_pka_margin", 1.0),
            pathway_floor_coef=0.0,
            pathway_floor_min=rl_cfg.get("pathway_floor_min", 1.0),
            c_snc_floor_coef=0.0,
            c_snc_floor_min=rl_cfg.get("c_snc_floor_min", 0.0),
            gpe_floor_coef=rl_cfg.get("gpe_floor_coef", 0.0),
            gpe_floor_min=rl_cfg.get("gpe_floor_min", 0.0),
            dead_area_coef=rl_cfg.get("dead_area_coef", 0.0),
            dead_area_min=rl_cfg.get("dead_area_min", 0.0),
            dead_proj_coef=rl_cfg.get("dead_proj_coef", 1.0),
            dead_proj_floor=rl_cfg.get("dead_proj_floor", 0.0001),
        )

    out_path = cfg.pavlovian_params_path()
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    print(f"Saved Pavlovian params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        label = "Final mean accuracy" if mode == "supervised" else "Final mean reward"
        print(f"{label}: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
