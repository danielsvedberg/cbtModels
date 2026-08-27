"""Train the FINAL task (self-timed movement + Pavlovian mix) from FRESH init.

Same task as the last curriculum stage (train_from_hybrid.py / the "shaped" stage) --
N_STMT_TRIALS self-timed STMT trials + N_PAVLOVIAN_TRIALS Pavlovian trials, two cue
channels (col 0 = timing/preparatory cue, col 1 = Pavlovian/go cue) -- but WITHOUT the
Pavlovian -> hybrid -> shaped curriculum: params come straight from init_params (the
de-saturated logit init, n_input=2, both cue columns freshly randomized).

Use this to test whether the de-saturated loop can learn the final task end-to-end from
scratch, skipping the curriculum. The mixed batch (a few Pavlovian trials alongside the
self-timed ones) is reused verbatim from train_from_hybrid so the task is identical.

Saves to params_final_scratch.pkl.

    python train_final_scratch.py [--iters N] [--objective log_reward] [--seed S]
"""
import argparse
import pickle as pkl
import sys as _sys
import pathlib as _pl

import jax.random as jr
import optax

import cbt_rnn as cbtl

_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config_script as _config_script
cfg = _config_script.for_family("cbt_loop_noSCnoSTN")
import self_timed_movement_task as stmt
# Reuse the EXACT final-task batch builder from the curriculum's last stage so the task
# trained here is identical to train_from_hybrid -- only the init differs (fresh vs hybrid).
from train_from_hybrid import _build_mixed_batch, N_STMT_TRIALS, N_PAVLOVIAN_TRIALS


def _init_from_scratch(seed):
    """Fresh params from init_params with the 2-channel cue layout (both cue columns
    randomly initialized), no Pavlovian/hybrid bootstrap."""
    print("[init=scratch] fresh params from init_params (n_input=2, dual cue, no bootstrap)")
    return cbtl.init_params(jr.PRNGKey(seed), n_input=2)


def main(num_iters=None, objective=None, seed=None):
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = dict(cfg.RL_CONFIG)
    if objective is not None:
        rl_cfg["objective_mode"] = objective
        print(f"[override] objective_mode = {objective} (central: {cfg.RL_CONFIG['objective_mode']})")
    if train_cfg["mode"] != "reinforce":
        print(f"[warn] TRAINING_CONFIG['mode'] = {train_cfg['mode']!r}; this script trains with "
              f"reinforce (policy gradient) regardless.")
    n_iters = train_cfg["num_iters"] if num_iters is None else int(num_iters)
    seed = train_cfg["seed"] if seed is None else int(seed)

    params, config = _init_from_scratch(seed)
    inputs, targets, masks = _build_mixed_batch()
    print(f"Final task (fresh init): {N_STMT_TRIALS} STMT + {N_PAVLOVIAN_TRIALS} Pavlovian "
          f"trials -> inputs {tuple(inputs.shape)} (2 cue channels)")
    print(f"Training {n_iters} iters (objective={rl_cfg['objective_mode']}, seed={seed})")

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
        n_iters,
        log_interval=train_cfg["log_interval"],
        seed=seed,
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
        pathway_floor_coef=rl_cfg["pathway_floor_coef"],
        pathway_floor_min=rl_cfg["pathway_floor_min"],
        c_snc_floor_coef=rl_cfg["c_snc_floor_coef"],
        c_snc_floor_min=rl_cfg["c_snc_floor_min"],
        gpe_floor_coef=rl_cfg["gpe_floor_coef"],
        gpe_floor_min=rl_cfg["gpe_floor_min"],
        dead_area_coef=rl_cfg["dead_area_coef"],
        dead_area_min=rl_cfg["dead_area_min"],
        dead_proj_coef=rl_cfg["dead_proj_coef"],
        dead_proj_floor=rl_cfg["dead_proj_floor"],
    )

    out_path = cfg.shaped_params_path().with_name("params_final_scratch.pkl")
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)
    print(f"Saved final-from-scratch params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train the final (self-timed + Pavlovian) task from FRESH init.")
    ap.add_argument("--iters", type=int, default=None, help="override TRAINING_CONFIG['num_iters']")
    ap.add_argument("--objective", choices=("loss", "log_reward", "reward_prob"), default=None,
                    help="override RL_CONFIG['objective_mode']")
    ap.add_argument("--seed", type=int, default=None, help="init + train seed (default: train_seed)")
    a = ap.parse_args()
    main(num_iters=a.iters, objective=a.objective, seed=a.seed)
