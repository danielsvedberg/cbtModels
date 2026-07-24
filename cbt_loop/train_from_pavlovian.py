"""Shaping transfer: start from Pavlovian-trained parameters and retrain on a
mixed batch of self-timed movement (STMT) and Pavlovian trials.

Every training iteration sees N_STMT_TRIALS self-timed trials plus
N_PAVLOVIAN_TRIALS Pavlovian trials. Inputs carry two cue channels — the
timing (STMT) cue and the Pavlovian cue — and each drives cortex through its
own column of B_cue_c, so the two cues have separate, independently trained
input weight vectors.

The timing-cue weight vector is freshly randomized (it is a new cue); the
Pavlovian-cue weight vector carries over from Pavlovian training. Both are
trainable. Trained parameters are saved to params_shaped.pkl.
"""

import pickle as pkl

import jax.numpy as jnp
import jax.random as jr
import optax

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop')
import self_timed_movement_task as stmt

# Per-iteration batch composition.
N_STMT_TRIALS = 100
N_PAVLOVIAN_TRIALS = 10

# Seed for randomizing the new timing-cue weight vector.
CUE_RERANDOM_SEED = 1234


def _build_mixed_batch():
    """Build a mixed batch: N_STMT_TRIALS self-timed + N_PAVLOVIAN_TRIALS Pavlovian.

    Inputs have two cue channels — channel 0 is the timing (STMT) cue and
    channel 1 is the Pavlovian cue. Each task drives only its own channel; the
    other is held at zero.
    """
    task_cfg = cfg.TASK_CONFIG
    pav_cfg = cfg.PAVLOVIAN_CONFIG

    # Self-timed movement trials.
    stmt_in, stmt_tgt, stmt_mask = stmt.self_timed_movement_task(
        T_start=task_cfg["t_start"][:N_STMT_TRIALS],
        T_cue=task_cfg["t_cue"],
        T_wait=task_cfg["t_wait"],
        T_movement=task_cfg["t_movement"],
        T=task_cfg["t_total"],
    )
    # Pavlovian trials.
    pav_in, pav_tgt, pav_mask = stmt.pavlovian_task(
        T_start=pav_cfg["t_start"][:N_PAVLOVIAN_TRIALS],
        T_cue=pav_cfg["t_cue"],
        T_response=pav_cfg["t_response"],
        T=pav_cfg["t_total"],
    )

    # Two cue channels: [timing cue, Pavlovian cue].
    stmt_in2 = jnp.concatenate([stmt_in, jnp.zeros_like(stmt_in)], axis=-1)
    pav_in2 = jnp.concatenate([jnp.zeros_like(pav_in), pav_in], axis=-1)

    inputs = jnp.concatenate([stmt_in2, pav_in2], axis=0)   # (n_trials, T, 2)
    targets = jnp.concatenate([stmt_tgt, pav_tgt], axis=0)  # (n_trials, T, 1)
    masks = jnp.concatenate([stmt_mask, pav_mask], axis=0)  # (n_trials, T, 1)
    return inputs, targets, masks


def _make_dual_cue_weights(params, rng_key):
    """Build a two-column cue→cortex projection for the excitatory cortical pools.

    Column 0 (timing/STMT cue) is freshly randomized — it is a new cue whose
    meaning differs from the Pavlovian task. Column 1 (Pavlovian cue) carries
    over the weights learned during Pavlovian training. Both columns are
    trainable parameters and are updated during this run. The cue targets only
    the excitatory cortical pools (cU, cL) through two weight tensors
    (B_cue_cU / B_cue_cL); any legacy cue->c_inh weight is dropped.
    """
    p = dict(params)
    p.pop("B_cue_c_inh", None)  # cue no longer projects to the inhibitory pool
    k_U, k_L = jr.split(rng_key, 2)
    for key, sub in (("B_cue_cU", k_U), ("B_cue_cL", k_L)):
        pav_col = jnp.asarray(params[key])[:, -1:]  # (n, 1) carried-over Pavlovian cue
        n = pav_col.shape[0]
        timing_col = jr.normal(sub, (n, 1))         # fresh timing-cue vector
        p[key] = jnp.concatenate([timing_col, pav_col], axis=1)  # (n, 2)
    return p


def main():
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG
    rnn_cfg = cfg.RNN_CONFIG

    pav_path = cfg.pavlovian_params_path()
    print(f"Loading Pavlovian parameters from {pav_path}...")
    try:
        with pav_path.open("rb") as f:
            bundle = pkl.load(f)
    except FileNotFoundError:
        print("Error: Pavlovian params not found. Please run train_pavlovian.py first.")
        raise SystemExit(1)

    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        params = bundle["params"]
        config = bundle["config"]
    else:
        params = bundle
        _, config = cbtl.init_params(jr.PRNGKey(0), n_input=1)

    # Give the timing cue and the Pavlovian cue separate input weight vectors.
    params = _make_dual_cue_weights(params, jr.PRNGKey(CUE_RERANDOM_SEED))
    print("Built dual cue→cortex weights B_cue_c (col 0 = timing, col 1 = Pavlovian).")

    inputs, targets, masks = _build_mixed_batch()
    print(f"Mixed batch: {N_STMT_TRIALS} STMT + {N_PAVLOVIAN_TRIALS} Pavlovian "
          f"trials -> inputs {tuple(inputs.shape)}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )

    mode = train_cfg["mode"]
    if mode == "supervised":
        # Dense supervised regression onto the target trajectory; structural
        # priors carried over from RL_CONFIG.
        best_params, losses, rewards = stmt.fit_rnn_supervised(
            cbtl.rnn_func,
            params,
            config,
            inputs,
            masks,
            optimizer,
            train_cfg["num_iters"],
            batch_targets=targets,
            log_interval=train_cfg["log_interval"],
            seed=train_cfg["seed"],
            loss_type=train_cfg["loss_type"],
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
    else:
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

    out_path = cfg.shaped_params_path()
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    print(f"Saved shaped params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        label = "Final mean accuracy" if mode == "supervised" else "Final mean reward"
        print(f"{label}: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
