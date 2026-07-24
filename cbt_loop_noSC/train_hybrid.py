"""Hybrid shaping step: bridge Pavlovian -> fully self-timed.

Starts from the Pavlovian-trained parameters and retrains on the *hybrid* STMT
task (``stmt.hybrid_stmt``). Every trial has the full self-timed temporal
structure -- a preparatory cue at trial start on cue channel 0 (the cue used by
the final task) -- PLUS a Pavlovian-style "go" cue delivered during the movement
window on cue channel 1. The go cue tells the network *when* to move instead of
requiring it to self-time, so it scaffolds the transition from the Pavlovian
task (respond to a cue) to the fully self-timed task (no go cue).

The go-cue channel (column 1) reuses the Pavlovian-cue weight column carried
over from Pavlovian training, so the cue->response mapping learned there
transfers directly; the preparatory-cue column (column 0) is freshly
randomized. This is the same two-channel cue layout as train_from_pavlovian, so
the hybrid output can be handed straight to the final self-timed stage.

Trained parameters are saved to params_hybrid.pkl.
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
cfg = _config_script.for_family('cbt_loop_noSC')
import self_timed_movement_task as stmt

# Seed for randomizing the new preparatory-cue weight vector
# (matches train_from_pavlovian so the two stages share a cue layout).
CUE_RERANDOM_SEED = 1234


def _make_dual_cue_weights(params, rng_key):
    """Extend cue->cortex weights to two columns [preparatory, Pavlovian/go].

    Column 0 (preparatory / timing cue) is freshly randomized -- it is a new cue
    whose meaning differs from the Pavlovian task. Column 1 (Pavlovian "go" cue)
    carries over the weights learned during Pavlovian training. Both are
    trainable. Identical to train_from_pavlovian._make_dual_cue_weights so the
    hybrid and final stages use the same cue channels.

    The cue targets only the excitatory cortical pools (cU, cL); any legacy
    cue->c_inh weight is dropped so the cue no longer feeds the inhibitory pool.
    """
    p = dict(params)
    p.pop("B_cue_c_inh", None)  # cue no longer projects to the inhibitory pool
    k_U, k_L = jr.split(rng_key, 2)
    for key, sub in (("B_cue_cU", k_U), ("B_cue_cL", k_L)):
        pav_col = jnp.asarray(params[key])[:, -1:]  # (n, 1) carried-over Pavlovian cue
        n = pav_col.shape[0]
        timing_col = jr.normal(sub, (n, 1))         # fresh preparatory-cue vector
        p[key] = jnp.concatenate([timing_col, pav_col], axis=1)  # (n, 2)
    return p


def _build_hybrid_batch():
    """Hybrid STMT trials: preparatory cue at trial start (channel 0) plus a
    Pavlovian "go" cue during the movement window (channel 1). Reward follows the
    self-timed movement window opened by the go cue (see stmt.hybrid_stmt)."""
    task_cfg = cfg.TASK_CONFIG
    return stmt.hybrid_stmt(
        T_start=task_cfg["t_start"],
        T_cue=task_cfg["t_cue"],
        T_wait=task_cfg["t_wait"],
        T_movement=task_cfg["t_movement"],
        T=task_cfg["t_total"],
    )


def main():
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG

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

    # Give the preparatory cue and the Pavlovian/go cue separate input vectors.
    params = _make_dual_cue_weights(params, jr.PRNGKey(CUE_RERANDOM_SEED))
    print("Built dual cue->cortex weights B_cue_c (col 0 = preparatory, col 1 = Pavlovian/go).")

    inputs, targets, masks = _build_hybrid_batch()
    print(f"Hybrid batch: inputs {tuple(inputs.shape)} (2 cue channels: preparatory + go)")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )

    mode = train_cfg["mode"]
    if mode == "supervised":
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

    out_path = cfg.hybrid_params_path()
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)

    print(f"Saved hybrid params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        label = "Final mean accuracy" if mode == "supervised" else "Final mean reward"
        print(f"{label}: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
