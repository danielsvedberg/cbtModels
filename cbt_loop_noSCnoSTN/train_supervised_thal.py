"""Supervised training against a soft step target, read out from the THALAMUS.

Differs from train_hybrid / training_script in three ways:

  1. READOUT   -- config["readout_source"] = "thalamus" swaps the medulla readout for
     cbt_rnn's C_thal weight vector, taken off the thalamic excitatory relay pool.
     C_thal is wrapped in exc() = sigmoid in the forward, so its effective weights are
     strictly positive and bounded (0,1); training cannot flip a sign.
  2. TARGET    -- stmt.selftimed_step_target: hold 0.25, then a 50-timestep plateau at
     0.75 spanning [cue+300, cue+350), back to 0.25. Every timestep is supervised
     (all-ones mask), so holding baseline off-window is part of the objective -- and
     the 300-step gap must be bridged by the network's own dynamics.
  3. TASK      -- SELF-TIMED ONLY. The step is anchored to the single STMT cue, so the
     hybrid / pavlovian two-cue variants are not valid here and are never built.

Usage:  python train_supervised_thal.py [--iters N] [--delay N] [--loss mse|bce]
"""
import argparse
import pickle as pkl
import sys as _sys, pathlib as _pl

import jax.numpy as jnp
import jax.random as jr
import optax

import cbt_rnn as cbtl

_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config_script as _config_script
import self_timed_movement_task as stmt

cfg = _config_script.for_family("cbt_loop_noSCnoSTN")


def _build_task(sup_cfg):
    t = cfg.TASK_CONFIG
    return stmt.selftimed_step_target(
        T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"],
        T_movement=t["t_movement"], T=t["t_total"],
        lo=sup_cfg["target_lo"], hi=sup_cfg["target_hi"],
        hold=sup_cfg["hold"], delay=sup_cfg["delay"],
    )


def main(num_iters=None, delay=None, loss_type=None):
    sup = dict(cfg.SUPERVISED_THAL_CONFIG)
    if delay is not None:
        sup["delay"] = int(delay)
    if loss_type is not None:
        sup["loss_type"] = loss_type
    n_iters = sup["num_iters"] if num_iters is None else int(num_iters)
    rl = cfg.RL_CONFIG

    inputs, targets, masks = _build_task(sup)
    # Rebalance the 95/5 baseline/plateau split: supervised_loss treats the mask as a
    # per-timestep weight, so scaling in-window steps by (1-f)/f removes the trivial
    # constant-output basin at 0.2750. See config_script SUPERVISED_THAL_CONFIG.
    w = float(sup.get("in_window_weight", 1.0))
    if w != 1.0:
        import numpy as _np
        _hi = _np.asarray(targets)[..., 0] > (sup["target_lo"] + sup["target_hi"]) / 2
        masks = jnp.asarray(_np.where(_hi, w, 1.0)[..., None].astype(_np.float32))
        print(f"[supervised-thal] in-window loss weight = {w}")
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]),
                                      n_input=inputs.shape[-1])
    config = dict(config)
    config["readout_source"] = "thalamus"   # <- the whole point of this mode

    print(f"[supervised-thal] readout=thalamus  target {sup['target_lo']}->{sup['target_hi']} "
          f"for {sup['hold']} steps, opening {sup['delay']} steps after cue onset")
    print(f"[supervised-thal] task=self-timed  inputs {tuple(inputs.shape)}  "
          f"loss={sup['loss_type']}  iters={n_iters}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )
    if sup.get("freeze_init_states", False):
        optimizer = cbtl.freeze_init_state_optimizer(optimizer, params)
        print(f"[freeze] init-state params frozen: {len(cbtl.INIT_STATE_KEYS)} keys")
    best_params, losses, accs = stmt.fit_rnn_supervised(
        cbtl.rnn_func, params, config, inputs, masks, optimizer, n_iters,
        batch_targets=targets,
        log_interval=sup["log_interval"],
        seed=cfg.TRAINING_CONFIG["seed"],
        loss_type=sup["loss_type"],
        asym_coef=rl["asym_coef"], asym_margin=rl["asym_margin"],
        rest_pka_coef=rl["rest_pka_coef"], rest_pka_margin=rl["rest_pka_margin"],
        pathway_floor_coef=rl["pathway_floor_coef"], pathway_floor_min=rl["pathway_floor_min"],
        c_snc_floor_coef=rl["c_snc_floor_coef"], c_snc_floor_min=rl["c_snc_floor_min"],
        gpe_floor_coef=rl["gpe_floor_coef"], gpe_floor_min=rl["gpe_floor_min"],
        dead_area_coef=rl["dead_area_coef"], dead_area_min=rl["dead_area_min"],
        dead_proj_coef=rl["dead_proj_coef"], dead_proj_floor=rl["dead_proj_floor"],
    )
    out_path = cfg.params_path().with_name("params_supervised_thal.pkl")
    with out_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)
    print(f"Saved to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    return best_params, losses, accs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--delay", type=int, default=None,
                    help="steps after CUE ONSET at which the step opens "
                         "(default t_wait=300; use t_cue+t_wait=310 to match the "
                         "reinforce reward window exactly)")
    ap.add_argument("--loss", choices=("mse", "bce"), default=None)
    a = ap.parse_args()
    main(num_iters=a.iters, delay=a.delay, loss_type=a.loss)
