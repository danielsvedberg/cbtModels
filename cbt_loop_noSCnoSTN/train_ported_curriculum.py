"""Curriculum training of noSCnoSTN starting FROM the ported max-PCI loop.

    python train_ported_curriculum.py --stage pavlovian   # from params_pci_loop_init.pkl
    python train_ported_curriculum.py --stage hybrid       # from the pavlovian result

Brevity / silence / tail penalties are set to ZERO in both stages: those terms push the
network toward a quiet, brief output, which would erode the rich sustained dynamics of
the transplanted high-PCI loop. The dead-area / dead-projection regularizers are KEPT
(they encourage activity, so they help preserve PCI, not spoil it).
"""
import argparse
import pickle as pkl
import sys
import pathlib

import jax.random as jr
import optax

import cbt_rnn as cbtl
import train_hybrid  # _build_hybrid_batch, _make_dual_cue_weights, CUE_RERANDOM_SEED
_root = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as _config_script
import self_timed_movement_task as stmt

cfg = _config_script.for_family("cbt_loop_noSCnoSTN")
HERE = pathlib.Path(__file__).resolve().parent
INIT = HERE / "params_pci_loop_init.pkl"
PAV_OUT = HERE / "params_ported_pavlovian.pkl"
HYB_OUT = HERE / "params_ported_hybrid.pkl"


def _fit(params, config, inputs, targets, masks, n_iters, rl_cfg, entropy_coef=None):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )
    ent = rl_cfg["entropy_coef"] if entropy_coef is None else entropy_coef
    print(f"  entropy_coef = {ent} (config {rl_cfg['entropy_coef']})")
    best_params, losses, rewards = stmt.fit_rnn_reinforce(
        cbtl.rnn_func, params, config, inputs, masks, optimizer, n_iters,
        log_interval=cfg.TRAINING_CONFIG["log_interval"],
        seed=cfg.TRAINING_CONFIG["seed"],
        baseline_momentum=rl_cfg["baseline_momentum"],
        entropy_coef=ent,
        objective_mode=rl_cfg["objective_mode"],
        batch_targets=targets,
        brevity_coef=0.0,   # OFF: would shorten/quiet the response -> erodes PCI dynamics
        silence_coef=0.0,   # OFF: would suppress baseline/off-window activity
        tail_coef=0.0,      # OFF: would suppress post-response activity
        dead_area_coef=rl_cfg["dead_area_coef"],
        dead_area_min=rl_cfg["dead_area_min"],
        dead_proj_coef=rl_cfg["dead_proj_coef"],
        dead_proj_floor=rl_cfg["dead_proj_floor"],
    )
    return best_params, rewards


def run_pavlovian(entropy_coef=None):
    t = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=t["t_start"], T_cue=t["t_cue"], T_response=t["t_response"], T=t["t_total"])
    with INIT.open("rb") as f:
        d = pkl.load(f)
    params, config = d["params"], d["config"]
    print(f"[pavlovian] from {INIT.name} (ported loop, abs BG), inputs {tuple(inputs.shape)}, "
          f"{cfg.TRAINING_CONFIG['num_iters']} iters, brevity/silence/tail=0 -> {PAV_OUT.name}")
    best, rewards = _fit(params, config, inputs, targets, masks,
                         cfg.TRAINING_CONFIG["num_iters"], cfg.RL_CONFIG, entropy_coef)
    with PAV_OUT.open("wb") as f:
        pkl.dump({"params": best, "config": config}, f)
    print(f"Saved -> {PAV_OUT}. Final reward {float(rewards[-1]):.4f}" if rewards else "done")


def run_hybrid(entropy_coef=None):
    with PAV_OUT.open("rb") as f:
        d = pkl.load(f)
    params, config = d["params"], d["config"]
    params = train_hybrid._make_dual_cue_weights(params, jr.PRNGKey(train_hybrid.CUE_RERANDOM_SEED))
    inputs, targets, masks = train_hybrid._build_hybrid_batch()
    print(f"[hybrid] from {PAV_OUT.name} (dual-cue), inputs {tuple(inputs.shape)}, "
          f"{cfg.TRAINING_CONFIG['num_iters']} iters, brevity/silence/tail=0 -> {HYB_OUT.name}")
    best, rewards = _fit(params, config, inputs, targets, masks,
                         cfg.TRAINING_CONFIG["num_iters"], cfg.RL_CONFIG, entropy_coef)
    with HYB_OUT.open("wb") as f:
        pkl.dump({"params": best, "config": config}, f)
    print(f"Saved -> {HYB_OUT}. Final reward {float(rewards[-1]):.4f}" if rewards else "done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pavlovian", "hybrid"], required=True)
    ap.add_argument("--entropy", type=float, default=None,
                    help="override entropy_coef (config default 0.01; try ~0.04 to avoid "
                         "premature policy collapse)")
    args = ap.parse_args()
    (run_pavlovian if args.stage == "pavlovian" else run_hybrid)(args.entropy)
