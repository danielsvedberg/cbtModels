"""Train noSCnoSTN on the hybrid (go-cue) task starting FROM the ported max-PCI loop
init (params_pci_loop_init.pkl). Tests whether transplanting the high-PCI thalamocortical
loop helps the full model learn. Non-clobbering: saves to params_pci_ported_hybrid.pkl.
"""
import pickle as pkl
import sys
import pathlib

import optax

import cbt_rnn as cbtl
import train_hybrid  # reuse its hybrid-batch builder
_root = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as _config_script
import self_timed_movement_task as stmt

cfg = _config_script.for_family("cbt_loop_noSCnoSTN")
HERE = pathlib.Path(__file__).resolve().parent
INIT = HERE / "params_pci_loop_init.pkl"
OUT = HERE / "params_pci_ported_hybrid.pkl"


def main():
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG

    with INIT.open("rb") as f:
        d = pkl.load(f)
    params, config = d["params"], d["config"]
    print(f"[train_ported_hybrid] init from {INIT.name} (ported max-PCI loop + "
          f"log-normal BG); {train_cfg['num_iters']} iters -> {OUT.name}")

    inputs, targets, masks = train_hybrid._build_hybrid_batch()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )
    best_params, losses, rewards = stmt.fit_rnn_reinforce(
        cbtl.rnn_func, params, config, inputs, masks, optimizer,
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
        dead_area_coef=rl_cfg["dead_area_coef"],
        dead_area_min=rl_cfg["dead_area_min"],
        dead_proj_coef=rl_cfg["dead_proj_coef"],
        dead_proj_floor=rl_cfg["dead_proj_floor"],
    )
    with OUT.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)
    print(f"Saved -> {OUT}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")


if __name__ == "__main__":
    main()
