from pathlib import Path
import pickle as pkl

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

import vanilla_rnn as vr
import self_timed_movement_task as stmt
import vanilla_config as cfg


def test_and_plot_trained(params, config, inputs, targets, noise_std=0.01, n_seeds=5):
    """Quick visual check of trained output traces against targets."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    all_ys, all_xs, all_as = vr.evaluate(params, config, inputs, noise_std=noise_std, n_seeds=n_seeds)

    ax = axs[0]
    colors = plt.cm.tab20(jnp.linspace(0, 1, inputs.shape[0]))
    for trial_idx in range(min(inputs.shape[0], 10)):
        mean_y = jnp.mean(all_ys[:, trial_idx, :, 0], axis=0)
        std_y = jnp.std(all_ys[:, trial_idx, :, 0], axis=0)
        ax.plot(mean_y, color=colors[trial_idx], label=f"Trial {trial_idx}", linewidth=1)
        ax.fill_between(
            jnp.arange(mean_y.shape[0]),
            mean_y - std_y,
            mean_y + std_y,
            color=colors[trial_idx],
            alpha=0.2,
        )
        ax.plot(targets[trial_idx, :, 0], color=colors[trial_idx], linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Output (y)")
    ax.set_title("Network Output vs Target (first 10 trials)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[1]
    mean_ys = jnp.mean(all_ys, axis=(0, 1))
    std_ys = jnp.std(all_ys, axis=(0, 1))
    mean_targets = jnp.mean(targets, axis=0)
    std_targets = jnp.std(targets, axis=0)

    ax.plot(mean_ys[:, 0], label="Network Output", color="blue", linewidth=2)
    ax.fill_between(
        jnp.arange(mean_ys.shape[0]),
        mean_ys[:, 0] - std_ys[:, 0],
        mean_ys[:, 0] + std_ys[:, 0],
        color="blue",
        alpha=0.2,
    )
    ax.plot(mean_targets[:, 0], label="Target", color="red", linestyle="--", linewidth=2)
    ax.fill_between(
        jnp.arange(mean_targets.shape[0]),
        mean_targets[:, 0] - std_targets[:, 0],
        mean_targets[:, 0] + std_targets[:, 0],
        color="red",
        alpha=0.2,
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Output (y)")
    ax.set_title("Average Network Output vs Target (all trials)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, all_ys, all_xs, all_as


def main():
    task = cfg.TASK_CONFIG
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG

    inputs, targets, masks = stmt.self_timed_movement_task(
        task["t_start"],
        task["t_cue"],
        task["t_wait"],
        task["t_movement"],
        task["t_total"],
    )

    '''
        pretrain_path = cfg.pretrain_params_path()
    if pretrain_path.exists():
        with pretrain_path.open("rb") as f:
            params = pkl.load(f)
        print(f"Loaded pretrained params from: {pretrain_path}")
        _, config = vr.init_params(
            jr.PRNGKey(train_cfg["seed"]),
            n_hidden=cfg.RNN_CONFIG["n_hidden"],
            n_input=inputs.shape[-1],
            n_output=1,
        )
    else:
    '''
    params, config = vr.init_params(
        jr.PRNGKey(train_cfg["seed"]),
        n_hidden=cfg.RNN_CONFIG["n_hidden"],
        n_input=inputs.shape[-1],
        n_output=1,
    )
    print("No pretrained params found; training from scratch.")

    optimizer = optax.adam(cfg.OPTIM_CONFIG["learning_rate"])

    def rnn_func(cur_params, config, batch_inputs, stim, rng_keys):
        batch_stim = None
        ys, xs = vr.batched_rnn(cur_params, config, batch_inputs, batch_stim, rng_keys)
        return ys, xs, None

    best_params, losses, rewards = stmt.fit_rnn_reinforce(
        rnn_func,
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
        objective_mode=rl_cfg.get("objective_mode", "log_reward"),
        batch_targets=targets,
        brevity_coef=rl_cfg.get("brevity_coef", 0.0),
        silence_coef=rl_cfg.get("silence_coef", 0.0),
        tail_coef=rl_cfg.get("tail_coef", 0.0),
    )

    out_path = cfg.pretrain_params_path()
    with out_path.open("wb") as f:
        pkl.dump(best_params, f)
    out_path = cfg.params_path()
    with out_path.open("wb") as f:
        pkl.dump(best_params, f)

    print(f"Saved params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.6f}")

    print("\nTesting and plotting trained model...")
    test_and_plot_trained(
        best_params,
        config,
        inputs,
        targets,
        noise_std=cfg.RNN_CONFIG["noise_std"],
        n_seeds=5,
    )


if __name__ == "__main__":
    main()

