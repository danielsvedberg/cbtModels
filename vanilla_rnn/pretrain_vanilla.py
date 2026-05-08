import pickle as pkl

import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt

import vanilla_rnn as vr
import self_timed_movement_task as stmt
import vanilla_config as cfg


def test_and_plot_pretrained(params, config, inputs, targets, masks, noise_std=0.01, n_seeds=5):
    """
    Test pretrained model and plot outputs vs targets.

    Args:
        params: Trained model parameters
        config: RNN runtime config (x0, tau, noise_std)
        inputs: Input data (batch_size, T, 1)
        targets: Target outputs (batch_size, T, 1)
        masks: Loss masks (batch_size, T, 1)
        noise_std: Noise standard deviation for evaluation
        n_seeds: Number of random seeds to evaluate
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Run evaluation with multiple seeds
    all_ys, all_xs, all_as = vr.evaluate(params, config, inputs, noise_std=noise_std, n_seeds=n_seeds)

    # all_ys shape: (n_seeds, batch_size, T, 1)
    # all_xs shape: (n_seeds, batch_size, T, n_hidden)

    # Plot individual trials
    ax = axs[0]
    colors = plt.cm.tab20(jnp.linspace(0, 1, inputs.shape[0]))
    for trial_idx in range(min(inputs.shape[0], 10)):  # Plot first 10 trials
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
        # Also plot target for this trial
        ax.plot(targets[trial_idx, :, 0], color=colors[trial_idx], linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Output (y)')
    ax.set_title('Network Output vs Target (first 10 trials)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot average across trials
    ax = axs[1]
    mean_ys = jnp.mean(all_ys, axis=(0, 1))  # Average across seeds and trials
    std_ys = jnp.std(all_ys, axis=(0, 1))
    mean_targets = jnp.mean(targets, axis=0)
    std_targets = jnp.std(targets, axis=0)

    ax.plot(mean_ys[:, 0], label='Network Output', color='blue', linewidth=2)
    ax.fill_between(
        jnp.arange(mean_ys.shape[0]),
        mean_ys[:, 0] - std_ys[:, 0],
        mean_ys[:, 0] + std_ys[:, 0],
        color='blue',
        alpha=0.2,
    )
    ax.plot(mean_targets[:, 0], label='Target', color='red', linestyle='--', linewidth=2)
    ax.fill_between(
        jnp.arange(mean_targets.shape[0]),
        mean_targets[:, 0] - std_targets[:, 0],
        mean_targets[:, 0] + std_targets[:, 0],
        color='red',
        alpha=0.2,
    )

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Output (y)')
    ax.set_title('Average Network Output vs Target (all trials)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, all_ys, all_xs, all_as


def main():
    task = cfg.PRETRAIN_TASK_CONFIG
    pretrain_cfg = cfg.PRETRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG
    rnn_cfg = cfg.RNN_CONFIG

    inputs, targets, masks = stmt.hybrid_stmt(
        task["t_start"],
        task["t_cue"],
        task["t_wait"],
        task["t_movement"],
        task["t_total"],
    )

    params, config = vr.init_params(
        jr.PRNGKey(pretrain_cfg["seed"]),
        n_hidden=rnn_cfg["n_hidden"],
        n_input=inputs.shape[-1],
        n_output=1,
        noise_std=rnn_cfg["noise_std"],
    )

    optimizer = optax.adam(cfg.OPTIM_CONFIG["learning_rate"])

    # Create RNN function wrapper for REINFORCE
    def rnn_func(cur_params, config, batch_inputs, opto_stim, rng_keys):
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
        pretrain_cfg["num_iters"],
        log_interval=pretrain_cfg["log_interval"],
        seed=pretrain_cfg["seed"],
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

    print(f"Saved pretrained params to: {out_path}")
    if losses:
        print(f"Final logged loss: {float(losses[-1]):.6f}")
    if rewards:
        print(f"Final mean reward: {float(rewards[-1]):.4f}")

    # Test and plot the pretrained model
    print("\nTesting and plotting pretrained model...")
    test_and_plot_pretrained(best_params, config, inputs, targets, masks,
                            noise_std=cfg.RNN_CONFIG["noise_std"], n_seeds=5)


if __name__ == "__main__":
    main()

