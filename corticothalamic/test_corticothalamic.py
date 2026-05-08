import pickle as pkl

import jax.numpy as jnp

import corticothalamic_rnn as ctrnn
import corticothalamic_config as cfg


def main():
    params_path = cfg.params_path()
    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing {params_path}. Run train_corticothalamic.py first."
        )

    with params_path.open("rb") as f:
        bundle = pkl.load(f)

    params = bundle["params"]
    config = bundle["config"]

    starts = jnp.arange(100, 221, 30)
    inputs, targets, _ = ctrnn.stmt.self_timed_movement_task(starts, 10, 120, 40, 420)
    ys, x_ctx, x_t, actions = ctrnn.evaluate(params, config, inputs, noise_std=0.0, n_seeds=4)

    print("ys shape:", ys.shape)  # (n_seeds, n_conditions, T, 1)
    print("ctx shape:", x_ctx.shape)  # (n_seeds, n_conditions, T, n_ctx)
    print("t shape:", x_t.shape)  # (n_seeds, n_conditions, T, n_t)
    print("actions shape:", actions.shape)  # (n_seeds, n_conditions, T, 1)
    print("target shape:", targets.shape)
    print("max y:", float(jnp.max(ys)))


if __name__ == "__main__":
    main()

