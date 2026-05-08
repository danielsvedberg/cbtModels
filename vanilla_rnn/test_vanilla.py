from pathlib import Path
import pickle as pkl

import jax.numpy as jnp

import vanilla_rnn as vr


def main():
    params_path = Path(__file__).resolve().parent / "params_vanilla.pkl"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing {params_path}. Run train_vanilla.py first."
        )

    with params_path.open("rb") as f:
        params = pkl.load(f)

    n_hidden = params["b_h"].shape[0]
    config = {
        "noise_std": 0.0,
        "x0": jnp.ones((n_hidden,)) * 0.1,
        "tau": 20.0,
    }

    starts = jnp.arange(100, 221, 30)
    inputs, targets, _ = vr.stmt.self_timed_movement_task(starts, 10, 120, 40, 420)
    ys, xs, actions = vr.evaluate(params, config, inputs, noise_std=0.0, n_seeds=4)

    print("ys shape:", ys.shape)  # (n_seeds, n_conditions, T, 1)
    print("xs shape:", xs.shape)  # (n_seeds, n_conditions, T, n_hidden)
    print("actions shape:", actions.shape)  # (n_seeds, n_conditions, T, 1)
    print("target shape:", targets.shape)
    print("max y:", float(jnp.max(ys)))


if __name__ == "__main__":
    main()

