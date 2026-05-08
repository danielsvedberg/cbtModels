from pathlib import Path

import jax.random as jr


# Shared seeds
SEED_CONFIG = {
    "task_seed": 13,
    "train_seed": 0,
}

# Shared model/noise settings used by pretraining, training, and plotting.
RNN_CONFIG = {
    "n_hidden": 32,
    "noise_std": 0.01,
}

OPTIM_CONFIG = {
    "learning_rate": 1e-3,
}

RL_CONFIG = {
    "entropy_coef": 1e-2,
    "baseline_momentum": 0.99,
    "objective_mode": "log_reward",
    "brevity_coef": 0.1,
    "silence_coef": 0.02,
    "tail_coef": 0.08,
}

# Canonical STMT task used for main training/evaluation.
TASK_CONFIG = {
    "t_start": jr.randint(jr.PRNGKey(SEED_CONFIG["task_seed"]), shape=(100,), minval=50, maxval=250),
    "t_cue": 10,
    "t_wait": 300,
    "t_movement": 100,
    "t_total": 900,
}

# Hybrid shaping pretraining task.
PRETRAIN_TASK_CONFIG = {
    "t_start": jr.randint(jr.PRNGKey(SEED_CONFIG["task_seed"] + 1), shape=(100,), minval=50, maxval=250),
    "t_cue": 10,
    "t_wait": 300,
    "t_movement": 100,
    "t_total": 900,
}

PRETRAINING_CONFIG = {
    "num_iters": 5000,
    "log_interval": 200,
    "seed": SEED_CONFIG["train_seed"],
}

TRAINING_CONFIG = {
    "num_iters": 5000,
    "log_interval": 200,
    "seed": SEED_CONFIG["train_seed"],
}

PARAMS_FILENAME = "params_vanilla.pkl"
PRETRAIN_PARAMS_FILENAME = "pretrain_params_vanilla.pkl"


def params_path() -> Path:
    return Path(__file__).resolve().parent / PARAMS_FILENAME


def pretrain_params_path() -> Path:
    return Path(__file__).resolve().parent / PRETRAIN_PARAMS_FILENAME

