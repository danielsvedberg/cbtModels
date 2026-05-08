from pathlib import Path

import jax.random as jr


SEED_CONFIG = {
    "task_seed": 13,
    "train_seed": 0,
}

RNN_CONFIG = {
    "n_ctx": 30,
    "n_t": 30,
    "noise_std": 0.01,
}

OPTIM_CONFIG = {
    "learning_rate": 2e-3,
}

RL_CONFIG = {
    "entropy_coef": 1e-2,
    "baseline_momentum": 0.99,
    "objective_mode": "log_reward",
    "brevity_coef": 0.1,
    "silence_coef": 0.02,
    "tail_coef": 0.08,
}

TASK_CONFIG = {
    "task_mode": "self_timed",  # one of: self_timed, hybrid, pavlovian
    "t_start": jr.randint(jr.PRNGKey(SEED_CONFIG["task_seed"]), shape=(200,), minval=100, maxval=400),
    "t_cue": 10,
    "t_wait": 300,
    "t_movement": 300,
    "t_total": 900,
}

TRAINING_CONFIG = {
    "num_iters": 2000,
    "log_interval": 200,
    "seed": SEED_CONFIG["train_seed"],
}

PARAMS_FILENAME = "params_corticothalamic.pkl"


def params_path() -> Path:
    return Path(__file__).resolve().parent / PARAMS_FILENAME

