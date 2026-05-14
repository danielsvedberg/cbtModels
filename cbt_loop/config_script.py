from pathlib import Path
import os

import jax.numpy as jnp
import jax.random as jr
import optax


SEED_CONFIG = {
    "task_seed": 13,
    "train_seed": 0,
}

RNN_CONFIG = {
    "n_c": 10,
    "n_d1": 4,
    "n_d2": 4,
    "n_snc": 3,
    "n_snr": 6,
    "n_gpe": 3,
    "n_stn": 3,
    "n_t": 10,
    "g_bg": 1.2,
    "g_nm": 0.5,
    "noise_std": 0.05,
}

OPTIM_CONFIG = {
    "learning_rate": 1e-3,
}

RL_CONFIG = {
    "entropy_coef": 0.1,
    "baseline_momentum": 0.99,
    "objective_mode": "log_reward",
    "brevity_coef": 0.1,
    "silence_coef": 0.1,
    "tail_coef": 0.1,
}

TASK_CONFIG = {
    "task_mode": "self-timed",  # one of: self_timed, hybrid, pavlovian
    "t_start": jr.randint(jr.PRNGKey(SEED_CONFIG["task_seed"]), shape=(100,), minval=50, maxval=300),
    "t_cue": 10,
    "t_wait": 300,
    "t_movement": 300,
    "t_total": 900,
    "dt_ms": 10,
}

TRAINING_CONFIG = {
    "num_iters": 10000,
    "log_interval": 200,
    "seed": SEED_CONFIG["train_seed"],
}

TEST_CONFIG = {
    "n_seeds": 5,
    "noise_std": 0.05,
    "start_t": jnp.arange(270, 330, 10),
}

PARAMS_FILENAME = "params_nm.pkl"


def params_path() -> Path:
    return Path(__file__).resolve().parent / PARAMS_FILENAME


# ---------------------------------------------------------------------------
# Compatibility aliases for existing cbt_loop plotting scripts.
# ---------------------------------------------------------------------------
default_config = {
    "noise_std": RNN_CONFIG["noise_std"],
    "dt": TASK_CONFIG["dt_ms"],
}

config = {
    "T_start": TASK_CONFIG["t_start"],
    "T_cue": TASK_CONFIG["t_cue"],
    "T_wait": TASK_CONFIG["t_wait"],
    "T_movement": TASK_CONFIG["t_movement"],
    "T": TASK_CONFIG["t_total"],
    "dt": TASK_CONFIG["dt_ms"],
}

test_start_t = TEST_CONFIG["start_t"]
n_seeds = TEST_CONFIG["n_seeds"]
test_noise_std = TEST_CONFIG["noise_std"]

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=OPTIM_CONFIG["learning_rate"]),
)

# Lightweight placeholders retained for scripts that still reference them.
params = {}
x0 = None
z0 = None

# Optogenetic stimulation defaults retained for existing scripts.
n_opto_seeds = 1000
opto_tstart = 250
opto_start = opto_tstart + 100
opto_end = opto_start + 175

d1_stim_strength = jnp.arange(0.0, 1.6, 0.3)
d2_stim_strength = jnp.arange(0.0, 0.8, 0.15)
d1_suppress_strength = -d1_stim_strength
d2_suppress_strength = -d2_stim_strength

# Construct spatial stim vectors in a model-agnostic way (n_d1 + n_d2 channels).
_n_d1 = RNN_CONFIG["n_d1"]
_n_d2 = RNN_CONFIG["n_d2"]
suppress_d1 = [
    jnp.concatenate([jnp.full((_n_d1,), i), jnp.zeros((_n_d2,))])
    for i in d1_suppress_strength
]
suppress_d2 = [
    jnp.concatenate([jnp.zeros((_n_d1,)), jnp.full((_n_d2,), i)])
    for i in d2_suppress_strength
]
enhance_d1 = [
    jnp.concatenate([jnp.full((_n_d1,), i), jnp.zeros((_n_d2,))])
    for i in d1_stim_strength
]
enhance_d2 = [
    jnp.concatenate([jnp.zeros((_n_d1,)), jnp.full((_n_d2,), i)])
    for i in d2_stim_strength
]
spatial_stim_list = suppress_d1 + suppress_d2 + enhance_d1 + enhance_d2

stim_strengths = jnp.concatenate([
    d1_suppress_strength,
    d2_suppress_strength,
    d1_stim_strength,
    d2_stim_strength,
])
stim_labels = (
    ["inh dMSN"] * len(d1_suppress_strength)
    + ["inh iMSN"] * len(d2_suppress_strength)
    + ["stim dMSN"] * len(d1_stim_strength)
    + ["stim iMSN"] * len(d2_stim_strength)
)

plots_folder = str(Path(__file__).resolve().parent / "plots")
svg_folder = str(Path(plots_folder) / "svg")
png_folder = str(Path(plots_folder) / "png")
os.makedirs(svg_folder, exist_ok=True)
os.makedirs(png_folder, exist_ok=True)
