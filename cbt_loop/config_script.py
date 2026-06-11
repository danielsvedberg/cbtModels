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
    # Cortex excitatory pool split into two PT-like populations (Economo 2018).
    "n_c_U": 10,
    "n_c_L": 10,
    "n_c_inh": 4,
    "n_d1": 8,
    "n_d2": 8,
    "n_snc": 4,
    "n_snr": 6,
    "n_gpe": 6,
    "n_stn": 6,
    "n_sc": 6,
    "n_t_exc": 5,
    "n_t_inh": 2,
    "n_med": 4,
    "n_input":1,
    "n_output":1,
    "g_bg": 1.0,
    "g_nm": 1.0,
    "noise_std": 0.05,
}

OPTIM_CONFIG = {
    "learning_rate": 1e-3,
}

RL_CONFIG = {
    "entropy_coef": 0.01,
    "baseline_momentum": 0.99,
    "objective_mode": "log_reward",
    "brevity_coef": 0.5,
    "silence_coef": 0.5,
    "tail_coef": 0.5,
    "asym_coef": 0.1,
    "asym_margin": 0.5,
    "rest_pka_coef": 0.0,
    "rest_pka_margin": 0.9,
    "pathway_floor_coef": 0.1,
    "pathway_floor_min": 1.5,
    "c_snc_floor_coef": 0.2,
    "c_snc_floor_min": 0.2,
    # GPe kept alive structurally by the pacer floor (gpe_pacer_min >= 1 in
    # cbt_rnn config), so the activity-floor penalty is disabled (coef=0). Raise
    # gpe_floor_coef to re-enable it as a soft push above the pacer-set level.
    "gpe_floor_coef": 0.2,
    "gpe_floor_min": 0.2,
    # Dead-area inactivity floor: require every region (Cortex, D1, D2, SNc, GPe,
    # SNr, Thalamus, Medulla) to keep mean activity above dead_area_min over the
    # latter half of each trial. Each region below the floor adds
    # max(0, dead_area_min - mean_late_activity)^2; the summed hinge penalizes a
    # single silenced region. Latter-half window prevents gaming via a high
    # initial transient that then decays to zero.
    "dead_area_coef": 0.2,
    "dead_area_min": 0.1,
    # Dead-projection floor: keep every synaptic projection (each 2-D weight
    # matrix) from collapsing to zero. A projection is dead when its mean
    # absolute weight < dead_proj_floor / n_connections, i.e. its total |weight|
    # (L1) < dead_proj_floor; the penalty is dead_proj_coef * max(0,
    # dead_proj_floor - sum|W|)^2 summed over projections.
    "dead_proj_coef": 0.2,
    "dead_proj_floor": 0.1,
}

TASK_CONFIG = {
    "task_mode": "self-timed",  # one of: self_timed, hybrid, pavlovian
    "t_start": jr.randint(jr.PRNGKey(SEED_CONFIG["task_seed"]), shape=(100,), minval=50, maxval=400),
    "t_cue": 10,
    "t_wait": 300,
    "t_movement": 300,
    "t_total": 1000,
    "dt_ms": 10,
}

# Pavlovian conditioning task: a single cue arrives at a random time within
# the trial and the network is reinforced for responding immediately after.
PAVLOVIAN_CONFIG = {
    "t_start": jr.randint(
        jr.PRNGKey(SEED_CONFIG["task_seed"]),
        shape=(100,),
        minval=50,
        maxval=TASK_CONFIG["t_total"] - 50,
    ),
    "t_cue": 10,
    "t_response": 100,
    "t_total": TASK_CONFIG["t_total"],
}

TRAINING_CONFIG = {
    "num_iters": 20000,
    "log_interval": 200,
    "seed": SEED_CONFIG["train_seed"],
    # Training objective for train_pavlovian: "reinforce" (policy gradient) or
    # "supervised" (dense BCE/MSE regression onto the target trajectory).
    "mode": "reinforce",
    "loss_type": "bce",  # supervised mode only: "bce" or "mse"
}

TEST_CONFIG = {
    "n_seeds": 5,
    "noise_std": 0.05,
    "start_t": jnp.arange(270, 330, 10),
}

PARAMS_FILENAME = "params_shaped.pkl"
PAVLOVIAN_PARAMS_FILENAME = "params_pavlovian.pkl"
SHAPED_PARAMS_FILENAME = "params_shaped.pkl"


def params_path() -> Path:
    return Path(__file__).resolve().parent / PARAMS_FILENAME


def pavlovian_params_path() -> Path:
    return Path(__file__).resolve().parent / PAVLOVIAN_PARAMS_FILENAME


def shaped_params_path() -> Path:
    return Path(__file__).resolve().parent / SHAPED_PARAMS_FILENAME


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

d1_stim_strength = jnp.arange(0.0, 1.0, 0.2)
d2_stim_strength = jnp.arange(0.0, 1.0, 0.2)
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
