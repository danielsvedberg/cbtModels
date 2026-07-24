"""Central configuration for ALL model families.

Single source of truth. Every family (cbt_loop, cbt_loop_noSC,
cbt_loop_noSCnoSTN, corticothalamic, vanilla_rnn) loads its config from here via

    import config_script
    cfg = config_script.for_this_file(__file__)   # picks family from the dir name

`cfg` then exposes the same names the old per-family config_script.py did
(cfg.RNN_CONFIG, cfg.RL_CONFIG, cfg.params_path(), cfg.plots_folder, ...), so
call sites are unchanged apart from the two import lines.

HOMOGENIZATION (per user, 2026-07-24): shared knobs are ONE value, taken from
cbt_loop (the canonical/reference family) — objective_mode, coefs, floors, task
timing, training, test, optim, seeds, AND the biophysical runtime constants
(tau_*, *_pacer_*, m_floor*). The ONLY per-family differences that remain are
STRUCTURAL, i.e. forced by the architecture:
  * which nuclei exist  (noSC has no SC; noSCnoSTN has no SC and no STN),
  * keys only one architecture reads (noSC's DA/adenosine concentration dynamics
    tau_da/tau_ado/da_release/ado_release),
  * the two non-CBT architectures (corticothalamic = 2-node ctx/thal; vanilla =
    single hidden layer) which have their own structural sizes + init scales.
Those are noted inline as "architecture-specific", not discrepancies.
"""
from pathlib import Path
import types

import jax.numpy as jnp
import jax.random as jr
import optax

ROOT = Path(__file__).resolve().parent

# Families and their subdirectories (path helpers resolve relative to these).
FAMILY_DIRS = {
    "cbt_loop": ROOT / "cbt_loop",
    "cbt_loop_noSC": ROOT / "cbt_loop_noSC",
    "cbt_loop_noSCnoSTN": ROOT / "cbt_loop_noSCnoSTN",
    "corticothalamic": ROOT / "corticothalamic",
    "vanilla_rnn": ROOT / "vanilla_rnn",
}
CBT_FAMILIES = ("cbt_loop", "cbt_loop_noSC", "cbt_loop_noSCnoSTN")


# =========================================================================== #
# SHARED knobs (identical for every family; canonical = cbt_loop).
# =========================================================================== #
SEED_CONFIG = {
    "task_seed": 13,
    "train_seed": 0,
}

OPTIM_CONFIG = {
    "learning_rate": 1e-3,
}

RL_CONFIG = {
    "entropy_coef": 0.01,
    "baseline_momentum": 0.99,
    "objective_mode": "loss",   # supervised BCE against the (brief-transient) target
    "brevity_coef": 0.5,
    "silence_coef": 0.5,
    "tail_coef": 0.5,
    "asym_coef": 0.0,
    "asym_margin": 0.5,
    "rest_pka_coef": 0.0,
    "rest_pka_margin": 0.9,
    "pathway_floor_coef": 0.0,
    "pathway_floor_min": 1.5,
    "c_snc_floor_coef": 0.0,
    "c_snc_floor_min": 0.2,
    "gpe_floor_coef": 0.0,
    "gpe_floor_min": 0.2,
    "dead_area_coef": 0.0,
    "dead_area_min": 0.1,
    "dead_proj_coef": 0.0,
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
    # Brief-transient supervised target: a 10-step pulse at movement onset.
    "t_pulse": 10,
}

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
    "num_iters": 10000,
    "log_interval": 200,
    "seed": SEED_CONFIG["train_seed"],
    "mode": "reinforce",   # "reinforce" (policy gradient) or "supervised"
    "loss_type": "bce",    # supervised mode only: "bce" or "mse"
}

TEST_CONFIG = {
    "n_seeds": 5,
    "noise_std": 0.05,
    "start_t": jnp.arange(270, 330, 10),
}

# vanilla-only hybrid-shaping pretraining task (architecture-specific extra).
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


# =========================================================================== #
# ARCHITECTURE configs.
# CBT trio share ONE canonical RNN_CONFIG + RUNTIME_CONFIG (cbt_loop values);
# each family drops the nuclei it lacks (below).
# =========================================================================== #
CBT_RNN_CONFIG = {
    # Cortex excitatory pool split into two PT-like populations (Economo 2018).
    "n_c_U": 10,
    "n_c_L": 10,
    "n_c_inh": 10,
    "n_d1": 8,
    "n_d2": 8,
    "n_snc": 4,
    "n_snr": 6,
    "n_gpe": 6,
    "n_stn": 6,      # dropped for noSCnoSTN
    "n_sc": 6,       # dropped for noSC and noSCnoSTN
    "n_t_exc": 10,
    "n_t_inh": 5,
    "n_med": 4,
    "n_input": 1,
    "n_output": 1,
    "g_bg": 1.0,
    "g_nm": 1.0,
    "noise_std": 0.01,
    "balanced_init": False,
}

# init_params runtime dict (biophysical constants). Built into `config` by
# init_params in each cbt_rnn.py so there is one place to edit them.
CBT_RUNTIME_CONFIG = {
    "tau_c": 5.0,
    "tau_med": 5.0,
    "tau_d1": 5.0,
    "tau_d2": 5.0,
    "tau_t": 5.0,
    "tau_snr": 5.0,
    "tau_gpe": 5.0,
    "tau_stn": 5.0,
    "tau_sc": 5.0,
    "tau_snc": 10.0,
    "tau_pka_fall": 1440.0,
    "tau_pka_rise": 10.0,
    "m_floor": 0.001,
    "snr_med_floor": 0.1,
    "m_floor_a1": 0.001,
    "m_floor_a2": 0.001,
    "da_pka_gain": 1.0,
    "k_a_floor": 0.001,
    "k_a_cap": 1.0,
    "snc_pacer_min": 0.05,
    "snc_pacer_max": 0.2,
    "snr_pacer_max": 0.85,
    "snr_pacer_min": 0.4,
    "gpe_pacer_min": 0.45,
    "gpe_pacer_max": 0.8,
    "stn_pacer_max": 0.3,
}

# init_params SIGNATURE defaults (fallbacks when a caller omits a kwarg). Kept
# equal to the canonical RNN_CONFIG so defaults never diverge from it.
CBT_INIT_DEFAULTS = {
    "n_c_U": CBT_RNN_CONFIG["n_c_U"],
    "n_c_L": CBT_RNN_CONFIG["n_c_L"],
    "n_c_inh": CBT_RNN_CONFIG["n_c_inh"],
    "n_d1": CBT_RNN_CONFIG["n_d1"],
    "n_d2": CBT_RNN_CONFIG["n_d2"],
    "n_snc": CBT_RNN_CONFIG["n_snc"],
    "n_snr": CBT_RNN_CONFIG["n_snr"],
    "n_gpe": CBT_RNN_CONFIG["n_gpe"],
    "n_stn": CBT_RNN_CONFIG["n_stn"],
    "n_sc": CBT_RNN_CONFIG["n_sc"],
    "n_t_exc": CBT_RNN_CONFIG["n_t_exc"],
    "n_t_inh": CBT_RNN_CONFIG["n_t_inh"],
    "n_med": CBT_RNN_CONFIG["n_med"],
    "n_input": CBT_RNN_CONFIG["n_input"],
    "n_output": CBT_RNN_CONFIG["n_output"],
    "g_bg": CBT_RNN_CONFIG["g_bg"],
    "g_nm": CBT_RNN_CONFIG["g_nm"],
    "noise_std": CBT_RNN_CONFIG["noise_std"],
    "balanced_init": CBT_RNN_CONFIG["balanced_init"],
    "balanced_target_rho": 0.997,
    "persistent_self_gain": None,
}

# Architecture-unique keys per CBT family (nuclei dropped; extra runtime keys).
_CBT_FAMILY_STRUCTURE = {
    "cbt_loop": {"drop_rnn": (), "extra_runtime": {}},
    "cbt_loop_noSC": {
        "drop_rnn": ("n_sc",),
        # noSC models DA/adenosine as dynamic concentrations (only it reads these).
        "extra_runtime": {
            "tau_da": 20.0, "tau_ado": 200.0,
            "da_release": 1.0, "ado_release": 1.0,
            "stn_pacer_min": 0.05,
        },
    },
    "cbt_loop_noSCnoSTN": {"drop_rnn": ("n_sc", "n_stn"), "extra_runtime": {}},
}

# --- corticothalamic (2-node ctx/thalamus RNN; architecture-specific) ---
CORTICOTHALAMIC_RNN_CONFIG = {
    "n_ctx": 30,
    "n_t": 30,
    "noise_std": 0.01,
}
CORTICOTHALAMIC_RUNTIME_CONFIG = {
    "tau_ctx": 20.0,
    "tau_t": 20.0,
    "rec_scale": 0.15,
    "cross_scale": 0.15,
    "in_scale": 0.25,
    "out_scale": 0.2,
}

# --- vanilla (single hidden layer RNN; architecture-specific) ---
VANILLA_RNN_CONFIG = {
    "n_hidden": 32,
    "noise_std": 0.01,
}
VANILLA_RUNTIME_CONFIG = {
    "tau": 20.0,
    "rec_scale": 0.15,
    "in_scale": 0.25,
    "out_scale": 0.5,
}

_ARCH_RNN_CONFIG = {"corticothalamic": CORTICOTHALAMIC_RNN_CONFIG, "vanilla_rnn": VANILLA_RNN_CONFIG}

# Per-family output filenames.
_CBT_FILENAMES = {
    "params": "params_shaped.pkl",
    "pavlovian": "params_pavlovian.pkl",
    "hybrid": "params_hybrid.pkl",
    "shaped": "params_shaped.pkl",
    "pretrain": "pretrain_params_vanilla.pkl",
}
_FAMILY_FILENAMES = {
    "corticothalamic": {**_CBT_FILENAMES, "params": "params_corticothalamic.pkl",
                        "shaped": "params_corticothalamic.pkl"},
    "vanilla_rnn": {**_CBT_FILENAMES, "params": "params_vanilla.pkl"},
}


def _filenames_for(family):
    return _FAMILY_FILENAMES.get(family, _CBT_FILENAMES)


def rnn_config_for(family):
    """The RNN_CONFIG a given family should use (canonical, minus dropped nuclei)."""
    if family in CBT_FAMILIES:
        cfg = dict(CBT_RNN_CONFIG)
        for k in _CBT_FAMILY_STRUCTURE[family]["drop_rnn"]:
            cfg.pop(k, None)
        return cfg
    return dict(_ARCH_RNN_CONFIG[family])


def runtime_config_for(family):
    """The init_params biophysical runtime dict for a CBT family (canonical +
    that family's architecture-unique extras)."""
    if family in CBT_FAMILIES:
        cfg = dict(CBT_RUNTIME_CONFIG)
        cfg.update(_CBT_FAMILY_STRUCTURE[family]["extra_runtime"])
        return cfg
    if family == "corticothalamic":
        return dict(CORTICOTHALAMIC_RUNTIME_CONFIG)
    if family == "vanilla_rnn":
        return dict(VANILLA_RUNTIME_CONFIG)
    return {}


def _make_optimizer():
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=OPTIM_CONFIG["learning_rate"]),
    )


def for_family(family):
    """Return a config view for `family` exposing the same names the old
    per-family config_script.py did (dicts, path helpers, opto/plot surface)."""
    if family not in FAMILY_DIRS:
        raise KeyError(f"unknown family {family!r}; known: {sorted(FAMILY_DIRS)}")
    d = FAMILY_DIRS[family]
    ns = types.SimpleNamespace()

    # --- shared config dicts (copied so callers can mutate without leaking) ---
    ns.SEED_CONFIG = dict(SEED_CONFIG)
    ns.OPTIM_CONFIG = dict(OPTIM_CONFIG)
    ns.RL_CONFIG = dict(RL_CONFIG)
    ns.TASK_CONFIG = dict(TASK_CONFIG)
    ns.PAVLOVIAN_CONFIG = dict(PAVLOVIAN_CONFIG)
    ns.TRAINING_CONFIG = dict(TRAINING_CONFIG)
    ns.TEST_CONFIG = dict(TEST_CONFIG)
    ns.PRETRAIN_TASK_CONFIG = dict(PRETRAIN_TASK_CONFIG)
    ns.PRETRAINING_CONFIG = dict(PRETRAINING_CONFIG)
    ns.RNN_CONFIG = rnn_config_for(family)
    ns.RUNTIME_CONFIG = runtime_config_for(family)

    # --- filenames ---
    fn = _filenames_for(family)
    ns.PARAMS_FILENAME = fn["params"]
    ns.PAVLOVIAN_PARAMS_FILENAME = fn["pavlovian"]
    ns.HYBRID_PARAMS_FILENAME = fn["hybrid"]
    ns.SHAPED_PARAMS_FILENAME = fn["shaped"]
    ns.PRETRAIN_PARAMS_FILENAME = fn["pretrain"]

    # --- path helpers (resolve to the family directory) ---
    ns.params_path = lambda: d / fn["params"]
    ns.pavlovian_params_path = lambda: d / fn["pavlovian"]
    ns.hybrid_params_path = lambda: d / fn["hybrid"]
    ns.shaped_params_path = lambda: d / fn["shaped"]
    ns.pretrain_params_path = lambda: d / fn["pretrain"]

    # --- plotting / analysis aliases ---
    ns.default_config = {"noise_std": ns.RNN_CONFIG.get("noise_std", 0.01), "dt": TASK_CONFIG["dt_ms"]}
    ns.config = {
        "T_start": TASK_CONFIG["t_start"], "T_cue": TASK_CONFIG["t_cue"],
        "T_wait": TASK_CONFIG["t_wait"], "T_movement": TASK_CONFIG["t_movement"],
        "T": TASK_CONFIG["t_total"], "dt": TASK_CONFIG["dt_ms"],
    }
    ns.test_start_t = TEST_CONFIG["start_t"]
    ns.n_seeds = TEST_CONFIG["n_seeds"]
    ns.test_noise_std = TEST_CONFIG["noise_std"]
    ns.optimizer = _make_optimizer()
    ns.params = {}
    ns.x0 = None
    ns.z0 = None

    # --- optogenetic stim surface (needs n_d1/n_d2; CBT families only) ---
    ns.n_opto_seeds = 1000
    ns.opto_tstart = 250
    ns.opto_start = ns.opto_tstart + 100
    ns.opto_end = ns.opto_start + 175
    n_d1 = ns.RNN_CONFIG.get("n_d1", 0)
    n_d2 = ns.RNN_CONFIG.get("n_d2", 0)
    if n_d1 and n_d2:
        d1_stim = jnp.arange(0.0, 1.0, 0.2)
        d2_stim = jnp.arange(0.0, 1.0, 0.2)
        suppress_d1 = [jnp.concatenate([jnp.full((n_d1,), -i), jnp.zeros((n_d2,))]) for i in d1_stim]
        suppress_d2 = [jnp.concatenate([jnp.zeros((n_d1,)), jnp.full((n_d2,), -i)]) for i in d2_stim]
        enhance_d1 = [jnp.concatenate([jnp.full((n_d1,), i), jnp.zeros((n_d2,))]) for i in d1_stim]
        enhance_d2 = [jnp.concatenate([jnp.zeros((n_d1,)), jnp.full((n_d2,), i)]) for i in d2_stim]
        ns.spatial_stim_list = suppress_d1 + suppress_d2 + enhance_d1 + enhance_d2
        ns.stim_strengths = jnp.concatenate([-d1_stim, -d2_stim, d1_stim, d2_stim])
        ns.stim_labels = (["inh dMSN"] * len(d1_stim) + ["inh iMSN"] * len(d2_stim)
                          + ["stim dMSN"] * len(d1_stim) + ["stim iMSN"] * len(d2_stim))
    else:
        ns.spatial_stim_list, ns.stim_strengths, ns.stim_labels = [], jnp.array([]), []

    # --- plots folders (under the family directory) ---
    plots = d / "plots"
    (plots / "svg").mkdir(parents=True, exist_ok=True)
    (plots / "png").mkdir(parents=True, exist_ok=True)
    ns.plots_folder = str(plots)
    ns.svg_folder = str(plots / "svg")
    ns.png_folder = str(plots / "png")

    ns.family = family
    ns.family_dir = d
    return ns


def for_this_file(file):
    """Convenience: derive the family from the calling file's parent dir name."""
    return for_family(Path(file).resolve().parent.name)
