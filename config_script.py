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
    "train_seed": 4,
}

OPTIM_CONFIG = {
    "learning_rate": 1e-3,
}

RL_CONFIG = {
    "entropy_coef": 0.01,
    "baseline_momentum": 0.99,
    # Optimize log P(first response lands in the reward window) directly.
    # NOT "loss" (dense BCE): BCE scores timesteps independently and is nearly blind
    # to what actually sets the reward. Cutting pre-window firing 0.0163 -> 0.001
    # improves mean BCE by only ~0.011 (noise on a ~0.5 loss) but improves the
    # hazard reward ~39,000x, because pre-window firing compounds over ~684 steps.
    # Measured: hybrid-from-scratch reward 6e-4 under BCE vs 0.88 under log_reward.
    "objective_mode": "log_reward",
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
    # t_start range widened and the movement window narrowed so that reward
    # actually measures SELF-TIMING. With the old values (window 300, t_start in
    # [50,400)) the per-trial reward windows overlapped so heavily that a single
    # FIXED response time ignoring the cue scored 0.87 -- i.e. a model with zero
    # timing ability looked near-perfect, and gradient descent duly took that
    # shortcut (measured: slope 0.26, see docs/parameter_findings.md section 8).
    # Ceiling ~ window / t_start_range: 100/490 -> ~0.20 (empirically ~0.25).
    # t_start max 540 keeps the hybrid window (t_go = t_start+360, +100) inside
    # t_total as well. See corticothalamic/task_design.py.
    "t_start": jr.randint(jr.PRNGKey(SEED_CONFIG["task_seed"]), shape=(100,), minval=50, maxval=540),
    "t_cue": 10,
    "t_wait": 300,
    "t_movement": 100,
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
    # Spectral-normalize the cortico-thalamic loop at init (loop_init.normalize_loop):
    # scale the 17 loop blocks so rho of the update map M=(1-1/tau)I+(1/tau)W equals
    # balanced_target_rho. Raw init is rho~1.76 -> the loop runs away until the sigmoid
    # nln saturates, and saturated cortex (gain ~0.2) passes neither the cue forward nor
    # the gradient backward, so the task gradient vanishes. Normalizing to ~1.0
    # de-saturates cortex and restores a usable cue->output gradient (~1000x larger).
    "balanced_init": True,
    # TUNING KNOB, NOT TRAINED: applied once at init; never enters params, so the
    # optimizer never sees it and there is no gradient w.r.t. it. Higher rho = longer
    # memory (tau_eff = -1/ln(rho)) but closer to instability/saturation.
    "balanced_target_rho": 1.0,
}

# init_params runtime dict (biophysical constants). Built into `config` by
# init_params in each cbt_rnn.py so there is one place to edit them.
CBT_RUNTIME_CONFIG = {
    "tau_c": 7.0,    # ~20 ms EPSP-like single-neuron decay (nln-modified, dt=10 ms)
    "tau_med": 10.0,
    "tau_d1": 10.0,
    "tau_d2": 10.0,
    "tau_t": 7.0,
    "tau_snr": 10.0,
    "tau_gpe": 10.0,
    "tau_stn": 10.0,
    "tau_sc": 10.0,
    "tau_snc": 10.0,
    "tau_pka_fall": 900.0,  # moderate lengthening: longer memory than 500, less saturated than 1440
    "tau_pka_rise": 10.0,
    "m_floor": 0.001,
    "snr_med_floor": 0.1,
    "m_floor_a1": 0.001,
    "m_floor_a2": 0.001,
    # DA->PKA drive gain. Must be large enough that the DA term beats the tonic
    # adenosine term at the operating point, else max(DA - adenosine, 0) clamps to
    # zero and PKA has neither drive NOR gradient w.r.t. SNc (measured: DA 0.0225
    # vs adenosine 0.0593 at gain 1.0 -> production exactly 0, PKA cue response
    # exactly 0.00000). See corticothalamic/pka_timer_probe.py.
    "da_pka_gain": 4.0,
    # PKA as a genuine leaky integrator: keep the STATE unsquashed (tau_pka_fall
    # then really sets the timescale) and squash only where it is USED, by clipping
    # into bg_nln's valid (0,1) excitability range. With the legacy behaviour
    # (nln applied to the state every step) the measured half-life was ~3 steps despite
    # tau_pka_fall=1440. This is the model's only delay-scale variable, hence the
    # natural substrate for an interval timer.
    "pka_integrator": True,
    # PKA state saturation rule (read only by cbt_loop/cbt_rnn.py):
    #   "linear"      - unbounded leaky integrator (canonical/default); only the
    #                   readout gate saturates the signal, the state can ramp freely.
    #   "mass_action" - bounded pool: production throttled by (1 - pka/pka_max) so
    #                   the STATE saturates at ~pka_max while the leak stays linear
    #                   (slow timescale preserved through the delay). See cbt_loop
    #                   override below. pka_max is unused when saturation == "linear".
    "pka_saturation": "linear",
    "pka_max": 4.0,
    # Numerical safety inset when PKA is fed DIRECTLY as bg_nln's excitability b
    # (cbt_loop): b = clip(pka, eps, 1-eps) keeps c=3/(1-b) and d=(1/6)(1-b)/b
    # finite at the (0,1) endpoints. Only bites at the extremes; rest sits at ~0.5.
    "pka_clip_eps": 0.02,
    "pka_gate_min": 0.05,
    "pka_gate_max": 0.95,
    # Slope of the soft threshold; small because the linear integrator spans ~0-15.
    # (cbt_loop's mass_action path overrides this steeper — the bounded state spans
    # only ~0-pka_max, so a wider slope is needed for a comparable gate transition.)
    "pka_gate_slope": 1.0,
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

# Per-area initial state (resting/baseline activity each area starts at, before
# the per-step noise+nln). One value per area, canonical = cbt_loop. Families that
# store these as trainable params (cbt_loop, noSCnoSTN) use them as the init value;
# noSC uses them directly as fixed initial conditions.
CBT_INIT_STATE = {
    "x_c0_U": 0.1,
    "x_c0_L": 0.1,
    "x_c0_inh": 0.1,
    "x_d10": 0.1,
    "x_d20": 0.1,
    "x_snc0": 0.1,
    "x_gpe0": 0.1,
    "x_stn0": 0.1,   # STN families only
    "x_snr0": 0.1,
    "x_sc0": 0.1,    # SC families only
    "x_t0_exc": 0.3,
    "x_t0_inh": 0.3,
    "x_med0": 0.1,
    "pka_d10": 0.3,
    "pka_d20": 0.3,
}

# Scalar initial VALUES of trainable weight params that aren't fan-in-scaled
# (canonical = cbt_loop). Each family reads the subset it has: cbt_loop uses all;
# noSC has no k_a (it uses DA/adenosine concentration dynamics); noSCnoSTN has no
# out_gain/out_bias (plain nln readout).
CBT_WEIGHT_INIT = {
    "m_a1": 0.05,          # A1R inhibitory drive on D1 PKA (per-SPN gain)
    "m_a2": 0.01,          # A2R excitatory drive on D2 PKA (per-SPN gain)
    "out_gain": 4.0,       # readout gain
    "out_bias": -1.0986123,  # readout bias = logit(0.25)
    "k_a": 1.0,            # tonic adenosine level (pre-sigmoid/exc)
    # Initial PKA soft-threshold. The integrator ramps ~0.3->12 over a trial,
    # so a mid-range init puts the gate crossing inside the trial where there
    # is gradient to move it toward the correct interval.
    "pka_thresh": 4.0,
}


# Architecture-unique keys per CBT family (nuclei dropped; extra runtime keys;
# extra initial-state keys).
_CBT_FAMILY_STRUCTURE = {
    "cbt_loop": {
        "drop_rnn": (),
        # PKA redesign (cbt_loop only): PKA is a mass-action-bounded pool in (0,1)
        # fed DIRECTLY as bg_nln's excitability b — NO separate soft-threshold gate.
        # pka_max=1 keeps it a valid b; both D1 and D2 PKA rest ~0.5 (so bg_nln≈nln
        # at rest) via the rebalanced tonic adenosine drive m_a1/m_a2. Dopamine
        # raises D1 PKA and brakes D2 PKA; adenosine does the inverse. noSC /
        # noSCnoSTN keep the canonical linear-integrator + soft-threshold-gate path.
        "extra_runtime": {"pka_saturation": "mass_action", "pka_max": 1.0, "m_a1_cap": 0.08,
                          "pka_init_floor": 0.4, "pka_init_cap": 0.6},
        # Start PKA LOW so it ramps up over the trial (a rising clock), rather than
        # resting at its ~0.5 equilibrium from t=0. Mirrors promising_version
        # (pka_d10=0.1). The tonic-adenosine tuning (m_a1/m_a2) still sets the
        # equilibrium near 0.5, so PKA ramps 0.1 -> ~0.5 across the delay.
        "extra_init": {"pka_d10": 0.5, "pka_d20": 0.5},  # clamped to [0.4,0.6] at use
        # Adenosine drives BALANCED (m_a1 ~= m_a2) so A1R inhibition on D1 doesn't
        # swamp its (small) DA drive; m_a1 also CAPPED (extra_runtime m_a1_cap) so
        # training can't regrow A1R and collapse dSPN excitability. Keeps D1 alive.
        "extra_weight_init": {"m_a1": 0.06, "m_a2": 0.07},
    },
    "cbt_loop_noSC": {
        "drop_rnn": ("n_sc",),
        # noSC models DA/adenosine as dynamic concentrations (only it reads these).
        # Both concentrations AND PKA use MASS-ACTION kinetics: production is
        # throttled by available substrate (1 - C/C_max) so each pool saturates at
        # its C_max instead of growing without bound. PKA is then fed directly into
        # bg_nln as excitability b (no legacy per-step state squash). Plus the
        # cbt_loop D1-preservation guards (mid-range PKA init, capped A1R gain,
        # balanced m_a1/m_a2). da_max/ado_max bound the DA/adenosine pools.
        "extra_runtime": {
            "tau_da": 20.0, "tau_ado": 200.0,
            "da_release": 1.0, "ado_release": 1.0,
            "da_max": 1.0, "ado_max": 1.0,
            "stn_pacer_min": 0.05,
            "pka_saturation": "mass_action", "pka_max": 1.0, "m_a1_cap": 0.08,
        },
        "extra_init": {"x_da0": 0.1, "x_ado0": 0.1, "pka_d10": 0.5, "pka_d20": 0.5},
        "extra_weight_init": {"m_a1": 0.06, "m_a2": 0.07},
    },
    "cbt_loop_noSCnoSTN": {
        "drop_rnn": ("n_sc", "n_stn"),
        # PKA redesign PORTED from cbt_loop: mass-action-bounded PKA fed directly to
        # bg_nln (no soft-threshold gate / no per-step state squash), capped A1R,
        # clamped PKA inits, balanced adenosine. Same values as cbt_loop.
        # Dynamic DA/adenosine concentration model PORTED from noSC (mass-action
        # x_da/x_ado states; DA fast, adenosine slow; substrate-bounded at *_max).
        "extra_runtime": {"pka_saturation": "mass_action", "pka_max": 1.0, "m_a1_cap": 0.08,
                          "pka_init_floor": 0.4, "pka_init_cap": 0.6,
                          "tau_da": 20.0, "tau_ado": 200.0,
                          "da_release": 1.0, "ado_release": 1.0,
                          "da_max": 1.0, "ado_max": 1.0},
        "extra_init": {"pka_d10": 0.5, "pka_d20": 0.5, "x_da0": 0.1, "x_ado0": 0.1},
        "extra_weight_init": {"m_a1": 0.06, "m_a2": 0.07},
    },
}

# --- corticothalamic (2-node ctx/thalamus RNN; architecture-specific) ---
# Dale's-law corticothalamic testbed: cortex cU/cL/cI + thalamus t_exc/t_inh, all
# populations sign-constrained (exc/inh). Sizes match the CBT cortex ratios
# (cU=cL=cI=10) with thalamus 20/10 (2:1 E/I). g is the shared weight gain
# (fan-in-scaled like the CBT families).
CORTICOTHALAMIC_RNN_CONFIG = {
    "n_c_U": 10,
    "n_c_L": 10,
    "n_c_inh": 10,
    "n_t_exc": 10,   # homogenized to the CBT canonical (CBT_RNN_CONFIG)
    "n_t_inh": 5,    # homogenized to the CBT canonical (CBT_RNN_CONFIG)
    "n_output": 1,
    "noise_std": 0.01,
    "g": 1.0,          # shared weight gain (fan-in-scaled)
    # Spectral-normalize the assembled loop at init to rho(M)=balanced_target_rho
    # (reuses loop_init.normalize_loop, same 17-block structure as the CBT families).
    # Seed-invariant + desaturating, vs hand-tuning g. Applied once; weights trainable after.
    "balanced_init": True,
    "balanced_target_rho": 1.0,
}
CORTICOTHALAMIC_RUNTIME_CONFIG = {
    "tau_ctx": 7.0,   # ~20 ms single-neuron decay (nln-modified) at dt=10 ms
    "tau_t": 7.0,
    "in_scale": 0.25,   # cue -> cortex (free-sign external drive)
    "out_scale": 0.2,   # thalamus -> readout (free-sign)
    "x_init": 0.1,      # initial state (all populations)
}

# --- vanilla (single hidden layer RNN; architecture-specific) ---
VANILLA_RNN_CONFIG = {
    "n_hidden": 32,
    "n_output": 1,
    "noise_std": 0.01,
}
VANILLA_RUNTIME_CONFIG = {
    "tau": 20.0,
    "rec_scale": 0.15,
    "in_scale": 0.25,
    "out_scale": 0.5,
    "x_init": 0.1,   # initial hidden state
}

_ARCH_RNN_CONFIG = {"corticothalamic": CORTICOTHALAMIC_RNN_CONFIG, "vanilla_rnn": VANILLA_RNN_CONFIG}

# Per-family output filenames.
_CBT_FILENAMES = {
    "params": "params_shaped.pkl",
    "pavlovian": "params_pavlovian.pkl",
    "hybrid": "params_hybrid.pkl",
    # hybrid trained FROM SCRATCH (init_params, no Pavlovian bootstrap); kept
    # separate so it never clobbers the curriculum-trained params_hybrid.pkl.
    "hybrid_scratch": "params_hybrid_scratch.pkl",
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


def init_state_for(family):
    """Per-area initial state for a CBT family (canonical + architecture extras,
    e.g. noSC's DA/adenosine concentration states)."""
    d = dict(CBT_INIT_STATE)
    d.update(_CBT_FAMILY_STRUCTURE[family]["extra_init"])
    return d


def weight_init_for(family):
    """Scalar init values of non-fan-in-scaled trainable weights (m_a1/m_a2,
    out_gain/out_bias, k_a). Canonical is shared across the CBT trio; a family may
    override specific values via its structure's optional "extra_weight_init"."""
    d = dict(CBT_WEIGHT_INIT)
    if family in CBT_FAMILIES:
        d.update(_CBT_FAMILY_STRUCTURE[family].get("extra_weight_init", {}))
    return d


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
    ns.hybrid_scratch_params_path = lambda: d / fn["hybrid_scratch"]
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
