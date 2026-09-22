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
    "entropy_coef": 0.1,
    "baseline_momentum": 0.99,
    # Optimize log P(first response lands in the reward window) directly.
    # NOT "loss" (dense BCE): BCE scores timesteps independently and is nearly blind
    # to what actually sets the reward. Cutting pre-window firing 0.0163 -> 0.001
    # improves mean BCE by only ~0.011 (noise on a ~0.5 loss) but improves the
    # hazard reward ~39,000x, because pre-window firing compounds over ~684 steps.
    # Measured: hybrid-from-scratch reward 6e-4 under BCE vs 0.88 under log_reward.
    "objective_mode": "log_reward",
    "brevity_coef": 1.0,
    "silence_coef": 1.0,
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

# Supervised THALAMIC-READOUT mode (cbt_loop_noSCnoSTN). A dense-target alternative to
# the REINFORCE path: the readout is a separate positive-weight vector off the thalamic
# excitatory relay pool (cbt_rnn "C_thal", selected by runtime key readout_source =
# "thalamus"), and the target is a soft step rather than the 0/1 response window.
# Deliberately SELF-TIMED ONLY -- the step is anchored to the single STMT cue, so the
# hybrid / pavlovian two-cue variants are not valid targets for it and train_supervised_thal
# builds the self-timed task unconditionally.
SUPERVISED_THAL_CONFIG = {
    # Baseline == nln(out_bias) with a SILENT thalamus. Under the nln readout the bias must
    # be atanh(target_lo), NOT logit(): logit(0.25) = -1.0986 puts the pre-activation below
    # zero on 100% of timesteps, where max(0,tanh(.)) returns exactly 0 with exactly 0
    # gradient -- the readout then passes NO gradient back and nothing upstream can learn.
    # 0.10 rather than 0.0: a positive baseline keeps the 95% of off-window timesteps
    # pushing the readout AWAY from the dead zone. target_lo = 0.0 would make baseline free
    # (any z<=0 matches exactly) but would drive the readout INTO the absorbing dead zone.
    "target_lo": 0.10,
    "target_hi": 0.75,     # step height
    "hold": 50,            # timesteps held at target_hi
    # Offset from CUE ONSET at which the step opens. t_wait == 300, so the plateau runs
    # [cue+300, cue+350): the network must bridge a 300-step delay on its own, which is
    # what actually exercises the PKA clock. (Was t_cue = 10, i.e. the step opened as the
    # cue ended -- a ~10-step cue-following reflex that never tested self-timing.)
    # NOTE the reinforce reward window opens 10 steps later, at t_cue + t_wait = 310;
    # use that instead if you want the supervised target to coincide with it exactly.
    "delay": TASK_CONFIG["t_wait"],
    # IN-WINDOW LOSS WEIGHT. The target is lo for ~95% of timesteps and hi for ~5%, so
    # unweighted, 95% of the gradient says "hold baseline" and the optimal CONSTANT output is
    # 0.95*lo + 0.05*hi = 0.2750 -- which is exactly where every collapsed run converged.
    # supervised_loss uses the mask as a per-timestep WEIGHT (sum(per_t*mask)/sum(mask)), so
    # upweighting in-window steps rebalances it; (1-f)/f = 19 makes the two classes equal.
    # Measured on the 300-step task at 4200 iters: 4/10 breakthroughs unweighted -> 9/10 at 19x.
    "in_window_weight": 19.0,
    # FREEZE the t=0 state parameters (cbt_rnn.INIT_STATE_KEYS) so the optimiser cannot tune
    # them. They are ordinary trainable params by default, which has caused two silent
    # failures: x_da0 trains slightly negative and the [0,1] clip rectifies the INITIAL
    # dopamine to exactly 0 (seed 42 ended there), and pka_d10/d20 train ABOVE pka_init_cap
    # so the clip discards the learned value. Frozen, every trial starts from the
    # CBT_INIT_STATE values and the network must produce its dynamics from the weights.
    "freeze_init_states": True,
    "loss_type": "mse",    # soft targets: squared error, not BCE (whose min is a nonzero floor)
    "num_iters": 10000,
    "log_interval": 200,
}

TEST_CONFIG = {
    "n_seeds": 5,
    "noise_std": 0.1,
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
    "n_c_inh": 20,
    "n_d1": 10,
    "n_d2": 10,
    "n_snc": 6,
    "n_snr": 8,
    "n_gpe": 6,
    "n_stn": 6,      # dropped for noSCnoSTN
    "n_sc": 6,       # dropped for noSC and noSCnoSTN
    "n_t_exc": 10,
    "n_t_inh": 10,
    "n_med": 4,
    "n_input": 1,
    "n_output": 1,
    "g_bg": 1.0,
    "g_nm": 1.0,
    "noise_std": 0.05,
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
    "tau_pka_fall": 500.0,  # moderate lengthening: longer memory than 500, less saturated than 1440
    "tau_pka_rise": 50.0,
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
    "snr_pacer_min": 0.1,
    "gpe_pacer_min": 0.45,
    "gpe_pacer_max": 0.8,
    "stn_pacer_max": 0.3,
}

# Per-area initial state (resting/baseline activity each area starts at, before
# the per-step noise+nln). One value per area, canonical = cbt_loop. Families that
# store these as trainable params (cbt_loop, noSCnoSTN) use them as the init value;
# noSC uses them directly as fixed initial conditions.
CBT_INIT_STATE = {
    "x_c0_U": 0.01,
    "x_c0_L": 0.01,
    "x_c0_inh": 0.4,
    "x_d10": 0.15,
    "x_d20": 0.15,
    "x_snc0": 0.01,
    "x_gpe0": 0.05,
    "x_stn0": 0.1,   # STN families only
    "x_snr0": 0.25,
    "x_sc0": 0.1,    # SC families only
    "x_t0_exc": 0.1,
    "x_t0_inh": 0.4,
    "x_med0": 0.1,
    "pka_d10": 0.5,
    "pka_d20": 0.5,
}

# Scalar initial VALUES of trainable weight params that aren't fan-in-scaled
# (canonical = cbt_loop). Each family reads the subset it has: cbt_loop uses all;
# noSC has no k_a (it uses DA/adenosine concentration dynamics); noSCnoSTN has no
# out_gain/out_bias (plain nln readout).
CBT_WEIGHT_INIT = {
    # Per-SPN DA-sensitivity gains (mean of the exponential init). Explicit scale so the
    # gain does NOT depend on the SNc pool size (replaces the vestigial 1/n_snc fan-in
    # scaling that dated from a per-connection (n_d, n_snc) design). 0.25 == old 1/n_snc
    # at n_snc=4, preserving the current magnitude.
    "m_d1": 0.9,          # D1R excitatory drive on D1 PKA (per-SPN gain)
    "m_d2": 0.09,          # D2R inhibitory drive on D2 PKA (per-SPN gain)
    "m_a1": 0.05,          # A1R inhibitory drive on D1 PKA (per-SPN gain)
    "m_a2": 0.5,          # A2R excitatory drive on D2 PKA (per-SPN gain)
    "out_gain": 4.0,       # readout gain
    "out_bias": 0.1003353,   # readout bias = atanh(0.10), paired with the nln readout
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
                          "pka_init_floor": 0.2, "pka_init_cap": 0.25},
        # Start PKA LOW so it ramps up over the trial (a rising clock), rather than
        # resting at its ~0.5 equilibrium from t=0. Mirrors promising_version
        # (pka_d10=0.1). The tonic-adenosine tuning (m_a1/m_a2) still sets the
        # equilibrium near 0.5, so PKA ramps 0.1 -> ~0.5 across the delay.
        "extra_init": {"pka_d10": 0.25, "pka_d20": 0.25},  # clamped to [0.4,0.6] at use
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
        # bg_nln as excitability b (no legacy per-step state squash). da_max/ado_max
        # bound the DA/adenosine pools.
        # DA/adenosine + PKA knobs are kept IN LOCKSTEP with cbt_loop_noSCnoSTN
        # (da_pka_gain, da/ado_release, m_a1_cap, pka_init_floor/cap, pka_d10/d20);
        # only stn_pacer_min is noSC-specific (STN).
        "extra_runtime": {
            "tau_da": 50.0, "tau_ado": 50.0,
            "da_release": 0.5, "ado_release": 0.5,
            "da_max": 1.0, "ado_max": 1.0,
            "stn_pacer_min": 0.05, "nt_mode": "forward_euler",
            "pka_saturation": "mass_action", "pka_max": 1.0, "m_a1_cap": 1.0,
            "pka_init_floor": 0.2, "pka_init_cap": 0.25,
            "da_pka_gain": 1.0,
        },
        "extra_init": {"x_da0": 0.1, "x_ado0": 0.1, "pka_d10": 0.2, "pka_d20": 0.2},
        "extra_weight_init": {"m_a1": 0.06, "m_a2": 0.07},
    },
    "cbt_loop_noSCnoSTN": {
        "drop_rnn": ("n_sc", "n_stn"),
        # PKA redesign PORTED from cbt_loop: mass-action-bounded PKA fed directly to
        # bg_nln (no soft-threshold gate / no per-step state squash), capped A1R,
        # clamped PKA inits, balanced adenosine. Same values as cbt_loop.
        # Dynamic DA/adenosine concentration model PORTED from noSC (mass-action
        # x_da/x_ado states; DA fast, adenosine slow; substrate-bounded at *_max).
        "extra_runtime": {"nt_mode": "forward_euler", "pka_saturation": "mass_action",
                          "pka_max": 1.0, "m_a1_cap": 1.0, "pka_init_floor": 0.25, "pka_init_cap": 0.75,
                          # Adenosine CLAMPED to a fixed tonic level: x_ado is held here every
                          # step instead of integrating mean_stri. Severs the pka_d2 -> D2 ->
                          # x_ado -> prod_d2 positive feedback, so pka_d2 is monostable (with
                          # dynamic adenosine it has an UNSTABLE fixed point at 0.267,
                          # lambda = 1.0045). extra_weight_init below is SOLVED for this value
                          # -- change one and the PKA rest point moves. Set to None to restore
                          # the dynamic pool. tau_ado/ado_release are then unused.
                          # Adenosine CLAMPED. Tried unpinned (2026-09-10): training stalled
                          # at the trivial constant (separation 0.0000 at step 1800, where the
                          # working run was already at 0.4698), so the clamp went back on.
                          # ado_release / tau_ado are INERT while this is set.
                          "pin_ado": 0.256,
                          # SNc pacer cap raised 0.2 -> 0.5 (family-scoped; the shared
                          # CBT_RUNTIME_CONFIG value is untouched so cbt_loop / noSC keep 0.2).
                          # snc_pacer = snc_pacer_min + sigmoid(P_snc)*(max-min), so the cap is a
                          # HARD ceiling on SNc's intrinsic drive -- training P_snc cannot exceed
                          # it. At 0.2 the pacer (~0.13) loses to ~0.5 of D2+GPe inhibition and
                          # max(0,tanh(.)) pins SNc at exactly 0, which zeroes x_da and makes
                          # pka_d1 unreachable at any gain. See tests/pka_stability/.
                          # Measured SNc / x_da / pkaD1-at-trial-end over the cap (fresh init,
                          # train seed, hybrid batch, no pins):
                          #   0.2 -> 0.000 / 0.000 / 0.085      0.5 -> 0.010 / 0.006 / 0.111
                          #   1.0 -> 0.119 / 0.063 / 0.273      2.0 -> 0.171 / 0.089 / 0.333
                          # 0.8 is chosen because it lands on the SNc ~ 0.1 operating point the
                          # gains below were solved at almost exactly: pkaD1 0.236 vs 0.235 and
                          # D1 0.167 vs 0.168 against the pinned search reference. 1.0 also clears
                          # the floor but overshoots (D1 0.323, ~2x the searched value, eating the
                          # D1 headroom the gains were tuned for); 0.6 undershoots (SNc 0.037).
                          # The init search itself ran with pin_snc/pin_ctx = 0.1, which are
                          # deliberately NOT set here -- pinning cortex would stop the cue driving
                          # the loop entirely and the model could not do the task.
                          "snc_pacer_max": 0.8,
                          "tau_da": 50.0, "tau_ado": 50.0,
                          # DA release gain: RAW (exc=sigmoid wraps it), effective
                          # sigmoid(0.1845) = 0.546. From the init search rank-1 config
                          # (tests/init_search/, tag ado0sweep_pinSNC_pinCTX).
                          # ado_release is INERT while pin_ado is set (the clamp overwrites the
                          # x_ado integrator every step); kept for the pin_ado=None path.
                          "da_release": 0.4973, "ado_release": 0.9892,  # raw = atanh(effective 0.460)
                          "da_max": 1.0, "ado_max": 1.0,
                          # de-saturate D1: base da_pka_gain=4.0 drove pkaD1 to 0.77 (D1
                          # pinned ~0.99, no dynamic range under the clip/exp init). 1.0
                          # rests pkaD1 ~0.5 (design target) with D1 headroom to gate.
                          "da_pka_gain": 1.0,
                          # --- striatal E/I (8-D init search, tag ei8d_bgnln_recttanh) ---
                          # The 6 DA/adenosine gains only set PKA, which reaches the striatum
                          # solely as the bg_nln slope a = 0.75/(1-b) -- a MULTIPLIER. It cannot
                          # make a net-negative input positive, so no gain value revives a dead
                          # pathway: the 6-D sweep capped at alive_both = 0.156 over 38 configs.
                          # These two can, and with them alive_both reaches 0.999:
                          #   stri_cross_scale -- scales BOTH D1<->D2 cross-projections
                          #     (B_d1_d2, B_d2_d1) at init. D1 and D2 mutually inhibit, so
                          #     weakening one side alone just moves activity across; both move
                          #     together. 0.0 (delete collaterals) scored only 0.3% better, so
                          #     this keeps them.
                          #   stri_tonic -- constant added to the cortex->striatum init
                          #     magnitudes (B_cU_d1, B_cU_d2): the tonic drive the loop lacks.
                          # Applied to MAGNITUDES in init_params before the wrapper-aware
                          # logit init, so exc(raw) reproduces the intended effective weight.
                          "stri_cross_scale": 0.423, "stri_tonic": 0.158,
                          # FLOOR on the four DA/adenosine PKA gains. exc = clip(tanh(w),0,None)
                          # lets the optimiser walk a gain just past zero, where it is rectified
                          # to exactly 0 AND loses its gradient permanently -- m_a1 did precisely
                          # this in the seed-42 20k run (raw +0.036 -> -0.0085, effective 0.036 ->
                          # 0.0000), deleting the A1R adenosine brake on D1 PKA entirely.
                          # Applied as floor + (1-floor)*exc(w), NOT max(exc(w), floor): the
                          # affine form keeps a nonzero gradient everywhere, whereas a hard max
                          # would recreate the same dead zone one step lower.
                          "m_gain_floor": 0.1},
        # normalize_loop scales the 17 cortico-thalamic blocks but NOT B_snr_t_exc, the
        # SNr->thalamus inhibition that opposes them (see its docstring: "every other
        # projection (cue, BG, readout) is left alone"). At rho=1.0 that leaves the thalamic
        # net input at -0.139 -- SNr row-sum -0.902 vs cortical drive +0.157 -- so the pool
        # rectifies to EXACTLY zero on 95% of timesteps and the thalamic readout has nothing
        # to read (C_thal gradient 8e-5 vs out_bias 8.8e-3; training flatlines on the best
        # constant). NOTE the loop is NOT sub-critical: measured rho* ~= 0.99 here, and it
        # FALLS as gain rises (tanh saturation). This is a DC operating-point fix, not a
        # criticality fix -- 1.107 adds enough excitation to clear the rectifier threshold.
        # Measured at 1.107: thalamus 99.7% alive (mean 0.255), cortex 0.384, D1 0.428,
        # D2 0.576, rho* 0.915. The principled fix is to rebalance B_snr_t_exc against the
        # cortical drive instead; this is the blunt version that unblocks training.
        "extra_rnn": {"balanced_target_rho": 1.107},
        # PKA starts LOW (0.25), not 0.5: with the de-saturated loop, pka=0.5 sits in the
        # SATURATING bg_nln regime (D1/D2->~0.95) and, because pka rises fast but falls
        # slowly (tau_pka_fall>>rise), lingers there as a mid-trial spike. A low start
        # (below the production level) lets pka rise into band with no saturated transient,
        # and gives the PKA clock room to ramp UP over the trial (the self-timing signal)
        # rather than sitting flat. See tests/init_state_fix/.
        # 0.25 == pka_init_cap: the init search scores the trial with BOTH PKAs held at
        # 0.25, so start them there (tests/init_search/).
        "extra_init": {"pka_d10": 0.5, "pka_d20": 0.5, "x_da0": 0.1, "x_ado0": 0.062},
        # DA/adenosine PKA gains: rank-1 config of the init search under the CLAMPED-adenosine
        # model (tests/init_search/, tag ado0sweep_pinSNC_pinCTX; 168 configs, 6-D sweep where
        # ado0 replaced the -- inert under the clamp -- g_ado_release).
        #   effective:  m_d1 0.643   m_d2 0.074   m_a1 0.024   m_a2 0.619
        #               g_da_release 0.546 (extra_runtime da_release)   ado0 0.062 (pin_ado)
        #   RAW below = logit(effective), since exc()=sigmoid wraps them in the forward.
        # Measured for this config: track_pka 1.000, alive_both 1.000, pkaD1 0.24+-0.01,
        # pkaD2 0.24+-0.01, D1 0.18, D2 0.18, x_da 0.05 -- i.e. BOTH PKAs hold the 0.25 target
        # all trial with both pathways mid-band. All six values landed INTERIOR to their search
        # ranges (no railing), and the winners cluster on the closed-form D2 locus
        # m_a2 * ado0 = P_req = 0.0370 (rank 1 within 3%; corr(score, log-distance) = -0.83),
        # so the search and the algebra agree.
        # CAVEAT: the search ran with SNc and cortex CLAMPED at 0.1 (cbt_rnn pin_snc / pin_ctx,
        # both left None here) to supply the DA drive that a rectified-tanh nln otherwise kills.
        # These gains therefore assume x_da ~ 0.05, which requires SNc ~ 0.1. snc_pacer_max
        # above is the intended real mechanism for that; verify SNc actually leaves zero before
        # trusting the D1 side, since prod_d1 = max(G*m_d1*x_da - m_a1*pin_ado, 0) rectifies to
        # exactly 0 whenever x_da does.
        # RAW logits of the 8-D search winner (rank-1 with collaterals kept, cross>=0.15):
        #   effective  m_d1 0.757  m_d2 0.177  m_a1 0.032  m_a2 0.502
        #              g_da_release 0.574 (extra_runtime da_release)  ado0 0.093 (pin_ado)
        #              cross 0.162  tonic 0.220
        # Measured: alive_both 0.999, track_pka 1.000 (was 0.156 / 0.913 under the 6-D gains).
        # RAW = atanh(effective): exc/inh are +-clip(tanh(w), 0, None), NOT sigmoid. Writing
        # logit() here (as the pre-2026-09-10 values did) maps every NEGATIVE value to exactly
        # 0 under tanh-clip, which is what silently zeroed m_d2 and m_a1.
        # From the 8-D sweep re-run under tanh-clip WITH the pka_d2 saddle constraint
        # (tests/init_search, tag ei8d_tanhclip_saddle, 173 configs).
        #   effective: m_d1 0.602  m_d2 0.283  m_a1 0.036  m_a2 0.139
        #              da_release 0.460   pin_ado 0.256   cross 0.423   tonic 0.158
        #   measured: saddle_safe 1.000, alive_both 0.998, track_pka 1.000, pkaD2 0.188.
        # NOT the top-ranked config (2.458 vs 2.467): ranks 1-6 are statistically tied while
        # m_a2 spans 0.12-0.82, and m_a2 is the parameter implicated in BOTH collapses --
        # training drove it to 0 in the model that trained stably and amplified it 0.502 ->
        # 0.589 in the one that died at step 1400. This picks the low-m_a2 end of the tie.
        "extra_weight_init": {"m_d1": 0.6963, "m_d2": 0.2909,
                              "m_a1": 0.0360, "m_a2": 0.1399},
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
    "noise_std": 0.05,
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
        # Family-scoped RNN_CONFIG overrides, same shape as extra_runtime above. Lets one
        # family retune a canonical value (e.g. balanced_target_rho) without moving it for
        # the others.
        cfg.update(_CBT_FAMILY_STRUCTURE[family].get("extra_rnn", {}))
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
    ns.SUPERVISED_THAL_CONFIG = dict(SUPERVISED_THAL_CONFIG)
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
