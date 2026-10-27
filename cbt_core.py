"""Shared code for every CBT family (cbt_simple / cbt_STN / cbt_SC_STN).

Anything that is the same in every family lives here, so a change made once applies to all
of them. Reference for every definition is cbt_loop_noSCnoSTN (cbt_simple); see
docs/CHANGELOG_opto_recovery.md.

Not named cbt_rnn.py on purpose: each family directory keeps a cbt_rnn.py, and a top-level
module of the same name would be shadowed by it (scripts run from a family dir put that dir
first on sys.path), so the family file could never import this one.

Contents:
  * shared helpers (sign wrappers, nonlinearities, freezing, evaluation/analysis);
  * the CORE CIRCUIT: init_core_params + multiregion_rnn, written as one update function
    per area run in CORE_AREA_ORDER, with W["ext"] hooks where area modules plug in.
Family modules bind these with one-line wrappers (their own STATE_AREA_ORDER, batched_rnn,
rnn_func), so each family's public API is unchanged.
"""
import math

import jax.numpy as jnp
import jax.random as jr
import optax
from jax import lax
from jax.nn import sigmoid

import config_script
import loop_init
import self_timed_movement_task as stmt


# --------------------------------------------------------------------------- #
# Sign wrappers and nonlinearities. self_timed_movement_task holds the
# definitions (its losses use them too); these names are what the circuit code calls.
# --------------------------------------------------------------------------- #
def exc(w):
    return stmt.exc(w)


def inh(w):
    return stmt.inh(w)


def nln(x):
    return stmt.nln(x)


def bg_nln(x, b):
    return stmt.bg_nln(x, b)


def no_autapse(m):
    """Zero the diagonal of a square within-population recurrence (no self-synapse)."""
    return m * (1.0 - jnp.eye(m.shape[0], dtype=m.dtype))


def match_input_channels(inputs, params):
    """Zero-pad cue input channels to match the model's cue→cortex width.

    Tasks that emit fewer cue channels than the model expects (e.g. a 1-channel
    STMT cue fed to a model trained with an extra Pavlovian-cue channel) are
    padded with zeros — the missing cues are simply treated as absent.
    """
    n_expected = params["B_cue_cU"].shape[1]
    n_have = inputs.shape[-1]
    if n_have >= n_expected:
        return inputs
    pad = jnp.zeros(inputs.shape[:-1] + (n_expected - n_have,), dtype=inputs.dtype)
    return jnp.concatenate([inputs, pad], axis=-1)


# --------------------------------------------------------------------------- #
# Freezing the trainable t=0 state parameters.
# --------------------------------------------------------------------------- #
def init_state_mask(params, init_state_keys):
    """Pytree of bools, True for every initial-state param present in `params`."""
    return {k: (k in init_state_keys) for k in params}


def freeze_init_state_optimizer(optimizer, params, init_state_keys):
    """Wrap `optimizer` so `init_state_keys` receive zero updates (frozen at their init values).

    Applied AFTER the inner optimiser, so adam/adamw statistics are untouched and only the
    emitted update is zeroed. Returns `optimizer` unchanged if no init-state params exist.
    """
    mask = init_state_mask(params, init_state_keys)
    if not any(mask.values()):
        return optimizer
    return optax.chain(optimizer, optax.masked(optax.set_to_zero(), mask))


# --------------------------------------------------------------------------- #
# Evaluation and analysis helpers.
# --------------------------------------------------------------------------- #
def evaluate(batched_rnn, params, config, all_inputs, noise_std=None, n_seeds=8):
    all_ys = []
    all_xs = []
    all_as = []

    eval_config = dict(config)
    if noise_std is not None:
        eval_config["noise_std"] = noise_std

    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    batch_stim = jnp.zeros((all_inputs.shape[0], all_inputs.shape[1], n_d1 + n_d2))

    for seed in range(n_seeds):
        rng_key = jr.PRNGKey(seed)
        rng_key, action_key = jr.split(rng_key)
        batch_rng_keys = jr.split(rng_key, all_inputs.shape[0])
        ys, xs = batched_rnn(params, eval_config, all_inputs, batch_stim, batch_rng_keys)
        actions = jr.bernoulli(action_key, p=ys).astype(ys.dtype)
        all_ys.append(ys)
        all_xs.append(xs)
        all_as.append(actions)

    all_ys = jnp.stack(all_ys)
    all_xs = [jnp.stack(parts) for parts in zip(*all_xs)]
    all_as = jnp.stack(all_as)
    return all_ys, all_xs, all_as


def get_brain_area(brain_area, xs, area_order):
    """Return the state array for ``brain_area`` from a family's state tuple ``xs``.

    ``area_order`` is the family's STATE_AREA_ORDER.
    """
    return xs[area_order.index(brain_area)]


def get_response_times_opto(ys, cue_start=0, dt=0.01, threshold=0.5, exclude_nan=True, rng_key=None):
    """First-crossing response times for opto-format outputs.

    Args:
        ys: array of shape (n_seeds, T, 1) or (n_seeds, T).
        cue_start: timestep to start looking for a response.
        dt: timestep duration in seconds.
        threshold: output threshold counted as a response.
        exclude_nan: if True, drop trials with no response; otherwise keep NaN.
        rng_key: if provided, sample Bernoulli actions from ``ys`` (treated as
            probabilities) before thresholding. The network is a stochastic
            policy whose probability output rarely exceeds 0.5 deterministically,
            so thresholding the raw probabilities yields empty CDFs. If None,
            threshold raw values directly (legacy behavior).

    Returns:
        1-D array of response times (seconds) when exclude_nan, else (n_seeds,).
    """
    if ys.ndim == 3:
        ys = ys[..., 0]
    if rng_key is not None:
        signal = jr.bernoulli(rng_key, p=jnp.clip(ys, 0.0, 1.0)).astype(ys.dtype)
    else:
        signal = ys
    post = signal[:, cue_start:]
    crossed = post > threshold
    has_resp = jnp.any(crossed, axis=1)
    first_idx = jnp.argmax(crossed, axis=1)
    rts = jnp.where(has_resp, first_idx.astype(jnp.float32) * dt, jnp.nan)
    if exclude_nan:
        return rts[~jnp.isnan(rts)]
    return rts


def get_d1_d2_ratio(all_xs, t_start, t_end, area_order, avg_time=True, remove_outliers=False):
    """D1 minus D2 mean activity over a time window.

    Args:
        all_xs: state list/tuple of arrays shaped (n_seeds, n_conditions, T, N).
        t_start, t_end: time-window bounds (timesteps).
        area_order: the family's STATE_AREA_ORDER.
        avg_time: if True, average over the window -> (n_seeds, n_conditions).
        remove_outliers: if True, NaN out per-condition outliers (z > 3).

    Returns:
        (n_seeds, n_conditions[, win]) array of D1-D2 activity differences.
    """
    d1 = get_brain_area("D1", all_xs, area_order)
    d2 = get_brain_area("D2", all_xs, area_order)
    d1m = jnp.mean(d1[..., t_start:t_end, :], axis=-1)
    d2m = jnp.mean(d2[..., t_start:t_end, :], axis=-1)
    ratio = d1m - d2m
    if avg_time:
        ratio = jnp.mean(ratio, axis=-1)
    if remove_outliers:
        z = jnp.abs((ratio - jnp.nanmean(ratio)) / (jnp.nanstd(ratio) + 1e-8))
        ratio = jnp.where(z > 3, jnp.nan, ratio)
    return ratio


def get_slope(all_xs, t_start, t_end, area_order, avg_neurons=True, remove_outliers=False):
    """Least-squares ramp slope of cortical activity over a time window.

    Returns:
        (n_seeds, n_conditions) array of slopes (per unit time-step).
    """
    cortex = get_brain_area("Cortex", all_xs, area_order)  # (n_seeds, n_conditions, T, N)
    seg = cortex[..., t_start:t_end, :]
    # Averaged over neurons either way; avg_neurons is kept for the call signature.
    seg = jnp.mean(seg, axis=-1)  # (n_seeds, n_conditions, win)
    win = seg.shape[-1]
    t = jnp.arange(win, dtype=seg.dtype)
    t_centered = t - jnp.mean(t)
    y_centered = seg - jnp.mean(seg, axis=-1, keepdims=True)
    num = jnp.sum(t_centered * y_centered, axis=-1)
    den = jnp.sum(t_centered ** 2)
    slope = num / (den + 1e-8)
    if remove_outliers:
        z = jnp.abs((slope - jnp.nanmean(slope)) / (jnp.nanstd(slope) + 1e-8))
        slope = jnp.where(z > 3, jnp.nan, slope)
    return slope


# =========================================================================== #
# CORE CIRCUIT (cbt_simple, formerly cbt_loop_noSCnoSTN)
# =========================================================================== #
def init_core_params(rng_key, n_input, family):
    """Initialize the CORE circuit's params and runtime config for `family`.

    The core is cbt_simple (formerly cbt_loop_noSCnoSTN): cortex cU/cL/c_inh, D1/D2,
    GPe, SNr, SNc + DA/adenosine/PKA, thalamus exc/inh, medulla. Area modules (STN, SC)
    add their own blocks on top; they never change how a core block is drawn, so at the
    same seed the core weights start identical in every family.

    Cortex and thalamus follow Dale's law as fully separate populations:
    ``n_*_exc`` excitatory projection neurons and ``n_*_inh`` local inhibitory
    interneurons live in their own state vectors with their own connectivity
    blocks. Only the excitatory pool sends out-of-area projections.

    Every structural/runtime value comes from the central config (crashes on a
    missing key — no silent defaults). n_input is the only per-call argument.
    """
    _rc = config_script.rnn_config_for(family)
    _is = config_script.init_state_for(family)
    _wi = config_script.weight_init_for(family)
    _rt = config_script.runtime_config_for(family)   # for the DA/adenosine release-gain inits
    n_c_U = _rc["n_c_U"]
    n_c_L = _rc["n_c_L"]
    n_c_inh = _rc["n_c_inh"]
    n_d1 = _rc["n_d1"]
    n_d2 = _rc["n_d2"]
    n_snc = _rc["n_snc"]
    n_snr = _rc["n_snr"]
    n_gpe = _rc["n_gpe"]
    n_t_exc = _rc["n_t_exc"]
    n_t_inh = _rc["n_t_inh"]
    n_med = _rc["n_med"]
    n_output = _rc["n_output"]
    g_bg = _rc["g_bg"]
    g_nm = _rc["g_nm"]
    noise_std = _rc["noise_std"]
    balanced_init = _rc["balanced_init"]
    balanced_target_rho = _rc["balanced_target_rho"]
    skeys = jr.split(rng_key, 60)

    # Fan-in scaling (adapted from the promising_version design):
    #  - Feed-forward sign-constrained projections (wrapped in exc()/inh() = |w|)
    #    sum same-signed weights against non-negative rates — nothing cancels, so
    #    the drive would grow as sqrt(fan_in) under 1/sqrt(n) and saturate nln
    #    downstream of a large population. They scale by 1/fan_in, keeping the mean
    #    drive O(1) in area size.
    #  - Recurrent EXCITATORY self-connection blocks (cortex J_cU/J_cL/B_cU_cL/
    #    B_cL_cU, thalamus J_t_ee) keep 1/sqrt(n): the E->E loop gain must stay
    #    >~1 to sustain persistent/ramping activity. 1/fan_in makes it subcritical
    #    (~0.8*g_bg), so the cortex decays to silence and the whole loop starves.
    #  - Pacemaker vectors (P_snc/P_snr/P_gpe) are per-unit biases, not fan-in
    #    sums, so they keep 1/sqrt(n).

    def _mag(key, shape):
        # HALF-NORMAL synaptic MAGNITUDE |N(0,1)| (mean sqrt(2/pi) = 0.798, std 0.603).
        # Same role as before: a one-signed (Dale) magnitude that each block scales by its
        # 1/fan_in (or 1/sqrt(fan_in)) prefactor, so a fan-in-n sum of same-signed weights
        # against non-negative rates keeps its MEAN drive O(1) -- nothing cancels. Dale sign
        # is applied by the +/- prefactor per block (exc +, inh -), so the sign never comes
        # from the draw and nothing is zeroed at init under a clip-based exc/inh.
        # Reverted from jr.exponential (Exp(1), mean 1, std 1): the half-normal has the same
        # shape at the origin but a much lighter tail (std/mean 0.755 vs 1.0), so it produces
        # far fewer outsized synapses. Note the mean magnitude drops 1.0 -> 0.798, i.e. every
        # block is ~20% weaker than under Exp(1). The 17 cortico-thalamic loop blocks are
        # unaffected in aggregate because balanced_init/normalize_loop rescales them to
        # rho = balanced_target_rho afterwards; the BG blocks (striatum, GPe, SNr, SNc,
        # medulla, readout) are NOT normalized and do carry the 0.8x through.
        return jnp.abs(jr.normal(key, shape))

    params = {
        # Cortex split into two excitatory PT-like populations — cU (upper) and
        # cL (lower) — plus a shared inhibitory pool c_inh (Economo et al. 2018,
        # see README). Within-population excitatory recurrence:
        "J_cU": (g_bg / math.sqrt(n_c_U)) * _mag(skeys[2],  (n_c_U, n_c_U)),   # exc
        "J_cL": (g_bg / math.sqrt(n_c_L)) * _mag(skeys[52], (n_c_L, n_c_L)),   # exc
        # Reciprocal excitatory cU <-> cL (post, pre).
        "B_cU_cL": (g_bg / math.sqrt(n_c_U)) * _mag(skeys[53], (n_c_L, n_c_U)),  # cU -> cL exc
        "B_cL_cU": (g_bg / math.sqrt(n_c_L)) * _mag(skeys[54], (n_c_U, n_c_L)),  # cL -> cU exc
        # cU / cL -> c_inh (excitatory); c_inh -> cU / cL and c_inh -> c_inh (inhibitory).
        "J_cU_ci": (g_bg / (n_c_U)) * _mag(skeys[43], (n_c_inh, n_c_U)),        # exc
        "J_cL_ci": (g_bg / (n_c_L)) * _mag(skeys[56], (n_c_inh, n_c_L)),        # exc
        "J_ci_cU": (g_bg / (n_c_inh)) * _mag(skeys[42], (n_c_U, n_c_inh)),     # inh
        "J_ci_cL": (g_bg / (n_c_inh)) * _mag(skeys[55], (n_c_L, n_c_inh)),     # inh
        "J_c_ii": (g_bg / (n_c_inh)) * _mag(skeys[44], (n_c_inh, n_c_inh)),    # inh
        "J_d1": (g_bg / (n_d1)) * _mag(skeys[0], (n_d1, n_d1)),               # inh
        "J_d2": (g_bg / (n_d2)) * _mag(skeys[8], (n_d2, n_d2)),               # inh
        # Pacemaker vectors (no within-area recurrence for SNc/SNr).
        "P_snc": (g_nm / math.sqrt(n_snc)) * _mag(skeys[7], (n_snc,)),         # exc
        "J_gpe": (g_bg / (n_gpe)) * _mag(skeys[19], (n_gpe, n_gpe)),          # inh
        "P_snr": (g_bg / math.sqrt(n_snr)) * _mag(skeys[18], (n_snr,)),        # exc
        "P_gpe": (g_bg / math.sqrt(n_gpe)) * _mag(skeys[34], (n_gpe,)),        # exc
        # Thalamus E/I recurrence.
        "J_t_ee": (g_bg / math.sqrt(n_t_exc)) * _mag(skeys[5],  (n_t_exc, n_t_exc)),  # exc
        "J_t_ei": (g_bg / (n_t_inh)) * _mag(skeys[45], (n_t_exc, n_t_inh)),   # inh
        "J_t_ie": (g_bg / (n_t_exc)) * _mag(skeys[46], (n_t_inh, n_t_exc)),    # exc
        "J_t_ii": (g_bg / (n_t_inh)) * _mag(skeys[47], (n_t_inh, n_t_inh)),   # inh
        # Cue → cortex (all three pools receive) -- excitatory.
        "B_cue_cU": (1 / (n_input)) * _mag(skeys[3],  (n_c_U, n_input)),
        "B_cue_cL": (1 / (n_input)) * _mag(skeys[57], (n_c_L, n_input)),
        "B_cue_c_inh": (1 / (n_input)) * _mag(skeys[48], (n_c_inh, n_input)),
        # Thalamus exc → cortex: reciprocal with cU; feedforward inhibition to c_inh (exc synapse).
        "B_t_cU": (g_bg / (n_t_exc)) * _mag(skeys[4],  (n_c_U, n_t_exc)),
        "B_t_c_inh": (g_bg / (n_t_exc)) * _mag(skeys[49], (n_c_inh, n_t_exc)),
        # cU → thalamus (both thalamic pools receive) -- excitatory.
        "B_cU_t_exc": (1 / (n_c_U)) * _mag(skeys[29], (n_t_exc, n_c_U)),
        "B_cU_t_inh": (1 / (n_c_U)) * _mag(skeys[50], (n_t_inh, n_c_U)),
        # cU → striatum / GPe (basal-ganglia-projecting upper population) -- excitatory.
        "B_cU_d1": (g_bg / (n_c_U)) * _mag(skeys[1], (n_d1, n_c_U)),
        "B_cU_d2": (g_bg / (n_c_U)) * _mag(skeys[12], (n_d2, n_c_U)),
        "B_cU_gpe": (g_bg / (n_c_U)) * _mag(skeys[58], (n_gpe, n_c_U)),
        # cL → SNc (descending lower population; also → medulla E units below) -- excitatory.
        "B_cL_snc": (1 / (n_c_L)) * _mag(skeys[32], (n_snc, n_c_L)),
        "B_d1_snc": (1 / (n_d1)) * _mag(skeys[17], (n_snc, n_d1)),   # D1 -> SNc inh
        "B_d2_snc": (1 / (n_d2)) * _mag(skeys[28], (n_snc, n_d2)),   # D2 -> SNc inh
        "B_d1_snr": (1 / (n_d1)) * _mag(skeys[22], (n_snr, n_d1)),   # D1 -> SNr inh
        "B_d2_gpe": (1 / (n_d2)) * _mag(skeys[24], (n_gpe, n_d2)),   # D2 -> GPe inh
        "B_gpe_snr": (1 / (n_gpe)) * _mag(skeys[40], (n_snr, n_gpe)),  # GPe → SNr inh
        "B_gpe_snc": (1 / (n_gpe)) * _mag(skeys[33], (n_snc, n_gpe)),  # GPe → SNc inh
        # SNr → thalamus (both pools receive) -- inhibitory.
        "B_snr_t_exc": (1 / (n_snr)) * _mag(skeys[6],  (n_t_exc, n_snr)),
        "B_snr_t_inh": (1 / (n_snr)) * _mag(skeys[51], (n_t_inh, n_snr)),
        # Dopamine→PKA gains: a SINGLE scalar per population (one trainable m_d1 shared by
        # all D1 units, one m_d2 shared by all D2 units) -- a whole-population DA
        # sensitivity to the broadcast scalar DA concentration (volume transmission). It
        # broadcasts over the per-neuron pka_d1/pka_d2 states in prod_d1/prod_d2.
        "m_d1": jnp.array(_wi["m_d1"]),
        "m_d2": jnp.array(_wi["m_d2"]),

        # Lateral inhibition between D1 and D2 populations.
        "B_d1_d2": (g_bg / (n_d1)) * _mag(skeys[16], (n_d2, n_d1)),  # D1 → D2 inh
        "B_d2_d1": (g_bg / (n_d2)) * _mag(skeys[31], (n_d1, n_d2)),  # D2 → D1 inh
        # Medullary area: two E/I pairs (E0,I0) and (E1,I1) coupled reciprocally.
        # Each 2×2 block: col 0 = from E (exc, +), col 1 = from I (inh, -). _med_block
        # applies exc()/inh() per column, so the init must already carry those signs.
        "J_med_w1": (g_bg / (2)) * _mag(skeys[13], (2, 2)) * jnp.array([1.0, -1.0]),  # within pair 1
        "J_med_w2": (g_bg / (2)) * _mag(skeys[21], (2, 2)) * jnp.array([1.0, -1.0]),  # within pair 2
        "J_med_x":  (g_bg / (2)) * _mag(skeys[30], (2, 2)) * jnp.array([1.0, -1.0]),  # cross-pair
        "B_cL_med": (1 / (n_c_L)) * _mag(skeys[14], (n_med // 2, n_c_L)),  # cL → medulla E (exc)
        "B_snr_med": (1 / (n_snr)) * _mag(skeys[41], (n_med // 2, n_snr)),  # SNr → Medulla E (inh)
        "C_med": (1 / (n_med // 2)) * _mag(skeys[15], (n_output, n_med // 2)),  # readout (exc)
        # Thalamic readout weights, used when config["readout_source"] == "thalamus".
        # Reads the thalamic EXCITATORY relay pool only -- t_inh is a local (TRN-like)
        # interneuron pool that projects nowhere outside the area, matching how the rest
        # of the model treats thalamus (b_t_cU / b_t_c_inh both read x_t_exc). Wrapped in
        # exc() in the forward, so the effective weights are sigmoid(raw) and therefore
        # STRICTLY POSITIVE and bounded (0,1) -- training cannot flip a sign. 2-D, so the
        # wrapper-aware logit init below applies and sigmoid(raw) reproduces this magnitude.
        "C_thal": (1 / n_t_exc) * _mag(skeys[59], (n_output, n_t_exc)),  # readout (exc)
        #"rb": jnp.abs((1 / (n_med)) * jr.normal(skeys[16], (n_output,))),
        # Output readout gain/bias: y = sigmoid(out_gain*(c_med@x_med_E) + out_bias).
        # out_bias = logit(0.25) gives a nonzero resting response prob so the policy
        # can explore from the start; both trainable. (Ported from cbt_loop.)
        "out_gain": jnp.array(_wi["out_gain"]),
        "out_bias": jnp.array(_wi["out_bias"]),
        # Trainable initial states (resting/baseline activity per area); the
        # starting values are declared centrally (config_script.CBT_INIT_STATE).
        "x_c0_U": jnp.ones((n_c_U,)) * _is["x_c0_U"],
        "x_c0_L": jnp.ones((n_c_L,)) * _is["x_c0_L"],
        "x_c0_inh": jnp.ones((n_c_inh,)) * _is["x_c0_inh"],
        "x_d10":  jnp.ones((n_d1,))  * _is["x_d10"],
        "x_d20":  jnp.ones((n_d2,))  * _is["x_d20"],
        "x_snc0": jnp.ones((n_snc,)) * _is["x_snc0"],
        "x_gpe0": jnp.ones((n_gpe,)) * _is["x_gpe0"],
        "x_snr0": jnp.ones((n_snr,)) * _is["x_snr0"],
        "x_t0_exc": jnp.ones((n_t_exc,)) * _is["x_t0_exc"],
        "x_t0_inh": jnp.ones((n_t_inh,)) * _is["x_t0_inh"],
        "x_med0": jnp.ones((n_med,)) * _is["x_med0"],
        "pka_d10": jnp.ones((n_d1,)) * _is["pka_d10"],
        "pka_d20": jnp.ones((n_d2,)) * _is["pka_d20"],
        # Dynamic DA / adenosine concentration initial states (scalars).
        "x_da0": jnp.array(_is["x_da0"]),
        "x_ado0": jnp.array(_is["x_ado0"]),
        # Adenosine: one tunable tonic level k_a (scalar — will become a
        # dynamic state later) feeding per-SPN weights m_a1 / m_a2, mirroring
        # m_d1 / m_d2 for the broadcast DA gain.
        "k_a": jnp.array(_wi["k_a"]),
        # Single scalar A1R/A2R adenosine gains: one trainable m_a1 shared by all D1 units,
        # one m_a2 by all D2 units (a whole-population adenosine sensitivity, mirroring the
        # scalar m_d1/m_d2). Broadcasts over the per-neuron pka states.
        "m_a1": jnp.array(_wi["m_a1"]),  # A1R inhibitory drive on D1 PKA
        "m_a2": jnp.array(_wi["m_a2"]),  # A2R excitatory drive on D2 PKA
        # Trainable DA / adenosine RELEASE gains (scalars, init from the runtime config;
        # clipped >=0 in the forward). Set how much x_da / x_ado is produced per unit of
        # SNc / striatal activity. Moved out of config so training can tune them.
        "g_da_release": jnp.array(_rt["da_release"]),
        "g_ado_release": jnp.array(_rt["ado_release"]),
    }

    # Biophysical runtime constants are declared centrally (config_script.
    # CBT_RUNTIME_CONFIG + this family's architecture-unique extras).
    config = dict(config_script.runtime_config_for(family))
    config.update({
        "n_c_U": n_c_U,
        "n_c_L": n_c_L,
        "n_c_inh": n_c_inh,
        "n_t_exc": n_t_exc,
        "n_t_inh": n_t_inh,
        "noise_std": noise_std,
    })
    if balanced_init:
        # Spectral-normalize the cortico-thalamic loop so it starts near-critical
        # instead of strongly super-critical (rho ~ 1.76 raw). Without this the
        # loop grows until the sigmoid nln saturates; saturated cortex has ~zero
        # local gain and passes neither the cue forward nor the gradient backward,
        # so the task gradient vanishes. See ../loop_init.py and
        # ../corticothalamic/{loop_criticality,loop_gradient,desaturate_sweep}.py.
        # balanced_target_rho is a CONFIG CONSTANT applied once here, never trained.
        # This family no-autapses the loop self-recurrences at runtime; zero those
        # diagonals FIRST so the normalized (no-autapse) rho hits target exactly.
        for _k in ("J_cU", "J_cL", "J_c_ii", "J_t_ee", "J_t_ii"):
            _m = jnp.asarray(params[_k])
            params[_k] = _m * (1.0 - jnp.eye(_m.shape[0], dtype=_m.dtype))
        params, rho0, rho1 = loop_init.normalize_loop(
            params, n_c_U, n_c_L, n_c_inh, n_t_exc, n_t_inh,
            tau=config["tau_c"], target_rho=balanced_target_rho,
        )
        print(f"[balanced_init] cortico-thalamic loop rho(M): {rho0:.3f} -> {rho1:.3f} "
              f"(target {balanced_target_rho}, tau={config['tau_c']})")

    # Wrapper-aware magnitude init. exc/inh = +/-sigmoid(w) (mass-action, bounded (0,1)),
    # so the EFFECTIVE weight is sigmoid(raw). Every block above (and normalize_loop) was
    # built as an intended fan-in-scaled MAGNITUDE -- the value a linear clip/abs Dale would
    # use directly. Under sigmoid a small raw (~0.05 from the 1/fan_in scaling) collapses to
    # sigmoid~0.5, so ALL weights become ~0.5 -> dense over-excitation pins the whole loop at
    # the top rail (cortex/thal ~0.99, D2->1.0) and defeats normalize_loop (which scales raw
    # magnitudes that sigmoid then flattens). Invert the sigmoid so exc(raw) reproduces the
    # intended magnitude: raw <- logit(|magnitude|). Sign is still supplied by exc/inh in the
    # forward, so this touches connectivity matrices only (2-D; the scalar DA/adenosine/pacer
    # gains are excluded). If exc is linear (clip/abs, exc(0)=0) the magnitudes are already the
    # effective weights, so this is skipped. See tests/loop_desaturation/.
    # --- striatal E/I shaping (see config_script stri_cross_scale / stri_tonic) ---
    # Applied to the intended MAGNITUDES, before the logit init below, so exc(raw) ends up
    # reproducing the shaped effective weight. cross scales both D1<->D2 cross-projections
    # (they must move together -- weakening one side alone just shifts activity between the
    # pathways); tonic adds a constant cortex->striatum drive to both.
    _cross = _rt.get("stri_cross_scale", None)
    if _cross is not None and _cross != 1.0:
        for _k in ("B_d1_d2", "B_d2_d1"):
            params[_k] = jnp.asarray(params[_k]) * _cross
    _tonic = _rt.get("stri_tonic", 0.0)
    if _tonic:
        for _k in ("B_cU_d1", "B_cU_d2"):
            params[_k] = jnp.abs(jnp.asarray(params[_k])) + _tonic

    if float(stmt.exc(jnp.asarray(0.0))) > 0.25:
        _gain_keys = {"m_d1", "m_d2", "m_a1", "m_a2", "g_da_release", "g_ado_release",
                      "P_gpe", "P_snc", "P_snr"}
        for _k in list(params):
            _v = jnp.asarray(params[_k])
            if _k not in _gain_keys and _v.ndim >= 2:
                _m = jnp.clip(jnp.abs(_v), 1e-4, 1.0 - 1e-4)
                params[_k] = jnp.log(_m / (1.0 - _m))

    return params, config


# --------------------------------------------------------------------------- #
# Core circuit forward pass.
#
# One function per area, run in CORE_AREA_ORDER every time step. Each reads the shared
# state dict `s` and writes its own entries back, so an area sees the NEW value of any
# area updated before it in the step and the OLD value of any area after it. That
# ordering is part of the model (e.g. thalamus reads this step's cortex, SNc reads last
# step's D1/D2), so do not reorder without meaning to change the dynamics.
#
# Extension hooks: W["ext"][state_name] is a list of callables f(s, W) -> current. Each is
# added, scaled by the area's 1/tau, just before that area's nonlinearity. They are empty
# for the core; area modules (STN, SC) register their projections into core areas here.
#
# Two deliberate quirks are kept exactly as in the pre-refactor model:
#  * D1/D2 within-population recurrence and the medulla's recurrence read the state
#    AFTER this step's leak, i.e. J @ ((1 - 1/tau) x), not J @ x.
#  * Every area's initial-state noise uses the same init_key, so the t=0 jitter of
#    same-shaped areas is identical across areas.
# --------------------------------------------------------------------------- #
CORE_STATE_VAR_ORDER = (
    "x_c", "x_d1", "x_d2", "x_snc", "x_gpe", "x_snr", "x_t", "pka_d1", "pka_d2", "x_med",
    "x_da", "x_ado",
)

# Trainable INITIAL-STATE parameters (the network's state at t=0). These are ordinary
# params, so by default the optimiser tunes them -- which has bitten this model before:
# x_da0 trains negative and the [0,1] clip rectifies the initial dopamine to exactly 0, and
# pka_d10/d20 train ABOVE pka_init_cap so the clip silently discards the learned value.
# freeze_init_state_optimizer() wraps an optimiser so their updates are zeroed, pinning every
# area's t=0 state at the config_script CBT_INIT_STATE values.
CORE_INIT_STATE_KEYS = (
    "x_c0_U", "x_c0_L", "x_c0_inh", "x_d10", "x_d20", "x_snc0", "x_gpe0", "x_snr0",
    "x_t0_exc", "x_t0_inh", "x_med0", "x_da0", "x_ado0", "pka_d10", "pka_d20",
)

# Per-step process noise: which states get it, in which order the step key is split.
# The order fixes which random stream each area draws, so it is part of reproducibility.
_CORE_NOISE_ORDER = (
    "x_d1", "x_d2", "x_c_U", "x_c_L", "x_c_inh", "x_t_exc", "x_t_inh",
    "x_snr", "x_gpe", "x_snc", "x_med",
)


def core_initial_state(params, config, init_key):
    """t=0 state dict of the core circuit (noisy, rectified, clipped to [0,1])."""
    # Clamp the (trainable) PKA initial states to a sane range so training can't
    # push them to extreme values. (Ported from cbt_loop.)
    _pka_lo = config["pka_init_floor"]; _pka_hi = config["pka_init_cap"]
    s = {
        "pka_d1": jnp.clip(jnp.asarray(params["pka_d10"]), _pka_lo, _pka_hi),
        "pka_d2": jnp.clip(jnp.asarray(params["pka_d20"]), _pka_lo, _pka_hi),
        "x_da": jnp.asarray(params["x_da0"]),
        "x_ado": jnp.asarray(params["x_ado0"]),
    }
    noise_std = jnp.asarray(config["noise_std"])

    def _noisy(name):
        x = jnp.asarray(params[name])
        return nln(x + noise_std * jr.normal(init_key, x.shape))

    # Cortical pools are additionally capped at 0.5.
    s["x_c_U"] = jnp.minimum(_noisy("x_c0_U"), 0.5)
    s["x_c_L"] = jnp.minimum(_noisy("x_c0_L"), 0.5)
    s["x_c_inh"] = jnp.minimum(_noisy("x_c0_inh"), 0.5)
    for state, p in (("x_d1", "x_d10"), ("x_d2", "x_d20"), ("x_snc", "x_snc0"),
                     ("x_gpe", "x_gpe0"), ("x_snr", "x_snr0"), ("x_t_exc", "x_t0_exc"),
                     ("x_t_inh", "x_t0_inh"), ("x_med", "x_med0")):
        s[state] = _noisy(p)

    # HARD INVARIANT: every initial state is a rate or a concentration, so it lives in [0,1].
    # x_da0/x_ado0 bypass nln and are trainable: a 10k-iter train_hybrid run drove x_da0 to
    # -1.05, which rectifies prod_d1 to exactly zero for the whole trial. Clipping every state
    # here also makes the bound independent of nln (some nln options are unbounded above).
    # No-op for any state already in range.
    return {k: jnp.clip(v, 0.0, 1.0) for k, v in s.items()}


def prepare_core_weights(params, config):
    """Effective (sign-wrapped, floored, pacer-mapped) weights + constants for one forward pass.

    Everything here is computed once per trial, outside the time loop. Python floats and
    static options (taus, pins, nt_mode, ...) are carried alongside the arrays.
    """
    W = {"ext": {}}
    # Cortex blocks. Sign follows presynaptic identity (Dale's law). cU/cL are
    # excitatory PT-like populations; c_inh is the shared inhibitory pool.
    W["j_cU"] = no_autapse(exc(params["J_cU"]))        # cU → cU
    W["j_cL"] = no_autapse(exc(params["J_cL"]))        # cL → cL
    W["b_cU_cL"] = exc(params["B_cU_cL"])              # cU → cL
    W["b_cL_cU"] = exc(params["B_cL_cU"])              # cL → cU
    W["j_cU_ci"] = exc(params["J_cU_ci"])              # cU → c_inh
    W["j_cL_ci"] = exc(params["J_cL_ci"])              # cL → c_inh
    W["j_ci_cU"] = inh(params["J_ci_cU"])              # c_inh → cU
    W["j_ci_cL"] = inh(params["J_ci_cL"])              # c_inh → cL
    W["j_c_ii"] = no_autapse(inh(params["J_c_ii"]))    # c_inh → c_inh
    # Thalamus E/I recurrent blocks.
    W["j_t_ee"] = no_autapse(exc(params["J_t_ee"]))    # T_exc → T_exc
    W["j_t_ei"] = inh(params["J_t_ei"])                # T_inh → T_exc
    W["j_t_ie"] = exc(params["J_t_ie"])                # T_exc → T_inh
    W["j_t_ii"] = no_autapse(inh(params["J_t_ii"]))    # T_inh → T_inh

    W["j_d1"] = no_autapse(inh(params["J_d1"]))
    W["j_d2"] = no_autapse(inh(params["J_d2"]))
    # GPe recurrence is built but NOT used in the GPe update (commented out there).
    W["j_gpe"] = no_autapse(inh(params["J_gpe"]))

    # Cue → cortex (all three pools).
    W["b_cue_cU"] = exc(params["B_cue_cU"])
    W["b_cue_cL"] = exc(params["B_cue_cL"])
    W["b_cue_c_inh"] = exc(params["B_cue_c_inh"])
    # Thalamus exc → cU (reciprocal) and → c_inh (feedforward inhibition).
    W["b_t_cU"] = exc(params["B_t_cU"])
    W["b_t_c_inh"] = exc(params["B_t_c_inh"])
    # cU → thalamus (both thalamic pools).
    W["b_cU_t_exc"] = exc(params["B_cU_t_exc"])
    W["b_cU_t_inh"] = exc(params["B_cU_t_inh"])
    # cU → basal ganglia (striatum + GPe); cL → SNc / medulla.
    W["b_cU_d1"] = exc(params["B_cU_d1"])
    W["b_cU_d2"] = exc(params["B_cU_d2"])
    W["b_cU_gpe"] = exc(params["B_cU_gpe"])
    W["b_cL_snc"] = exc(params["B_cL_snc"])
    W["b_d1_snc"] = inh(params["B_d1_snc"])
    W["b_d2_snc"] = inh(params["B_d2_snc"])
    W["b_d1_snr"] = inh(params["B_d1_snr"])
    W["b_d2_gpe"] = inh(params["B_d2_gpe"])
    W["b_gpe_snr"] = inh(params["B_gpe_snr"])
    W["b_gpe_snc"] = inh(params["B_gpe_snc"])
    # SNr → thalamus (both pools).
    W["b_snr_t_exc"] = inh(params["B_snr_t_exc"])
    W["b_snr_t_inh"] = inh(params["B_snr_t_inh"])
    # D1 <-> D2 lateral inhibition.
    n_d1, n_d2 = W["j_d1"].shape[0], W["j_d2"].shape[0]
    W["b_d1_d2"] = inh(params.get("B_d1_d2", jnp.zeros((n_d2, n_d1))))  # D1 → D2
    W["b_d2_d1"] = inh(params.get("B_d2_d1", jnp.zeros((n_d1, n_d2))))  # D2 → D1

    # DA / adenosine → PKA gains. Affine floor (config m_gain_floor) maps exc(w) in [0,1)
    # onto [floor, 1): a gain can approach the floor but never reach 0 or lose its gradient.
    _mf = config.get("m_gain_floor", 0.0)
    _floored = lambda w: _mf + (1.0 - _mf) * exc(w)
    W["m_d1"] = _floored(params["m_d1"])
    W["m_d2"] = _floored(params["m_d2"])
    W["m_a1"] = _floored(params["m_a1"])   # A1R inhibitory drive on D1 PKA
    W["m_a2"] = _floored(params["m_a2"])   # A2R excitatory drive on D2 PKA
    # Trainable DA / adenosine RELEASE gains (>= 0 via exc).
    W["g_da_release"] = exc(params["g_da_release"])
    W["g_ado_release"] = exc(params["g_ado_release"])

    # Medullary E/I pairs. Each 2×2 block: col 0 from E (exc), col 1 from I (inh).
    def _med_block(raw):
        return jnp.concatenate([exc(raw[:, :1]), inh(raw[:, 1:])], axis=1)

    j_w1 = _med_block(params["J_med_w1"])  # within pair 1
    j_w2 = _med_block(params["J_med_w2"])  # within pair 2
    j_x = _med_block(params["J_med_x"])    # cross-pair
    # 4×4 matrix in [E0, E1, I0, I1] order. Pair 1 = (E0, I0); pair 2 = (E1, I1).
    W["j_med"] = jnp.stack([
        jnp.stack([j_w1[0, 0], j_x[0, 0],  j_w1[0, 1], j_x[0, 1]]),   # E0
        jnp.stack([j_x[0, 0],  j_w2[0, 0], j_x[0, 1],  j_w2[0, 1]]),  # E1
        jnp.stack([j_w1[1, 0], j_x[1, 0],  j_w1[1, 1], j_x[1, 1]]),   # I0
        jnp.stack([j_x[1, 0],  j_w2[1, 0], j_x[1, 1],  j_w2[1, 1]]),  # I1
    ])
    W["b_cL_med"] = exc(params["B_cL_med"])    # cL → medulla E units only
    # SNr → medulla E units: INHIBITORY (SNr is GABAergic), the tonic motor gate.
    W["b_snr_med"] = inh(params["B_snr_med"])
    W["c_med"] = exc(params["C_med"])          # readout from E units only

    # Readout source: "medulla" (default, motor-output path) or "thalamus" (positive-weight
    # vector off the thalamic relay pool). Static config value -> resolved at trace time.
    readout_source = config.get("readout_source", "medulla")
    if readout_source not in ("medulla", "thalamus"):
        raise ValueError(f"readout_source must be 'medulla' or 'thalamus', got {readout_source!r}")
    if readout_source == "thalamus" and "C_thal" not in params:
        raise KeyError("readout_source='thalamus' needs a 'C_thal' weight vector in params; "
                       "this bundle predates it -- re-init, or read out from the medulla.")
    W["readout_source"] = readout_source
    W["c_thal"] = exc(params["C_thal"]) if "C_thal" in params else None
    W["out_gain"] = jnp.asarray(params["out_gain"])
    W["out_bias"] = jnp.asarray(params["out_bias"])

    # Pacemaker biases are used RAW inside sigmoid (wrapping them in exc() first was a
    # double nonlinearity that confined every pacer to the top half of its range).
    def _pacer(p, lo, hi):
        return lo + sigmoid(p) * (hi - lo)

    W["snc_pacer"] = _pacer(params["P_snc"], config["snc_pacer_min"], config["snc_pacer_max"])
    W["snr_pacer"] = _pacer(params["P_snr"], config["snr_pacer_min"], config["snr_pacer_max"])
    W["gpe_pacer"] = _pacer(params.get("P_gpe", jnp.zeros(W["j_gpe"].shape[0])),
                            config["gpe_pacer_min"], config["gpe_pacer_max"])

    # Constants and static options.
    for k in ("tau_c", "tau_d1", "tau_d2", "tau_t", "tau_snr", "tau_gpe", "tau_snc",
              "tau_med", "tau_pka_fall", "tau_pka_rise", "da_pka_gain", "pka_saturation",
              "pka_max", "tau_da", "tau_ado", "da_max", "ado_max"):
        W[k] = config[k]
    W["noise_std"] = jnp.asarray(config["noise_std"])
    # Legacy bundles (saved before nt_mode joined the runtime config) omit this key.
    W["nt_mode"] = config.get("nt_mode", "forward_euler")
    # Optional clamps (None => normal dynamics). Static, so branches resolve at trace time.
    #   pin_pka_d1/d2: hold PKA (hence D1/D2 gain) fixed.
    #   pin_ado: hold adenosine at a tonic level, severing the pka_d2 -> D2 -> x_ado loop
    #            that makes pka_d2 bistable (unstable fixed point 0.267, lambda 1.0045).
    #   pin_snc / pin_ctx: open-loop probes; with cortex pinned the cue cannot drive the loop.
    for k in ("pin_pka_d1", "pin_pka_d2", "pin_ado", "pin_snc", "pin_ctx"):
        W[k] = config.get(k, None)
    return W


def _ext(s, W, name, tau, x):
    """Add every registered module input into state `name` (see W["ext"])."""
    for f in W["ext"].get(name, ()):
        x = x + (1.0 / tau) * f(s, W)
    return x


def _noise(s, W, step_rng):
    keys = jr.split(step_rng, len(_CORE_NOISE_ORDER))
    coef = W["noise_std"] / jnp.sqrt(2.0 * W["tau_c"])
    coef_snc = W["noise_std"] / jnp.sqrt(2.0 * W["tau_snc"])
    for name, k in zip(_CORE_NOISE_ORDER, keys):
        c = coef_snc if name == "x_snc" else coef
        s[name] = s[name] + c * jr.normal(k, s[name].shape)


def _cortex(s, W, u_t, stim_d1, stim_d2):
    # cU/cL excitatory PT populations + shared inhibitory c_inh. Recurrent currents are
    # taken from the pre-update snapshot; one nln per pool, at the end.
    tau = W["tau_c"]
    cU_rec = W["j_cU"] @ s["x_c_U"] + W["b_cL_cU"] @ s["x_c_L"] + W["j_ci_cU"] @ s["x_c_inh"]
    cL_rec = W["j_cL"] @ s["x_c_L"] + W["b_cU_cL"] @ s["x_c_U"] + W["j_ci_cL"] @ s["x_c_inh"]
    ci_rec = W["j_cU_ci"] @ s["x_c_U"] + W["j_cL_ci"] @ s["x_c_L"] + W["j_c_ii"] @ s["x_c_inh"]

    # cU: reciprocal thalamic input + cue.
    x = (1.0 - 1.0 / tau) * s["x_c_U"] + (1.0 / tau) * cU_rec
    x = x + (1.0 / tau) * (W["b_t_cU"] @ s["x_t_exc"])
    x = x + (1.0 / tau) * (W["b_cue_cU"] @ u_t)
    x_c_U = nln(_ext(s, W, "x_c_U", tau, x))

    # cL: cue only (no direct thalamic input).
    x = (1.0 - 1.0 / tau) * s["x_c_L"] + (1.0 / tau) * cL_rec
    x = x + (1.0 / tau) * (W["b_cue_cL"] @ u_t)
    x_c_L = nln(_ext(s, W, "x_c_L", tau, x))

    # c_inh: thalamic feedforward inhibition + cue.
    x = (1.0 - 1.0 / tau) * s["x_c_inh"] + (1.0 / tau) * ci_rec
    x = x + (1.0 / tau) * (W["b_t_c_inh"] @ s["x_t_exc"])
    x = x + (1.0 / tau) * (W["b_cue_c_inh"] @ u_t)
    x_c_inh = nln(_ext(s, W, "x_c_inh", tau, x))

    if W["pin_ctx"] is not None:
        x_c_U = jnp.full_like(x_c_U, W["pin_ctx"])
        x_c_L = jnp.full_like(x_c_L, W["pin_ctx"])
        x_c_inh = jnp.full_like(x_c_inh, W["pin_ctx"])
    s["x_c_U"], s["x_c_L"], s["x_c_inh"] = x_c_U, x_c_L, x_c_inh


def _thalamus(s, W, u_t, stim_d1, stim_d2):
    tau = W["tau_t"]
    t_rec_to_exc = W["j_t_ee"] @ s["x_t_exc"] + W["j_t_ei"] @ s["x_t_inh"]
    t_rec_to_inh = W["j_t_ie"] @ s["x_t_exc"] + W["j_t_ii"] @ s["x_t_inh"]

    x = (1.0 - 1.0 / tau) * s["x_t_exc"] + (1.0 / tau) * t_rec_to_exc
    x = x + (1.0 / tau) * (W["b_cU_t_exc"] @ s["x_c_U"])
    x = x + (1.0 / tau) * (W["b_snr_t_exc"] @ s["x_snr"])
    x_t_exc = nln(_ext(s, W, "x_t_exc", tau, x))

    x = (1.0 - 1.0 / tau) * s["x_t_inh"] + (1.0 / tau) * t_rec_to_inh
    x = x + (1.0 / tau) * (W["b_cU_t_inh"] @ s["x_c_U"])
    x = x + (1.0 / tau) * (W["b_snr_t_inh"] @ s["x_snr"])
    x_t_inh = nln(_ext(s, W, "x_t_inh", tau, x))
    s["x_t_exc"], s["x_t_inh"] = x_t_exc, x_t_inh


def _snc(s, W, u_t, stim_d1, stim_d2):
    tau = W["tau_snc"]
    x = (1.0 - (1.0 / tau)) * s["x_snc"]
    x = x + (1.0 / tau) * W["snc_pacer"]
    x = x + (1.0 / tau) * (W["b_cL_snc"] @ s["x_c_L"])
    x = x + (1.0 / tau) * (W["b_d1_snc"] @ s["x_d1"])
    x = x + (1.0 / tau) * (W["b_d2_snc"] @ s["x_d2"])
    x = x + (1.0 / tau) * (W["b_gpe_snc"] @ s["x_gpe"])
    x = nln(_ext(s, W, "x_snc", tau, x))
    if W["pin_snc"] is not None:
        x = jnp.full_like(x, W["pin_snc"])
    s["x_snc"] = x


def _neuromodulators(s, W, u_t, stim_d1, stim_d2):
    # DA is released by SNc (broadcast as a scalar); adenosine by overall striatal +
    # SNc + cU activity. Note mean_stri reads this step's SNc/cU but last step's D1/D2.
    da_release = jnp.mean(s["x_snc"])
    ado_release = jnp.mean(jnp.concatenate((s["x_d1"], s["x_d2"], s["x_snc"], s["x_c_U"])))
    x_da, x_ado = s["x_da"], s["x_ado"]
    tau_da, tau_ado = W["tau_da"], W["tau_ado"]
    if W["nt_mode"] == "mass_action":
        x_da = x_da + (1.0 / tau_da) * (W["g_da_release"] * da_release
                                        * jnp.maximum(1.0 - x_da / W["da_max"], 0.0) - x_da)
        x_ado = x_ado + (1.0 / tau_ado) * (W["g_ado_release"] * ado_release
                                           * jnp.maximum(1.0 - x_ado / W["ado_max"], 0.0) - x_ado)
    else:
        x_da = (1.0 - 1.0 / tau_da) * x_da + (1.0 / tau_da) * (W["g_da_release"] * da_release)
        x_ado = (1.0 - 1.0 / tau_ado) * x_ado + (1.0 / tau_ado) * (W["g_ado_release"] * ado_release)
        x_da = nln(x_da)
        x_ado = nln(x_ado)
    if W["pin_ado"] is not None:
        x_ado = jnp.full_like(x_ado, W["pin_ado"])
    s["x_da"], s["x_ado"] = x_da, x_ado


def _pka(s, W, u_t, stim_d1, stim_d2):
    # Mass-action-bounded leaky integrator: rectified production (receptor activation can't
    # make negative cAMP) throttled by available substrate (1 - pka/pka_max); the leak stays
    # linear so tau_pka_fall really sets the timescale (a per-step squash would destroy it).
    #   D1: D1R (DA) activates PKA; A1R (adenosine) inhibits.
    #   D2: A2R (adenosine) activates PKA; D2R (DA) inhibits.
    g = W["da_pka_gain"]
    pka_d1, pka_d2 = s["pka_d1"], s["pka_d2"]
    prod_d1 = jnp.maximum(g * W["m_d1"] * s["x_da"] - W["m_a1"] * s["x_ado"], 0)
    prod_d2 = jnp.maximum(W["m_a2"] * s["x_ado"] - g * W["m_d2"] * s["x_da"], 0)
    if W["pka_saturation"] == "mass_action":
        prod_d1 = prod_d1 * jnp.maximum(1.0 - pka_d1 / W["pka_max"], 0.0)
        prod_d2 = prod_d2 * jnp.maximum(1.0 - pka_d2 / W["pka_max"], 0.0)
    pka_d1 = (1.0 - 1.0 / W["tau_pka_fall"]) * pka_d1 + (1.0 / W["tau_pka_rise"]) * prod_d1
    pka_d2 = (1.0 - 1.0 / W["tau_pka_fall"]) * pka_d2 + (1.0 / W["tau_pka_rise"]) * prod_d2
    if W["pka_saturation"] != "mass_action":
        pka_d1 = sigmoid(4 * (pka_d1 - 0.5))
        pka_d2 = sigmoid(4 * (pka_d2 - 0.5))
    if W["pin_pka_d1"] is not None:
        pka_d1 = jnp.full_like(pka_d1, W["pin_pka_d1"])
    if W["pin_pka_d2"] is not None:
        pka_d2 = jnp.full_like(pka_d2, W["pin_pka_d2"])
    s["pka_d1"], s["pka_d2"] = pka_d1, pka_d2


def _striatum(s, W, u_t, stim_d1, stim_d2):
    # PKA enters as gains on the SPN input currents (gain system, commit 403033c).
    # NOTE the D2 pair is SWAPPED relative to D1 (higher pka_d2 -> LESS excitatory gain),
    # so DA, which lowers pka_d2, raises D2 excitability. Flagged in
    # docs/CHANGELOG_opto_recovery.md; kept as-is until its own fix.
    # NOTE D1's gain is also applied inside its nln, D2's is not.
    pka_d1, pka_d2 = s["pka_d1"], s["pka_d2"]
    gain_ed1 = pka_d1 + 0.5
    gain_id2 = pka_d2 + 0.5
    gain_id1 = (1.0 - pka_d1) + 0.5
    gain_ed2 = (1.0 - pka_d2) + 0.5

    tau = W["tau_d1"]
    x_d1 = (1.0 - (1.0 / tau)) * s["x_d1"]
    x_d1 = x_d1 + (1.0 / tau) * (gain_id1 * (W["j_d1"] @ x_d1))   # reads the leaked x_d1
    x_d1 = x_d1 + (1.0 / tau) * (gain_id1 * (W["b_d2_d1"] @ s["x_d2"]))
    x_d1 = x_d1 + (1.0 / tau) * (gain_ed1 * (W["b_cU_d1"] @ s["x_c_U"]))
    x_d1 = x_d1 + (1.0 / tau) * (gain_ed1 * stim_d1)
    x_d1 = nln(gain_ed1 * _ext(s, W, "x_d1", tau, x_d1))
    s["x_d1"] = x_d1

    tau = W["tau_d2"]
    x_d2 = (1.0 - (1.0 / tau)) * s["x_d2"]
    x_d2 = x_d2 + (1.0 / tau) * (gain_id2 * (W["j_d2"] @ x_d2))   # reads the leaked x_d2
    x_d2 = x_d2 + (1.0 / tau) * (gain_id2 * (W["b_d1_d2"] @ x_d1))  # this step's D1
    x_d2 = x_d2 + (1.0 / tau) * (gain_ed2 * (W["b_cU_d2"] @ s["x_c_U"]))
    x_d2 = x_d2 + (1.0 / tau) * (gain_ed2 * stim_d2)
    s["x_d2"] = nln(_ext(s, W, "x_d2", tau, x_d2))


def _gpe(s, W, u_t, stim_d1, stim_d2):
    tau = W["tau_gpe"]
    x = (1.0 - (1.0 / tau)) * s["x_gpe"]   # GPe recurrence (W["j_gpe"]) is off
    x = x + (1.0 / tau) * W["gpe_pacer"]
    x = x + (1.0 / tau) * (W["b_d2_gpe"] @ s["x_d2"])
    x = x + (1.0 / tau) * (W["b_cU_gpe"] @ s["x_c_U"])   # cU → GPe (exc)
    s["x_gpe"] = nln(_ext(s, W, "x_gpe", tau, x))


def _snr(s, W, u_t, stim_d1, stim_d2):
    tau = W["tau_snr"]
    x = (1.0 - (1.0 / tau)) * s["x_snr"]
    x = x + (1.0 / tau) * W["snr_pacer"]
    x = x + (1.0 / tau) * (W["b_d1_snr"] @ s["x_d1"])
    x = x + (1.0 / tau) * (W["b_gpe_snr"] @ s["x_gpe"])
    s["x_snr"] = nln(_ext(s, W, "x_snr", tau, x))


def _medulla(s, W, u_t, stim_d1, stim_d2):
    # Two E/I pairs with reciprocal coupling; SNr (inh) and cL (exc) drive target the E
    # units only. Module hooks on "x_med_E" likewise return an E-unit (first 2) current.
    tau = W["tau_med"]
    x = (1.0 - (1.0 / tau)) * s["x_med"]
    x = x + (1.0 / tau) * (W["j_med"] @ x)   # reads the leaked x_med
    x = x.at[:2].add((1.0 / tau) * (W["b_snr_med"] @ s["x_snr"]))
    x = x.at[:2].add((1.0 / tau) * (W["b_cL_med"] @ s["x_c_L"]))
    for f in W["ext"].get("x_med_E", ()):
        x = x.at[:2].add((1.0 / tau) * f(s, W))
    s["x_med"] = nln(x)


def _readout(s, W, u_t, stim_d1, stim_d2):
    # LINEAR readout: no squashing nonlinearity, so y is >= 0 but unbounded above (the
    # REINFORCE path's bernoulli(p=ys) and loss_type="bce" assume a probability and are
    # not valid against it). out_gain/out_bias are applied in BOTH modes and trainable.
    if W["readout_source"] == "thalamus":
        s["y"] = W["out_gain"] * (W["c_thal"] @ s["x_t_exc"]) + W["out_bias"]
    else:
        s["y"] = W["out_gain"] * (W["c_med"] @ s["x_med"][:2]) + W["out_bias"]


CORE_AREA_ORDER = (
    _cortex, _thalamus, _snc, _neuromodulators, _pka, _striatum, _gpe, _snr, _medulla, _readout,
)


def multiregion_rnn(params, config, inputs, opto_stimulation=None, rng_key=None,
                    var_order=CORE_STATE_VAR_ORDER):
    """Simulate one trial of the core circuit.

    Returns ``(ys, states)``: ys (T, n_output) and a tuple of state traces in `var_order`
    (default: the core's). x_c packs [cU, cL, c_inh] and x_t packs [t_exc, t_inh].
    """
    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    W = prepare_core_weights(params, config)
    s0 = core_initial_state(params, config, init_key)
    n_d1 = W["j_d1"].shape[0]
    n_steps = inputs.shape[0]
    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_d1 + W["j_d2"].shape[0]))

    def _step(carry, inp_stim_rng):
        u_t, stim_t, step_rng = inp_stim_rng
        stim_d1, stim_d2 = stim_t[:n_d1], stim_t[n_d1:]
        s = dict(carry)
        _noise(s, W, step_rng)
        for area in CORE_AREA_ORDER:
            area(s, W, u_t, stim_d1, stim_d2)
        y_t = s.pop("y")
        out = {k: s[k] for k in ("x_d1", "x_d2", "x_snc", "x_gpe", "x_snr", "pka_d1", "pka_d2",
                                 "x_med", "x_da", "x_ado")}
        out["x_c"] = jnp.concatenate([s["x_c_U"], s["x_c_L"], s["x_c_inh"]])
        out["x_t"] = jnp.concatenate([s["x_t_exc"], s["x_t_inh"]])
        return s, (y_t, out)

    step_keys = jr.split(step_key, n_steps)
    _, (ys, outs) = lax.scan(_step, s0, (inputs, opto_stimulation, step_keys))
    return ys, tuple(outs[k] for k in var_order)
