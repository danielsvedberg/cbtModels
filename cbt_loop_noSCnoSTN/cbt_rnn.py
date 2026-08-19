import math
from pathlib import Path
import sys

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.nn import sigmoid, tanh, softplus

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt
import config_script as _rootcfg
import loop_init as _loop_init
_FAMILY = Path(__file__).resolve().parent.name


def exc(w):
    return stmt.exc(w)


def inh(w):
    return stmt.inh(w)


def no_autapse(m):
    """Zero the diagonal of a square within-population recurrence (no self-synapse)."""
    return m * (1.0 - jnp.eye(m.shape[0], dtype=m.dtype))


def nln(x):
    return stmt.nln(x)


def bg_nln(x, b):
    return stmt.bg_nln(x, b)
    #return sigmoid(4*(x-1+b))
    #return jnp.maximum(0, tanh(b*2*x))


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


# Canonical state tuple order used across CBT loop code.
STATE_VAR_ORDER = (
    "x_c", "x_d1", "x_d2", "x_snc", "x_gpe", "x_snr", "x_t", "pka_d1", "pka_d2", "x_med",
    "x_da", "x_ado",
)
STATE_AREA_ORDER = (
    "Cortex", "D1", "D2", "SNc", "GPe", "SNr", "Thalamus", "pkaD1", "pkaD2", "Medulla",
    "DA", "Adenosine",
)
# DA / adenosine are neuromodulator concentration states (not firing rates), so
# skip them (with PKA + Medulla) from the dead-area inactivity floor.
DEAD_AREA_SKIP_INDICES = tuple(
    STATE_AREA_ORDER.index(a) for a in ("pkaD1", "pkaD2", "Medulla", "DA", "Adenosine")
)


def init_params(rng_key, n_input):
    """Initialize multiregion CBT loop params and runtime config.

    Cortex and thalamus follow Dale's law as fully separate populations:
    ``n_*_exc`` excitatory projection neurons and ``n_*_inh`` local inhibitory
    interneurons live in their own state vectors with their own connectivity
    blocks. Only the excitatory pool sends out-of-area projections.

    Every structural/runtime value comes from the central config (crashes on a
    missing key — no silent defaults). n_input is the only per-call argument.
    """
    _rc = _rootcfg.rnn_config_for(_FAMILY)
    _is = _rootcfg.init_state_for(_FAMILY)
    _wi = _rootcfg.weight_init_for(_FAMILY)
    _rt = _rootcfg.runtime_config_for(_FAMILY)   # for the DA/adenosine release-gain inits
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
        # exponential synaptic MAGNITUDE ~ Exp(1) (mean 1). Combined with each block's
        # 1/fan_in prefactor this is Exp with mean 1/fan_in (rate = fan_in) -- the
        # one-signed (Dale) analog of He init: a fan-in-n sum of positive weights against
        # positive rates keeps its MEAN drive O(1) (same-signed terms don't cancel). Dale
        # sign is applied by the +/- prefactor per block (exc +, inh -), so under the
        # clip-based exc/inh nothing is zeroed at init.
        return jr.exponential(key, shape)

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
    config = dict(_rootcfg.runtime_config_for(_FAMILY))
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
        params, rho0, rho1 = _loop_init.normalize_loop(
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
    if float(stmt.exc(jnp.asarray(0.0))) > 0.25:
        _gain_keys = {"m_d1", "m_d2", "m_a1", "m_a2", "g_da_release", "g_ado_release",
                      "P_gpe", "P_snc", "P_snr"}
        for _k in list(params):
            _v = jnp.asarray(params[_k])
            if _k not in _gain_keys and _v.ndim >= 2:
                _m = jnp.clip(jnp.abs(_v), 1e-4, 1.0 - 1e-4)
                params[_k] = jnp.log(_m / (1.0 - _m))

    return params, config


def multiregion_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    # Trainable initial states come straight from params (crashes if absent — no
    # fallback); their starting values were set from config_script.CBT_INIT_STATE.
    x_c0_U   = jnp.asarray(params["x_c0_U"])
    x_c0_L   = jnp.asarray(params["x_c0_L"])
    x_c0_inh = jnp.asarray(params["x_c0_inh"])
    x_d10    = jnp.asarray(params["x_d10"])
    x_d20    = jnp.asarray(params["x_d20"])
    x_snc0   = jnp.asarray(params["x_snc0"])
    x_gpe0   = jnp.asarray(params["x_gpe0"])
    x_snr0   = jnp.asarray(params["x_snr0"])
    x_t0_exc = jnp.asarray(params["x_t0_exc"])
    x_t0_inh = jnp.asarray(params["x_t0_inh"])
    x_med0   = jnp.asarray(params["x_med0"])
    # Clamp the (trainable) PKA initial states to a sane range so training can't
    # push them to extreme bg_nln shifts. (Ported from cbt_loop.)
    _pka_lo = config["pka_init_floor"]; _pka_hi = config["pka_init_cap"]
    pka_d10  = jnp.clip(jnp.asarray(params["pka_d10"]), _pka_lo, _pka_hi)
    pka_d20  = jnp.clip(jnp.asarray(params["pka_d20"]), _pka_lo, _pka_hi)
    x_da0    = jnp.asarray(params["x_da0"])
    x_ado0   = jnp.asarray(params["x_ado0"])


    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    noise_std = jnp.asarray(config["noise_std"])
    x_c0_U = jnp.minimum(nln(x_c0_U + noise_std * jr.normal(init_key, x_c0_U.shape)), 0.5)
    x_c0_L = jnp.minimum(nln(x_c0_L + noise_std * jr.normal(init_key, x_c0_L.shape)), 0.5)
    x_c0_inh = jnp.minimum(nln(x_c0_inh + noise_std * jr.normal(init_key, x_c0_inh.shape)), 0.5)
    x_d10 = nln(x_d10 + noise_std * jr.normal(init_key, x_d10.shape))
    x_d20 = nln(x_d20 + noise_std * jr.normal(init_key, x_d20.shape))
    x_snc0 = nln(x_snc0 + noise_std * jr.normal(init_key, x_snc0.shape))
    x_gpe0 = nln(x_gpe0 + noise_std * jr.normal(init_key, x_gpe0.shape))
    x_snr0 = nln(x_snr0 + noise_std * jr.normal(init_key, x_snr0.shape))
    x_t0_exc = nln(x_t0_exc + noise_std * jr.normal(init_key, x_t0_exc.shape))
    x_t0_inh = nln(x_t0_inh + noise_std * jr.normal(init_key, x_t0_inh.shape))
    x_med0 = nln(x_med0 + noise_std * jr.normal(init_key, x_med0.shape))

    # Cortex blocks. Sign follows presynaptic identity (Dale's law). cU/cL are
    # excitatory PT-like populations; c_inh is the shared inhibitory pool.
    j_cU = no_autapse(exc(params["J_cU"]))        # cU → cU
    j_cL = no_autapse(exc(params["J_cL"]))        # cL → cL
    b_cU_cL = exc(params["B_cU_cL"])  # cU → cL
    b_cL_cU = exc(params["B_cL_cU"])  # cL → cU
    j_cU_ci = exc(params["J_cU_ci"])  # cU → c_inh
    j_cL_ci = exc(params["J_cL_ci"])  # cL → c_inh
    j_ci_cU = inh(params["J_ci_cU"])  # c_inh → cU
    j_ci_cL = inh(params["J_ci_cL"])  # c_inh → cL
    j_c_ii = no_autapse(inh(params["J_c_ii"]))    # c_inh → c_inh
    # Thalamus E/I recurrent blocks.
    j_t_ee = no_autapse(exc(params["J_t_ee"]))  # T_exc → T_exc
    j_t_ei = inh(params["J_t_ei"])  # T_inh → T_exc
    j_t_ie = exc(params["J_t_ie"])  # T_exc → T_inh
    j_t_ii = no_autapse(inh(params["J_t_ii"]))  # T_inh → T_inh

    j_d1 = no_autapse(inh(params["J_d1"]))
    j_d2 = no_autapse(inh(params["J_d2"]))
    j_gpe = no_autapse(inh(params["J_gpe"]))

    p_snr = exc(params["P_snr"])
    p_snc = exc(params["P_snc"])
    p_gpe = exc(params.get("P_gpe", jnp.zeros(j_gpe.shape[0])))

    # Cue → cortex (all three pools).
    b_cue_cU = exc(params["B_cue_cU"])
    b_cue_cL = exc(params["B_cue_cL"])
    b_cue_c_inh = exc(params["B_cue_c_inh"])
    # Thalamus exc → cU (reciprocal) and → c_inh (feedforward inhibition).
    b_t_cU = exc(params["B_t_cU"])
    b_t_c_inh = exc(params["B_t_c_inh"])
    # cU → thalamus (both thalamic pools).
    b_cU_t_exc = exc(params["B_cU_t_exc"])
    b_cU_t_inh = exc(params["B_cU_t_inh"])
    # cU → basal ganglia (striatum + GPe); cL → SNc / medulla (below).
    b_cU_d1 = exc(params["B_cU_d1"])#+(0.1/n_c_U_)
    b_cU_d2 = exc(params["B_cU_d2"])#+(0.1/n_c_U_)
    b_cU_gpe = exc(params["B_cU_gpe"])#+(0.1/n_c_U_)
    b_cL_snc = exc(params["B_cL_snc"])#+(0.1/n_c_L_)
    b_d1_snc = inh(params["B_d1_snc"])
    b_d2_snc = inh(params["B_d2_snc"])
    b_d1_snr = inh(params["B_d1_snr"])#-(0.1/n_d1_)
    b_d2_gpe = inh(params["B_d2_gpe"])#-(0.1/n_d2_)
    b_gpe_snr = inh(params["B_gpe_snr"])#-(0.1/n_gpe_)
    b_gpe_snc = inh(params["B_gpe_snc"])
    # SNr → thalamus (both pools).
    b_snr_t_exc = inh(params["B_snr_t_exc"])#-(0.1/n_snr_)
    b_snr_t_inh = inh(params["B_snr_t_inh"])
    # Dopamine / adenosine→PKA gains. Floored exc keeps the per-SPN weights
    # ≥ m_floor with a live gradient (no dead zone) for both DA and tonic
    # adenosine drives; k_a is the (currently scalar) adenosine level shared
    # by all SPNs.
    #m_floor = config["m_floor"]
    # Adenosine weights get their own floor so the tonic A1R/A2R drive can be
    # decoupled from the DA floor. The DA→D1 PKA term is structurally weak
    # (mean_snc is small), so a high shared floor pins A1R inhibition above it
    # and kills pka_d1. m_floor_a1 lets the A1R floor drop (revive D1) while
    # m_floor_a2 preserves the A2R drive that keeps pka_d2 alive.
    #m_floor_a1 = config["m_floor_a1"]
    #m_floor_a2 = config["m_floor_a2"]
    m_d1 = exc(params["m_d1"])# + m_floor
    m_d2 = exc(params["m_d2"]) #+ m_floor
    # Cap the A1R gain so training can't grow adenosine inhibition on D1 PKA past
    # the DA drive and collapse dSPN excitability. (Ported from cbt_loop.)
    m_a1 = exc(params["m_a1"])   # cap lifted: A1R gain free to train per-neuron
    m_a2 = exc(params["m_a2"])
    _zeros_d1_d2 = jnp.zeros((j_d2.shape[0], j_d1.shape[0]))
    _zeros_d2_d1 = jnp.zeros((j_d1.shape[0], j_d2.shape[0]))
    b_d1_d2 = inh(params.get("B_d1_d2", _zeros_d1_d2))  # D1 → D2 lateral inhibition
    b_d2_d1 = inh(params.get("B_d2_d1", _zeros_d2_d1))  # D2 → D1 lateral inhibition
    # Medullary E/I pairs. Each 2×2 block: col 0 from E (exc), col 1 from I (inh).
    def _med_block(raw):
        return jnp.concatenate([exc(raw[:, :1]), inh(raw[:, 1:])], axis=1)

    j_w1 = _med_block(params["J_med_w1"])  # within pair 1
    j_w2 = _med_block(params["J_med_w2"])  # within pair 2
    j_x  = _med_block(params["J_med_x"])   # cross-pair

    # Assemble 4×4 matrix in [E0, E1, I0, I1] order.
    # Pair 1 = (E0 idx 0, I0 idx 2); Pair 2 = (E1 idx 1, I1 idx 3).
    j_med = jnp.stack([
        jnp.stack([j_w1[0, 0], j_x[0, 0],  j_w1[0, 1], j_x[0, 1]]),   # E0
        jnp.stack([j_x[0, 0],  j_w2[0, 0], j_x[0, 1],  j_w2[0, 1]]),   # E1
        jnp.stack([j_w1[1, 0], j_x[1, 0],  j_w1[1, 1], j_x[1, 1]]),    # I0
        jnp.stack([j_x[1, 0],  j_w2[1, 0], j_x[1, 1],  j_w2[1, 1]]),   # I1
    ])
    b_cL_med = exc(params["B_cL_med"])  # shape (n_med//2, n_c_L): cL → medulla E units only
    # SNr → Medulla E units: inhibitory with a minimum magnitude (floored exc,
    # negated) so each weight stays ≤ -snr_med_floor and the tonic gate persists.
    #snr_med_floor = config["snr_med_floor"]
    b_snr_med = -inh(params["B_snr_med"])# + snr_med_floor)  # shape (n_med//2, n_snr)
    c_med = exc(params["C_med"])  # shape (n_output, 2): reads from E units only
    out_gain = jnp.asarray(params["out_gain"])
    out_bias = jnp.asarray(params["out_bias"])
    # Readout gain/bias (fall back to constants for legacy bundles without them).
    #rb = params["rb"]


    tau_c = config["tau_c"]
    tau_d1 = config["tau_d1"]
    tau_d2 = config["tau_d2"]
    tau_t = config["tau_t"]
    tau_snr = config["tau_snr"]
    tau_gpe = config["tau_gpe"]
    tau_snc = config["tau_snc"]
    tau_pka_fall = config["tau_pka_fall"]
    tau_pka_rise = config["tau_pka_rise"]
    # Gain on the DA→PKA drive (tanh(da_pka_gain * m_d * mean_snc)). mean_snc is
    # small, so the raw DA term barely registers against tonic adenosine; a gain
    # >1 lets phasic DA actually drive D1 PKA (and inhibit D2 PKA) within range.
    da_pka_gain = config["da_pka_gain"]
    # Optional PKA pin: if set to a float (e.g. 0.5), pka_d1/pka_d2 are FORCED to that
    # constant every step -- overriding the DA/adenosine dynamics -- so D1/D2 excitability
    # is held fixed (bg_nln's b). None/absent => normal dynamics. Static config value, so
    # the branch resolves at trace time (JAX-safe).
    pin_pka_d1 = config.get("pin_pka_d1", None)
    pin_pka_d2 = config.get("pin_pka_d2", None)
    # PKA saturation rule (ported from cbt_loop): mass-action-bounded pool fed
    # directly into bg_nln as excitability b (no per-step state squash).
    pka_saturation = config["pka_saturation"]
    pka_max = config["pka_max"]
    pka_clip_eps = config["pka_clip_eps"]

    # Tonic adenosine level: a single tunable scalar shared by both SPN
    # populations. Kept explicit so it can later be promoted to a dynamic
    # state (e.g. activity-dependent A1R/A2R modulation) without touching the
    # m_a1 / m_a2 connection weights. Sigmoid bound keeps k_a ∈ [floor, cap]
    # with a live gradient across the full range.
    k_a_floor = config["k_a_floor"]
    k_a_cap = config["k_a_cap"]
    k_a = k_a_floor + exc(jnp.asarray(params["k_a"])) * (k_a_cap - k_a_floor)  # (legacy; unused with dynamic DA/adenosine)
    # Dynamic DA / adenosine concentration model (mass-action), ported from noSC.
    tau_da = config["tau_da"]; tau_ado = config["tau_ado"]
    g_da_release = exc(params["g_da_release"]); g_ado_release = exc(params["g_ado_release"])  # trainable, >=0
    da_max = config["da_max"]; ado_max = config["ado_max"]

    snc_pacer_max = config["snc_pacer_max"]
    snc_pacer_min = config["snc_pacer_min"]
    snr_pacer_max = config["snr_pacer_max"]
    snr_pacer_min = config["snr_pacer_min"]
    gpe_pacer_max = config["gpe_pacer_max"]
    gpe_pacer_min = config["gpe_pacer_min"]

    snc_pacer = snc_pacer_min + sigmoid(p_snc) * (snc_pacer_max - snc_pacer_min)
    snr_pacer = snr_pacer_min + sigmoid(p_snr) * (snr_pacer_max - snr_pacer_min)
    gpe_pacer = gpe_pacer_min + sigmoid(p_gpe) * (gpe_pacer_max - gpe_pacer_min)

    tau_med = config["tau_med"]

    n_steps = inputs.shape[0]
    n_d1_cells = j_d1.shape[0]
    n_d2_cells = j_d2.shape[0]
    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((n_steps, n_d1_cells + n_d2_cells))

    def _step(carry, inp_stim_rng):
        (x_d1, x_d2,
         x_c_U, x_c_L, x_c_inh,
         x_t_exc, x_t_inh,
         x_snr, x_gpe, x_snc, pka_d1, pka_d2, x_med, x_da, x_ado) = carry
        u_t, stim_t, step_rng = inp_stim_rng
        stim_d1 = stim_t[:n_d1_cells]
        stim_d2 = stim_t[n_d1_cells:]

        # add noise
        (rng_d1, rng_d2,
         rng_c_U, rng_c_L, rng_c_inh,
         rng_t_exc, rng_t_inh,
         rng_snr, rng_gpe, rng_snc, rng_med) = jr.split(step_rng, 11)
        coef = noise_std / jnp.sqrt(2.0 * tau_c)
        x_d1 = x_d1 + coef * jr.normal(rng_d1, x_d1.shape)
        x_d2 = x_d2 + coef * jr.normal(rng_d2, x_d2.shape)
        x_c_U = x_c_U + coef * jr.normal(rng_c_U, x_c_U.shape)
        x_c_L = x_c_L + coef * jr.normal(rng_c_L, x_c_L.shape)
        x_c_inh = x_c_inh + coef * jr.normal(rng_c_inh, x_c_inh.shape)
        x_t_exc = x_t_exc + coef * jr.normal(rng_t_exc, x_t_exc.shape)
        x_t_inh = x_t_inh + coef * jr.normal(rng_t_inh, x_t_inh.shape)
        x_snr = x_snr + coef * jr.normal(rng_snr, x_snr.shape)
        x_gpe = x_gpe + coef * jr.normal(rng_gpe, x_gpe.shape)
        coef_snc = noise_std / jnp.sqrt(2.0 * tau_snc)
        x_snc = x_snc + coef_snc * jr.normal(rng_snc, x_snc.shape)
        x_med = x_med + coef * jr.normal(rng_med, x_med.shape)

        # cortex: cU/cL excitatory PT populations + shared inhibitory c_inh.
        # Raw signed recurrent/cross-pool currents; one nln per area, at the end.
        cU_rec = j_cU @ x_c_U + b_cL_cU @ x_c_L + j_ci_cU @ x_c_inh
        cL_rec = j_cL @ x_c_L + b_cU_cL @ x_c_U + j_ci_cL @ x_c_inh
        ci_rec = j_cU_ci @ x_c_U + j_cL_ci @ x_c_L + j_c_ii @ x_c_inh

        # cU: reciprocal thalamic input + cue.
        x_c_U = (1.0 - 1.0 / tau_c) * x_c_U + (1.0 / tau_c) * cU_rec
        x_c_U = x_c_U + (1.0 / tau_c) * (b_t_cU @ x_t_exc)
        x_c_U = x_c_U + (1.0 / tau_c) * (b_cue_cU @ u_t)
        x_c_U = nln(x_c_U)

        # cL: cue only (no direct thalamic input).
        x_c_L = (1.0 - 1.0 / tau_c) * x_c_L + (1.0 / tau_c) * cL_rec
        x_c_L = x_c_L + (1.0 / tau_c) * (b_cue_cL @ u_t)
        x_c_L = nln(x_c_L)

        # c_inh: thalamic feedforward inhibition + cue.
        x_c_inh = (1.0 - 1.0 / tau_c) * x_c_inh + (1.0 / tau_c) * ci_rec
        x_c_inh = x_c_inh + (1.0 / tau_c) * (b_t_c_inh @ x_t_exc)
        x_c_inh = x_c_inh + (1.0 / tau_c) * (b_cue_c_inh @ u_t)
        x_c_inh = nln(x_c_inh)

        # thalamus: same pre-step snapshot trick.
        t_rec_to_exc = j_t_ee @ x_t_exc + j_t_ei @ x_t_inh
        t_rec_to_inh = j_t_ie @ x_t_exc + j_t_ii @ x_t_inh

        x_t_exc = (1.0 - 1.0 / tau_t) * x_t_exc + (1.0 / tau_t) * t_rec_to_exc
        x_t_exc = x_t_exc + (1.0 / tau_t) * (b_cU_t_exc @ x_c_U)
        x_t_exc = x_t_exc + (1.0 / tau_t) * (b_snr_t_exc @ x_snr)
        x_t_exc = nln(x_t_exc)

        x_t_inh = (1.0 - 1.0 / tau_t) * x_t_inh + (1.0 / tau_t) * t_rec_to_inh
        x_t_inh = x_t_inh + (1.0 / tau_t) * (b_cU_t_inh @ x_c_U)
        x_t_inh = x_t_inh + (1.0 / tau_t) * (b_snr_t_inh @ x_snr)
        x_t_inh = nln(x_t_inh)

        x_snc = (1.0 - (1.0 / tau_snc)) * x_snc
        x_snc = x_snc + (1.0 / tau_snc) * snc_pacer
        x_snc = x_snc + (1.0 / tau_snc) * (b_cL_snc @ x_c_L)
        x_snc = x_snc + (1.0 / tau_snc) * (b_d1_snc @ x_d1)
        x_snc = x_snc + (1.0 / tau_snc) * (b_d2_snc @ x_d2)
        x_snc = x_snc + (1.0 / tau_snc) * (b_gpe_snc @ x_gpe)
        x_snc = nln(x_snc)
        # SNc is broadcast as a single scalar to every SPN; each SPN scales it
        # by its own per-neuron gain m_d1[i] / m_d2[i].
        mean_snc = jnp.mean(x_snc)
        mean_spn = jnp.mean(jnp.concatenate((x_d1, x_d2)))
        # Mass-action DA / adenosine concentrations
        # (tau_da), adenosine slow (tau_ado). Substrate-throttled to saturate at *_max.
        da_release = mean_snc
        ado_release = mean_spn
        x_da  = x_da  + (1.0 / tau_da)  * (g_da_release  * da_release * jnp.maximum(1.0 - x_da / da_max, 0.0)  - x_da)
        x_ado = x_ado + (1.0 / tau_ado) * (g_ado_release * ado_release * jnp.maximum(1.0 - x_ado / ado_max, 0.0) - x_ado)

        # PKA dynamics (leaky saturating integrator):
        # exponential leak with tau_pka_fall, rectified DA-driven production
        # (receptor activation can't make negative cAMP), tanh-saturating output.
        # Asymmetric timescales emerge from the gain ratio tau_fall/tau_rise.
        #   D1: D1R (DA) activates PKA; A1R (tonic adenosine) inhibits.
        # Mass-action-bounded leaky integrator (ported from cbt_loop): production
        # rectified (biological), throttled by available-substrate (1-pka/pka_max)
        # so the STATE stays in (0,1); the leak stays linear so tau_pka_fall really
        # sets the timescale (no per-step sigmoid squash, which would destroy it).
        prod_d1 = jnp.maximum(da_pka_gain * m_d1 * x_da - m_a1 * x_ado, 0)
        prod_d2 = jnp.maximum(m_a2 * x_ado - da_pka_gain * m_d2 * x_da, 0)
        if pka_saturation == "mass_action":
            prod_d1 = prod_d1 * jnp.maximum(1.0 - pka_d1 / pka_max, 0.0)
            prod_d2 = prod_d2 * jnp.maximum(1.0 - pka_d2 / pka_max, 0.0)
        pka_d1 = (1.0 - 1.0 / tau_pka_fall) * pka_d1 + (1.0 / tau_pka_rise) * prod_d1
        pka_d2 = (1.0 - 1.0 / tau_pka_fall) * pka_d2 + (1.0 / tau_pka_rise) * prod_d2
        # Optional pin: hold pka_d1/pka_d2 at a fixed value (overrides the dynamics above).
        if pin_pka_d1 is not None:
            pka_d1 = jnp.full_like(pka_d1, pin_pka_d1)
        if pin_pka_d2 is not None:
            pka_d2 = jnp.full_like(pka_d2, pin_pka_d2)

        # PKA is bounded to (0,1), so it IS bg_nln's excitability b directly (no
        # soft-threshold gate). Clip only insets off the (0,1) endpoints.
        pka_gate_d1 = jnp.clip(pka_d1, pka_clip_eps, 1.0 - pka_clip_eps)
        pka_gate_d2 = jnp.clip(pka_d2, pka_clip_eps, 1.0 - pka_clip_eps)

        # PKA shifts rheobase in bg_nln: higher PKA → lower threshold → more excitable.
        x_d1 = (1.0 - (1.0 / tau_d1)) * x_d1
        x_d1 = x_d1 + (1.0 / tau_d1) * (j_d1 @ x_d1)
        x_d1 = x_d1 + (1.0 / tau_d1) * (b_d2_d1 @ x_d2)
        x_d1 = x_d1 + (1.0 / tau_d1) * (b_cU_d1 @ x_c_U)
        x_d1 = x_d1 + (1.0 / tau_d1) * stim_d1
        x_d1 = bg_nln(x_d1, pka_gate_d1)

        x_d2 = (1.0 - (1.0 / tau_d2)) * x_d2
        x_d2 = x_d2 + (1.0 / tau_d2) * (j_d2 @ x_d2)
        x_d2 = x_d2 + (1.0 / tau_d2) * (b_d1_d2 @ x_d1)
        x_d2 = x_d2 + (1.0 / tau_d2) * (b_cU_d2 @ x_c_U)
        x_d2 = x_d2 + (1.0 / tau_d2) * stim_d2
        x_d2 = bg_nln(x_d2, pka_gate_d2)

        x_gpe = (1.0 - (1.0 / tau_gpe)) * x_gpe #+ (1.0 / tau_gpe) * (j_gpe @ x_gpe)
        x_gpe = x_gpe + (1.0 / tau_gpe) * gpe_pacer
        x_gpe = x_gpe + (1.0 / tau_gpe) * (b_d2_gpe @ x_d2)
        x_gpe = x_gpe + (1.0 / tau_gpe) * (b_cU_gpe @ x_c_U)  # cU → GPe (exc)
        x_gpe = nln(x_gpe)

        x_snr = (1.0 - (1.0 / tau_snr)) * x_snr
        x_snr = x_snr + (1.0 / tau_snr) * snr_pacer
        x_snr = x_snr + (1.0 / tau_snr) * (b_d1_snr @ x_d1)
        x_snr = x_snr + (1.0 / tau_snr) * (b_gpe_snr @ x_gpe)
        x_snr = nln(x_snr)

        # medulla: two E/I pairs with reciprocal coupling; cortical (exc) and
        # inhibitory SNr drive both target the E units only
        x_med = (1.0 - (1.0 / tau_med)) * x_med 
        x_med = x_med + (1.0 / tau_med) * (j_med @ x_med)
        x_med = x_med.at[:2].add((1.0 / tau_med) * (b_snr_med @ x_snr))  # SNr → Medulla E units only
        x_med = x_med.at[:2].add((1.0 / tau_med) * (b_cL_med @ x_c_L))  # cL → medulla E units only
        x_med = nln(x_med)

        # Biased sigmoid readout (ported from cbt_loop): nonzero resting prob
        # (~sigmoid(out_bias)) so the policy can explore; out_gain/out_bias trainable.
        y_t = sigmoid(out_gain * (c_med @ x_med[:2]) + out_bias)  # readout from E units only

        # Pack the full cortex/thalamus state ([cU..., cL..., c_inh...]) into the
        # output so downstream analysis code (get_brain_area, slope, ratios) still
        # sees a single Cortex / Thalamus array.
        x_c = jnp.concatenate([x_c_U, x_c_L, x_c_inh])
        x_t = jnp.concatenate([x_t_exc, x_t_inh])

        new_carry = (x_d1, x_d2,
                     x_c_U, x_c_L, x_c_inh,
                     x_t_exc, x_t_inh,
                     x_snr, x_gpe, x_snc, pka_d1, pka_d2, x_med, x_da, x_ado)
        out = (y_t, x_c, x_d1, x_d2, x_snc, x_gpe, x_snr, x_t, pka_d1, pka_d2, x_med, x_da, x_ado)
        return new_carry, out

    step_keys = jr.split(step_key, n_steps)
    _, (ys, xc, xd1, xd2, xsnc, xgpe, xsnr, xt, pkad1, pkad2, xmed, xda, xado) = lax.scan(
        _step,
        (x_d10, x_d20,
         x_c0_U, x_c0_L, x_c0_inh,
         x_t0_exc, x_t0_inh,
         x_snr0, x_gpe0, x_snc0, pka_d10, pka_d20, x_med0, x_da0, x_ado0),
        (inputs, opto_stimulation, step_keys),
    )
    return ys, (xc, xd1, xd2, xsnc, xgpe, xsnr, xt, pkad1, pkad2, xmed, xda, xado)


batched_rnn = vmap(multiregion_rnn, in_axes=(None, None, 0, 0, 0))


def rnn_func(params, config, batch_inputs, opto_stim, rng_keys):
    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    if opto_stim is None:
        batch_stim = jnp.zeros((batch_inputs.shape[0], batch_inputs.shape[1], n_d1 + n_d2))
    else:
        batch_stim = opto_stim
    ys, xs = batched_rnn(params, config, batch_inputs, batch_stim, rng_keys)
    # State tuple ends with (..., pkad1, pkad2, xmed); expose PKA traces plus the
    # GPe trajectory for loss shaping (PKA asymmetry + GPe activity floor). The
    # full state tuple ``xs`` is returned last so the loss can enforce an
    # activity floor on *every* area (dead-area / inactivity penalty), not just GPe.
    gpe = xs[STATE_AREA_ORDER.index("GPe")]
    return ys, xs[STATE_AREA_ORDER.index("pkaD1")], xs[STATE_AREA_ORDER.index("pkaD2")], gpe, xs, DEAD_AREA_SKIP_INDICES


def evaluate(params, config, all_inputs, noise_std=None, n_seeds=8):
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


def get_brain_area(brain_area, xs, zs=None):
    """Return the state array for ``brain_area``.

    ``zs`` is accepted for backward compatibility with older call sites that
    passed a separate neuromodulator-state tuple; the current model packs all
    states (including PKA excitability and Medulla) into ``xs``, so ``zs`` is
    ignored.
    """
    return xs[STATE_AREA_ORDER.index(brain_area)]


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


def get_d1_d2_ratio(all_xs, t_start, t_end, avg_time=True, remove_outliers=False):
    """D1 minus D2 mean activity over a time window.

    Args:
        all_xs: state list/tuple of arrays shaped (n_seeds, n_conditions, T, N).
        t_start, t_end: time-window bounds (timesteps).
        avg_time: if True, average over the window -> (n_seeds, n_conditions).
        remove_outliers: if True, NaN out per-condition outliers (z > 3).

    Returns:
        (n_seeds, n_conditions[, win]) array of D1-D2 activity differences.
    """
    d1 = get_brain_area("D1", all_xs)
    d2 = get_brain_area("D2", all_xs)
    d1m = jnp.mean(d1[..., t_start:t_end, :], axis=-1)
    d2m = jnp.mean(d2[..., t_start:t_end, :], axis=-1)
    ratio = d1m - d2m
    if avg_time:
        ratio = jnp.mean(ratio, axis=-1)
    if remove_outliers:
        z = jnp.abs((ratio - jnp.nanmean(ratio)) / (jnp.nanstd(ratio) + 1e-8))
        ratio = jnp.where(z > 3, jnp.nan, ratio)
    return ratio


def get_slope(all_xs, t_start, t_end, avg_neurons=True, remove_outliers=False):
    """Least-squares ramp slope of cortical activity over a time window.

    Returns:
        (n_seeds, n_conditions) array of slopes (per unit time-step).
    """
    cortex = get_brain_area("Cortex", all_xs)  # (n_seeds, n_conditions, T, N)
    seg = cortex[..., t_start:t_end, :]
    if avg_neurons:
        seg = jnp.mean(seg, axis=-1)  # (n_seeds, n_conditions, win)
    else:
        seg = jnp.mean(seg, axis=-1)
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
