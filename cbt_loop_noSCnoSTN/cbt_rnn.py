"""cbt_simple (cbt_loop_noSCnoSTN): the CORE circuit with no added areas.

The circuit itself lives in /cbt_core.py; this module binds it to this family's config and
state order and keeps the family's public API (init_params, rnn_func, evaluate, ...).
"""
from pathlib import Path
import sys

import jax.numpy as jnp
import jax.random as jr  # noqa: F401  (kept for callers using cbtl.jr)
from jax import vmap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt
import cbt_core as core
# Shared helpers live in /cbt_core.py (one definition for every family). Re-exported
# here so `cbtl.exc`, `cbtl.nln`, ... keep working for existing callers.
from cbt_core import exc, inh, nln, bg_nln, no_autapse, match_input_channels  # noqa: F401
_FAMILY = Path(__file__).resolve().parent.name


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
    """Initialize this family's params + runtime config (core circuit, see cbt_core)."""
    return core.init_core_params(rng_key, n_input, _FAMILY)


# Trainable t=0 state params (frozen by freeze_init_state_optimizer); see cbt_core.
INIT_STATE_KEYS = core.CORE_INIT_STATE_KEYS


def init_state_mask(params):
    """Pytree of bools, True for every initial-state param present in `params`."""
    return core.init_state_mask(params, INIT_STATE_KEYS)


def freeze_init_state_optimizer(optimizer, params):
    """Wrap `optimizer` so INIT_STATE_KEYS receive zero updates (see cbt_core)."""
    return core.freeze_init_state_optimizer(optimizer, params, INIT_STATE_KEYS)


def multiregion_rnn(params, config, inputs, opto_stimulation=None, rng_key=None):
    """One trial of the core circuit (cbt_core.multiregion_rnn), states in STATE_VAR_ORDER."""
    return core.multiregion_rnn(params, config, inputs, opto_stimulation, rng_key,
                                var_order=STATE_VAR_ORDER)


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


# Evaluation / analysis helpers: shared implementations in cbt_core, bound to this
# family's batched_rnn and STATE_AREA_ORDER so the call signatures are unchanged.
def evaluate(params, config, all_inputs, noise_std=None, n_seeds=8):
    return core.evaluate(batched_rnn, params, config, all_inputs, noise_std, n_seeds)


def get_brain_area(brain_area, xs, zs=None):
    """Return the state array for ``brain_area``. ``zs`` is ignored (legacy call sites)."""
    return core.get_brain_area(brain_area, xs, STATE_AREA_ORDER)


get_response_times_opto = core.get_response_times_opto


def get_d1_d2_ratio(all_xs, t_start, t_end, avg_time=True, remove_outliers=False):
    return core.get_d1_d2_ratio(all_xs, t_start, t_end, STATE_AREA_ORDER, avg_time, remove_outliers)


def get_slope(all_xs, t_start, t_end, avg_neurons=True, remove_outliers=False):
    return core.get_slope(all_xs, t_start, t_end, STATE_AREA_ORDER, avg_neurons, remove_outliers)
