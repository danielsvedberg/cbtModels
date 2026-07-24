"""Test the structural hypothesis: the cortico-thalamic loop has NO constant
input term, so x=0 is its only stable fixed point and it can never leave it.

Every drive into cortex/thalamus is multiplicative on rates (W @ x). With
nln(0)=0 and spectral radius <= 1, x*=0 is stable and the loop is silent
regardless of E/I balance. GPe/SNr/SNc each have a `pacer` (a constant term)
and are the ONLY areas alive at rest -- cortex, thalamus, SC, D1, D2 and
medulla have none.

Here a tonic drive is injected via an extra constant-1 input channel (so no
change to cbt_rnn.py is needed) and we ask whether the whole loop wakes up.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop')
import self_timed_movement_task as stmt

AREAS = cbtl.STATE_AREA_ORDER


def build(n_input):
    rnn_cfg = cfg.RNN_CONFIG
    t = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=t["t_start"], T_cue=t["t_cue"], T_response=t["t_response"], T=t["t_total"])
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=n_input)
    return params, config, inputs, targets, masks


def with_tonic(params, config, bias, snc_pacer=None, da_gain=None):
    """Set the 2nd input column to a constant tonic drive of magnitude `bias`."""
    p, c = dict(params), dict(config)
    for k in ("B_cue_cU", "B_cue_cL"):
        w = np.asarray(p[k]).copy()
        w[:, 1] = bias
        p[k] = jnp.asarray(w)
    if snc_pacer is not None:
        c["snc_pacer_min"], c["snc_pacer_max"] = snc_pacer
    if da_gain is not None:
        c["da_pka_gain"] = da_gain
    return p, c


def report(p, c, inputs, targets, masks, label):
    B = 16
    T = inputs.shape[1]
    n_d1 = p["J_d1"].shape[0]; n_d2 = p["J_d2"].shape[0]
    ones = jnp.ones((B, T, 1))
    cue = jnp.concatenate([inputs[:B], ones], axis=-1)
    null = jnp.concatenate([jnp.zeros_like(inputs[:B]), ones], axis=-1)
    stim = jnp.zeros((B, T, n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)

    ys_n, xs_n = cbtl.batched_rnn(p, c, null, stim, keys)
    ys_c, _ = cbtl.batched_rnn(p, c, cue, stim, keys)
    rest = {a: float(np.asarray(x[:, -200:]).mean()) for a, x in zip(AREAS, xs_n)}

    starts = np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:B].astype(int)
    a, b = np.asarray(ys_c), np.asarray(ys_n)
    dy = float(np.mean([np.abs(a[i, s:s + 100] - b[i, s:s + 100]).mean()
                        for i, s in enumerate(starts)]))

    def loss_fn(pp):
        y, _ = cbtl.batched_rnn(pp, c, cue, stim, keys)
        e = 1e-6
        return jnp.mean((-(targets[:B] * jnp.log(y + e)
                           + (1 - targets[:B]) * jnp.log(1 - y + e))) * masks[:B])

    g = jax.grad(loss_fn)(p)
    tot = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in g.values())))
    fb = 100.0 * float(jnp.linalg.norm(g["out_bias"])) / max(tot, 1e-30)

    print(f"{label:<30} " + " ".join(f"{k[:4]}={rest[k]:.3f}" for k in
          ("Cortex", "Thalamus", "SNc", "pkaD1", "D1", "SNr", "SC", "Medulla"))
          + f" out={float(np.asarray(ys_n)[:, -200:].mean()):.3f}"
            f" d_out={dy:.4f} |g|={tot:.2e} bias%={fb:5.1f}")


def main():
    params, config, inputs, targets, masks = build(n_input=2)
    print("\nlegend: Cort/Thal/SNc/pkaD/D1/SNr/SC/Medu rest rates | d_out = cue-driven "
          "output change | bias% = share of ||grad|| on the scalar out_bias\n")

    print(">>> tonic drive to cortex ONLY (dopamine still dead)")
    for bias in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        p, c = with_tonic(params, config, bias)
        report(p, c, inputs, targets, masks, f"tonic={bias}")

    print("\n>>> tonic drive + revived dopamine source")
    for bias in (0.2, 0.3, 0.5, 1.0):
        p, c = with_tonic(params, config, bias, snc_pacer=(0.5, 0.6), da_gain=5.0)
        report(p, c, inputs, targets, masks, f"tonic={bias} snc=.5 da=5")

    print("\n>>> the same, with a weaker SNr pacer (loosen the tonic gate)")
    for bias in (0.3, 0.5):
        for snr in ((0.2, 0.4), (0.05, 0.2)):
            p, c = with_tonic(params, config, bias, snc_pacer=(0.5, 0.6), da_gain=5.0)
            c = dict(c); c["snr_pacer_min"], c["snr_pacer_max"] = snr
            report(p, c, inputs, targets, masks, f"tonic={bias} snr={snr}")


if __name__ == "__main__":
    main()
