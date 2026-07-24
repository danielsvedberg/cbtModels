"""Test candidate fixes for the dead resting fixed point, applied POST-init
(no changes to cbt_rnn.py) so each can be evaluated before being wired in.

Knobs:
  inh_scale : multiply the local inhibitory blocks of the cortico-thalamic loop
              by <1, so per-row E/I balance leaves a small NET EXCITATORY drive.
              This is the only way the loop gets any tonic drive at all -- the
              three pacers (SNc/SNr/GPe) are the model's only drive terms and
              two of them are inhibitory onto the loop.
  snc_pacer : raise the SNc pacer above the tonic GPe->SNc inhibition, so the
              dopamine source is not rectified off.
  da_gain   : da_pka_gain, sets whether DA can beat tonic adenosine at the
              max(DA - adenosine, 0) gate.

Reports the resting state of every area and how the task gradient is
distributed (>90% on out_bias == the network is only learning a constant).
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
LOOP_INH = ["J_ci_cU", "J_ci_cL", "J_c_ii", "J_t_ei", "J_t_ii"]


def build():
    t = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=t["t_start"], T_cue=t["t_cue"], T_response=t["t_response"], T=t["t_total"])
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=inputs.shape[-1])
    return params, config, inputs, targets, masks


def patch(params, config, inh_scale=1.0, snc_pacer=None, da_gain=None, k_a_cap=None):
    p, c = dict(params), dict(config)
    for k in LOOP_INH:
        p[k] = p[k] * inh_scale
    if snc_pacer is not None:
        c["snc_pacer_min"], c["snc_pacer_max"] = snc_pacer
    if da_gain is not None:
        c["da_pka_gain"] = da_gain
    if k_a_cap is not None:
        c["k_a_cap"] = k_a_cap
    return p, c


def evaluate(p, c, inputs, targets, masks, label):
    B = 16
    n_d1 = p["J_d1"].shape[0]; n_d2 = p["J_d2"].shape[0]
    null = jnp.zeros_like(inputs[:B])
    stim = jnp.zeros((B, inputs.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)

    ys_n, xs_n = cbtl.batched_rnn(p, c, null, stim, keys)
    ys_c, xs_c = cbtl.batched_rnn(p, c, inputs[:B], stim, keys)
    rest = {a: float(np.asarray(x[:, -200:]).mean()) for a, x in zip(AREAS, xs_n)}

    # cue-driven change in output, cue-aligned
    starts = np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:B].astype(int)
    a, b = np.asarray(ys_c), np.asarray(ys_n)
    dy = float(np.mean([np.abs(a[i, s:s + 100] - b[i, s:s + 100]).mean()
                        for i, s in enumerate(starts)]))

    def loss_fn(pp):
        y, _ = cbtl.batched_rnn(pp, c, inputs[:B], stim, keys)
        e = 1e-6
        return jnp.mean((-(targets[:B] * jnp.log(y + e)
                           + (1 - targets[:B]) * jnp.log(1 - y + e))) * masks[:B])

    g = jax.grad(loss_fn)(p)
    tot = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in g.values())))
    ob = float(jnp.linalg.norm(g["out_bias"]))
    frac_bias = 100.0 * ob / max(tot, 1e-30)

    print(f"{label:<34} " + " ".join(
        f"{k}={rest[k]:.3f}" for k in ("Cortex", "Thalamus", "SNc", "pkaD1", "D1", "SNr", "Medulla"))
        + f"  out={float(np.asarray(ys_n)[:, -200:].mean()):.3f}"
          f"  d_out={dy:.4f}  |g|={tot:.2e}  bias%={frac_bias:5.1f}")
    return rest, dy, frac_bias


def main():
    params, config, inputs, targets, masks = build()
    print("\nlegend: rest rates of each area | out = resting output | d_out = cue-driven "
          "output change | bias% = share of ||grad|| sitting on the single scalar out_bias\n")

    print(">>> baseline (current repo state)")
    evaluate(params, config, inputs, targets, masks, "as-is")

    print("\n>>> A. revive the dopamine source only (SNc pacer above GPe inhibition)")
    for pacer in [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]:
        p, c = patch(params, config, snc_pacer=pacer)
        evaluate(p, c, inputs, targets, masks, f"snc_pacer={pacer}")

    print("\n>>> B. + DA gain so DA beats tonic adenosine at the max() gate")
    for gain in [1.0, 2.0, 5.0]:
        p, c = patch(params, config, snc_pacer=(0.5, 0.6), da_gain=gain)
        evaluate(p, c, inputs, targets, masks, f"snc=(0.5,0.6) da_gain={gain}")

    print("\n>>> C. give the cortico-thalamic loop a net excitatory drive "
          "(scale local inhibition below exact balance)")
    for s in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]:
        p, c = patch(params, config, inh_scale=s)
        evaluate(p, c, inputs, targets, masks, f"inh_scale={s}")

    print("\n>>> D. combined: live cortex + live dopamine")
    for s in [0.9, 0.85, 0.8]:
        for gain in [2.0, 5.0]:
            p, c = patch(params, config, inh_scale=s, snc_pacer=(0.5, 0.6), da_gain=gain)
            evaluate(p, c, inputs, targets, masks, f"inh={s} snc=(.5,.6) da={gain}")

    print("\n>>> E. combined + lower tonic adenosine (k_a_cap)")
    for cap in [0.5, 0.2, 0.1]:
        p, c = patch(params, config, inh_scale=0.85, snc_pacer=(0.5, 0.6),
                     da_gain=2.0, k_a_cap=cap)
        evaluate(p, c, inputs, targets, masks, f"inh=.85 da=2 k_a_cap={cap}")


if __name__ == "__main__":
    main()
