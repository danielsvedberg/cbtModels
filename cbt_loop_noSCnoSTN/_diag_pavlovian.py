"""Diagnose why train_pavlovian is stuck in cbt_loop_noSCnoSTN.

Checks, at the actual init used by train_pavlovian:
  1. resting state of every area + the readout floor
  2. the log_reward objective value and total gradient norm
  3. is the readout on the FLAT (zero-gradient) side? -> silence trap
  4. does the cue reach the output at all?
  5. a short training run (few hundred iters) to see if reward moves
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import jax, jax.numpy as jnp, jax.random as jr, numpy as np
import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSCnoSTN')
import self_timed_movement_task as stmt

AREAS = cbtl.STATE_AREA_ORDER


def build():
    rc = cfg.RNN_CONFIG; tc = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=tc["t_start"], T_cue=tc["t_cue"], T_response=tc["t_response"], T=tc["t_total"])
    params, config = cbtl.init_params(
        jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]),
        n_c_U=rc["n_c_U"], n_c_L=rc["n_c_L"], n_c_inh=rc["n_c_inh"],
        n_d1=rc["n_d1"], n_d2=rc["n_d2"], n_snc=rc["n_snc"], n_snr=rc["n_snr"],
        n_gpe=rc["n_gpe"], n_t_exc=rc["n_t_exc"], n_t_inh=rc["n_t_inh"],
        n_input=inputs.shape[-1], n_output=1,
        g_bg=rc["g_bg"], g_nm=rc["g_nm"], noise_std=rc["noise_std"])
    return params, config, inputs, targets, masks


def run(params, config, inputs, seed=0):
    B = inputs.shape[0]
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inputs.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(seed), B)
    return cbtl.batched_rnn(params, config, inputs, stim, keys)


def main():
    params, config, inputs, targets, masks = build()
    B = 16
    inp, tgt, msk = inputs[:B], targets[:B], masks[:B]

    print("=" * 70, "\n1. REST STATE (cued trials, mean over all steps)\n", "=" * 70)
    ys, xs = run(params, config, inp)
    for name, x in zip(AREAS, xs):
        m = float(np.asarray(x).mean())
        print(f"  {name:<10} {m:.4f}  {'DEAD' if m < 1e-3 else ''}")
    print(f"  OUTPUT     {float(np.asarray(ys).mean()):.4f}   "
          f"(min {float(np.asarray(ys).min()):.4f}, max {float(np.asarray(ys).max()):.4f})")

    print("\n" + "=" * 70, "\n2/3. OBJECTIVE + GRADIENT + readout-flatness\n", "=" * 70)
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inp.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)

    def loss_fn(p):
        l, aux = stmt.reinforce_loss(
            cbtl.rnn_func, p, config, inp, tgt, msk, keys,
            entropy_coef=cfg.RL_CONFIG["entropy_coef"],
            objective_mode=cfg.RL_CONFIG["objective_mode"],
            brevity_coef=cfg.RL_CONFIG["brevity_coef"],
            silence_coef=cfg.RL_CONFIG["silence_coef"],
            tail_coef=cfg.RL_CONFIG["tail_coef"])
        return l, aux

    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    gn = {k: float(jnp.linalg.norm(v)) for k, v in g.items()}
    tot = float(np.sqrt(sum(v ** 2 for v in gn.values())))
    print(f"  total_loss   = {float(l):.5f}")
    print(f"  reward_mean  = {float(aux['reward_mean']):.6f}")
    print(f"  log_reward   = {float(aux['log_reward_mean']):.4f}   "
          f"(floor = log(window*1e-6) ~ {np.log(cfg.PAVLOVIAN_CONFIG['t_response']*1e-6):.3f})")
    print(f"  ||grad||     = {tot:.3e}")
    # readout flatness: fraction of the medulla-E drive on the flat (<=0) side of nln
    argin = np.asarray(xs[AREAS.index('Medulla')])  # rate, already nln'd
    print(f"  medulla-E rate mean = {argin[:, :, :2].mean():.5f}  "
          f"(readout = tanh(c_med @ med_E); silent -> output ~ 0, "
          f"and log_reward's descent direction is MORE silence)")
    live = sorted(((v, k) for k, v in gn.items()), reverse=True)[:8]
    print("  largest grads:", ", ".join(f"{k}={v:.2e}" for v, k in live))

    print("\n" + "=" * 70, "\n4. CUE -> OUTPUT (cued vs null, aligned to cue onset)\n", "=" * 70)
    null = jnp.zeros_like(inp)
    yc, _ = run(params, config, inp)
    yn, _ = run(params, config, null)
    starts = np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:B].astype(int)
    a, b = np.asarray(yc), np.asarray(yn)
    dy = np.mean([np.abs(a[i, s:s+100] - b[i, s:s+100]).mean() for i, s in enumerate(starts)])
    print(f"  |cued - null| output in 100 steps post-cue = {dy:.6f}  "
          f"{'<-- NO CUE->OUTPUT PATH' if dy < 1e-4 else ''}")

    print("\n" + "=" * 70, "\n5. SHORT TRAINING RUN (400 iters)\n", "=" * 70)
    import optax
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]))
    best, losses, rewards = stmt.fit_rnn_reinforce(
        cbtl.rnn_func, params, config, inp, msk, opt, 400,
        log_interval=100, seed=0,
        baseline_momentum=cfg.RL_CONFIG["baseline_momentum"],
        entropy_coef=cfg.RL_CONFIG["entropy_coef"],
        objective_mode=cfg.RL_CONFIG["objective_mode"],
        batch_targets=tgt,
        brevity_coef=cfg.RL_CONFIG["brevity_coef"],
        silence_coef=cfg.RL_CONFIG["silence_coef"],
        tail_coef=cfg.RL_CONFIG["tail_coef"])
    print(f"  reward:  start={float(rewards[0]):.5f}  end={float(rewards[-1]):.5f}")
    print(f"  loss:    start={float(losses[0]):.4f}  end={float(losses[-1]):.4f}")


if __name__ == "__main__":
    main()


def per_stage_cue():
    """Per-area cue response (cued minus null), aligned to cue onset."""
    params, config, inputs, targets, masks = build()
    B = 16
    cued, null = inputs[:B], jnp.zeros_like(inputs[:B])
    _, xc = run(params, config, cued)
    _, xn = run(params, config, null)
    starts = np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:B].astype(int)
    print("\n" + "=" * 70, "\nPER-STAGE cue response + saturation\n", "=" * 70)
    print(f"  {'area':<10}{'rest':>8}{'%>0.9':>8}{'cue delta':>12}")
    for name, a, b in zip(AREAS, xc, xn):
        a = np.asarray(a); b = np.asarray(b)
        d = np.mean([np.abs(a[i, s:s+100] - b[i, s:s+100]).mean() for i, s in enumerate(starts)])
        sat = 100.0 * (b > 0.9).mean()
        print(f"  {name:<10}{b.mean():8.3f}{sat:7.1f}%{d:12.6f}"
              f"   {'SATURATED' if sat > 40 else ''}")


if __name__ == "__main__":
    per_stage_cue()
