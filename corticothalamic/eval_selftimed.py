"""Is the self-timed response REALLY timed to the cue, or is it a fixed latency?

    python corticothalamic/eval_selftimed.py cbt_loop params_shaped.pkl

On this task a response at one FIXED time (ignoring the cue) already lands
in-window on ~87% of trials, because the per-trial windows overlap heavily
(t_start in [52,399] -> windows [362,709]..[662,1009]). So a high reward is NOT
evidence of self-timing. The decisive test is whether the response SHIFTS with
the cue:

    regress  response_time  on  t_start
      slope ~ 1  -> genuine self-timing (fixed latency AFTER the cue)
      slope ~ 0  -> fixed-time response, cue ignored (degenerate)

We also report the latency distribution (response_time - t_start); real
self-timing should concentrate near T_cue + T_wait = 310.
"""
import os
import pickle as pkl
import sys

import numpy as np
import jax.numpy as jnp
import jax.random as jr

FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop"
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "params_shaped.pkl"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, FAM)); sys.path.insert(0, ROOT)

import cbt_rnn as cbtl          # noqa: E402
import config_script            # noqa: E402
import self_timed_movement_task as stmt  # noqa: E402

cfg = config_script.for_family(FAM)
THRESH = 0.5


def main():
    path = os.path.join(ROOT, FAM, BUNDLE)
    with open(path, "rb") as f:
        b = pkl.load(f)
    params, config = b["params"], b["config"]
    print(f"loaded {path}\n")

    t = cfg.TASK_CONFIG
    ts = np.asarray(t["t_start"])
    inputs, targets, masks = stmt.self_timed_movement_task(
        T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"],
        T_movement=t["t_movement"], T=t["t_total"])
    # the model was trained with a 2-channel cue layout; pad if needed
    inputs = cbtl.match_input_channels(inputs, params)

    B = inputs.shape[0]
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inputs.shape[1], n_d1 + n_d2))
    ys = cbtl.rnn_func(params, config, inputs, stim, jr.split(jr.PRNGKey(0), B))[0]
    ys = np.asarray(ys[..., 0] if ys.ndim == 3 else ys)

    # first threshold crossing per trial
    over = ys > THRESH
    has = over.any(1)
    rt = np.where(has, over.argmax(1), -1)
    ok = has
    print(f"trials with a response (p>{THRESH}): {ok.sum()}/{B}")
    if ok.sum() < 5:
        print("too few responses to assess timing; peak-based fallback:")
        rt = ys.argmax(1); ok = np.ones(B, bool)

    x, y = ts[ok].astype(float), rt[ok].astype(float)
    lat = y - x
    win_lo = x + t["t_cue"] + t["t_wait"]
    win_hi = win_lo + t["t_movement"]
    in_win = (y >= win_lo) & (y < win_hi)

    slope, icpt = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    print(f"\nresponse_time = {slope:.3f} * t_start + {icpt:.1f}   (r = {r:.3f})")
    print(f"latency (response - t_start): mean {lat.mean():.1f}, sd {lat.std():.1f} "
          f"(target ~{t['t_cue'] + t['t_wait']})")
    print(f"response_time itself:          mean {y.mean():.1f}, sd {y.std():.1f}")
    # Degenerate ceiling, computed from the CURRENT task config (not hardcoded):
    # the best single fixed response time that ignores the cue entirely.
    w_lo = ts + t["t_cue"] + t["t_wait"]
    w_hi = w_lo + t["t_movement"]
    ceiling = max(float(np.mean((T >= w_lo) & (T < w_hi)))
                  for T in range(int(t["t_total"])))
    print(f"in-window fraction: {in_win.mean():.1%}   "
          f"(cue-ignoring fixed-time ceiling for this task: {ceiling:.1%})")

    print("\nVERDICT:")
    if slope > 0.7 and r > 0.7:
        print(f"  SELF-TIMED — the response shifts with the cue (slope {slope:.2f}).")
    elif slope < 0.3:
        print(f"  NOT self-timed — slope {slope:.2f}: the response sits at a roughly")
        print("  FIXED time regardless of when the cue arrived. The cue is being")
        print("  ignored; in-window hits come from window overlap, not timing.")
    else:
        print(f"  PARTIAL — slope {slope:.2f}: some cue dependence, well short of 1.0.")
    # a tighter discriminator: sd of latency vs sd of absolute response time
    print(f"\n  sd(latency) = {lat.std():.1f} vs sd(response_time) = {y.std():.1f}")
    print("  (self-timing => sd(latency) << sd(response_time); "
          "fixed-time => the reverse)")


if __name__ == "__main__":
    main()
