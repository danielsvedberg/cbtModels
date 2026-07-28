"""Is the PKA trace usable as an interval timer?

    python corticothalamic/pka_timer_probe.py cbt_loop [bundle.pkl]

PKA is the only variable in the model with a delay-scale time constant
(`tau_pka_fall = 1440`), and it is already wired as a gate: D1/D2 excitability is
`bg_nln(x, pka)`. So the architecture ALREADY contains a candidate clock. For it
to work as one, three things must hold:

  1. SLOW   - pka must actually evolve on a ~100s of steps timescale. Note the
              update squashes through a nonlinearity every step
              (pka <- nln((1-1/tau_fall) pka + (1/tau_rise) production)), which can
              destroy the slow leak: the effective per-step gain is
              (1-1/tau_fall) * nln'(.), and nln' collapses when pka saturates.
  2. CUE-DRIVEN - pka must actually move when the cue arrives (via cortex -> SNc
              -> mean_snc -> production), otherwise it is a clock with no start.
  3. CONNECTED - the pka change must reach the output.

This measures all three on cued vs uncued trials.
"""
import os
import pickle as pkl
import sys

import numpy as np
import jax.numpy as jnp
import jax.random as jr

FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop"
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, FAM)); sys.path.insert(0, ROOT)

import cbt_rnn as cbtl          # noqa: E402
import config_script            # noqa: E402
import self_timed_movement_task as stmt  # noqa: E402

cfg = config_script.for_family(FAM)
A = cbtl.STATE_AREA_ORDER


def load():
    if BUNDLE:
        with open(os.path.join(ROOT, FAM, BUNDLE), "rb") as f:
            b = pkl.load(f)
        print(f"loaded {BUNDLE}")
        return b["params"], b["config"]
    print("using fresh init_params")
    return cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=2)


def run(params, config, inputs, noise=0.0):
    c = dict(config); c["noise_std"] = noise
    B = inputs.shape[0]
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inputs.shape[1], n_d1 + n_d2))
    ys, xs = cbtl.batched_rnn(params, c, inputs, stim, jr.split(jr.PRNGKey(0), B))
    return np.asarray(ys), xs


def main():
    params, config = load()
    t = cfg.TASK_CONFIG
    tw = t["t_cue"] + t["t_wait"]
    print(f"tau_pka_fall={config['tau_pka_fall']}  tau_pka_rise={config['tau_pka_rise']}  "
          f"target interval={tw} steps\n")

    # single fixed cue time so traces are directly comparable
    t0 = 200
    T = t["t_total"]
    cued = np.zeros((1, T, 2), np.float32)
    cued[0, t0:t0 + t["t_cue"], 0] = 1.0          # preparatory cue only (no go cue)
    null = np.zeros_like(cued)
    inp_c = cbtl.match_input_channels(jnp.asarray(cued), params)
    inp_n = cbtl.match_input_channels(jnp.asarray(null), params)

    ys_c, xs_c = run(params, config, inp_c)
    ys_n, xs_n = run(params, config, inp_n)

    def tr(xs, name):
        return np.asarray(xs[A.index(name)][0]).mean(-1)   # (T,) pop mean

    print("=" * 74)
    print("1. IS PKA SLOW?  (uncued trial, trace after settling)")
    print("=" * 74)
    for name in ("pkaD1", "pkaD2"):
        y = tr(xs_n, name)
        # effective time constant from the autocorrelation decay of the settled tail
        seg = y[300:] - y[300:].mean()
        if seg.std() < 1e-9:
            print(f"  {name}: CONSTANT (std {seg.std():.2e}) -> carries no signal")
            continue
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        ac /= ac[0]
        below = np.argmax(ac < np.exp(-1)) if (ac < np.exp(-1)).any() else len(ac)
        print(f"  {name}: mean {y.mean():.3f}, range [{y.min():.3f},{y.max():.3f}], "
              f"autocorr tau ~{below} steps")

    print("\n" + "=" * 74)
    print("2. IS PKA CUE-DRIVEN?  (cued minus uncued, cue at t=200)")
    print("=" * 74)
    print(f"{'signal':>10}" + "".join(f"{f't+{d}':>10}" for d in (0, 50, 100, 200, 310, 500)))
    for name in ("Cortex", "SNc", "pkaD1", "pkaD2", "D1", "Medulla"):
        c_, n_ = tr(xs_c, name), tr(xs_n, name)
        d = c_ - n_
        print(f"{name:>10}" + "".join(f"{d[min(t0+off, T-1)]:>10.5f}"
                                      for off in (0, 50, 100, 200, 310, 500)))
    dy = (ys_c - ys_n)[0, :, 0]
    print(f"{'OUTPUT':>10}" + "".join(f"{dy[min(t0+off, T-1)]:>10.5f}"
                                      for off in (0, 50, 100, 200, 310, 500)))

    print("\n" + "=" * 74)
    print("3. VERDICT")
    print("=" * 74)
    d1 = tr(xs_c, "pkaD1") - tr(xs_n, "pkaD1")
    peak = np.abs(d1).max()
    at_target = abs(d1[min(t0 + tw, T - 1)])
    print(f"  peak |delta pka_d1| after cue      = {peak:.5f}")
    print(f"  |delta pka_d1| at the target time  = {at_target:.5f}")
    if peak < 1e-4:
        print("  -> PKA DOES NOT RESPOND TO THE CUE: it cannot act as a timer,")
        print("     because the clock never starts. Fix the cue->SNc->PKA drive first.")
    elif at_target < 0.1 * peak:
        print("  -> PKA responds but has DECAYED by the target time: too fast.")
    else:
        print("  -> PKA carries a cue-triggered signal that survives to the target")
        print("     time: usable as a timer; the readout must learn to threshold it.")


if __name__ == "__main__":
    main()
