"""Did the hybrid-trained network actually learn to USE the go cue, or did it
settle on a cue-blind constant output? Run:

    python corticothalamic/eval_hybrid.py cbt_loop params_hybrid_scratch.pkl

Ablation test on the trained bundle: compare the in-window output with the go
cue present vs removed (channel 1 zeroed), and vs both cues removed. A network
that learned the task shows a large in-window response that COLLAPSES when the
go cue is ablated. A cue-blind solution shows the same output either way.
"""
import os
import pickle as pkl
import sys

import numpy as np
import jax.numpy as jnp
import jax.random as jr

FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop"
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "params_hybrid_scratch.pkl"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, FAM)); sys.path.insert(0, ROOT)

import cbt_rnn as cbtl          # noqa: E402
import config_script            # noqa: E402
import self_timed_movement_task as stmt  # noqa: E402

cfg = config_script.for_family(FAM)


def run(params, config, inputs):
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    B = inputs.shape[0]
    stim = jnp.zeros((B, inputs.shape[1], n_d1 + n_d2))
    ys = cbtl.rnn_func(params, config, inputs, stim, jr.split(jr.PRNGKey(0), B))[0]
    return np.asarray(ys[..., 0] if ys.ndim == 3 else ys)


def main():
    path = os.path.join(ROOT, FAM, BUNDLE)
    with open(path, "rb") as f:
        bundle = pkl.load(f)
    params, config = bundle["params"], bundle["config"]
    print(f"loaded {path}")

    t = cfg.TASK_CONFIG
    inputs, targets, masks = stmt.hybrid_stmt(
        T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"],
        T_movement=t["t_movement"], T=t["t_total"])
    B = 32
    inputs, targets, masks = inputs[:B], targets[:B], masks[:B]
    win = ((np.asarray(targets)[..., 0] > 0) & (np.asarray(masks)[..., 0] > 0))
    out_win = ~win

    variants = {
        "both cues (trained condition)": inputs,
        "go cue ABLATED (ch1=0)": inputs.at[..., 1].set(0.0),
        "prep cue ABLATED (ch0=0)": inputs.at[..., 0].set(0.0),
        "both cues ablated": jnp.zeros_like(inputs),
    }
    print(f"\n{'condition':<32}{'in-window':>11}{'out-window':>12}{'ratio':>8}")
    base_in = None
    for name, inp in variants.items():
        ys = run(params, config, inp)
        i_m = float(ys[win].mean()); o_m = float(ys[out_win].mean())
        if base_in is None:
            base_in = i_m
        print(f"{name:<32}{i_m:>11.4f}{o_m:>12.4f}{i_m / max(o_m, 1e-9):>8.2f}")

    ys_full = run(params, config, inputs)
    ys_nogo = run(params, config, inputs.at[..., 1].set(0.0))
    drop = float(ys_full[win].mean() - ys_nogo[win].mean())
    print(f"\ngo-cue effect on in-window output = {drop:+.4f}")
    if abs(drop) < 1e-3:
        print("VERDICT: CUE-BLIND — ablating the go cue changes nothing; the network")
        print("         learned a constant output, not the task.")
    else:
        print("VERDICT: the in-window output DEPENDS on the go cue -> the network is")
        print("         using the cue (a real, if partial, solution).")

    # selectivity: does it fire more inside the window than outside?
    r = float(ys_full[win].mean() / max(ys_full[out_win].mean(), 1e-9))
    print(f"in/out-window ratio = {r:.2f}  "
          f"({'selective' if r > 1.2 else 'NOT selective (flat in time)'})")


if __name__ == "__main__":
    main()
