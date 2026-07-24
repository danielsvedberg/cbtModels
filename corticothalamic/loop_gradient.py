"""Is there a GRADIENT for the thalamocortical loop to learn the task, at init
(training from scratch)? Run per family:

    python corticothalamic/loop_gradient.py cbt_loop
    python corticothalamic/loop_gradient.py cbt_loop_noSCnoSTN

Two independent measurements at the from-scratch init on the Pavlovian task:

1. d(loss)/d(params), grouped. If almost all of ||grad|| sits on the readout
   bias (out_bias / C_med) and the thalamocortical + cue-input weights get ~0,
   then the optimizer can only tune a cue-blind constant output — there is no
   learning signal flowing back into the loop (vanished through saturation).

2. d(mean in-window output)/d(cue input): the sensitivity of the REWARDED-window
   output to the cue, backpropagated through the whole loop. ~0 => no gradient
   path from cue to the rewarded output => the loop cannot learn to use the cue.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, FAM))
sys.path.insert(0, ROOT)

import cbt_rnn as cbtl          # noqa: E402
import config_script            # noqa: E402
import self_timed_movement_task as stmt  # noqa: E402

cfg = config_script.for_family(FAM)

# Thalamocortical recurrent + cue-input parameter blocks (present in all families).
TC_RECUR = ["J_cU", "J_cL", "B_cU_cL", "B_cL_cU", "J_cU_ci", "J_cL_ci", "J_ci_cU",
            "J_ci_cL", "J_c_ii", "J_t_ee", "J_t_ei", "J_t_ie", "J_t_ii",
            "B_t_cU", "B_t_c_inh", "B_cU_t_exc", "B_cU_t_inh"]
CUE = ["B_cue_cU", "B_cue_cL", "B_cue_c_inh"]
READOUT = ["C_med", "out_gain", "out_bias"]


def main():
    B = 16
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=1)
    tc = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=tc["t_start"], T_cue=tc["t_cue"], T_response=tc["t_response"], T=tc["t_total"])
    inp, tgt, msk = inputs[:B], targets[:B], masks[:B]
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inp.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)
    rl = cfg.RL_CONFIG

    # ---- 1. gradient of the training loss w.r.t. params ----
    def loss_fn(p):
        l, _ = stmt.reinforce_loss(
            cbtl.rnn_func, p, config, inp, tgt, msk, keys,
            entropy_coef=rl["entropy_coef"], objective_mode=rl["objective_mode"],
            brevity_coef=rl["brevity_coef"], silence_coef=rl["silence_coef"],
            tail_coef=rl["tail_coef"])
        return l

    l, g = jax.value_and_grad(loss_fn)(params)
    gn = {k: float(jnp.linalg.norm(v)) for k, v in g.items()}
    tot = float(np.sqrt(sum(v * v for v in gn.values())))
    grp = lambda ks: float(np.sqrt(sum(gn.get(k, 0.0) ** 2 for k in ks)))
    tc_g, cue_g, read_g = grp(TC_RECUR), grp(CUE), grp(READOUT)
    ob = gn.get("out_bias", 0.0)
    other = float(np.sqrt(max(tot**2 - tc_g**2 - cue_g**2 - read_g**2, 0.0)))

    print("=" * 72)
    print(f"THALAMOCORTICAL GRADIENT AT INIT — family: {FAM}")
    print(f"  task=Pavlovian  objective={rl['objective_mode']}  loss={float(l):.4f}")
    print("=" * 72)
    print(f"total ||grad||                         = {tot:.3e}")
    print(f"  thalamocortical recurrent (17 blocks)= {tc_g:.3e}  ({100*tc_g/tot:5.2f}%)")
    print(f"  cue->cortex input                    = {cue_g:.3e}  ({100*cue_g/tot:5.2f}%)")
    print(f"  readout (C_med/out_gain/out_bias)    = {read_g:.3e}  ({100*read_g/tot:5.2f}%)")
    print(f"    of which out_bias alone            = {ob:.3e}  ({100*ob/tot:5.2f}%)")
    print(f"  everything else (BG, etc.)           = {other:.3e}  ({100*other/tot:5.2f}%)")

    # ---- 2. cue -> rewarded-window output sensitivity, through the loop ----
    in_window = ((tgt[..., 0] > 0.0) & (msk[..., 0] > 0.0)).astype(jnp.float32)  # (B,T)

    def window_out(inp_arr):
        ys = cbtl.rnn_func(params, config, inp_arr, stim, keys)[0]  # (B,T,1)
        ys = ys[..., 0] if ys.ndim == 3 else ys
        return jnp.sum(ys * in_window) / (jnp.sum(in_window) + 1e-8)

    val = float(window_out(inp))
    gcue = jax.grad(window_out)(inp)
    cue_sens = float(jnp.linalg.norm(gcue))
    # also the finite cue effect: cued vs null window output
    null = jnp.zeros_like(inp)
    dwin = float(window_out(inp) - window_out(null))

    print(f"\nmean rewarded-window output            = {val:.4f}")
    print(f"d(window output)/d(cue input) norm     = {cue_sens:.3e}")
    print(f"cued - null window output              = {dwin:+.5f}")

    # ---- verdict ----
    loop_frac = (tc_g + cue_g) / tot
    print("\nVERDICT:")
    if ob / tot > 0.9:
        print(f"  ~{100*ob/tot:.0f}% of the gradient is on the single scalar out_bias.")
        print("  The optimizer can only move a CUE-BLIND CONSTANT output — essentially")
        print("  NO learning signal reaches the thalamocortical loop.")
    elif cue_sens < 1e-4:
        print("  The rewarded-window output is INSENSITIVE to the cue (d/dcue ~ 0):")
        print("  no gradient path from cue -> loop -> rewarded output. Loop can't learn the cue.")
    elif loop_frac < 0.05:
        print(f"  Only {100*loop_frac:.1f}% of ||grad|| reaches the loop+cue weights — a weak,")
        print("  likely-unusable signal (attenuated through the saturated loop).")
    else:
        print(f"  {100*loop_frac:.1f}% of ||grad|| reaches the loop+cue weights and the window")
        print("  output responds to the cue -> a usable gradient exists for the loop.")


if __name__ == "__main__":
    main()
