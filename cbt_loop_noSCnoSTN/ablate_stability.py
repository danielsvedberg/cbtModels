"""ABLATION: which single parameter breaks training?

Every collapse so far happened by step ~1200-1400, so 2000 iters decides stability and
costs ~17 min instead of 83. Condition A is the config that trained stably to 10k
(separation 0.486); B/C/D each change exactly ONE thing relative to it.

  A  control          -- the solved config, unchanged
  B  m_a2  = 0.139    -- adenosine -> D2 PKA coupling (0 in the solved run)
  C  pin_ado = 0.256  -- clamped adenosine level (0.093 in the solved run)
  D  cross/tonic      -- striatal E/I from the sweep (0.162/0.220 in the solved run)

Usage:  python ablate_stability.py --condition A [--iters 2000]
"""
import argparse, sys, pathlib
import numpy as np, jax.random as jr, optax

_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as cs

# SOLVED config, in RAW units (exc/inh = clip(tanh(w),0,None), so negative raw -> 0).
SOLVED_GAINS = {"m_d1": 1.1363, "m_d2": -1.5368, "m_a1": -3.4095, "m_a2": 0.0080}
SOLVED_RT = {"da_release": 0.2982, "ado_release": 1.1363, "pin_ado": 0.093,
             "stri_cross_scale": 0.162, "stri_tonic": 0.220}

CONDITIONS = {
    "A": ({}, {}),
    "B": ({"m_a2": float(np.arctanh(0.139))}, {}),
    "C": ({}, {"pin_ado": 0.256}),
    "D": ({}, {"stri_cross_scale": 0.423, "stri_tonic": 0.158}),
    # E: control + the inactivity-floor penalty that exists precisely to stop areas going
    # silent. RL_CONFIG ships it at 0.0, so every run so far had it OFF (logs read
    # "dead_area: 0.0000"). If the collapse is the loop falling into the silent absorbing
    # state, this should raise the breakthrough rate.
    "E": ({}, {}),
}
DEAD_AREA_COEF = {"E": 0.1}   # condition -> dead_area_coef; absent means 0.0 (off)

def run_one(cond, seed, iters, cbtl, stmt, cfg):
    """One training run. Returns the separation trace so we can score breakthrough vs collapse."""
    sup, t = cfg.SUPERVISED_THAL_CONFIG, cfg.TASK_CONFIG
    inputs, targets, masks = stmt.selftimed_step_target(
        T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"], T_movement=t["t_movement"],
        T=t["t_total"], lo=sup["target_lo"], hi=sup["target_hi"], hold=sup["hold"], delay=sup["delay"])
    params, config = cbtl.init_params(jr.PRNGKey(seed), n_input=inputs.shape[-1])
    config = dict(config); config["readout_source"] = "thalamus"
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]))
    best, losses, accs = stmt.fit_rnn_supervised(
        cbtl.rnn_func, params, config, inputs, masks, opt, iters,
        batch_targets=targets, log_interval=200, seed=seed, loss_type=sup["loss_type"],
        dead_area_coef=DEAD_AREA_COEF.get(cond, 0.0), dead_area_min=0.1)
    return float(min(losses)) if losses else float("nan")


def main(cond, iters, n_seeds=5):
    gains, rt = CONDITIONS[cond]
    st = cs._CBT_FAMILY_STRUCTURE["cbt_loop_noSCnoSTN"]
    st["extra_weight_init"] = {**SOLVED_GAINS, **gains}
    st["extra_runtime"] = {**st["extra_runtime"], **SOLVED_RT, **rt}
    import cbt_rnn as cbtl, self_timed_movement_task as stmt
    cfg = cs.for_family("cbt_loop_noSCnoSTN")
    sup, t = cfg.SUPERVISED_THAL_CONFIG, cfg.TASK_CONFIG
    print(f"[{cond}] gains={ {k: round(v,4) for k,v in st['extra_weight_init'].items()} }")
    print(f"[{cond}] pin_ado={st['extra_runtime']['pin_ado']} "
          f"cross={st['extra_runtime']['stri_cross_scale']} tonic={st['extra_runtime']['stri_tonic']}")
    _p, _ = cbtl.init_params(jr.PRNGKey(0), n_input=1)
    for k in ("m_a2", "m_d2"):
        print(f"[{cond}] effective {k} = {float(cbtl.exc(_p[k])):.4f}", flush=True)
    print(f"[{cond}] dead_area_coef={DEAD_AREA_COEF.get(cond, 0.0)}  seeds=0..{n_seeds-1}  iters={iters}",
          flush=True)
    CONST = 0.011875          # loss of the trivial best-constant solution
    results = []
    for sd in range(n_seeds):
        ml = run_one(cond, sd, iters, cbtl, stmt, cfg)
        broke = ml < 0.9 * CONST
        results.append(broke)
        print(f"RESULT {cond} seed={sd} min_loss={ml:.6f} {'BREAKTHROUGH' if broke else 'collapsed'}",
              flush=True)
    print(f"SUMMARY {cond}: {sum(results)}/{len(results)} broke through", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    ap.add_argument("--iters", type=int, default=1400)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    main(a.condition, a.iters, a.seeds)
