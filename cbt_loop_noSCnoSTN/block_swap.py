"""CAUSAL test: which init weight blocks carry a seed's fate?

Seeds are deterministic, so a seed's outcome is fixed by its initial weights. Correlational
screening over 44 blocks at n=40 cannot identify WHICH weights matter (0 blocks survive
Bonferroni). This instead transplants blocks from a seed that BREAKS THROUGH into one that
COLLAPSES and asks whether the recipient is rescued.

Hypothesis under test (user's, sharpened by the block screen): breakthrough seeds have a
less saturation-prone thalamus -- stronger SNr->thalamus inhibition (B_snr_t_exc, r=+0.44)
and weaker thalamic self-excitation (J_t_ee, r=-0.42). If that is causal, transplanting
those two blocks alone should rescue a collapsing seed.

Controls are essential: J_d1 is a block with a weak correlation (r=-0.24) and should NOT
rescue; ALL_LOOP transplants every loop block and should rescue if anything does (if it
does not, fate lives outside the loop entirely).
"""
import argparse, json, pathlib, sys, time
import numpy as np, jax.random as jr, jax.numpy as jnp, optax

_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as cs
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from seed_stability import GAINS, RUNTIME, BREAKTHROUGH_SEP, outcome

import loop_init

SWAPS = {
    "none":        [],                                   # baseline: recipient unchanged
    "B_snr_t_exc": ["B_snr_t_exc"],
    "J_t_ee":      ["J_t_ee"],
    "thal_pair":   ["B_snr_t_exc", "J_t_ee"],            # the hypothesis
    "J_d1":        ["J_d1"],                             # negative control (weak r)
    "ALL_LOOP":    list(loop_init.LOOP_BLOCKS),          # positive control
}
DATA = pathlib.Path(__file__).with_name("block_swap_data.json")


def main(donor, recipient, iters, which):
    st = cs._CBT_FAMILY_STRUCTURE["cbt_loop_noSCnoSTN"]
    st["extra_weight_init"] = dict(GAINS)
    st["extra_runtime"] = {**st["extra_runtime"], **RUNTIME}
    import cbt_rnn as cbtl, self_timed_movement_task as stmt
    cfg = cs.for_family("cbt_loop_noSCnoSTN")
    sup, t = cfg.SUPERVISED_THAL_CONFIG, cfg.TASK_CONFIG
    inp, tg, mk = stmt.selftimed_step_target(
        T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"], T_movement=t["t_movement"],
        T=t["t_total"], lo=sup["target_lo"], hi=sup["target_hi"], hold=sup["hold"], delay=sup["delay"])
    dz = np.load(f"init_weights/seed_{donor:03d}.npz")
    rows = json.loads(DATA.read_text()) if DATA.exists() else []
    for name in (which or list(SWAPS)):
        blocks = SWAPS[name]
        t0 = time.time()
        params, config = cbtl.init_params(jr.PRNGKey(recipient), n_input=inp.shape[-1])
        params = dict(params); config = dict(config); config["readout_source"] = "thalamus"
        for b in blocks:
            if b in params and b in dz.files:
                params[b] = jnp.asarray(dz[b])
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]))
        best, losses, _ = stmt.fit_rnn_supervised(
            cbtl.rnn_func, params, config, inp, mk, opt, iters,
            batch_targets=tg, log_interval=200, seed=recipient, loss_type=sup["loss_type"])
        sep, mse = outcome(cbtl, stmt, cfg, best, config, inp[:32], tg[:32])
        rec = dict(donor=donor, recipient=recipient, swap=name, n_blocks=len(blocks),
                   separation=sep, mse=mse, rescued=bool(sep > BREAKTHROUGH_SEP),
                   secs=round(time.time() - t0, 1))
        rows = [r for r in rows if not (r["donor"] == donor and r["recipient"] == recipient
                                        and r["swap"] == name)] + [rec]
        DATA.write_text(json.dumps(rows, indent=1))
        print(f"swap {name:14s} ({len(blocks):2d} blocks)  sep={sep:+.4f}  mse={mse:.6f}  "
              f"{'RESCUED' if rec['rescued'] else 'still collapsed'}  [{rec['secs']:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", type=int, required=True)
    ap.add_argument("--recipient", type=int, required=True)
    ap.add_argument("--iters", type=int, default=1400)
    ap.add_argument("--which", nargs="*", default=None)
    a = ap.parse_args()
    main(a.donor, a.recipient, a.iters, a.which)
