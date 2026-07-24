"""Probe the dynamics that the CURRENT init actually produces.

Answers, in order:
  1. REST      -- what does every area settle to with no cue? (dead / saturated / alive)
  2. GAIN      -- what is the effective gain at each stage of cue -> cortex -> D1 ->
                  SNr -> medulla -> output? Where does the cue signal die?
  3. CHAOS     -- does a tiny perturbation grow (chaotic) or vanish (contractive)?
  4. GRADIENT  -- which parameter blocks receive gradient from the task loss, and
                  which are exactly zero (dead-gradient = untrainable)?

Run:  python cbt_loop/tests/init_dynamics_probe.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # cbt_loop
sys.path.insert(0, str(HERE.parent.parent))   # repo root

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


def build():
    task_cfg = cfg.PAVLOVIAN_CONFIG
    rnn_cfg = cfg.RNN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=task_cfg["t_start"], T_cue=task_cfg["t_cue"],
        T_response=task_cfg["t_response"], T=task_cfg["t_total"],
    )
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=inputs.shape[-1])
    return params, config, inputs, targets, masks


def run(params, config, inputs, seed=0):
    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((inputs.shape[0], inputs.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(seed), inputs.shape[0])
    return cbtl.batched_rnn(params, config, inputs, stim, keys)


def sec(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ---------------------------------------------------------------- 1. REST ----
def probe_rest(params, config, inputs):
    sec("1. REST STATE (cue-free trials, mean over last 300 steps)")
    null = jnp.zeros_like(inputs[:16])
    ys, xs = run(params, config, null)
    print(f"{'area':<10} {'mean':>8} {'std':>8} {'%silent':>8} {'%satur':>8}   verdict")
    for name, x in zip(AREAS, xs):
        tail = np.asarray(x[:, -300:, :])
        m, s = tail.mean(), tail.std()
        silent = 100.0 * (tail < 1e-3).mean()
        satur = 100.0 * (tail > 0.99).mean()
        v = "DEAD" if m < 1e-3 else ("saturated" if satur > 50 else
                                     ("weak" if m < 0.02 else "alive"))
        print(f"{name:<10} {m:8.4f} {s:8.4f} {silent:7.1f}% {satur:7.1f}%   {v}")
    y = np.asarray(ys[:, -300:, :])
    print(f"{'output':<10} {y.mean():8.4f} {y.std():8.4f}"
          f"{'':>18}   (fixed point of the readout)")
    return ys, xs


# --------------------------------------------------------- 2. STAGE GAIN ----
def probe_cue_pathway(params, config, inputs):
    sec("2. CUE PATHWAY -- per-stage response to the cue (cued minus null)")
    cued = inputs[:16]
    null = jnp.zeros_like(cued)
    t0 = int(np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:16].min())
    _, xs_c = run(params, config, cued)
    _, xs_n = run(params, config, null)
    ys_c, _ = run(params, config, cued)
    ys_n, _ = run(params, config, null)

    print("delta = |cued - null| averaged over units, in the 100 steps after cue onset")
    print(f"{'area':<10} {'baseline':>10} {'delta':>10} {'delta/base':>11}   verdict")
    prev = None
    for name, xc, xn in zip(AREAS, xs_c, xs_n):
        a = np.asarray(xc); b = np.asarray(xn)
        starts = np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:16]
        d, base = [], []
        for i, s in enumerate(starts):
            s = int(s); e = min(s + 100, a.shape[1])
            d.append(np.abs(a[i, s:e] - b[i, s:e]).mean())
            base.append(np.abs(b[i, s:e]).mean())
        d, base = float(np.mean(d)), float(np.mean(base))
        rel = d / max(base, 1e-9)
        v = ("SIGNAL LOST" if d < 1e-4 else "weak" if rel < 0.02 else "transmits")
        arrow = ""
        if prev is not None and prev > 1e-9:
            arrow = f"  (stage gain {d / prev:6.3f}x)"
        print(f"{name:<10} {base:10.5f} {d:10.6f} {rel:10.2%}   {v}{arrow}")
        prev = d
    starts = np.asarray(cfg.PAVLOVIAN_CONFIG["t_start"])[:16]
    dy = np.mean([np.abs(np.asarray(ys_c)[i, int(s):int(s) + 100]
                         - np.asarray(ys_n)[i, int(s):int(s) + 100]).mean()
                  for i, s in enumerate(starts)])
    by = np.mean([np.abs(np.asarray(ys_n)[i, int(s):int(s) + 100]).mean()
                  for i, s in enumerate(starts)])
    print(f"{'OUTPUT':<10} {by:10.5f} {dy:10.6f} {dy / max(by, 1e-9):10.2%}   "
          f"{'NO CUE->OUTPUT PATH' if dy < 1e-4 else 'transmits'}")


# ------------------------------------------------------- 2b. D1/PKA GATE ----
def probe_gate(params, config, inputs):
    sec("2b. MOTOR GATE (D1/PKA) -- is the direct pathway able to open?")
    ys, xs = run(params, config, inputs[:16])
    get = lambda n: np.asarray(xs[AREAS.index(n)])
    pka1, pka2 = get("pkaD1").mean(), get("pkaD2").mean()
    print(f"pka_d1 = {pka1:.4f}   (bg_nln gain = {pka1 / max(1 - pka1, 1e-9):.4f})")
    print(f"pka_d2 = {pka2:.4f}   (bg_nln gain = {pka2 / max(1 - pka2, 1e-9):.4f})")

    # Reconstruct the PKA production terms exactly as the step does.
    m_floor = config.get("m_floor", 0.1)
    fa1 = config.get("m_floor_a1", m_floor)
    fa2 = config.get("m_floor_a2", m_floor)
    cf = lambda x, f: f + np.abs(x) * (1 - f)
    m_d1 = cf(np.asarray(params["m_d1"]), m_floor).mean()
    m_d2 = cf(np.asarray(params["m_d2"]), m_floor).mean()
    m_a1 = cf(np.asarray(params["m_a1"]), fa1).mean()
    m_a2 = cf(np.asarray(params["m_a2"]), fa2).mean()
    k_a = config.get("k_a_floor", 0.05) + float(jax.nn.sigmoid(params["k_a"])) * (
        config.get("k_a_cap", 1.0) - config.get("k_a_floor", 0.05))
    snc = get("SNc").mean()
    g = config.get("da_pka_gain", 1.0)
    da, aden = g * m_d1 * snc, m_a1 * k_a
    print(f"\n  D1 production = max(da_pka_gain*m_d1*mean_snc - m_a1*k_a, 0)")
    print(f"    DA drive  = {g:.2f} * {m_d1:.3f} * {snc:.4f} = {da:.5f}")
    print(f"    adenosine = {m_a1:.3f} * {k_a:.3f} = {aden:.5f}")
    print(f"    net       = {max(da - aden, 0):.5f}"
          f"   {'<-- CLAMPED AT 0: D1 CANNOT BE PRODUCED' if da <= aden else ''}")
    print(f"    steady-state pka_d1 = tau_fall/tau_rise * net = "
          f"{config['tau_pka_fall'] / config['tau_pka_rise'] * max(da - aden, 0):.4f}")
    print(f"\n  D2 production = max(m_a2*k_a - da_pka_gain*m_d2*mean_snc, 0)"
          f" = {max(m_a2 * k_a - g * m_d2 * snc, 0):.5f}")
    print(f"\n  SNr = {get('SNr').mean():.4f}  D1 = {get('D1').mean():.4f}  "
          f"D2 = {get('D2').mean():.4f}  Medulla = {get('Medulla').mean():.4f}")


# --------------------------------------------------------------- 3. CHAOS ----
def probe_chaos(params, config, inputs):
    sec("3. PERTURBATION GROWTH (chaos vs contraction)")
    base = jnp.zeros_like(inputs[:1])
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((1, base.shape[1], n_d1 + n_d2))
    key = jr.split(jr.PRNGKey(7), 1)

    p2 = dict(params)
    eps = 1e-4
    p2["x_c0_U"] = jnp.asarray(params["x_c0_U"]) + eps

    _, xa = cbtl.batched_rnn(params, config, base, stim, key)
    _, xb = cbtl.batched_rnn(p2, config, base, stim, key)
    ca, cb = np.asarray(xa[0][0]), np.asarray(xb[0][0])
    d = np.linalg.norm(ca - cb, axis=-1)
    print(f"cortex |dx| after a {eps:.0e} perturbation of x_c0_U:")
    for t in (0, 10, 25, 50, 100, 200, 400, 700, 999):
        print(f"  t={t:4d}   |dx| = {d[t]:.3e}   ({d[t] / eps:8.3f}x)")
    if d[-1] > d[0]:
        print("  -> EXPANDING (chaotic / unstable)")
    else:
        half = np.argmax(d < d[0] / 2) if (d < d[0] / 2).any() else -1
        print(f"  -> CONTRACTING: perturbation halves in ~{half} steps "
              f"(memory timescale, cue-holding capacity)")


# ------------------------------------------------------------ 4. GRADIENT ----
def probe_gradients(params, config, inputs, targets, masks):
    sec("4. GRADIENT FLOW (which blocks are trainable at init?)")
    rl = cfg.RL_CONFIG
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    B = 16
    inp, tgt, msk = inputs[:B], targets[:B], masks[:B]
    stim = jnp.zeros((B, inp.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)

    def loss_fn(p):
        ys, _ = cbtl.batched_rnn(p, config, inp, stim, keys)
        eps = 1e-6
        bce = -(tgt * jnp.log(ys + eps) + (1 - tgt) * jnp.log(1 - ys + eps))
        return jnp.mean(bce * msk)

    l, g = jax.value_and_grad(loss_fn)(params)
    gn = {k: float(jnp.linalg.norm(v)) for k, v in g.items()}
    total = float(jnp.sqrt(sum(v ** 2 for v in gn.values())))
    print(f"BCE loss at init = {l:.6f}   ||grad|| = {total:.4e}\n")
    dead = [k for k, v in gn.items() if v < 1e-12]
    live = sorted(((v, k) for k, v in gn.items() if v >= 1e-12), reverse=True)
    print(f"{len(dead)} / {len(gn)} parameter blocks have EXACTLY ZERO gradient:")
    print("  " + ", ".join(sorted(dead)) if dead else "  (none)")
    print("\nlargest live gradients:")
    for v, k in live[:12]:
        print(f"  {k:<16} {v:.4e}")
    if live:
        print("\nsmallest live gradients:")
        for v, k in live[-6:]:
            print(f"  {k:<16} {v:.4e}")


def main():
    params, config, inputs, targets, masks = build()
    print(f"\nn_cU={params['J_cU'].shape[0]} n_cL={params['J_cL'].shape[0]} "
          f"n_cI={params['J_c_ii'].shape[0]} n_tE={params['J_t_ee'].shape[0]} "
          f"n_tI={params['J_t_ii'].shape[0]} n_d1={params['J_d1'].shape[0]}  "
          f"tau_c={config.get('tau_c')} noise={config.get('noise_std')}")
    probe_rest(params, config, inputs)
    probe_cue_pathway(params, config, inputs)
    probe_gate(params, config, inputs)
    probe_chaos(params, config, inputs)
    probe_gradients(params, config, inputs, targets, masks)


if __name__ == "__main__":
    main()
