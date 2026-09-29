"""Can the trained-in oscillator be trained back DOWN into the target step shape?

At 600 iters train_supervised_thal reaches obs_rise = 0.31 (95% of the required step) but
does it with a free-running limit cycle: period ~68, peak-to-peak 0.53, running before the
cue. Effective rho* has crossed 1 (0.912 at init -> 1.021), a trained-in Hopf bifurcation
(tests/rise_term_waveform/, memory: training-drives-hopf).

REASON FOR HOPE: the rise term is HINGED, max(0, required - observed)^2, so once the rise
is achieved it contributes exactly zero gradient and goes inert. The only term still
pulling is then the level MSE, which wants flat baseline + held plateau + no ripple -- and
the ripple currently costs 0.0258, more than the wrong mean (0.0173). If the Hopf is
supercritical, amplitude ~ sqrt(rho*-1) shrinks continuously to zero, so there is a smooth
downhill path with no barrier.

REASON FOR DOUBT: as the oscillation damps, obs_rise falls with it and the rise term
switches back on. The two can pin each other at a compromise that is neither flat nor a
step. Escaping that needs a genuinely cue-driven response, i.e. the 300-step credit
assignment problem the oscillation was avoiding.

This runs 3000 iters (5x the original budget; memory: budget-not-learning-rate says longer
budgets are what convert these) from a fresh init, logging every 200:
    level / rise / obs_rise   - is the rise held while the level improves?
    sep                       - full-plateau minus full-baseline (the real target metric)
    osc_pp                    - pre-cue peak-to-peak: the oscillation itself
    rho*                      - effective spectral radius; < 1 means the limit cycle is gone

VERDICT is decided on the last logged row: the oscillator is "trained down" only if
osc_pp has fallen well below its peak AND sep has risen toward the 0.333 band.

Writes: oscillator_trainability.png, trained_<iters>.pkl
"""
import sys, pathlib, time, pickle
_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
sys.path[:0] = [str(_root), str(_root / "cbt_loop_noSCnoSTN")]

import numpy as np, jax, jax.numpy as jnp, jax.random as jr, optax
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config_script as cs
cfg = cs.for_family("cbt_loop_noSCnoSTN")
import self_timed_movement_task as stmt
import cbt_rnn as c
import loop_init as li

t = cfg.TASK_CONFIG; sup = dict(cfg.SUPERVISED_THAL_CONFIG)
B = 20; ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
lo, hi = sup["target_lo"], sup["target_hi"]; hold = int(sup["hold"])

inputs, targets, _ = stmt.selftimed_step_target(
    T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"],
    T_movement=t["t_movement"], T=t["t_total"], lo=lo, hi=hi,
    hold=hold, delay=sup["delay"])
inputs, targets = inputs[:B], targets[:B]
hi_m = np.asarray(targets)[..., 0] > (lo + hi) / 2
masks = jnp.asarray(np.where(hi_m, float(sup["in_window_weight"]), 1.0)[..., None].astype(np.float32))
cue_on = np.argmax(np.asarray(inputs)[..., 0] > 0.5, axis=1)

params, config = c.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=inputs.shape[-1])
config = dict(config); config["readout_source"] = "thalamus"
opt = optax.chain(optax.clip_by_global_norm(1.0),
                  optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]))
opt = c.freeze_init_state_optimizer(opt, params)
st = opt.init(params)
# FRESH noise every iteration. A fixed `keys` here (which this script originally had) shows
# the network ONE frozen noise realization for the whole run, and it memorizes that trace:
# the 10k run reached sep = 0.328 (99% of the band) on seed 0 and 0.019-0.074 (6-22%) on any
# other seed at the SAME noise level. fit_rnn_supervised splits a new key per iteration for
# exactly this reason; the eval below is held at a fixed seed on purpose so the logged
# diagnostics stay comparable across iterations.
train_rng = jr.PRNGKey(0)
eval_keys = jr.split(jr.PRNGKey(12345), B)

TAU = config["tau_c"]; SHIFT = 1.0 - 1.0 / TAU
NN = dict(cU=config["n_c_U"], cL=config["n_c_L"], cI=config["n_c_inh"],
          tE=config["n_t_exc"], tI=config["n_t_inh"])
_AUTO = {"J_cU", "J_cL", "J_c_ii", "J_t_ee", "J_t_ii"}
def W_true(p):
    order = [("cU", NN["cU"]), ("cL", NN["cL"]), ("cI", NN["cI"]), ("tE", NN["tE"]), ("tI", NN["tI"])]
    idx, off = {}, 0
    for k_, s_ in order: idx[k_] = slice(off, off + s_); off += s_
    W = np.zeros((off, off))
    for post, pre, key, sgn in li.LOOP_EDGES:
        m = np.asarray(c.exc(p[key]) if sgn > 0 else c.inh(p[key]))
        if key in _AUTO: m = m * (1.0 - np.eye(m.shape[0]))
        W[idx[post], idx[pre]] = m
    return W

def diagnose(p):
    ys, xs = c.batched_rnn(p, config, inputs, jnp.zeros((B, inputs.shape[1], 20)), eval_keys)
    y = np.asarray(ys)[..., 0]
    xc = np.asarray(xs[c.STATE_AREA_ORDER.index("Cortex")])
    xt = np.asarray(xs[c.STATE_AREA_ORDER.index("Thalamus")])
    xf = np.concatenate([xc, xt], axis=-1); x = xf.reshape(-1, xf.shape[-1]).mean(0)
    g = np.where(x > 0, 1.0 - x ** 2, 0.0)
    J = g[:, None] * (SHIFT * np.eye(len(x)) + W_true(p) / TAU)
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    pp = np.mean([y[b, 10:cue_on[b] - 5].max() - y[b, 10:cue_on[b] - 5].min()
                  for b in range(B) if cue_on[b] > 60])
    return y, rho, float(pp)

def loss_fn(p, keys_):
    return stmt.supervised_loss(c.rnn_func, p, config, inputs, targets, masks, keys_,
                                loss_type="mse", rise_coef=sup["rise_coef"],
                                rise_window=sup["rise_window"])
@jax.jit
def step(p, st, keys_):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, keys_)
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, l, aux

hist = []
print(f"{ITERS} iters, rise_coef={sup['rise_coef']}, band {lo}->{hi}")
print(f"{'it':>6} {'total':>9} {'level':>8} {'rise':>8} {'obs_rise':>9} {'sep':>8} {'osc_pp':>8} {'rho*':>7}")
t0 = time.time()
for i in range(ITERS + 1):
    p_prev = params
    train_rng, sub = jr.split(train_rng)
    params, st, l, aux = step(params, st, jr.split(sub, B))
    if i % 200 == 0:
        y, rho, pp = diagnose(p_prev)
        sep = y[hi_m].mean() - y[~hi_m].mean()
        hist.append((i, float(l), float(aux["sup_loss"]), float(aux["rise_loss"]),
                     float(aux["observed_rise"]), sep, pp, rho))
        print(f"{i:6d} {float(l):9.5f} {float(aux['sup_loss']):8.5f} {float(aux['rise_loss']):8.5f} "
              f"{float(aux['observed_rise']):+9.4f} {sep:8.4f} {pp:8.4f} {rho:7.4f}", flush=True)
print(f"({time.time()-t0:.0f}s)")
with open(pathlib.Path(__file__).with_name(f"trained_{ITERS}.pkl"), "wb") as f:
    pickle.dump({k: np.asarray(v) for k, v in params.items()}, f)

H = np.array(hist)
pk = H[:, 6].max()
print(f"\n--- verdict ---")
print(f"osc_pp  peak {pk:.4f} -> final {H[-1,6]:.4f}  ({100*H[-1,6]/pk:.0f}% of peak)")
print(f"rho*    peak {H[:,7].max():.4f} -> final {H[-1,7]:.4f}  (<1.0 = limit cycle gone)")
print(f"sep     final {H[-1,5]:.4f}  ({100*H[-1,5]/(hi-lo):.0f}% of the {hi-lo:.3f} band)")
damped = H[-1, 6] < 0.5 * pk; shaped = H[-1, 5] > 0.5 * (hi - lo)
print("VERDICT:", "TRAINED DOWN into the step" if (damped and shaped)
      else "oscillation damped but no step" if damped
      else "step emerging under a persistent oscillation" if shaped
      else "STUCK as an oscillator")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(H[:,0], H[:,2], label="level MSE"); ax[0].plot(H[:,0], H[:,3], label="rise term")
ax[0].plot(H[:,0], H[:,1], 'k--', lw=1, label="total"); ax[0].set_yscale("log")
ax[0].set_title("loss terms"); ax[0].legend(fontsize=8)
ax[1].plot(H[:,0], H[:,5], color="#c0392b", label="sep (plateau-baseline)")
ax[1].plot(H[:,0], H[:,4], color="#2e86c1", label="obs_rise")
ax[1].plot(H[:,0], H[:,6], color="#8e44ad", label="osc peak-to-peak (pre-cue)")
ax[1].axhline(hi-lo, color="k", ls=":", lw=1, label=f"target band {hi-lo:.3f}")
ax[1].set_title("shape metrics"); ax[1].legend(fontsize=8)
ax[2].plot(H[:,0], H[:,7], color="#e67e22"); ax[2].axhline(1.0, color="k", ls=":", lw=1)
ax[2].set_title("effective rho*  (>1 = limit cycle)")
for a in ax: a.set_xlabel("iteration"); a.grid(alpha=0.25)
fig.tight_layout()
out = pathlib.Path(__file__).with_name("oscillator_trainability.png")
fig.savefig(out, dpi=140); print("wrote", out)
