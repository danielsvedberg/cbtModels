"""Does the rise term buy a real PLATEAU, or only a transient at the step onset?

supervised_loss's rise term (config rise_coef / rise_window) compares mean output over
[t0, t0+k) against mean over [t0-k, t0) and hinges on falling short of the step height.
It therefore prices the EDGE and nothing prices the plateau, so a network could satisfy
it with a brief transient at t0 and let the output decay back inside the 50-step window.

The 600-iter training run that motivated this test ended with:
    obs_rise = 0.3165   (95% of the required hi-lo = 0.333)
    sep      = 0.0824   (full-plateau mean minus full-baseline mean)
    level    = 0.0434   (WORSE than the 0.0277 it had while perfectly flat)
obs_rise >> sep is exactly the signature a decaying transient would produce, but it is
also what a slow-rising ramp would produce, so the two metrics alone cannot tell them
apart. This script retrains under the identical protocol, then plots the cue-aligned
mean output against the target so the waveform itself settles it.

Reads the live config/readout, so it reflects whatever cbt_rnn currently does (at the
time of writing: the bias-free thalamic readout y_t = c_thal @ x_t_exc).

VERDICT (600 iters, seed 42, B=20, bias-free thalamic readout): NEITHER. The network
learned a FREE-RUNNING OSCILLATOR -- period ~68 timesteps, peak-to-peak 0.53, running
before the cue arrives where the target is flat -- and collects the rise term from the
oscillation's upswing (trough in the pre-window, peak in the post-window) rather than
producing a step. rise_window = 25 is ~1/3 of that period, i.e. near the window that
maximizes an oscillator's post-minus-pre. Numbers:
    baseline [-150,0)      0.4539   (target 0.333)
    plateau 1st half       0.6259   (target 0.666)
    plateau 2nd half       0.4443   (target 0.666)   <- already falling
    off-window MSE from the wrong mean      0.0173
    off-window MSE from the variance/ripple 0.0258   <- oscillation dominates the loss
So the rise term fixes the flat-constant degeneracy and introduces an oscillatory one.

Writes: rise_term_waveform.png, trained_<iters>.pkl (params cache; --retrain to redo)
"""
import sys, pathlib, time
_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
_fam = _root / "cbt_loop_noSCnoSTN"
sys.path[:0] = [str(_root), str(_fam)]

import numpy as np, jax, jax.numpy as jnp, jax.random as jr, optax
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config_script as cs
cfg = cs.for_family("cbt_loop_noSCnoSTN")
import self_timed_movement_task as stmt
import cbt_rnn as c

t = cfg.TASK_CONFIG; sup = dict(cfg.SUPERVISED_THAL_CONFIG)
B = 20; ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
lo, hi = sup["target_lo"], sup["target_hi"]
k = int(sup["rise_window"]); hold = int(sup["hold"])

inputs, targets, _ = stmt.selftimed_step_target(
    T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"],
    T_movement=t["t_movement"], T=t["t_total"],
    lo=lo, hi=hi, hold=hold, delay=sup["delay"])
inputs, targets = inputs[:B], targets[:B]
hi_m = np.asarray(targets)[..., 0] > (lo + hi) / 2
masks = jnp.asarray(np.where(hi_m, float(sup["in_window_weight"]), 1.0)[..., None].astype(np.float32))

params, config = c.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=inputs.shape[-1])
config = dict(config); config["readout_source"] = "thalamus"
opt = optax.chain(optax.clip_by_global_norm(1.0),
                  optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]))
opt = c.freeze_init_state_optimizer(opt, params)
st = opt.init(params); keys = jr.split(jr.PRNGKey(0), B)

def loss_fn(p):
    return stmt.supervised_loss(c.rnn_func, p, config, inputs, targets, masks, keys,
                                loss_type="mse", rise_coef=sup["rise_coef"],
                                rise_window=sup["rise_window"])
@jax.jit
def step(p, st):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, l, aux

ys_init, *_ = c.rnn_func(params, config, inputs, None, keys)
_cache = pathlib.Path(__file__).with_name(f"trained_{ITERS}.pkl")
if _cache.exists() and "--retrain" not in sys.argv:
    import pickle
    with _cache.open("rb") as f:
        params = pickle.load(f)
    print(f"loaded trained params from {_cache.name} (pass --retrain to redo the {ITERS} iters)")
else:
    print(f"training {ITERS} iters (rise_coef={sup['rise_coef']}, window={k})...")
    t0_ = time.time()
    for i in range(ITERS):
        params, st, l, aux = step(params, st)
        if i % 100 == 0:
            print(f"  {i:5d} total {float(l):.5f} rise {float(aux['rise_loss']):.5f} "
                  f"obs_rise {float(aux['observed_rise']):+.4f}")
    print(f"  done ({time.time()-t0_:.0f}s)  final total {float(l):.5f} obs_rise {float(aux['observed_rise']):+.4f}")
    import pickle
    with _cache.open("wb") as f:
        pickle.dump({kk: np.asarray(vv) for kk, vv in params.items()}, f)
    print(f"  cached -> {_cache.name}")

ys_fin, *_ = c.rnn_func(params, config, inputs, None, keys)

# Cue-align every trial on its own step onset so the plateaus superimpose.
tg = np.asarray(targets)[..., 0]
onset = np.argmax(tg > (lo + hi) / 2, axis=1)
PRE, POST = 150, 200
T_full = tg.shape[1]
def align(y):
    """Cue-align on each trial's own onset, NaN-padding where the window runs off the
    trial. Onsets span 369-810 in a T=1000 trial, so a fixed slice is ragged; NaN + nanmean
    keeps every trial contributing wherever it has data (the [0, hold) plateau is in range
    for all of them) instead of dropping trials or silently truncating."""
    y = np.asarray(y)[..., 0]
    out = np.full((y.shape[0], PRE + POST), np.nan)
    for b in range(y.shape[0]):
        s0, s1 = onset[b] - PRE, onset[b] + POST
        lo_c, hi_c = max(s0, 0), min(s1, T_full)
        out[b, lo_c - s0: hi_c - s0] = y[b, lo_c:hi_c]
    return out
a_fin, a_init, a_tg = align(ys_fin), align(ys_init), align(targets)
tau = np.arange(-PRE, POST)

plateau = (tau >= 0) & (tau < hold)
first_half = (tau >= 0) & (tau < hold // 2)
second_half = (tau >= hold // 2) & (tau < hold)
base = tau < 0
m = np.nanmean(a_fin, axis=0)
print("\n--- verdict numbers (trained, cue-aligned mean) ---")
print(f"baseline        [-150,0) : {np.nanmean(m[base]):.4f}   (target {lo})")
print(f"plateau 1st half [0,{hold//2}) : {np.nanmean(m[first_half]):.4f}   (target {hi})")
print(f"plateau 2nd half [{hold//2},{hold}): {np.nanmean(m[second_half]):.4f}   (target {hi})")
print(f"plateau peak             : {np.nanmax(m[plateau]):.4f} at tau={tau[plateau][int(np.nanargmax(m[plateau]))]}")
decay = np.nanmean(m[first_half]) - np.nanmean(m[second_half])
rise = np.nanmean(m[plateau]) - np.nanmean(m[base])
print(f"\nplateau rise over baseline : {rise:+.4f}  ({100*rise/(hi-lo):.0f}% of the {hi-lo:.3f} band)")
print(f"decay across the plateau   : {decay:+.4f}  ({100*decay/max(rise,1e-9):.0f}% of the rise)")
print("VERDICT:", "TRANSIENT (decays >25% within the window)" if decay > 0.25*max(rise,1e-9)
      else "sustained plateau (holds within 25%)")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [2, 1]})
for a in ax:
    a.axvspan(0, hold, color="0.92", zorder=0, label=f"target window [0,{hold})")
    a.axvspan(-k, 0, color="#fde7c9", zorder=0, label=f"rise-term pre [-{k},0)")
    a.axvspan(0, k, color="#cfe8d4", zorder=0, label=f"rise-term post [0,{k})")
    a.plot(tau, np.nanmean(a_tg, axis=0), color="k", lw=2.2, ls="--", label="target")
    a.plot(tau, np.nanmean(a_init, axis=0), color="#9aa0a6", lw=1.3, label="output at init")
    a.plot(tau, np.nanmean(a_fin, axis=0), color="#c0392b", lw=2.0, label=f"output after {ITERS} iters")
    a.set_xlabel("timesteps from step onset")
    a.grid(alpha=0.25)
ax[0].plot(tau, a_fin.T, color="#c0392b", lw=0.4, alpha=0.25, zorder=1)
ax[0].set_ylabel("readout y")
ax[0].set_title(f"cue-aligned readout (thin = individual trials, n={B})")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[1].set_xlim(-k - 10, hold + 40); ax[1].set_title("zoom on the step")
lohi = [min(np.nanmin(a_fin), lo) - 0.03, max(np.nanmax(a_fin), hi) + 0.03]
for a in ax: a.set_ylim(*lohi)
fig.suptitle(f"Does the rise term give a plateau or a transient?   "
             f"rise={rise:+.3f} of {hi-lo:.3f} band, decay={decay:+.3f}", fontsize=10.5)
fig.tight_layout()
out = pathlib.Path(__file__).with_name("rise_term_waveform.png")
fig.savefig(out, dpi=140)
print("\nwrote", out)
