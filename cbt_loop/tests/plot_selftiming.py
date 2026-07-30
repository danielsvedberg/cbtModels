"""Demonstrate self-timing of the trained cbt_loop model (params_shaped.pkl).

Runs the self-timed task and shows (A) output traces for several cue times — the
response peak SHIFTS with the cue — and (B) response peak-time vs cue time, which
sits on the slope-1 line (self-timing) rather than flat (fixed-time).

Run:  python cbt_loop/tests/plot_selftiming.py [bundle.pkl]
"""
import sys, os, pathlib, pickle as pkl
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cbt_loop")); sys.path.insert(0, str(ROOT))
import cbt_rnn as cbtl
import config_script as C
import self_timed_movement_task as stmt

BUNDLE = sys.argv[1] if len(sys.argv) > 1 else "params_shaped.pkl"
OUT = pathlib.Path(__file__).resolve().parent / "plots" / "selftiming.png"
cfg = C.for_family("cbt_loop"); t = cfg.TASK_CONFIG

with open(ROOT / "cbt_loop" / BUNDLE, "rb") as f:
    b = pkl.load(f)
params, config = b["params"], b["config"]

inputs, targets, masks = stmt.self_timed_movement_task(
    T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"],
    T_movement=t["t_movement"], T=t["t_total"])
inputs = cbtl.match_input_channels(inputs, params)
ts = np.asarray(t["t_start"])
B = inputs.shape[0]; nd1 = params["J_d1"].shape[0]; nd2 = params["J_d2"].shape[0]
stim = jnp.zeros((B, inputs.shape[1], nd1 + nd2))
ys = cbtl.rnn_func(params, config, inputs, stim, jr.split(jr.PRNGKey(0), B))[0]
ys = np.asarray(ys[..., 0] if ys.ndim == 3 else ys)
rt = ys.argmax(1)                                   # peak response time per trial
slope, icpt = np.polyfit(ts.astype(float), rt.astype(float), 1)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

# A: example output traces for a spread of cue times
order = np.argsort(ts)
pick = order[np.linspace(0, B - 1, 7).astype(int)]
cmap = cm.viridis; norm = Normalize(ts.min(), ts.max())
for i in pick:
    c = cmap(norm(ts[i]))
    ax[0].plot(ys[i], color=c, lw=1.8)
    ax[0].axvline(ts[i], color=c, ls=":", alpha=.5)      # cue time
ax[0].set_title("A. Output traces (color = cue time)\nresponse peak shifts WITH the cue")
ax[0].set_xlabel("time step"); ax[0].set_ylabel("output (response prob)")
ax[0].grid(alpha=.25)
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax[0], label="cue time t_start")

# B: response peak-time vs cue time
ax[1].scatter(ts, rt, s=28, c=ts, cmap="viridis", zorder=3)
xx = np.array([ts.min(), ts.max()])
ax[1].plot(xx, slope * xx + icpt, "k-", lw=2, label=f"fit: slope={slope:.2f}")
ax[1].plot(xx, xx + t["t_cue"] + t["t_wait"], "r--", lw=1.5,
           label=f"ideal (cue + {t['t_cue']+t['t_wait']})")
ax[1].axhline(np.median(rt), color="grey", ls=":", label="fixed-time (slope 0)")
ax[1].set_title("B. Response time vs cue time\nslope 1 = self-timing; flat = fixed-time")
ax[1].set_xlabel("cue time t_start"); ax[1].set_ylabel("response peak time")
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.25)

fig.suptitle(f"cbt_loop self-timing ({BUNDLE}) — slope {slope:.2f}", fontsize=13)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print(f"slope={slope:.3f}  saved {OUT}")
