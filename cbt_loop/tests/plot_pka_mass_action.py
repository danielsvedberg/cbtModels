"""Plot the REAL PKA dynamics of the current cbt_loop model (forward pass, no
training), under the gate-free architecture: PKA is a mass-action-bounded pool in
(0,1) fed DIRECTLY as bg_nln's excitability b (b = clip(pka, eps, 1-eps)).

Runs the actual cbt_rnn forward on a cued vs uncued preparatory-cue trial and
shows: pka_d1 / pka_d2 both resting ~0.5, the OPPONENT cue response (dopamine
raises D1 PKA, brakes D2 PKA), and the downstream D1 / OUTPUT signals.

Run:  python cbt_loop/tests/plot_pka_mass_action.py
"""
import sys, pathlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cbt_loop")); sys.path.insert(0, str(ROOT))
import cbt_rnn as cbtl
import config_script as C
import self_timed_movement_task as stmt  # noqa

OUT = pathlib.Path(__file__).resolve().parent / "plots" / "pka_mass_action.png"
A = cbtl.STATE_AREA_ORDER
cfg = C.for_family("cbt_loop")
t = cfg.TASK_CONFIG
T = t["t_total"]; t0 = 200; tw = t["t_cue"] + t["t_wait"]; target = t0 + tw


def run(params, config, inputs, noise=0.0):
    c = dict(config); c["noise_std"] = noise
    B = inputs.shape[0]
    nd1 = params["J_d1"].shape[0]; nd2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inputs.shape[1], nd1 + nd2))
    ys, xs = cbtl.batched_rnn(params, c, inputs, stim, jr.split(jr.PRNGKey(0), B))
    return np.asarray(ys), xs


def tr(xs, name):
    return np.asarray(xs[A.index(name)][0]).mean(-1)


params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=2)
cued = np.zeros((1, T, 2), np.float32); cued[0, t0:t0 + t["t_cue"], 0] = 1.0
null = np.zeros_like(cued)
inp_c = cbtl.match_input_channels(jnp.asarray(cued), params)
inp_n = cbtl.match_input_channels(jnp.asarray(null), params)
ys_c, xs_c = run(params, config, inp_c)
ys_n, xs_n = run(params, config, inp_n)

pka1_c, pka1_n = tr(xs_c, "pkaD1"), tr(xs_n, "pkaD1")
pka2_c, pka2_n = tr(xs_c, "pkaD2"), tr(xs_n, "pkaD2")
d1_c, d1_n = tr(xs_c, "D1"), tr(xs_n, "D1")
out_c, out_n = ys_c[0, :, 0], ys_n[0, :, 0]
eps = config["pka_clip_eps"]
print(f"resting pka_d1={pka1_n[300:].mean():.3f}  pka_d2={pka2_n[300:].mean():.3f}  "
      f"(both fed to bg_nln as b=clip(pka,{eps},{1-eps}))")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
CUE, TGT = "#888", "#c44"


def marks(a):
    a.axvspan(t0, t0 + t["t_cue"], color=CUE, alpha=.25)
    a.axvline(target, ls="--", color=TGT, alpha=.7)


# A: pka_d1 & pka_d2 both rest ~0.5, bounded (0,1) — they ARE the bg_nln b
a = ax[0, 0]
a.plot(pka1_n, lw=2.2, color="#2a6fdb", label="pka_d1 (uncued)")
a.plot(pka1_c, lw=2.2, color="#2a6fdb", ls=":", label="pka_d1 (cued)")
a.plot(pka2_n, lw=2.2, color="#d1495b", label="pka_d2 (uncued)")
a.plot(pka2_c, lw=2.2, color="#d1495b", ls=":", label="pka_d2 (cued)")
a.axhline(0.5, ls="--", color="grey", alpha=.6); a.text(5, 0.5, " rest 0.5", fontsize=8, color="grey", va="bottom")
a.axhline(config["pka_max"], ls=":", color="green", alpha=.6); a.text(T*0.98, 1.0, "pka_max=1", color="green", ha="right", va="top", fontsize=8)
a.set_title("A. PKA rests ~0.5 (bounded 0-1) and IS bg_nln's b\n(no separate gate)")
a.set_ylabel("pka (= excitability b)"); a.set_xlabel("time step"); a.set_ylim(0, 1.02)
a.legend(fontsize=8); a.grid(alpha=.25); marks(a)

# B: OPPONENT cue response — DA raises D1 PKA, brakes D2 PKA
a = ax[0, 1]
a.plot(pka1_c - pka1_n, lw=2.2, color="#2a6fdb", label="Δ pka_d1 (DA excites D1) > 0")
a.plot(pka2_c - pka2_n, lw=2.2, color="#d1495b", label="Δ pka_d2 (DA brakes D2) < 0")
a.axhline(0, color="black", lw=.8, alpha=.5)
a.set_title("B. Opponent dopamine response (cued − uncued)")
a.set_ylabel("Δ pka"); a.set_xlabel("time step")
a.legend(fontsize=8.5); a.grid(alpha=.25); marks(a)

# C: downstream D1
a = ax[1, 0]
a.plot(d1_n, lw=2, color="#2a6fdb", label="D1 (uncued)")
a.plot(d1_c, lw=2, color="#d1495b", label="D1 (cued)")
a.plot(d1_c - d1_n, lw=1.6, color="black", ls="--", label="cue-evoked Δ")
a.set_title("C. Downstream D1 firing"); a.set_ylabel("D1 (pop mean)"); a.set_xlabel("time step")
a.legend(fontsize=8.5); a.grid(alpha=.25); marks(a)

# D: OUTPUT
a = ax[1, 1]
a.plot(out_n, lw=2, color="#2a6fdb", label="OUTPUT (uncued)")
a.plot(out_c, lw=2, color="#d1495b", label="OUTPUT (cued)")
a.plot(out_c - out_n, lw=1.6, color="black", ls="--", label="cue-evoked Δ")
a.set_title("D. Behavioral OUTPUT (readout)"); a.set_ylabel("response prob"); a.set_xlabel("time step")
a.legend(fontsize=8.5); a.grid(alpha=.25); marks(a)

plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
