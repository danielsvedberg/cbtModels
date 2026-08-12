"""2D grid: balanced_target_rho x noise_std, on the Dale corticothalamic loop.
Tests for critical slowing down -- do fluctuation amplitude AND timescale peak near
the critical operating point (rho_lin=1.0, where rho* is maximal)?

Per cell (mean over seeds): fluctuation std of the pop-mean, autocorrelation time
tau_ac (steps), rho* (operating-point radius). Plots heatmaps + line cuts.

Run:  python corticothalamic/rho_noise_sweep.py
"""
import os, sys
import numpy as np
import jax.numpy as jnp, jax.random as jr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corticothalamic")); sys.path.insert(0, ROOT)
import corticothalamic_rnn as ct
import config_script as C
import loop_init as LI

RHOS = [0.95, 1.0, 1.05, 1.1, 1.2, 1.35]
NOISES = [0.01, 0.025, 0.05, 0.1]
SEEDS = 3
T = 3000; BURN = 600
dt = C.TASK_CONFIG["dt_ms"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "rho_noise_sweep.png")


def autocorr_tau(x):
    x = x - x.mean(); ac = np.correlate(x, x, "full")[len(x) - 1:]; ac = ac / ac[0]
    return float(np.argmax(ac < np.exp(-1)) if (ac < np.exp(-1)).any() else len(ac))


def rho_star(p, conf, rest):
    szs = (conf["n_c_U"], conf["n_c_L"], conf["n_c_inh"], conf["n_t_exc"], conf["n_t_inh"])
    W = LI.loop_matrix(p, *szs); tau = conf["tau_ctx"]; N = W.shape[0]
    M = (1.0 - 1.0 / tau) * np.eye(N) + (1.0 / tau) * W
    g = 4.0 * rest * (1.0 - rest)
    return float(np.max(np.abs(np.linalg.eigvals(g[:, None] * M))))


FS = np.zeros((len(NOISES), len(RHOS)))
TA = np.zeros((len(NOISES), len(RHOS)))
RS = np.zeros((len(NOISES), len(RHOS)))
for j, rho in enumerate(RHOS):
    C.CORTICOTHALAMIC_RNN_CONFIG["balanced_target_rho"] = rho
    for i, noise in enumerate(NOISES):
        fs, ta, rs = [], [], []
        for s in range(SEEDS):
            p, conf = ct.init_params(jr.PRNGKey(s), n_input=1)
            c = dict(conf); c["noise_std"] = noise
            _, (xc, xt) = ct.corticothalamic_rnn(p, c, jnp.zeros((T, 1)), rng_key=jr.PRNGKey(100 + s))
            allx = np.concatenate([np.asarray(xc), np.asarray(xt)], axis=-1)[BURN:]
            popm = allx.mean(-1)
            fs.append(popm.std()); ta.append(autocorr_tau(popm)); rs.append(rho_star(p, conf, allx.mean(0)))
        FS[i, j] = np.mean(fs); TA[i, j] = np.median(ta); RS[i, j] = np.mean(rs)
    print(f"rho={rho:.2f}: rho*={RS[:,j].mean():.3f}  tau_ac(noise=.01)={TA[0,j]:.0f}  "
          f"tau_ac(noise=.1)={TA[-1,j]:.0f}  fluct_std(.05)={FS[2,j]:.4f}")

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
def heat(a, Z, title, fmt="%.3f"):
    im = a.imshow(Z, aspect="auto", origin="lower", cmap="viridis")
    a.set_xticks(range(len(RHOS))); a.set_xticklabels([f"{r:.2f}" for r in RHOS])
    a.set_yticks(range(len(NOISES))); a.set_yticklabels(NOISES)
    a.set_xlabel("balanced_target_rho"); a.set_ylabel("noise_std"); a.set_title(title)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            a.text(j, i, fmt % Z[i, j], ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=a, fraction=0.046)
heat(ax[0,0], FS, "Fluctuation std (amplitude)", "%.4f")
heat(ax[0,1], TA, "Autocorrelation tau (steps)", "%.0f")
# line cuts
for i, noise in enumerate(NOISES):
    ax[1,0].plot(RHOS, TA[i], "o-", label=f"noise={noise}")
ax[1,0].axvline(1.0, ls="--", color="grey", alpha=.6); ax[1,0].set_xlabel("balanced_target_rho")
ax[1,0].set_ylabel("autocorr tau (steps)"); ax[1,0].set_title("Critical slowing? tau_ac vs rho"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.25)
for i, noise in enumerate(NOISES):
    ax[1,1].plot(RHOS, FS[i]/noise, "o-", label=f"noise={noise}")
ax[1,1].axvline(1.0, ls="--", color="grey", alpha=.6); ax[1,1].set_xlabel("balanced_target_rho")
ax[1,1].set_ylabel("fluct_std / noise (susceptibility)"); ax[1,1].set_title("Susceptibility (amplitude per unit noise) vs rho"); ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
