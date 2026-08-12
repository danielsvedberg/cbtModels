"""How does noise affect criticality in the Dale corticothalamic loop?

For each noise_std, run a long stochastic trajectory and measure:
  mean_rate   - operating point (mean pop activity, after settling)
  fluct_std   - std of the population-mean activity over time (fluctuation amplitude)
  tau_ac      - autocorrelation time of the pop-mean fluctuations (steps): the
                NOISE-DRIVEN memory timescale (near-critical -> long-lived fluctuations)
  rho*        - spectral radius of diag(g)M at the noise-shifted mean operating point
                (g=4r(1-r)); deterministic criticality AS SEEN by the noisy network
  tau_eff     - -1/ln(rho*) (steps), the deterministic prediction, for comparison

Both the linear rho_lin=1.0 (balanced_init) is fixed; the question is whether noise
shifts the operating point / gain and how the fluctuation timescale behaves.

Run:  python corticothalamic/noise_sweep.py
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

NOISES = [0.01, 0.025, 0.05, 0.1]
SEEDS = 4
T = 4000
BURN = 800
dt = C.TASK_CONFIG["dt_ms"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "noise_sweep.png")


def autocorr_tau(x):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    below = np.argmax(ac < np.exp(-1)) if (ac < np.exp(-1)).any() else len(ac)
    return float(below), ac[:400]


def rho_star(p, conf, rest):
    szs = (conf["n_c_U"], conf["n_c_L"], conf["n_c_inh"], conf["n_t_exc"], conf["n_t_inh"])
    W = LI.loop_matrix(p, *szs); tau = conf["tau_ctx"]; N = W.shape[0]
    M = (1.0 - 1.0 / tau) * np.eye(N) + (1.0 / tau) * W
    g = 4.0 * rest * (1.0 - rest)
    return float(np.max(np.abs(np.linalg.eigvals(g[:, None] * M))))


print(f"corticothalamic noise sweep  (tau={C.CORTICOTHALAMIC_RUNTIME_CONFIG['tau_ctx']}, "
      f"rho_lin=1.0, dt={dt} ms, T={T}, {SEEDS} seeds)")
print(f"{'noise':>6} {'mean_rate':>10} {'fluct_std':>10} {'tau_ac(steps)':>14} {'tau_ac(ms)':>11} {'rho*':>7} {'tau_eff':>8}")
curves = {}
for noise in NOISES:
    mr, fs, tac, rs = [], [], [], []
    acs = []
    for s in range(SEEDS):
        p, conf = ct.init_params(jr.PRNGKey(s), n_input=1)
        c = dict(conf); c["noise_std"] = noise
        _, (xc, xt) = ct.corticothalamic_rnn(p, c, jnp.zeros((T, 1)), rng_key=jr.PRNGKey(100 + s))
        allx = np.concatenate([np.asarray(xc), np.asarray(xt)], axis=-1)[BURN:]  # (T-burn, N)
        popm = allx.mean(-1)                                    # population-mean over time
        mr.append(allx.mean())
        fs.append(popm.std())
        tau_ac, ac = autocorr_tau(popm); tac.append(tau_ac); acs.append(ac)
        rs.append(rho_star(p, conf, allx.mean(0)))
    rho = np.mean(rs); teff = (-1.0 / np.log(rho)) if 0 < rho < 1 else np.inf
    tacm = np.median(tac)
    curves[noise] = np.mean(acs, axis=0)
    print(f"{noise:>6.3f} {np.mean(mr):>10.3f} {np.mean(fs):>10.4f} {tacm:>14.1f} {tacm*dt:>11.0f} {rho:>7.3f} {teff:>8.1f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for noise in NOISES:
    ax[0].plot(curves[noise], lw=2, label=f"noise={noise}")
ax[0].axhline(np.exp(-1), ls="--", color="grey", alpha=.6); ax[0].text(300, np.exp(-1), " 1/e", color="grey", fontsize=8)
ax[0].set_title("Autocorrelation of spontaneous pop activity\n(longer decay = longer fluctuation memory)")
ax[0].set_xlabel("lag (steps)"); ax[0].set_ylabel("autocorrelation"); ax[0].legend(); ax[0].grid(alpha=.25)
# example traces
ax[1].set_title("Example population-mean traces per noise level")
for i, noise in enumerate(NOISES):
    p, conf = ct.init_params(jr.PRNGKey(0), n_input=1); c = dict(conf); c["noise_std"] = noise
    _, (xc, xt) = ct.corticothalamic_rnn(p, c, jnp.zeros((1500, 1)), rng_key=jr.PRNGKey(7))
    pm = np.concatenate([np.asarray(xc), np.asarray(xt)], axis=-1).mean(-1)
    ax[1].plot(pm[500:1200] + i * 0.15, lw=0.8, label=f"noise={noise}")
ax[1].set_xlabel("time step"); ax[1].set_ylabel("pop-mean (offset per level)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
