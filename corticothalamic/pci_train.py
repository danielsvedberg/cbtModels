"""Perturbational Complexity Index (PCI) test + training for the corticothalamic loop.

Idea (Casali/Massimini/Tononi PCI): perturb the network, record the evoked
spatiotemporal response, binarize it against the pre-stimulus baseline, and measure
the algorithmic complexity of the binary pattern by *compressing* it (zlib/DEFLATE is
the LZ-family stand-in for Lempel-Ziv). A larger compressed file == a richer, less
redundant ("more complex") ensemble transient.

We turn this into a training objective for the loop's weights, with three guards the
user asked for:

  1. raw POST-perturbation PCI as the first reward term.  PCI is still measured on both
     a PRE-stim and a POST-stim window (equal length, same per-unit baseline threshold),
     but the reward's first term is the raw post-stim complexity `comp(post)`, NOT the
     delta `comp(post) - comp(pre)`.  Being complex at baseline is instead held back by
     guards (2) and (3) below rather than by subtraction.  (The pre-window PCI is still
     computed and reported for diagnostics.)

  2. NO baseline penalty.  PCI already rewards a low->high transition and the zip measure
     already penalizes saturation (a uniformly "on" post-window compresses to almost
     nothing), so no explicit pre-stim quiescence term is used. `base_std` is reported as
     a diagnostic only.

  3. anti-spontaneous (catch-trial) penalty with jitter.  Half the trials are CATCH
     trials: identical timing distribution, no perturbation.  The perturbation time is
     JITTERED, so a network that fires a complex transient at a fixed time would fire it
     on catch trials too and get penalized.  Only complexity that is genuinely
     *evoked and time-locked to the stimulus* survives.  We penalize the catch-trial
     post-window complexity.

Because zlib-compression and binarization are non-differentiable, the weights are
trained by evolution strategies (OpenAI-ES: antithetic Gaussian perturbations, common
random numbers across the population for variance reduction, fitness rank-shaping,
Adam on the ES gradient estimate).  What is optimized is therefore *exactly* the
zip-based delta_PCI objective -- no differentiable surrogate.  Dale sign constraints are
enforced inside the forward pass, so ES perturbations remain physiological.

Usage:
  python pci_train.py --generations 300               # train, saves params + curve
  python pci_train.py --eval --load params_pci.pkl    # evaluate + diagnostic plot
"""
import argparse
import pickle as pkl
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import corticothalamic_rnn as ctrnn  # noqa: E402
import loop_init  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"


# --------------------------------------------------------------------------------------
# Trial construction
# --------------------------------------------------------------------------------------
def make_trials(rng, cfg, n_stim, n_catch):
    """Build a batch of (inputs, opto, rng, is_stim, t_stim).

    Trials are shared across the ES population (common random numbers), so the network
    noise realization and the jittered stim time are identical for theta+eps and
    theta-eps -- this cancels most of the ES gradient variance.
    """
    T = cfg["T"]
    N = cfg["N"]
    n_trials = n_stim + n_catch
    rng, k_t, k_net = jr.split(rng, 3)

    # jittered stim time, uniform in [t_lo, t_hi)
    t_stim = jr.randint(k_t, (n_trials,), cfg["t_lo"], cfg["t_hi"])
    is_stim = jnp.arange(n_trials) < n_stim  # first n_stim are real, rest are catch

    # perturbation: brief strong current pulse onto the cU (upper cortical) pool
    opto = jnp.zeros((n_trials, T, N))
    u_idx = jnp.asarray(cfg["stim_units"])
    dur = cfg["stim_dur"]

    def _fill(i, opto):
        t0 = t_stim[i]
        pulse = jnp.zeros((T, N))
        # add stim_amp on [t0, t0+dur) for the perturbed units, only if a stim trial
        tmask = (jnp.arange(T) >= t0) & (jnp.arange(T) < t0 + dur)
        col = jnp.zeros((N,)).at[u_idx].set(cfg["stim_amp"])
        pulse = jnp.where(tmask[:, None], col[None, :], 0.0) * is_stim[i]
        return opto.at[i].set(pulse)

    opto = jax.lax.fori_loop(0, n_trials, _fill, opto)
    inputs = jnp.zeros((n_trials, T, cfg["n_input"]))  # no task input; perturb via opto
    net_keys = jr.split(k_net, n_trials)
    return inputs, opto, net_keys, is_stim, t_stim


# --------------------------------------------------------------------------------------
# Population rollout: vmap the batched rollout over the ES population axis.
# Trials (inputs/opto/rng) are shared across the population (in_axes=None).
# --------------------------------------------------------------------------------------
_pop_rollout = vmap(ctrnn.batched_rnn, in_axes=(0, None, None, None, None))


def rollout(pop_params, config, inputs, opto, net_keys):
    ys, (x_ctx, x_t) = _pop_rollout(pop_params, config, inputs, opto, net_keys)
    X = jnp.concatenate([x_ctx, x_t], axis=-1)  # (pop, trials, T, N)
    return X


# --------------------------------------------------------------------------------------
# zip-based complexity of a binary spatiotemporal window
# --------------------------------------------------------------------------------------
def _zip_len(bits_NW):
    """Compressed length (bytes) of a (N, W) binary matrix, units-major flatten.

    Each unit's time series is laid out contiguously so zlib can exploit BOTH temporal
    redundancy (within a row) and spatial redundancy (repeated rows).  The window size
    is fixed across all trials, so the raw byte length is directly comparable.
    """
    b = np.ascontiguousarray(bits_NW.astype(np.uint8))
    return len(zlib.compress(b.tobytes(), 9))


def pci_windows(X_pop, t_stim, is_stim, cfg):
    """Compute per-(pop,trial) complexity of the pre- and post-stim windows.

    Binarization: for each trial and unit, take the PRE-window mean/std as the baseline
    distribution; a sample is 'significant' if it deviates > k*std (two-sided).  Both
    windows are binarized against this pre-stim baseline (the Tononi significance idea).

    Returns dict of arrays shaped (pop, n_trials): comp_pre, comp_post, base_std.
    """
    X = np.asarray(X_pop)  # (pop, trials, T, N)
    pop, n_trials, T, N = X.shape
    W = cfg["W"]
    k = cfg["thr_k"]
    floor = cfg["sd_floor"]
    t_stim = np.asarray(t_stim)

    comp_pre = np.zeros((pop, n_trials))
    comp_post = np.zeros((pop, n_trials))
    base_std = np.zeros((pop, n_trials))

    for tr in range(n_trials):
        t0 = int(t_stim[tr])
        pre = X[:, tr, t0 - W:t0, :]      # (pop, W, N)
        post = X[:, tr, t0:t0 + W, :]     # (pop, W, N)
        mu = pre.mean(axis=1, keepdims=True)                 # (pop,1,N)
        sd = pre.std(axis=1, keepdims=True)                  # (pop,1,N)
        thr = k * np.maximum(sd, floor)
        bin_pre = np.abs(pre - mu) > thr                     # (pop,W,N)
        bin_post = np.abs(post - mu) > thr
        base_std[:, tr] = sd[:, 0, :].mean(axis=-1)          # mean unit std at baseline
        for p in range(pop):
            comp_pre[p, tr] = _zip_len(bin_pre[p].T)         # (N,W) units-major
            comp_post[p, tr] = _zip_len(bin_post[p].T)
    return {"comp_pre": comp_pre, "comp_post": comp_post, "base_std": base_std,
            "is_stim": np.asarray(is_stim)}


def rewards_from_pci(pw, cfg):
    """Scalar reward per population member from the PCI window measurements.

    reward = post_PCI  -  lam_spont * spontaneous

      post_PCI    = mean over STIM trials of comp_post   (raw post-perturbation PCI)
      spontaneous = mean over CATCH trials of max(comp_post - comp_pre, 0)

    Nothing is subtracted from the stimulated response: the first term is the RAW
    post-perturbation complexity. The only penalty is the catch-trial spontaneous term,
    which ties the complexity to the stimulus (a network complex WITHOUT a perturbation
    pays here). `base_std` is still reported as a diagnostic but does not enter the
    reward -- PCI itself already rewards a low->high transition and the zip measure
    already penalizes saturation, so no explicit baseline term is needed.
    """
    is_stim = pw["is_stim"]
    post = pw["comp_post"][:, is_stim].mean(axis=1)   # (pop,)  raw post-stim PCI
    catch_d = (pw["comp_post"] - pw["comp_pre"])[:, ~is_stim]
    spont = np.maximum(catch_d, 0.0).mean(axis=1) if catch_d.shape[1] else np.zeros_like(post)
    calm = pw["base_std"].mean(axis=1)                # (pop,)  diagnostic only
    reward = post - cfg["lam_spont"] * spont
    return reward, {"post_pci": post, "spont": spont, "calm": calm}


# --------------------------------------------------------------------------------------
# Evolution strategies
# --------------------------------------------------------------------------------------
def _rank_utilities(rewards):
    """Centered-rank fitness shaping in [-0.5, 0.5] (robust to reward scale/outliers)."""
    n = len(rewards)
    ranks = np.empty(n)
    ranks[np.argsort(rewards)] = np.arange(n)
    return ranks / (n - 1) - 0.5


# The 5 within-population loop blocks whose self-diagonal the forward pass zeros
# (no_autapse). ES perturbs the diagonal back in, so we must zero it before any rho
# computation/renormalization to match the actual dynamics.
_DIAG_BLOCKS = ("J_cU", "J_cL", "J_c_ii", "J_t_ee", "J_t_ii")


def _zero_loop_diag(theta):
    out = dict(theta)
    for key in _DIAG_BLOCKS:
        m = jnp.asarray(out[key])
        out[key] = m * (1.0 - jnp.eye(m.shape[0], dtype=m.dtype))
    return out


def loop_rho(theta, config):
    """rho_lin of the loop AS THE FORWARD PASS SEES IT (self-diagonals zeroed)."""
    p = {k: np.asarray(v) for k, v in _zero_loop_diag(theta).items()}
    return loop_init.spectral_radius(
        p, config["n_c_U"], config["n_c_L"], config["n_c_inh"],
        config["n_t_exc"], config["n_t_inh"], config["tau_ctx"])


def pin_rho(theta, config, target):
    """Rescale the 17 loop blocks so rho_lin == target (restoring force vs ES drift).

    Only the loop blocks are touched (one global factor); the cue/readout/bias params
    and all *structure* within the loop are preserved. Self-diagonals are zeroed first
    so the pinned rho matches the no-autapse forward pass exactly.
    """
    p = {k: np.asarray(v) for k, v in _zero_loop_diag(theta).items()}
    p, _, _ = loop_init.normalize_loop(
        p, config["n_c_U"], config["n_c_L"], config["n_c_inh"],
        config["n_t_exc"], config["n_t_inh"], config["tau_ctx"], target)
    return {k: jnp.asarray(v) for k, v in p.items()}


def es_gradient(theta, sigma, half_eps, utilities, pop):
    """OpenAI-ES gradient estimate: g = 1/(pop*sigma) * sum_i u_i * eps_i (mirrored)."""
    u = jnp.asarray(np.concatenate([utilities[:pop // 2], utilities[pop // 2:]]))
    g = {}
    for key, eps in half_eps.items():                 # eps: (pop/2, *leaf)
        full = jnp.concatenate([eps, -eps], axis=0)   # (pop, *leaf)
        w = u.reshape((pop,) + (1,) * (full.ndim - 1))
        g[key] = jnp.sum(w * full, axis=0) / (pop * sigma)
    return g


def validation_reward(theta, config, cfg, n_val, seed):
    """Low-noise reward on a FIXED validation batch (same trials every call).

    Used to pick the best checkpoint -- the per-generation ES reward is too noisy
    (new small trial batch each gen) to select on directly.
    """
    inputs, opto, net_keys, is_stim, t_stim = make_trials(
        jr.PRNGKey(seed), cfg, n_val, n_val)
    pop_params = {k: v[None] for k, v in theta.items()}
    X = rollout(pop_params, config, inputs, opto, net_keys)
    pw = pci_windows(X, t_stim, is_stim, cfg)
    reward, comps = rewards_from_pci(pw, cfg)
    return float(reward[0]), {k: float(v[0]) for k, v in comps.items()}


def train(args):
    import optax

    rng = jr.PRNGKey(args.seed)
    rng, k_init = jr.split(rng)
    # optional pool-size override (patch central config before init so the loop is
    # built at the requested thalamus sizes, e.g. to match noSCnoSTN t_exc=10/t_inh=5)
    if args.n_t_exc is not None or args.n_t_inh is not None:
        import config_script as _cs
        if args.n_t_exc is not None:
            _cs.CORTICOTHALAMIC_RNN_CONFIG["n_t_exc"] = args.n_t_exc
        if args.n_t_inh is not None:
            _cs.CORTICOTHALAMIC_RNN_CONFIG["n_t_inh"] = args.n_t_inh
        print(f"[pool override] n_t_exc={_cs.CORTICOTHALAMIC_RNN_CONFIG['n_t_exc']}, "
              f"n_t_inh={_cs.CORTICOTHALAMIC_RNN_CONFIG['n_t_inh']}")
    params, config = ctrnn.init_params(k_init, n_input=1)
    if args.load:
        with open(args.load, "rb") as f:
            d = pkl.load(f)
        params, config = d["params"], d["config"]
    config = dict(config)
    config["noise_std"] = args.noise_std

    N = config["x_ctx0"].shape[0] + config["x_t0"].shape[0]
    nU = config["n_c_U"]
    cfg = dict(
        T=args.T, N=N, n_input=1, W=args.window,
        t_lo=args.t_lo, t_hi=args.t_hi,
        stim_units=list(range(nU)), stim_amp=args.stim_amp, stim_dur=args.stim_dur,
        thr_k=args.thr_k, sd_floor=args.sd_floor,
        lam_spont=args.lam_spont,
    )

    theta = {k: jnp.asarray(v) for k, v in params.items()}
    opt = optax.adam(args.lr)
    opt_state = opt.init(theta)
    pop = args.pop - (args.pop % 2)
    half = pop // 2

    history = {"gen": [], "reward": [], "post_pci": [], "spont": [], "calm": [], "rho": [],
               "val_reward": []}
    best = {"reward": -np.inf, "theta": None, "gen": -1}
    print(f"[pci_train] N={N} units, pop={pop}, sigma={args.sigma}, lr={args.lr}, "
          f"{args.stim_trials} stim + {args.catch_trials} catch trials/eval, "
          f"pin_rho={args.pin_rho}")
    t0 = time.time()
    for gen in range(args.generations):
        rng, k_eps, k_tr = jr.split(rng, 3)
        # antithetic Gaussian perturbations, one block per parameter leaf
        eps_keys = jr.split(k_eps, len(theta))
        half_eps = {k: jr.normal(eps_keys[i], (half,) + theta[k].shape)
                    for i, k in enumerate(theta)}
        pop_params = {k: theta[k][None] + args.sigma *
                      jnp.concatenate([half_eps[k], -half_eps[k]], axis=0)
                      for k in theta}

        inputs, opto, net_keys, is_stim, t_stim = make_trials(
            k_tr, cfg, args.stim_trials, args.catch_trials)
        X = rollout(pop_params, config, inputs, opto, net_keys)  # (pop,trials,T,N)
        pw = pci_windows(X, t_stim, is_stim, cfg)
        reward, comps = rewards_from_pci(pw, cfg)

        util = _rank_utilities(reward)
        g = es_gradient(theta, args.sigma, half_eps, util, pop)
        updates, opt_state = opt.update({k: -g[k] for k in g}, opt_state, theta)  # ascend
        theta = optax.apply_updates(theta, updates)
        if args.pin_rho is not None:
            theta = pin_rho(theta, config, args.pin_rho)  # restoring force vs rho drift

        if gen % args.log_interval == 0 or gen == args.generations - 1:
            rho = loop_rho(theta, config)
            val_r, val_c = validation_reward(theta, config, cfg, args.val_trials, 12345)
            if val_r > best["reward"]:
                best = {"reward": val_r, "theta": {k: np.asarray(v) for k, v in theta.items()},
                        "gen": gen}
            history["gen"].append(gen)
            history["reward"].append(float(reward.mean()))
            history["val_reward"].append(val_r)
            history["post_pci"].append(val_c["post_pci"])
            history["spont"].append(val_c["spont"])
            history["calm"].append(val_c["calm"])
            history["rho"].append(rho)
            print(f"  gen {gen:4d}  val_reward {val_r:7.2f}  "
                  f"post_PCI {val_c['post_pci']:7.2f}  "
                  f"spont {val_c['spont']:6.2f}  "
                  f"calm {val_c['calm']:.4f}  rho {rho:.3f}  "
                  f"{'*BEST' if best['gen']==gen else '':5s}[{time.time()-t0:.0f}s]")

    # save the BEST validation checkpoint (per-gen ES reward is too noisy to select on)
    out_params = best["theta"] if best["theta"] is not None else \
        {k: np.asarray(v) for k, v in theta.items()}
    print(f"[pci_train] best val_reward {best['reward']:.2f} at gen {best['gen']}")
    save = {"params": out_params, "config": config, "cfg": cfg, "history": history,
            "best_gen": best["gen"]}
    with open(args.out, "wb") as f:
        pkl.dump(save, f)
    print(f"[pci_train] saved -> {args.out}")
    return save


# --------------------------------------------------------------------------------------
# Evaluation + diagnostic plot
# --------------------------------------------------------------------------------------
def evaluate(args):
    with open(args.load, "rb") as f:
        d = pkl.load(f)
    params = {k: jnp.asarray(v) for k, v in d["params"].items()}
    config = dict(d["config"]); config["noise_std"] = args.noise_std
    cfg = d.get("cfg")
    if cfg is None:
        N = config["x_ctx0"].shape[0] + config["x_t0"].shape[0]
        nU = config["n_c_U"]
        cfg = dict(T=args.T, N=N, n_input=1, W=args.window, t_lo=args.t_lo, t_hi=args.t_hi,
                   stim_units=list(range(nU)), stim_amp=args.stim_amp, stim_dur=args.stim_dur,
                   thr_k=args.thr_k, sd_floor=args.sd_floor,
                   lam_spont=args.lam_spont)

    rng = jr.PRNGKey(args.seed)
    n_stim = n_catch = args.eval_trials
    inputs, opto, net_keys, is_stim, t_stim = make_trials(rng, cfg, n_stim, n_catch)
    # single "population" of one (the trained params)
    pop_params = {k: v[None] for k, v in params.items()}
    X = rollout(pop_params, config, inputs, opto, net_keys)  # (1, trials, T, N)
    pw = pci_windows(X, t_stim, is_stim, cfg)
    reward, comps = rewards_from_pci(pw, cfg)

    is_stim = np.asarray(is_stim)
    dcomp = (pw["comp_post"] - pw["comp_pre"])[0]      # (trials,)
    print("=" * 68)
    print(f"PCI evaluation ({n_stim} stim + {n_catch} catch trials, T={cfg['T']}, "
          f"window={cfg['W']}, jitter t_stim in [{cfg['t_lo']},{cfg['t_hi']}))")
    print("-" * 68)
    print(f"  STIM  : comp_pre {pw['comp_pre'][0][is_stim].mean():6.1f}  "
          f"comp_post {pw['comp_post'][0][is_stim].mean():6.1f}  "
          f"delta_PCI {dcomp[is_stim].mean():+6.1f} ± {dcomp[is_stim].std():.1f}")
    print(f"  CATCH : comp_pre {pw['comp_pre'][0][~is_stim].mean():6.1f}  "
          f"comp_post {pw['comp_post'][0][~is_stim].mean():6.1f}  "
          f"delta_PCI {dcomp[~is_stim].mean():+6.1f} ± {dcomp[~is_stim].std():.1f}")
    print(f"  baseline std (calm): {pw['base_std'][0].mean():.4f}")
    print(f"  evoked-minus-spontaneous delta_PCI: "
          f"{dcomp[is_stim].mean() - dcomp[~is_stim].mean():+6.1f}")
    print(f"  scalar reward: {float(reward[0]):.2f}")
    print("=" * 68)

    _plot_eval(X[0], pw, t_stim, is_stim, cfg, d.get("history"), args.out_plot)


def _binarize_trial(Xtr, t0, cfg):
    W, k, floor = cfg["W"], cfg["thr_k"], cfg["sd_floor"]
    pre = Xtr[t0 - W:t0]; post = Xtr[t0:t0 + W]
    mu = pre.mean(0, keepdims=True); sd = pre.std(0, keepdims=True)
    thr = k * np.maximum(sd, floor)
    full = Xtr[t0 - W:t0 + W]
    return (np.abs(full - mu) > thr)  # (2W, N)


def _plot_eval(X1, pw, t_stim, is_stim, cfg, history, out_plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X1 = np.asarray(X1); t_stim = np.asarray(t_stim)
    stim_i = int(np.where(is_stim)[0][0]); catch_i = int(np.where(~is_stim)[0][0])
    W = cfg["W"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    for col, (ti, label) in enumerate([(stim_i, "STIM"), (catch_i, "CATCH")]):
        t0 = int(t_stim[ti])
        # continuous activity raster
        act = X1[ti, t0 - W:t0 + W].T  # (N, 2W)
        im = ax[0, col].imshow(act, aspect="auto", cmap="viridis",
                               extent=[-W, W, act.shape[0], 0], vmin=0, vmax=1)
        ax[0, col].axvline(0, color="r", lw=1.5)
        ax[0, col].set_title(f"{label} trial — activity (units × time)")
        ax[0, col].set_xlabel("time from stim (steps)"); ax[0, col].set_ylabel("unit")
        plt.colorbar(im, ax=ax[0, col], fraction=0.04)
        # binarized significance raster
        b = _binarize_trial(X1[ti], t0, cfg).T  # (N, 2W)
        ax[1, col].imshow(b, aspect="auto", cmap="Greys",
                          extent=[-W, W, b.shape[0], 0])
        ax[1, col].axvline(0, color="r", lw=1.5)
        cp = _zip_len(b[:, :W]); cpo = _zip_len(b[:, W:])
        ax[1, col].set_title(f"{label} — significant (pre zip={cp}, post zip={cpo})")
        ax[1, col].set_xlabel("time from stim (steps)"); ax[1, col].set_ylabel("unit")

    # delta_PCI distributions
    dcomp = (pw["comp_post"] - pw["comp_pre"])[0]
    ax[0, 2].hist(dcomp[is_stim], bins=15, alpha=0.7, label="stim", color="C3")
    ax[0, 2].hist(dcomp[~is_stim], bins=15, alpha=0.7, label="catch", color="C0")
    ax[0, 2].axvline(0, color="k", lw=0.8)
    ax[0, 2].set_title("delta_PCI  (post − pre zip bytes)")
    ax[0, 2].set_xlabel("delta_PCI"); ax[0, 2].legend()

    # training curve
    if history and history["gen"]:
        h = history
        ax[1, 2].plot(h["gen"], h["post_pci"], label="post PCI (stim)", color="C3")
        ax[1, 2].plot(h["gen"], h["spont"], label="spontaneous ΔPCI", color="C0")
        ax[1, 2].set_xlabel("generation"); ax[1, 2].set_ylabel("PCI (zip bytes)")
        ax[1, 2].set_title("training (ES)"); ax[1, 2].legend(loc="upper left")
        axr = ax[1, 2].twinx()
        axr.plot(h["gen"], h["calm"], "--", color="C2", label="baseline std")
        axr.plot(h["gen"], h["rho"], ":", color="C4", label="rho_lin")
        axr.set_ylabel("baseline std / rho"); axr.legend(loc="lower right")
    else:
        ax[1, 2].axis("off")

    fig.suptitle("Corticothalamic perturbational complexity (PCI) — zip-based", y=1.0)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / out_plot
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[pci_train] plot -> {out}")


# --------------------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true", help="evaluate + plot instead of train")
    p.add_argument("--load", type=str, default=None, help="params pkl to load")
    p.add_argument("--out", type=str, default=str(HERE / "params_pci.pkl"))
    p.add_argument("--out-plot", type=str, default="pci_eval.png")
    # pool sizes (override the central corticothalamic config, e.g. to match noSCnoSTN's
    # thalamus t_exc=10/t_inh=5 so the full loop ports over)
    p.add_argument("--n-t-exc", type=int, default=None)
    p.add_argument("--n-t-inh", type=int, default=None)
    # trial / task geometry
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--window", type=int, default=150, help="pre/post window length")
    p.add_argument("--t-lo", type=int, default=225, help="earliest stim time (jitter)")
    p.add_argument("--t-hi", type=int, default=326, help="latest stim time (exclusive)")
    p.add_argument("--stim-amp", type=float, default=5.0)
    p.add_argument("--stim-dur", type=int, default=3)
    p.add_argument("--noise-std", type=float, default=0.01)
    # PCI binarization
    p.add_argument("--thr-k", type=float, default=2.5, help="significance = k*baseline std")
    p.add_argument("--sd-floor", type=float, default=1e-3)
    # objective weights
    p.add_argument("--lam-spont", type=float, default=1.0,
                   help="weight on the catch-trial spontaneous penalty (the only "
                        "penalty; ties evoked complexity to the stimulus)")
    # ES
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--pin-rho", type=float, default=None,
                   help="renormalize the loop to this rho_lin every generation "
                        "(restoring force against the ES rho drift; off if unset)")
    p.add_argument("--stim-trials", type=int, default=16)
    p.add_argument("--catch-trials", type=int, default=16)
    p.add_argument("--val-trials", type=int, default=40,
                   help="fixed validation batch size (per stim/catch) for best-checkpoint")
    p.add_argument("--eval-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=10)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.eval:
        assert args.load, "--eval needs --load <params_pci.pkl>"
        evaluate(args)
    else:
        train(args)
