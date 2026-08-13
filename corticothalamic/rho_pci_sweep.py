"""Sweep the pinned loop rho and map achievable perturbational complexity vs rho.

For each target rho_lin we train a FRESH model with --pin-rho held at that value (so the
ES rho-drift is neutralized and each point is a clean "best PCI achievable at this fixed
operating point"), then evaluate the best checkpoint on a fixed batch. Plots post-PCI,
evoked-minus-spontaneous ΔPCI, and spontaneous vs rho.

  python rho_pci_sweep.py                 # default sweep around 1.12
"""
import argparse
import types
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import pci_train as P

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
SCRATCH = Path("/tmp/claude-1000/-home-dsvedberg-Documents-CodeVault-cbtModels/"
               "02953a08-a57a-4eda-b9df-a53339ae0f03/scratchpad")


def eval_best(save, seed=999, n=60):
    """Clean PCI metrics for a trained checkpoint on a fixed held-out batch."""
    params = {k: jnp.asarray(v) for k, v in save["params"].items()}
    config, cfg = save["config"], save["cfg"]
    inputs, opto, keys, is_stim, t_stim = P.make_trials(jr.PRNGKey(seed), cfg, n, n)
    X = P.rollout({k: v[None] for k, v in params.items()}, config, inputs, opto, keys)
    pw = P.pci_windows(X, t_stim, is_stim, cfg)
    reward, comps = P.rewards_from_pci(pw, cfg)
    is_stim = np.asarray(is_stim)
    dcomp = (pw["comp_post"] - pw["comp_pre"])[0]
    return dict(
        post_pci=float(comps["post_pci"][0]),
        spont=float(comps["spont"][0]),
        evoked=float(dcomp[is_stim].mean()),
        catch=float(dcomp[~is_stim].mean()),
        evoked_minus_spont=float(dcomp[is_stim].mean() - dcomp[~is_stim].mean()),
        reward=float(reward[0]),
        rho_measured=P.loop_rho(params, config),
    )


def run(args):
    base = P.build_argparser().parse_args([])   # defaults
    rows = []
    for rho in args.rhos:
        a = types.SimpleNamespace(**vars(base))
        a.pin_rho = rho
        a.generations = args.generations
        a.pop = args.pop
        a.seed = args.seed
        a.log_interval = args.generations  # only first+last line
        a.load = None
        a.out = str(SCRATCH / f"sweep_rho_{rho:.2f}.pkl")
        print(f"\n===== training pinned rho = {rho:.2f} =====")
        save = P.train(a)
        m = eval_best(save)
        best_val = max(save["history"]["val_reward"]) if save["history"]["val_reward"] else float("nan")
        m["best_val"] = best_val
        m["rho"] = rho
        rows.append(m)
        print(f"  rho {rho:.2f}: post_PCI {m['post_pci']:.0f}  "
              f"evoked-spont {m['evoked_minus_spont']:.0f}  spont {m['spont']:.0f}  "
              f"best_val {best_val:.0f}  (rho_meas {m['rho_measured']:.3f})")

    print("\n" + "=" * 72)
    print(f"{'rho':>6} {'post_PCI':>9} {'evoked-spont':>13} {'spont':>7} {'best_val':>9}")
    for m in rows:
        print(f"{m['rho']:>6.2f} {m['post_pci']:>9.0f} {m['evoked_minus_spont']:>13.0f} "
              f"{m['spont']:>7.0f} {m['best_val']:>9.0f}")
    _plot(rows, args.out_plot)
    return rows


def _plot(rows, out_plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = [m["rho"] for m in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(r, [m["post_pci"] for m in rows], "o-", label="post-PCI (stim)", color="C3")
    ax[0].plot(r, [m["evoked_minus_spont"] for m in rows], "s-",
               label="evoked − spontaneous ΔPCI", color="C1")
    ax[0].plot(r, [m["spont"] for m in rows], "^--", label="spontaneous", color="C0")
    ax[0].set_xlabel("pinned rho_lin"); ax[0].set_ylabel("PCI (zip bytes)")
    ax[0].set_title("Achievable complexity vs pinned rho"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].axvline(1.0, color="grey", ls=":", lw=1)
    ax[1].plot(r, [m["best_val"] for m in rows], "o-", color="C2")
    ax[1].set_xlabel("pinned rho_lin"); ax[1].set_ylabel("best val reward")
    ax[1].set_title("Best training reward vs pinned rho"); ax[1].grid(alpha=0.3)
    ax[1].axvline(1.0, color="grey", ls=":", lw=1)
    fig.suptitle("PCI vs pinned loop spectral radius", y=1.0)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / out_plot
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[rho_pci_sweep] plot -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rhos", type=float, nargs="+",
                   default=[1.00, 1.04, 1.08, 1.12, 1.16, 1.20, 1.24])
    p.add_argument("--generations", type=int, default=130)
    p.add_argument("--pop", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-plot", type=str, default="rho_pci_sweep.png")
    run(p.parse_args())
