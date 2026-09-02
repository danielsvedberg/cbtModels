"""TEST: pka_stability -- is PKA in cbt_loop_noSCnoSTN inherently unstable?

QUESTION
--------
Is the PKA state (pka_d1 / pka_d2) an unstable / runaway variable in this model?

METHOD
------
1. ISOLATED MAP (analytic). The update is a mass-action-bounded leaky integrator
       pka' = (1 - 1/tau_f)*pka + (1/tau_r)*prod*(1 - pka/pka_max)
   With prod held fixed this is AFFINE in pka with slope  s = (1 - 1/tau_f) - prod/tau_r,
   so |s| < 1 for every 0 <= prod < 2*tau_r*(1 - 1/(2*tau_f)) ~ 199.9 and the fixed point is
   p* = 9*prod/(1 + 9*prod)  (tau_f=900, tau_r=100). The integrator ALONE cannot run away.

2. CLOSED LOOP (measured). PKA is not isolated: pka -> bg_nln excitability -> D1/D2 rates ->
   mean_spn / mean_snc -> x_ado / x_da -> prod_d1 / prod_d2 -> pka. Everything except PKA is
   fast (network taus 5-10, tau_da=20, tau_ado=200 vs pka's 900), so the loop is reduced on the
   PKA slow manifold: pin (pka_d1, pka_d2) via config pin_pka_d1/pin_pka_d2, run to steady
   state, read (da*, ado*), and rebuild the one-step PKA map. That gives an exact 2-D map
   F(p1,p2) whose fixed points and Jacobian eigenvalues ARE the stability of PKA in the loop.
   (Verified T-independent: identical roots at T=1500 and T=5000.)

3. Cross-checked against the un-reduced full system (15k-step free runs) and against the
   trained checkpoints in the family dir.

FINDING
-------
NOT inherently unstable -- but marginally stable, and BISTABLE along the D2 axis.

  * pka_d1: monostable. p1* = 0.2118, eigenvalue 0.99816 -> stable, relaxation ~543 steps.
  * pka_d2 (at p1 = p1*): THREE fixed points
        p2* = 0.0581  eig 0.99885  stable   (D2 dead,      ado 0.052)
        p2* = 0.2671  eig 1.00450  UNSTABLE (separatrix,   ado 0.163)   <-- the only |eig|>1
        p2* = 0.3954  eig 0.99841  stable   (D2 railed .94, ado 0.280)
    The unstable root is a genuine saddle: e-folding 1/0.0045 ~ 222 steps, i.e. a quarter of a
    1000-step trial, so it is dynamically real, not a numerical curiosity.
  * MECHANISM of the unstable root: the one positive-feedback branch in the model,
        pka_d2 -> bg_nln b -> D2 -> mean_spn -> x_ado -> prod_d2 = m_a2*ado - G*m_d2*da -> pka_d2.
    Slope decomposition at the separatrix: the leak+substrate part contributes (a - k*prod2) =
    0.99848 (< 1, stable), and the adenosine feedback term k*(1-p2)*d(prod2)/dp2 adds +0.0060,
    pushing it over 1. D2's sigmoid switch supplies the steepness, m_a2/g_ado_release the gain.
    pka_d1 has no such branch (adenosine ENTERS prod_d1 with a minus sign -> negative feedback),
    which is exactly why the D1 axis is monostable.
  * The bistability only exists for pka_d1 <~ 0.24; above that a saddle-node annihilates the low
    branch (see the phase plane) and pka_d2 rises to the railed state from anywhere. The model
    inits at pka_d10 = pka_d20 = 0.25 -- ON THE FOLD and ~0.017 below the separatrix -- so every
    free run, and both loadable trained checkpoints, drift into a RAILED corner:
        init weights       -> pka_d2 0.395, D2 0.936, D1 0.093
        params_pavlovian   -> pka_d2 0.501, D2 0.989, D1 0.035     (D2 corner)
        params_shaped      -> pka_d2 0.000, D2 0.000, D1 1.000     (D1 corner; training
                              collapsed g_ado_release 0.757 -> 0.047, killing prod_d2)
    Training does not repair the marginal stability; it just picks a corner.
  * WITHIN A TRIAL none of this looks like a runaway. All relaxation times (543 / 222 / 630
    steps) are comparable to t_total=1000, so PKA is quasi-frozen: over one trial pka_d2 moves
    0.25 -> 0.27 and D2 0.19 -> 0.43. The "PKA clock" ramp IS the leading edge of this slow
    drift toward the railed attractor -- stable in the Lyapunov sense, useless as an operating
    point, and impossible to hold mid-band without either lowering the ado->D2->ado gain or
    re-centering the init away from the fold.
"""
import sys
import pathlib
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FAMILY_DIR = HERE.parents[1]
_root = next(p for p in HERE.parents if (p / "config_script.py").exists())
for _p in (str(FAMILY_DIR), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cbt_rnn as cbtl
import self_timed_movement_task as stmt

A = list(cbtl.STATE_AREA_ORDER)
T_SETTLE = 1500       # >= 7 * tau_ado: enough for everything but PKA (which is pinned)
T_FREE = 15000        # free-run horizon (PKA needs ~10 * 900 to converge)
T_TRIAL = 1000        # config_script TASK_CONFIG["t_total"]


def _traces(params, config, T, overrides=None):
    c = dict(config); c["noise_std"] = 0.0
    if overrides:
        c.update(overrides)
    nd = params["J_d1"].shape[0] + params["J_d2"].shape[0]
    _, xs = cbtl.multiregion_rnn(params, c, jnp.zeros((T, 1)), jnp.zeros((T, nd)), jr.PRNGKey(0))
    def g(a):
        x = xs[A.index(a)]
        return np.asarray(jnp.mean(x, axis=-1) if x.ndim > 1 else x)
    return {a: g(a) for a in ("pkaD1", "pkaD2", "D1", "D2", "DA", "Adenosine")}


def make_map(params, config):
    """Quasi-static one-step PKA map F(p1,p2) with the rest of the loop at steady state."""
    af = 1.0 - 1.0 / config["tau_pka_fall"]
    kr = 1.0 / config["tau_pka_rise"]
    G = config["da_pka_gain"]
    pmax = config["pka_max"]
    E = lambda k: stmt.exc(jnp.asarray(params[k]))
    nd = params["J_d1"].shape[0] + params["J_d2"].shape[0]

    def one(p1, p2):
        c = dict(config); c["noise_std"] = 0.0
        c["pin_pka_d1"] = p1; c["pin_pka_d2"] = p2
        _, xs = cbtl.multiregion_rnn(params, c, jnp.zeros((T_SETTLE, 1)),
                                     jnp.zeros((T_SETTLE, nd)), jr.PRNGKey(0))
        da = jnp.mean(xs[A.index("DA")][-1]); ado = jnp.mean(xs[A.index("Adenosine")][-1])
        pr1 = jnp.maximum(G * E("m_d1") * da - E("m_a1") * ado, 0.0)
        pr2 = jnp.maximum(E("m_a2") * ado - G * E("m_d2") * da, 0.0)
        f1 = af * p1 + kr * pr1 * jnp.maximum(1.0 - p1 / pmax, 0.0)
        f2 = af * p2 + kr * pr2 * jnp.maximum(1.0 - p2 / pmax, 0.0)
        return jnp.stack([f1, f2, da, ado, pr1, pr2,
                          jnp.mean(xs[A.index("D1")][-1]), jnp.mean(xs[A.index("D2")][-1])])
    #  sweep p2 (p1 fixed) and sweep p1 (p2 fixed)
    return (jax.jit(jax.vmap(one, in_axes=(None, 0))),
            jax.jit(jax.vmap(one, in_axes=(0, None))))


def roots_and_slopes(grid, f_of_p):
    """Fixed points of a 1-D map given F(p) sampled on `grid`, with local slopes."""
    d = f_of_p - grid
    out = []
    for i in np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        x0, x1 = grid[i], grid[i + 1]
        root = x0 - d[i] * (x1 - x0) / (d[i + 1] - d[i])
        out.append((root, (f_of_p[i + 1] - f_of_p[i]) / (x1 - x0)))
    return out


def main():
    params, config = cbtl.init_params(jr.PRNGKey(0), n_input=1)
    fmap, fmap_p1 = make_map(params, config)
    af = 1.0 - 1.0 / config["tau_pka_fall"]; kr = 1.0 / config["tau_pka_rise"]

    # --- 1-D scans through each axis -----------------------------------------------------
    fine = np.linspace(0.01, 0.99, 197)
    p1_fp = roots_and_slopes(fine, np.asarray(fmap_p1(jnp.asarray(fine), 0.3954))[:, 0])
    # iterate once so the p1 pin used for the p2 scan is self-consistent
    p1_star = p1_fp[0][0]
    scan2 = np.asarray(fmap(float(p1_star), jnp.asarray(fine)))
    p2_fp = roots_and_slopes(fine, scan2[:, 1])

    print(f"pka_d1 axis (p2 pinned 0.3954): " +
          "  ".join(f"p1*={r:.4f} eig={s:.5f} [{'UNSTABLE' if s > 1 else 'stable'}]" for r, s in p1_fp))
    print(f"pka_d2 axis (p1 pinned {p1_star:.4f}): " +
          "  ".join(f"p2*={r:.4f} eig={s:.5f} [{'UNSTABLE' if s > 1 else 'stable'}]" for r, s in p2_fp))

    # slope decomposition at the unstable root: leak/substrate part vs adenosine feedback
    unst = [(r, s) for r, s in p2_fp if s > 1.0]
    if unst:
        r, s = unst[0]
        j = int(np.argmin(np.abs(fine - r)))
        leak = af - kr * scan2[j, 5]
        print(f"  separatrix p2*={r:.4f}: leak+substrate part {leak:.5f} (<1) "
              f"+ adenosine feedback {s - leak:+.5f} = {s:.5f}; "
              f"d(ado)/dp2 = {np.gradient(scan2[:, 3], fine)[j]:.3f}")

    # --- 2-D phase plane ------------------------------------------------------------------
    gp = np.linspace(0.02, 0.60, 30)
    grid = np.stack([np.asarray(fmap(float(a), jnp.asarray(gp))) for a in gp])  # [p1, p2, 8]
    dP1 = grid[:, :, 0] - gp[:, None]
    dP2 = grid[:, :, 1] - gp[None, :]

    # --- free runs -------------------------------------------------------------------------
    free = {"init weights": _traces(params, config, T_FREE)}
    for name in ("params_pavlovian.pkl", "params_shaped.pkl"):
        f = FAMILY_DIR / name
        if not f.exists():
            continue
        try:
            b = pickle.load(open(f, "rb"))
            pr = b["params"] if isinstance(b, dict) and "params" in b else b
            cf = b["config"] if isinstance(b, dict) and "config" in b else config
            free[name.replace("params_", "").replace(".pkl", "")] = _traces(pr, cf, T_FREE)
        except Exception as e:                                   # pool-size-incompatible pkls
            print(f"  (skipped {name}: {type(e).__name__})")

    for k, tr in free.items():
        print(f"free run [{k:10s}] pka_d2 {tr['pkaD2'][0]:.3f} -> {tr['pkaD2'][T_TRIAL]:.3f} (1 trial) "
              f"-> {tr['pkaD2'][-1]:.3f} | pka_d1 -> {tr['pkaD1'][-1]:.3f} | "
              f"D1 {tr['D1'][-1]:.3f} D2 {tr['D2'][-1]:.3f}")

    # ================================ figure =================================================
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) 1-D map on the pka_d2 axis
    a0 = ax[0, 0]
    a0.axhline(0, color="k", lw=0.8)
    a0.plot(fine, scan2[:, 1] - fine, color="crimson", lw=2, label=r"$F_2(p_2)-p_2$")
    for r, s in p2_fp:
        unstable = s > 1.0
        a0.plot([r], [0], "o", ms=9, mfc="white" if unstable else "crimson",
                mec="crimson", mew=2, zorder=5)
        a0.annotate(f"{r:.3f}\n{'UNSTABLE' if unstable else 'stable'}\n$\\lambda$={s:.5f}",
                    (r, 0), textcoords="offset points",
                    xytext=(34, 30) if unstable else (0, -48),
                    ha="center", fontsize=8, fontweight="bold" if unstable else "normal",
                    color="darkred" if unstable else "black",
                    arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8) if unstable else None)
    a0.set_xlim(0.0, 0.55)
    a0.set_ylim(-4.5e-4, 4.5e-4)
    a0.axvline(0.25, color="steelblue", ls=":", lw=1.6)
    a0.text(0.248, -4.3e-4, "init 0.25 ", color="steelblue", fontsize=8, ha="right")
    a0.annotate("", xy=(0.20, 1.5e-4), xytext=(0.245, 1.5e-4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.4))
    a0.annotate("", xy=(0.33, 1.5e-4), xytext=(0.285, 1.5e-4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.4))
    a0.set_xlabel(r"$p_2$  (pka_d2)"); a0.set_ylabel(r"one-step drift $F_2(p_2)-p_2$")
    a0.set_title(f"(a) PKA-D2 map on the slow manifold ($p_1$ pinned {p1_star:.3f})\n"
                 "three fixed points -> bistable, middle root has $\\lambda>1$", fontsize=10)
    a0.grid(alpha=0.3); a0.legend(fontsize=8, loc="lower left")

    # (b) phase plane
    a1 = ax[0, 1]
    a1.contourf(gp, gp, np.sign(dP2.T), levels=[-1.5, 0, 1.5], colors=["#dfe9f5", "#f9e0e0"], alpha=0.9)
    a1.contour(gp, gp, dP1.T, levels=[0], colors="tab:blue", linewidths=2)
    a1.contour(gp, gp, dP2.T, levels=[0], colors="crimson", linewidths=2)
    tr = free["init weights"]
    a1.plot(tr["pkaD1"], tr["pkaD2"], color="k", lw=1.4, alpha=0.8)
    a1.plot(tr["pkaD1"][:T_TRIAL], tr["pkaD2"][:T_TRIAL], color="k", lw=3.5, label="1 trial (1000 steps)")
    a1.plot([tr["pkaD1"][0]], [tr["pkaD2"][0]], "o", color="k", ms=7, label="init (0.25, 0.25)")
    a1.plot([tr["pkaD1"][-1]], [tr["pkaD2"][-1]], "*", color="k", ms=16, label="attractor")
    a1.plot([], [], color="tab:blue", lw=2, label=r"$p_1$ nullcline")
    a1.plot([], [], color="crimson", lw=2, label=r"$p_2$ nullcline")
    a1.set_xlabel(r"$p_1$  (pka_d1)"); a1.set_ylabel(r"$p_2$  (pka_d2)")
    a1.set_xlim(gp[0], gp[-1]); a1.set_ylim(gp[0], gp[-1])
    a1.set_title("(b) PKA phase plane: bistable only for $p_1\\lesssim0.24$;\n"
                 "the init sits on the saddle-node fold", fontsize=10)
    a1.legend(fontsize=7.5, loc="lower right"); a1.grid(alpha=0.25)

    # (c) timescales
    a2 = ax[1, 0]
    labels, taus, cols = [], [], []
    for r, s in p1_fp:
        labels.append(f"$p_1^*$={r:.3f}\n$\\lambda$={s:.5f}"); taus.append(1.0 / abs(1 - s)); cols.append("tab:blue")
    for r, s in p2_fp:
        labels.append(f"$p_2^*$={r:.3f}\n$\\lambda$={s:.5f}"); taus.append(1.0 / abs(1 - s))
        cols.append("darkred" if s > 1 else "crimson")
    a2.bar(range(len(taus)), taus, color=cols, alpha=0.85)
    a2.axhline(T_TRIAL, color="k", ls="--", lw=1.5)
    a2.text(len(taus) - 0.4, T_TRIAL, " trial (1000)", va="bottom", ha="right", fontsize=8)
    a2.axhline(300, color="gray", ls=":", lw=1.5)
    a2.text(len(taus) - 0.4, 300, " delay (300)", va="bottom", ha="right", fontsize=8, color="gray")
    a2.set_xticks(range(len(taus))); a2.set_xticklabels(labels, fontsize=7.5)
    a2.set_ylabel(r"$1/|1-\lambda|$   (steps to relax / diverge)")
    a2.set_title("(c) every PKA eigen-timescale is trial-length or longer\n"
                 "-> PKA never equilibrates inside a trial", fontsize=10)
    a2.grid(alpha=0.3, axis="y")

    # (d) free runs
    a3 = ax[1, 1]
    styles = {"init weights": "-", "pavlovian": "--", "shaped": ":"}
    for k, tr in free.items():
        st = styles.get(k, "-")
        a3.plot(tr["pkaD2"], st, color="crimson", lw=1.8, label=f"pka_d2 [{k}]")
        a3.plot(tr["pkaD1"], st, color="tab:blue", lw=1.8, label=f"pka_d1 [{k}]")
    a3.axvspan(0, T_TRIAL, color="green", alpha=0.10)
    a3.text(T_TRIAL, 0.97, " one trial", fontsize=8, color="green", va="top")
    a3.set_xscale("symlog", linthresh=1000)
    a3.set_xlabel("step"); a3.set_ylabel("PKA")
    a3.set_ylim(-0.02, 1.02)
    a3.set_title("(d) un-reduced free runs: bounded (no runaway) but every one\n"
                 "drifts into a railed corner, over ~10x the trial length", fontsize=10)
    a3.legend(fontsize=7, ncol=2); a3.grid(alpha=0.3)

    fig.suptitle("noSCnoSTN PKA stability: contractive in isolation, marginally stable in the loop, "
                 "bistable along D2 (one $\\lambda=1.0045$ saddle)", y=0.995, fontsize=11)
    fig.tight_layout()
    out = HERE / "pka_stability.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
