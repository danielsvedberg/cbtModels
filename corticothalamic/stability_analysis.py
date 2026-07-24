"""Stability analysis of the cortico-thalamic loop (and the striatal gate),
with the ACTUAL nonlinearity nln = max(0, tanh) from self_timed_movement_task.

Consolidates the former cbt_loop/tests/eigen_ramp_probe.py into the
corticothalamic/ folder and fixes its central limitation: the old probe took the
spectral radius of the *linear* update map J = (1-1/tau)I + (1/tau)W, i.e. it
assumed a nonlinearity gain of 1 everywhere (a small-signal linearization about
x = 0). That tells you whether x = 0 is marginally (un)stable but says NOTHING
about where the network actually settles, because nln is contractive (tanh'(z)<1
for z>0) and rectifying (kills negative currents). The operating point can be a
high, SATURATED fixed point whose local gain -> 0 -- a strongly stable attractor
that the linear rho completely misses.

This script therefore works with the nonlinear map directly:

  pre_i = (1 - 1/tau) x_i + (1/tau) (W x + b)_i
  x_i   = nln(pre_i)                                     nln(z) = max(0, tanh z)

  fixed point   x*      : iterate the map to convergence
  local gain    g_i     : nln'(pre_i*) = (1 - tanh(pre_i*)^2) if pre_i* > 0 else 0
  Jacobian      J*      : diag(g) @ [ (1 - 1/tau) I + (1/tau) W ]
  stability     rho*    : max |eig(J*)|      (governs the ACTUAL operating point)

Sections:
  1. Cortico-thalamic loop: linear rho (about 0) vs nln operating-point rho*,
     plus the mean fixed-point activity -- shows the "faithful" loop self-limits
     by SATURATING (rho* < 1 but at a high, signal-dead fixed point), while the
     balanced loop settles low with rho* ~ near-critical and gain ~ 1.
  2. Cue-evoked persistence of the loop under nln (unchanged convention).
  3. Size sweep using the nln operating-point rho*.
  4. Striatal gate: why the per-term-nln wiring pins the striatum to saturation
     (each inhibitory projection is rectified to zero before it can subtract).

Run:  python corticothalamic/stability_analysis.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root on path so we use the SAME nonlinearity the models use.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import self_timed_movement_task as stmt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

TAU = 10.0
G = 1.0  # base gain (matches cbt_loop g_bg = 1.0)


# --------------------------------------------------------------------------- #
# The real nonlinearity and its derivative (numpy mirrors of stmt.nln).
# NOTE: stmt.nln was changed from max(0, tanh) to a SIGMOID: nln(x) =
# sigmoid(4(x-0.5)). It no longer rectifies and has a nonzero floor nln(0)=0.119,
# so x=0 is not a fixed point and negative currents are squashed toward ~0 (not
# to exactly 0). We mirror the live definition and verify it at import.
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nln(z):
    """sigmoid(4*(z-0.5)) -- mirrors the current stmt.nln (verified at import)."""
    return _sigmoid(4.0 * (z - 0.5))


def nln_prime(z):
    """d/dz sigmoid(4(z-0.5)) = 4 s (1-s), s = nln(z)."""
    s = nln(z)
    return 4.0 * s * (1.0 - s)


def bg_nln(x, b):
    """sigmoid(c(x-d)) with c=3/(1-b), d=(1/6)((1-b)/b) -- mirrors stmt.bg_nln."""
    c = 3.0 / (1.0 - b)
    d = (1.0 / 6.0) * ((1.0 - b) / b)
    return _sigmoid(c * (x - d))


# Sanity: the numpy mirrors must equal the jax stmt versions the models use.
_z = np.linspace(-3, 3, 41)
assert np.allclose(nln(_z), np.asarray(stmt.nln(_z)), atol=1e-5), "nln mismatch vs stmt.nln"
assert np.allclose(bg_nln(_z, 0.4), np.asarray(stmt.bg_nln(_z, 0.4)), atol=1e-5), "bg_nln mismatch"


# --------------------------------------------------------------------------- #
# Thalamocortical connectivity (state order cU, cL, cI, tE, tI), from the
# Dale's-law init_params of cbt_rnn. (post, pre, sign, faithful-scaling-kind).
# --------------------------------------------------------------------------- #
EDGES = [
    ("cU", "cU", +1, "sqrt"), ("cU", "cL", +1, "sqrt"), ("cU", "cI", -1, "lin"), ("cU", "tE", +1, "lin"),
    ("cL", "cL", +1, "sqrt"), ("cL", "cU", +1, "sqrt"), ("cL", "cI", -1, "lin"),
    ("cI", "cU", +1, "lin"),  ("cI", "cL", +1, "lin"),  ("cI", "cI", -1, "lin"), ("cI", "tE", +1, "lin"),
    ("tE", "tE", +1, "sqrt"), ("tE", "tI", -1, "lin"),  ("tE", "cU", +1, "lin"),
    ("tI", "tE", +1, "lin"),  ("tI", "tI", -1, "lin"),  ("tI", "cU", +1, "lin"),
]


def build_tc(n, scaling="faithful", balance=False, g=G, seed=0):
    """Full thalamocortical recurrent matrix W (state order cU,cL,cI,tE,tI).

    scaling: "faithful" = exact init scalings (E->E ~ g/sqrt(n); I / cross-area
             ~ g/n); "matched" = g/sqrt(n) on every block.
    balance: per-row E/I balance (each cell's local inhibition rescaled to cancel
             its total excitation -> zero mean recurrent drive)."""
    nI = max(1, n // 2)
    sizes = [("cU", n), ("cL", n), ("cI", n), ("tE", n), ("tI", nI)]
    idx, off = {}, 0
    for name, sz in sizes:
        idx[name] = slice(off, off + sz); off += sz
    N = off
    W = np.zeros((N, N))
    rng = np.random.default_rng(seed)
    size_of = {s[0]: s[1] for s in sizes}
    for post, pre, sign, kind in EDGES:
        pre_n = size_of[pre]
        scale = g / np.sqrt(pre_n) if (scaling == "matched" or kind == "sqrt") else g / pre_n
        r, c = idx[post], idx[pre]
        block = np.abs(rng.standard_normal((r.stop - r.start, c.stop - c.start))) * scale
        W[r, c] = sign * block
    if balance:
        for i in range(N):
            row = W[i]
            eE = row[row > 0].sum(); eI = -row[row < 0].sum()
            if eI > 1e-9:
                W[i, row < 0] *= eE / eI
    return W, idx, N


# --------------------------------------------------------------------------- #
# Linear vs nonlinear (operating-point) stability.
# --------------------------------------------------------------------------- #
def update_map(W):
    """Linear update Jacobian J = (1-1/tau) I + (1/tau) W (nln gain assumed 1)."""
    n = W.shape[0]
    return (1.0 - 1.0 / TAU) * np.eye(n) + (1.0 / TAU) * W


def rho_linear(W):
    """Spectral radius of the LINEAR map about x=0 (the old probe's quantity)."""
    return float(np.max(np.abs(np.linalg.eigvals(update_map(W)))))


def fixed_point(W, b=0.0, x0=None, iters=4000, tol=1e-10):
    """Settle the NONLINEAR map x <- nln((1-1/tau)x + (1/tau)(W x + b)).

    b is a constant background drive (scalar or vector). Returns (x*, pre*,
    converged). A small positive b is used to probe a nonzero operating point
    (the loop's tonic input); b=0 leaves the trivial x*=0."""
    N = W.shape[0]
    b = np.broadcast_to(np.asarray(b, dtype=float), (N,))
    x = np.full(N, 0.1) if x0 is None else x0.copy()
    pre = x
    for _ in range(iters):
        pre = (1.0 - 1.0 / TAU) * x + (1.0 / TAU) * (W @ x + b)
        x_new = nln(pre)
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x, pre, bool(np.max(np.abs(nln(pre) - x)) < 1e-6)


def rho_operating(W, pre_star):
    """Spectral radius of the Jacobian at the operating point:
       J* = diag(nln'(pre*)) @ [(1-1/tau)I + (1/tau)W].
    This is the quantity that actually governs stability of the fixed point."""
    g = nln_prime(pre_star)
    J = g[:, None] * update_map(W)
    return float(np.max(np.abs(np.linalg.eigvals(J)))), g


def persistence(W, idx, N, T=400, cue_t0=20, cue_len=10, amp=1.0, b=0.0):
    """Cue-evoked loop activity under the true nln map (cue targets cU, cL)."""
    x = np.zeros(N)
    inp = np.full((T, N), float(b))
    for name in ("cU", "cL"):
        inp[cue_t0:cue_t0 + cue_len, idx[name]] += amp
    tr = np.zeros(T)
    for t in range(T):
        x = nln((1.0 - 1.0 / TAU) * x + (1.0 / TAU) * (W @ x) + inp[t])
        tr[t] = x.mean()
    return tr


# =========================================================================== #
# Section 1: linear vs operating-point stability, faithful vs balanced.
# =========================================================================== #
def section1_operating_point(n=8, b_tonic=0.15):
    print("=" * 78)
    print("1. LINEAR (about x=0) vs NONLINEAR OPERATING-POINT stability")
    print(f"   loop size n={n} (cU=cL=cI=tE={n}, tI={max(1, n//2)}), tonic drive b={b_tonic}")
    print("=" * 78)
    print(f"{'architecture':<22}{'rho_linear':>11}{'x*_mean':>10}{'gain_mean':>10}"
          f"{'rho_operating':>15}   verdict")
    results = {}
    for label, (scaling, balance, g) in {
        "faithful (as-is)":     ("faithful", False, 1.0),
        "matched scaling":      ("matched",  False, 1.0),
        "balanced + matched":   ("matched",  True,  1.0),
    }.items():
        W, idx, N = build_tc(n, scaling, balance, g, seed=0)
        rl = rho_linear(W)
        xs, pre, conv = fixed_point(W, b=b_tonic)
        ro, gain = rho_operating(W, pre)
        sat = 100.0 * np.mean(xs > 0.9)
        verdict = ("SATURATED fixed point (signal-dead)" if sat > 40 else
                   "near-critical, live gain" if 0.9 <= ro < 1.02 else
                   "sub-critical / low")
        print(f"{label:<22}{rl:>11.3f}{xs.mean():>10.3f}{gain.mean():>10.3f}"
              f"{ro:>15.3f}   {verdict}")
        results[label] = (rl, xs.mean(), gain.mean(), ro, sat)
    print("\n  Reading: the faithful loop has rho_linear >> 1 (x=0 is unstable), but it")
    print("  does NOT run away to infinity -- nln saturates it to a HIGH fixed point")
    print("  where the local gain collapses (gain_mean -> 0), so rho_operating < 1.")
    print("  That saturated state is a stable attractor with ~zero gain: it neither")
    print("  drifts nor transmits a cue. 'Stable' by the linear test would be wrong")
    print("  about WHY it fails -- it fails by saturation, not by blow-up.\n")
    return results


# =========================================================================== #
# Section 2: cue-evoked persistence (already nln).
# =========================================================================== #
def section2_persistence(n=8, b_tonic=0.15, g_bal=1.0):
    Wf, idxf, Nf = build_tc(n, "faithful", False, 1.0, 0)
    Wb, idxb, Nb = build_tc(n, "matched", True, g_bal, 0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(persistence(Wf, idxf, Nf, b=b_tonic), color="#c0392b", lw=2, label="faithful (as-is)")
    ax.plot(persistence(Wb, idxb, Nb, b=b_tonic), color="#27ae60", lw=2, label="balanced + matched")
    ax.axvspan(20, 30, color="gray", alpha=0.2, label="cue pulse")
    ax.axvline(320, color="k", ls=":", lw=1.5, label="response time (+300)")
    ax.set_xlabel("timestep"); ax.set_ylabel("mean loop activity (nln)")
    ax.set_title(f"Cue-evoked persistence under nln  (n={n}, tonic b={b_tonic})")
    ax.legend(fontsize=9); fig.tight_layout()
    p = os.path.join(PLOTS, "persistence_nln.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"2. persistence plot -> {p}")
    return p


# =========================================================================== #
# Section 3: size sweep on the operating-point rho*.
# =========================================================================== #
def section3_size_sweep(b_tonic=0.15, seeds=40):
    NS = [2, 4, 6, 8, 12, 16, 24, 32]
    print("\n" + "=" * 78)
    print("3. SIZE SWEEP -- operating-point rho* (mean over seeds), faithful vs balanced")
    print("=" * 78)
    print(f"{'n':>3}{'N':>5} | {'faithful x*':>11}{'f rho*':>8} | {'balanced x*':>12}{'b rho*':>8}{'b gain':>8}")
    f_rho, b_rho, f_x, b_x, f_lin = [], [], [], [], []
    for n in NS:
        N = 4 * n + max(1, n // 2)
        fr, br, fx, bx, fl = [], [], [], [], []
        for s in range(seeds):
            Wf = build_tc(n, "faithful", False, 1.0, s)[0]
            Wb = build_tc(n, "matched", True, 1.0, s)[0]
            _, pf, _ = fixed_point(Wf, b=b_tonic)
            _, pb, _ = fixed_point(Wb, b=b_tonic)
            ro_f, _ = rho_operating(Wf, pf)
            ro_b, _ = rho_operating(Wb, pb)
            fr.append(ro_f); br.append(ro_b)
            fx.append(nln(pf).mean()); bx.append(nln(pb).mean())
            fl.append(rho_linear(Wf))
        f_rho.append(np.mean(fr)); b_rho.append(np.mean(br))
        f_x.append(np.mean(fx)); b_x.append(np.mean(bx)); f_lin.append(np.mean(fl))
        gb = nln_prime(fixed_point(build_tc(n, "matched", True, 1.0, 0)[0], b=b_tonic)[1]).mean()
        print(f"{n:>3}{N:>5} | {np.mean(fx):>11.3f}{np.mean(fr):>8.3f} | "
              f"{np.mean(bx):>12.3f}{np.mean(br):>8.3f}{gb:>8.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(NS, f_lin, "-o", color="#7f8c8d", label="faithful rho_LINEAR (about 0)")
    ax[0].plot(NS, f_rho, "-o", color="#c0392b", label="faithful rho* (operating)")
    ax[0].plot(NS, b_rho, "-o", color="#27ae60", label="balanced rho* (operating)")
    ax[0].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xticks(NS); ax[0].set_xticklabels(NS)
    ax[0].set_xlabel("n (per-pool size)"); ax[0].set_ylabel("spectral radius")
    ax[0].set_title("Linear rho overstates instability;\noperating-point rho* is what governs the fixed point")
    ax[0].legend(fontsize=8)
    ax[1].plot(NS, f_x, "-o", color="#c0392b", label="faithful x*_mean")
    ax[1].plot(NS, b_x, "-o", color="#27ae60", label="balanced x*_mean")
    ax[1].axhline(0.9, color="k", ls=":", lw=1, label="saturation")
    ax[1].set_xscale("log"); ax[1].set_xticks(NS); ax[1].set_xticklabels(NS); ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("n (per-pool size)"); ax[1].set_ylabel("mean fixed-point activity")
    ax[1].set_title("Faithful loop saturates at every size; balanced stays low")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Thalamocortical size sweep under nln (tonic b={b_tonic})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(PLOTS, "size_sweep_nln.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\n   size-sweep plot -> {p}")
    return p


# =========================================================================== #
# Section 4: striatal gate -- per-term nln deletes inhibition -> saturation.
# =========================================================================== #
def section4_striatum(pka=0.5, tau=TAU):
    """The striatum update applies a nonlinearity to EACH input term separately:
         x_d1 <- (1-1/tau)x + (1/tau)[ nln(J_self @ x)            # inhibitory: arg<0
                                     + nln(B_D2 @ x_d2)           # inhibitory: arg<0
                                     + bg_nln(B_cortex @ x_c, pka)]# excitatory:  arg>0
         x_d1 <- bg_nln(x_d1, pka)

    nln and bg_nln both map into (0,1) -- their output is ALWAYS non-negative.
    So wrapping an inhibitory projection in nln means it can never subtract: a
    strong negative current -A becomes nln(-A) ~ a small POSITIVE floor instead
    of -A. Inhibition is destroyed (converted to a small positive add). With the
    current sigmoid nln this floor is nln(-A) in (0, 0.12], not exactly 0, but
    the effect is the same: nothing opposes the excitatory drive, so the map has
    a single high fixed point. We sweep the excitatory drive and compare the
    per-term wiring against the correct signed-current wiring (one nonlinearity
    on the summed drive, so inhibition actually subtracts)."""
    a = 1.0 - 1.0 / tau
    exc_drive = 0.5                       # fixed moderate cortical excitation
    inh = np.linspace(0.0, 3.0, 31)       # sweep the inhibitory current magnitude

    def fp_perterm(inh_drive):
        # As coded: inhibition passes through nln (-> a small positive floor),
        # then the summed drive passes through bg_nln.
        x = 0.3
        for _ in range(3000):
            D = nln(-inh_drive) + bg_nln(exc_drive, pka)
            x = bg_nln(a * x + D / tau, pka)
        return x

    def fp_signed(inh_drive):
        # Correct: signed currents summed, ONE bg_nln on the total (inh subtracts).
        x = 0.3
        for _ in range(3000):
            x = bg_nln(a * x + (exc_drive - inh_drive) / tau, pka)
        return x

    per = [fp_perterm(v) for v in inh]
    sign = [fp_signed(v) for v in inh]

    print("\n" + "=" * 78)
    print("4. STRIATAL GATE -- why per-term nln pins the striatum high")
    print(f"   pka={pka}, fixed excitation {exc_drive}; sweeping the inhibitory current.")
    print(f"   nln(-A) is a POSITIVE floor, e.g. nln(-1)={nln(-1.0):.4f}, nln(-3)={nln(-3.0):.4f}"
          f" -> inhibition can't subtract.")
    print("=" * 78)
    print(f"{'inh current':>12}{'per-term x*':>14}{'signed x*':>12}")
    for v in (0.0, 0.5, 1.0, 2.0, 3.0):
        print(f"{v:>12.2f}{fp_perterm(v):>14.4f}{fp_signed(v):>12.4f}")
    print("\n  Per-term: x* is FLAT vs inhibition -- a growing inhibitory current is")
    print("  squashed to a ~constant positive floor by nln, so it never gates D1 down.")
    print("  Signed current: the same inhibition monotonically shuts the gate.")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(inh, per, "-o", color="#c0392b", ms=4, label="per-term nln (as coded): inhibition -> positive floor")
    ax.plot(inh, sign, "-o", color="#27ae60", ms=4, label="signed current (fix): inhibition subtracts")
    ax.axhline(0.9, color="k", ls=":", lw=1, label="saturation")
    ax.set_xlabel("inhibitory current magnitude"); ax.set_ylabel("striatal fixed point x*")
    ax.set_title(f"Striatal gate (pka={pka}, exc={exc_drive}): per-term nln ignores inhibition\n"
                 f"(nln of a negative current is a small POSITIVE floor) -> pinned high")
    ax.legend(fontsize=9); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    p = os.path.join(PLOTS, "striatum_saturation.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\n   striatum plot -> {p}")
    return p


def main():
    section1_operating_point()
    section2_persistence()
    section3_size_sweep()
    section4_striatum()
    print("\nAll sections complete. Plots in:", PLOTS)


if __name__ == "__main__":
    main()
