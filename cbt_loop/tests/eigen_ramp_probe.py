"""Eigenvalue + persistence probe for the THALAMOCORTICAL LOOP of the cbt_loop family.

Unlike an isolated cortex, the real recurrent loop spans cortex (cU, cL, cI) and
thalamus (tE, tI) with reciprocal cortico-thalamic excitation. This script builds
that 5-population loop with the exact connectivity and Dale signs of cbt_rnn.py's
init_params, then:
  * plots the eigenvalue spectrum of the update map J = (1-1/tau)I + (1/tau)W,
  * simulates cue-evoked persistence of the loop,
  * sweeps network size to find which sizes yield (reproducible) near-critical
    dynamics.

Size constraint (per user): n_cU = n_cL = n_cI = n_tE = n ; n_tI = n/2.

Two architectures are compared at each step:
  faithful : exact init scalings (E->E ~ g/sqrt(n); every I / cross-area ~ 1/n),
             g_bg = 1, NO balancing  -> the architecture as it currently stands.
  balanced : matched g/sqrt(n) scaling on all blocks + per-row E/I balance
             (each cell's local inhibition cancels its total excitation) + a
             calibrated global gain -> the "fixed" near-critical architecture.

Connectivity (post <- pre), signs and faithful scalings, from init_params:
  cU <- cU (+, g/sqrt) cU <- cL (+, g/sqrt) cU <- cI (-, g/n)  cU <- tE (+, g/n)
  cL <- cL (+, g/sqrt) cL <- cU (+, g/sqrt) cL <- cI (-, g/n)
  cI <- cU (+, g/n)    cI <- cL (+, g/n)    cI <- cI (-, g/n)  cI <- tE (+, g/n)
  tE <- tE (+, g/sqrt) tE <- tI (-, g/n)    tE <- cU (+, 1/n)
  tI <- tE (+, g/n)    tI <- tI (-, g/n)    tI <- cU (+, 1/n)
(cL receives no thalamic input; thalamus is driven only by cU -- as in cbt_rnn.)

NOTE: linearized (nln gain ~ 1) small-signal probe about a low-activity fixed
point; uniform tau (config default tau_t = tau_c). External drive (cue, SNr, SC)
is not part of the recurrent W.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_script as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

G = cfg.RNN_CONFIG.get("g_bg", 1.0)
TAU = 10.0

# (post, pre, sign, kind) with kind in {"sqrt","lin"} selecting the faithful scaling.
EDGES = [
    ("cU", "cU", +1, "sqrt"), ("cU", "cL", +1, "sqrt"), ("cU", "cI", -1, "lin"), ("cU", "tE", +1, "lin"),
    ("cL", "cL", +1, "sqrt"), ("cL", "cU", +1, "sqrt"), ("cL", "cI", -1, "lin"),
    ("cI", "cU", +1, "lin"),  ("cI", "cL", +1, "lin"),  ("cI", "cI", -1, "lin"), ("cI", "tE", +1, "lin"),
    ("tE", "tE", +1, "sqrt"), ("tE", "tI", -1, "lin"),  ("tE", "cU", +1, "lin"),
    ("tI", "tE", +1, "lin"),  ("tI", "tI", -1, "lin"),  ("tI", "cU", +1, "lin"),
]


def build_tc(n, scaling="faithful", balance=False, g=G, seed=0):
    """Full thalamocortical recurrent matrix W (state order cU,cL,cI,tE,tI)."""
    nI = max(1, n // 2)
    sizes = [("cU", n), ("cL", n), ("cI", n), ("tE", n), ("tI", nI)]
    idx, off = {}, 0
    for name, sz in sizes:
        idx[name] = slice(off, off + sz); off += sz
    N = off
    W = np.zeros((N, N))
    rng = np.random.default_rng(seed)
    for post, pre, sign, kind in EDGES:
        pre_n = sizes[[s[0] for s in sizes].index(pre)][1]
        scale = g / np.sqrt(pre_n) if (scaling == "matched" or kind == "sqrt") else g / pre_n
        r, c = idx[post], idx[pre]
        block = np.abs(rng.standard_normal((r.stop - r.start, c.stop - c.start))) * scale
        W[r, c] = sign * block
    if balance:                       # per-row: local inhibition cancels total excitation
        for i in range(N):
            row = W[i]
            eE = row[row > 0].sum(); eI = -row[row < 0].sum()
            if eI > 1e-9:
                W[i, row < 0] *= eE / eI
    return W, idx, N


def update_map(W):
    n = W.shape[0]
    return (1.0 - 1.0 / TAU) * np.eye(n) + (1.0 / TAU) * W


def rho_of(W):
    return np.max(np.abs(np.linalg.eigvals(update_map(W))))


def persistence(W, idx, N, T=400, cue_t0=20, cue_len=10, amp=1.0):
    x = np.zeros(N)
    inp = np.zeros((T, N))
    for name in ("cU", "cL"):                     # cue targets cU and cL
        inp[cue_t0:cue_t0 + cue_len, idx[name]] = amp
    tr = np.zeros(T)
    for t in range(T):
        x = np.maximum(0.0, np.tanh((1.0 - 1.0 / TAU) * x + (1.0 / TAU) * (W @ x) + inp[t]))
        tr[t] = x.mean()
    return tr


# ---- calibrate balanced gain so mean rho ~ 0.98 at the largest swept size ----
SEEDS = 60
NS = [2, 4, 6, 8, 12, 16, 24, 32]        # base size n (n_tI = n/2); total units = 4.5 n
shift = 1.0 - 1.0 / TAU
r1 = np.mean([rho_of(build_tc(32, "matched", True, 1.0, s)[0]) for s in range(SEEDS)])
g_bal = (0.98 - shift) / (r1 - shift)
print(f"balanced gain calibrated: g = {g_bal:.3f}  (mean rho -> ~0.98 at n=32)")

# ---- figure 1: eigenvalue spectra (faithful vs balanced) at a representative n
REP = 8
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
theta = np.linspace(0, 2 * np.pi, 200)
for ax, (title, W) in zip(axes, [
        (f"faithful (as-is), n={REP}", build_tc(REP, "faithful", False, 1.0, 0)[0]),
        (f"balanced+matched, n={REP}", build_tc(REP, "matched", True, g_bal, 0)[0])]):
    ev = np.linalg.eigvals(update_map(W))
    rho = np.max(np.abs(ev))
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1, alpha=0.6)
    ax.scatter(ev.real, ev.imag, s=28, c="#c0392b", alpha=0.8, edgecolors="none")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.text(0.03, 0.97, f"rho={rho:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
fig.suptitle("Thalamocortical loop eigenvalues  J=(1-1/tau)I+(1/tau)W  (unit circle = persistence boundary)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
f1 = os.path.join(PLOTS, "eigenvalue_spectra.png"); fig.savefig(f1, dpi=130); plt.close(fig)

# ---- figure 2: cue-evoked persistence of the loop ----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
Wf, idxf, Nf = build_tc(REP, "faithful", False, 1.0, 0)
Wb, idxb, Nb = build_tc(REP, "matched", True, g_bal, 0)
ax.plot(persistence(Wf, idxf, Nf), color="#c0392b", lw=2, label=f"faithful (as-is), n={REP}")
ax.plot(persistence(Wb, idxb, Nb), color="#27ae60", lw=2, label=f"balanced+matched, n={REP}")
ax.axvspan(20, 30, color="gray", alpha=0.2, label="cue pulse")
ax.axvline(320, color="k", ls=":", lw=1.5, label="response time (+300 steps)")
ax.set_xlabel("timestep"); ax.set_ylabel("mean loop activity")
ax.set_title("Cue-evoked persistence in the thalamocortical loop")
ax.legend(fontsize=9); fig.tight_layout()
f2 = os.path.join(PLOTS, "ramp_probe.png"); fig.savefig(f2, dpi=130); plt.close(fig)

# ---- figure 3: size sweep ----------------------------------------------------
BAND = (0.95, 1.0)
print("\n=== thalamocortical size sweep (n_cU=n_cL=n_cI=n_tE=n, n_tI=n/2) ===")
print(f"{'n':>3s} {'N_tot':>6s} {'faithful rho':>13s} | {'bal mean rho':>12s} {'std':>6s} {'P(band)':>8s}")
fmeans, bmeans, bp10, bp90, bstd, buse = [], [], [], [], [], []
for n in NS:
    Ntot = 4 * n + max(1, n // 2)
    fr = np.mean([rho_of(build_tc(n, "faithful", False, 1.0, s)[0]) for s in range(SEEDS)])
    br = np.array([rho_of(build_tc(n, "matched", True, g_bal, s)[0]) for s in range(SEEDS)])
    fmeans.append(fr); bmeans.append(br.mean()); bstd.append(br.std())
    bp10.append(np.percentile(br, 10)); bp90.append(np.percentile(br, 90))
    buse.append(np.mean((br >= BAND[0]) & (br < BAND[1])))
    print(f"{n:3d} {Ntot:6d} {fr:13.3f} | {br.mean():12.3f} {br.std():6.3f} {buse[-1]:8.2f}")

fmeans, bmeans, bp10, bp90, bstd, buse = map(np.array, (fmeans, bmeans, bp10, bp90, bstd, buse))
ok = (bstd < 0.02) & (buse >= 0.90)
nstar = NS[int(np.argmax(ok))] if ok.any() else None

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
axL.plot(NS, fmeans, "-o", color="#c0392b", label="faithful (as-is)")
axL.plot(NS, bmeans, "-o", color="#27ae60", label="balanced (mean)")
axL.fill_between(NS, bp10, bp90, alpha=0.2, color="#27ae60", label="balanced 10-90 pctile")
axL.axhline(1.0, color="k", ls="--", lw=1, label="critical (rho=1)")
axL.axhspan(*BAND, color="gray", alpha=0.08)
axL.set_yscale("log"); axL.set_xscale("log"); axL.set_xticks(NS); axL.set_xticklabels(NS)
axL.set_xlabel("n (per-pool size; total = 4.5 n)"); axL.set_ylabel("spectral radius rho(J)")
axL.set_title("Spectral radius vs size: faithful vs balanced"); axL.legend(fontsize=8)

axR.plot(NS, 100 * buse, "-o", color="#27ae60")
axR.axhline(90, color="k", ls="--", lw=1, label="90% reliability")
if nstar is not None:
    axR.axvline(nstar, color="#c0392b", ls=":", lw=2, label=f"n* = {nstar}")
axR.set_xscale("log"); axR.set_xticks(NS); axR.set_xticklabels(NS); axR.set_ylim(0, 101)
axR.set_xlabel("n (per-pool size)"); axR.set_ylabel("% seeds in near-critical band")
axR.set_title("Reliability of near-critical dynamics (balanced loop)"); axR.legend(fontsize=8)
fig.suptitle("Thalamocortical loop: smallest size for reproducible near-critical dynamics", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
f3 = os.path.join(PLOTS, "size_sweep.png"); fig.savefig(f3, dpi=130); plt.close(fig)

if nstar is not None:
    print(f"\n--> balanced loop: smallest reliable n* = {nstar}  "
          f"(cU/cL/cI/tE={nstar}, tI={max(1,nstar//2)}, total={4*nstar+max(1,nstar//2)} units)")
else:
    print("\n--> balanced loop: no swept size met both criteria; widen the range.")
print(f"faithful (as-is) rho ranges {fmeans.min():.2f}-{fmeans.max():.2f} over swept sizes "
      f"({'never' if (fmeans<1).sum()==0 else 'sometimes'} < 1).")
print(f"\nsaved:\n  {f1}\n  {f2}\n  {f3}")
