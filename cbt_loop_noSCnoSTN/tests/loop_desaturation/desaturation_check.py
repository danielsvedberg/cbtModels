"""TEST: loop_desaturation -- the sigmoid-Dale init trap and the wrapper-aware logit fix.

WHAT & WHY
----------
The Dale wrapper is exc/inh = +/-sigmoid(w) (mass-action, bounded (0,1)), so the EFFECTIVE
weight is sigmoid(raw). The init builds every block as an intended fan-in-scaled MAGNITUDE
(~0.05, the value a linear clip/abs Dale would use directly). Under sigmoid a small raw
collapses to sigmoid(0.05)~=0.5, so EVERY weight becomes ~0.5 -> a dense, uniformly
over-excited net. The whole cortico-thalamic loop (cU, cL, cI, tE, tI -- excitatory AND
inhibitory pools alike) pins at the top rail ~0.99, which drives D2 -> 1.0. This is invariant
to the 6 DA/adenosine gains, to balanced_target_rho (1.0->0.5), to loop-block scale (1.0->0),
and to input on/off, because all of those act on RAW weights and sigmoid flattens them to 0.5.
normalize_loop and the 1/fan_in scaling are defeated for the same reason. cbt_rnn.py warns a
saturated cortex zeros the task gradient -- so this also explains why hybrid barely trains.

THE FIX (in cbt_rnn.init_params, gated on exc(0)>0.25 so it is a no-op for clip/abs Dale):
invert the sigmoid at init so exc(raw) reproduces the intended magnitude: raw <- logit(|mag|).
Sign is still supplied by exc/inh in the forward. Applied to connectivity matrices only (2-D;
the scalar DA/adenosine/pacer gains are excluded).

FINDING
-------
logit-init de-saturates the whole loop and hands the striatum a clean substrate:
  cortex 0.99 -> 0.38, thalamus 0.99 -> 0.18, GPe 0.84 -> 0.28  (all in-band, responsive)
  D1 0.46 -> 0.41 in-band (pka_d1 rests 0.30); loop operating-point rho* = 0.97 (near-critical)
  D2 STAYS ~1.0 -- but now purely because pka_d2 rests high (~0.86). bg_nln is cleanly pka-tuned
  (D1=D2~0.5 at pka~0.30, dead below 0.25, saturated above 0.4), so bringing D2 in-band is a
  well-defined target for the m_a2/m_d2/g_ado/g_da search -- which was impossible while the
  cortex itself was railed. De-saturating the loop was the prerequisite; the gain search is next.
"""
import sys
import pathlib
import numpy as np
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
import train_hybrid

A = list(cbtl.STATE_AREA_ORDER)
GAIN_KEYS = {"m_d1", "m_d2", "m_a1", "m_a2", "g_da_release", "g_ado_release", "P_gpe", "P_snc", "P_snr"}
PLOT_AREAS = ["Cortex", "Thalamus", "D1", "D2", "pkaD1", "pkaD2"]
BAR_AREAS = ["Cortex", "Thalamus", "GPe", "SNr", "D1", "D2", "DA", "Adenosine"]


def trap_params(fixed):
    """Reconstruct the pre-fix 'sigmoid trap': set connectivity raw = sigmoid(logit-raw) = the
    small intended magnitude, so exc(raw)=sigmoid(small)~=0.5 for every weight (over-excited)."""
    p = dict(fixed)
    for k in list(p):
        v = jnp.asarray(p[k])
        if k not in GAIN_KEYS and v.ndim >= 2:
            p[k] = stmt.exc(v)          # sigmoid(logit|mag|) = |mag| -> back into the trap regime
    return p


def timecourses(params, config, inp):
    B, T, _ = inp.shape
    nd1 = params["J_d1"].shape[0]; nd2 = params["J_d2"].shape[0]
    _, xs = cbtl.batched_rnn(params, config, inp, jnp.zeros((B, T, nd1 + nd2)),
                             jr.split(jr.PRNGKey(0), B))
    out = {}
    for a in set(PLOT_AREAS + BAR_AREAS):
        x = np.asarray(xs[A.index(a)]); x = x.mean(-1) if x.ndim > 2 else x
        out[a] = x.mean(0)              # (T,)
    return out


def main():
    inp = train_hybrid._build_hybrid_batch()[0][:16]
    fixed, config = cbtl.init_params(jr.PRNGKey(0), n_input=inp.shape[-1])
    assert float(stmt.exc(jnp.asarray(0.0))) > 0.25, "exc is not sigmoid; trap demo N/A"
    tc_fix = timecourses(fixed, config, inp)
    tc_trap = timecourses(trap_params(fixed), config, inp)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for a in PLOT_AREAS:
        ln, = ax[0].plot(tc_fix[a], lw=1.8, label=a)
        ax[0].plot(tc_trap[a], lw=1.2, ls="--", color=ln.get_color(), alpha=0.6)
    ax[0].axhspan(0.15, 0.85, color="green", alpha=0.06)
    ax[0].set_title("timecourse: logit-init (solid) vs sigmoid trap (dashed)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("rate"); ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)

    x = np.arange(len(BAR_AREAS)); w = 0.38
    mf = [tc_fix[a].mean() for a in BAR_AREAS]; mt = [tc_trap[a].mean() for a in BAR_AREAS]
    ax[1].bar(x - w / 2, mt, w, label="sigmoid trap", color="crimson", alpha=0.75)
    ax[1].bar(x + w / 2, mf, w, label="logit-init (fix)", color="steelblue")
    ax[1].axhspan(0.15, 0.85, color="green", alpha=0.08, label="operating band")
    ax[1].set_xticks(x); ax[1].set_xticklabels(BAR_AREAS, rotation=35, ha="right")
    ax[1].set_ylabel("mean rate over trial"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("mean level per area"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")

    fig.suptitle("noSCnoSTN loop de-saturation: sigmoid-Dale trap vs wrapper-aware logit-init", y=1.0)
    fig.tight_layout()
    out = HERE / "desaturation_check.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("trap  :", "  ".join(f"{a}={tc_trap[a].mean():.3f}" for a in BAR_AREAS))
    print("fixed :", "  ".join(f"{a}={tc_fix[a].mean():.3f}" for a in BAR_AREAS))
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
