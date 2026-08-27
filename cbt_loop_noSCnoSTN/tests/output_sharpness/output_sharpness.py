"""TEST: output_sharpness -- why noSCnoSTN produces a broad PLATEAU, not a sharp pulse.

The readout is instantaneous (y = sigmoid(out_gain * C_med @ x_med + out_bias)), so the
shape of y is the shape of the MEDULLA, whose shape is the shape of its DRIVE. Tracing the
chain on the trained self-timed model (params_shaped.pkl) for one representative trial shows
two independent problems:

1) NO TERMINATION (the plateau). The direct pathway is a "go" that never turns off: cortex
   and D1 fire AT the cue (argmax ~t_cue) and stay elevated -- persistent cortico-thalamic
   activity with no auto-off. The only available "stop" is the indirect pathway (D2 -> SNr
   gate closing), but D2/SNr ramp up SLOWLY (argmax near the trial END, ~t=997) instead of
   snapping the gate shut ~100 steps after it opened. So the SNr gate is open from the cue
   until ~end -> the medulla (and y) are ON that whole span -> a ~400-step plateau vs the
   100-step target. noSCnoSTN also LACKS the STN, the BG's fast hyperdirect brake, so the
   slow adenosine/PKA-paced indirect ramp is the ONLY brake available -- it cannot carve a
   sharp pulse.

2) WEAK output (amplitude). Even the plateau only reaches y~0.09 (never crosses 0.3): the
   medulla readout C_med is small (logit-init shrank it toward its intended magnitude) and
   out_gain/out_bias don't compensate, so the readout barely lifts off resting.

TAKEAWAY: sharpness needs a FAST termination (a brief SNr re-closure ~100 steps after
opening -- a real "stop"), which the current slow D2/adenosine ramp and the missing STN
can't provide; amplitude needs a stronger medulla readout.
"""
import sys
import pathlib
import pickle
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
import config_script as C
import self_timed_movement_task as stmt

cfg = C.for_family("cbt_loop_noSCnoSTN")
A = list(cbtl.STATE_AREA_ORDER)
BUNDLE = "params_shaped.pkl"


def main():
    d = pickle.load(open(FAMILY_DIR / BUNDLE, "rb"))
    p, c = d["params"], d["config"]
    ncue = int(np.asarray(p["B_cue_cU"]).shape[1])
    tc = cfg.TASK_CONFIG
    inp, tgt, _ = stmt.self_timed_movement_task(
        T_start=tc["t_start"], T_cue=tc["t_cue"], T_wait=tc["t_wait"],
        T_movement=tc["t_movement"], T=tc["t_total"])
    if ncue == 2:
        inp = jnp.concatenate([inp, jnp.zeros_like(inp)], axis=-1)
    B, T, _ = inp.shape
    nd1 = p["J_d1"].shape[0]; nd2 = p["J_d2"].shape[0]
    ys, xs = cbtl.batched_rnn(p, c, inp, jnp.zeros((B, T, nd1 + nd2)), jr.split(jr.PRNGKey(0), B))
    y = np.asarray(ys); y = y[..., 0] if y.ndim == 3 else y
    tgtn = np.asarray(tgt)[..., 0]

    def area(nm):
        x = np.asarray(xs[A.index(nm)]); return x.mean(-1) if x.ndim > 2 else x

    ts = np.asarray(tc["t_start"]); k = int(np.argmin(np.abs(ts - np.median(ts))))
    t = np.arange(T)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    ax[0].plot(t, tgtn[k], color="k", lw=2, label="target (sharp pulse)")
    ax[0].plot(t, y[k], color="crimson", lw=2, label="output y (broad + weak)")
    ax[0].axvline(ts[k], color="gray", ls=":", lw=1, label="cue (t_start)")
    ax[0].set_title(f"output vs target (trial {k}, t_start={ts[k]})")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("value"); ax[0].set_ylim(-0.02, 1.05)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    for nm, col in [("Cortex", "tab:blue"), ("D1", "tab:green"), ("D2", "tab:red"),
                    ("SNr", "tab:purple"), ("Medulla", "tab:orange")]:
        ax[1].plot(t, area(nm)[k], lw=1.7, color=col, label=nm)
    ax[1].axvline(ts[k], color="gray", ls=":", lw=1)
    ax[1].set_title("drive chain: D1/cortex 'go' fire at cue & sustain; D2/SNr 'stop' too slow")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("rate"); ax[1].set_ylim(-0.02, 1.05)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    fig.suptitle("noSCnoSTN output plateau: sustained 'go' + slow/late 'stop' (no STN brake) "
                 "-> broad response; weak medulla readout -> low amplitude", y=1.01)
    fig.tight_layout()
    out = HERE / "output_sharpness.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    W = lambda s: int((s > 0.5 * (s.max() + s.min())).sum())
    print(f"trial {k}: target width={W(tgtn[k])}  y width={W(y[k])}  y max={y[k].max():.3f}")
    print(f"D1 argmax={int(area('D1')[k].argmax())}  D2 argmax={int(area('D2')[k].argmax())}  "
          f"SNr argmax={int(area('SNr')[k].argmax())}  (stop peaks late => no sharp pulse)")
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
