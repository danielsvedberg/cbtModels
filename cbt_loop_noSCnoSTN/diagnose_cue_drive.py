"""Structural diagnostic: why does the model plateau at the ~7% cue-ignoring ceiling?

Runs a trained pavlovian model on jittered-cue trials and asks:
  1. Does the response track the CUE (cue-locked) or fire at a FIXED time (degenerate)?
  2. Is the cue even reaching the output? (cue-present vs cue-absent output)
  3. Is the motor gate open? (Cortex->D1->SNr-|Thalamus->Medulla->output chain state,
     plus any dead (~0) or saturated (~1) areas)
"""
import pickle as pkl
import sys
import pathlib

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import cbt_rnn as cbtl
_root = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as _config_script
import self_timed_movement_task as stmt

cfg = _config_script.for_family("cbt_loop_noSCnoSTN")
HERE = pathlib.Path(__file__).resolve().parent
PLOTS = HERE / "plots"
MODEL = HERE / "params_pavlovian.pkl"   # the naive baseline (0.070 ceiling)
AREAS = list(cbtl.STATE_AREA_ORDER)


def run(params, config, inputs):
    inputs = jnp.asarray(inputs)
    B, T, _ = inputs.shape
    n_d1 = np.asarray(params["J_d1"]).shape[0]
    n_d2 = np.asarray(params["J_d2"]).shape[0]
    stim = jnp.zeros((B, T, n_d1 + n_d2))          # opto channel (D1+D2), zeroed
    keys = jr.split(jr.PRNGKey(0), B)
    ys, xs = cbtl.batched_rnn(params, config, inputs, stim, keys)
    return np.asarray(ys), [np.asarray(x) for x in xs]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with MODEL.open("rb") as f:
        d = pkl.load(f)
    params, config = d["params"], d["config"]

    t = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=t["t_start"], T_cue=t["t_cue"], T_response=t["t_response"], T=t["t_total"])
    inputs = np.asarray(inputs)
    B, T, _ = inputs.shape
    cue_on = np.array([np.argmax(inputs[i, :, 0] > 0) for i in range(B)])  # first cue step

    ys, xs = run(params, config, inputs)
    yflat = ys[:, :, 0]  # (B, T)

    # cue-absent control (does the cue matter at all?)
    inp0, _, _ = stmt.pavlovian_task(T_start=t["t_start"], T_cue=t["t_cue"],
                                     T_response=t["t_response"], T=t["t_total"], null_trial=True)
    ys0, _ = run(params, config, jnp.asarray(inp0))
    y0flat = ys0[:, :, 0]

    # response = output peak AFTER cue; response time relative to cue
    resp_t, peak_amp = [], []
    for i in range(B):
        seg = yflat[i, cue_on[i]:]
        resp_t.append(int(np.argmax(seg)))
        peak_amp.append(float(seg.max()))
    resp_t = np.array(resp_t); peak_amp = np.array(peak_amp)
    slope = np.polyfit(cue_on, cue_on + resp_t, 1)[0]  # d(abs response time)/d(cue time)

    # per-area mean activity (over units & trials & time) -> dead/saturated scan
    area_mean = {a: float(x.mean()) for a, x in zip(AREAS, xs)}

    print("=" * 64)
    print(f"model: {MODEL.name}   trials: {B}, T={T}")
    print(f"cue-locking slope d(resp_time)/d(cue_time) = {slope:+.3f}  "
          f"(1=cue-locked, 0=fixed-time degenerate)")
    print(f"mean |output cue-present - cue-absent| = {np.abs(yflat - y0flat).mean():.4f}  "
          f"(0 => cue does nothing)")
    print("-" * 64)
    print("per-area mean activity (flag DEAD<0.05 / SAT>0.95):")
    for a in AREAS:
        m = area_mean[a]; flag = "  <-- DEAD" if m < 0.05 else ("  <-- SAT" if m > 0.95 else "")
        print(f"   {a:10s} {m:.3f}{flag}")
    print("=" * 64)

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    order = np.argsort(cue_on)
    for i in order[::max(1, B // 8)]:
        ax[0, 0].plot(yflat[i], lw=0.9)
        ax[0, 0].axvline(cue_on[i], color="grey", ls=":", lw=0.6)
    ax[0, 0].set_title("Output y for trials with different cue onsets (dotted = cue)")
    ax[0, 0].set_xlabel("time"); ax[0, 0].set_ylabel("output prob")

    ax[0, 1].scatter(cue_on, cue_on + resp_t, s=14, alpha=0.6)
    lim = [cue_on.min(), cue_on.max()]
    ax[0, 1].plot(lim, lim, "k--", lw=1, label="cue-locked (slope 1)")
    ax[0, 1].set_title(f"Response(peak) time vs cue time  — slope {slope:+.2f}")
    ax[0, 1].set_xlabel("cue onset"); ax[0, 1].set_ylabel("output peak time"); ax[0, 1].legend()

    # cue-aligned mean activity for the motor-gate chain
    W = 200
    chain = ["Cortex", "D1", "SNr", "Thalamus", "Medulla"]
    for a in chain:
        x = xs[AREAS.index(a)]
        aligned = []
        for i in range(B):
            c = cue_on[i]
            if c - 40 >= 0 and c + W < T:
                aligned.append(x[i, c - 40:c + W].mean(-1))
        m = np.mean(aligned, 0)
        ax[1, 0].plot(np.arange(-40, W), m, label=a)
    ax[1, 0].axvline(0, color="r", lw=1)
    ax[1, 0].set_title("Cue-aligned mean activity — motor-gate chain")
    ax[1, 0].set_xlabel("time from cue"); ax[1, 0].set_ylabel("mean rate"); ax[1, 0].legend()

    ax[1, 1].bar(range(len(AREAS)), [area_mean[a] for a in AREAS])
    ax[1, 1].axhline(0.05, color="r", ls=":", lw=0.8); ax[1, 1].axhline(0.95, color="r", ls=":", lw=0.8)
    ax[1, 1].set_xticks(range(len(AREAS))); ax[1, 1].set_xticklabels(AREAS, rotation=60, ha="right")
    ax[1, 1].set_title("Mean activity per area (red = dead/sat lines)")

    fig.suptitle("noSCnoSTN cue-drive diagnostic (naive pavlovian, 0.07 ceiling)", y=1.0)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / "cue_drive_diagnostic.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
