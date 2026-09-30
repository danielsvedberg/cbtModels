"""Evaluate every checkpoint from the seed / adenosine experiment on the metric that decides.

Separation is NOT sufficient: a cue-ignoring fixed response time scores ~87% on STMT. The
decisive metric is SLOPE -- regress absolute response time on cue onset over a range of
onsets wider than TASK_CONFIG["t_start"]. slope ~1.0 = cue-locked self-timing, ~0.0 = fixed
trial time. Reported alongside r, latency, threshold-crossing rate and separation.

Usage:  python evaluate_runs.py
"""
import sys, pickle, pathlib, glob
_here = pathlib.Path(__file__).resolve().parent
_root = next(p for p in _here.parents if (p / "config_script.py").exists())
sys.path[:0] = [str(_root), str(_root / "cbt_loop_noSCnoSTN")]
import numpy as np, jax.numpy as jnp, jax.random as jr
import config_script as cs; cfg = cs.for_family("cbt_loop_noSCnoSTN")
import cbt_rnn as c

t = cfg.TASK_CONFIG; sup = cfg.SUPERVISED_THAL_CONFIG
lo, hi = sup["target_lo"], sup["target_hi"]; THR = 0.5 * (lo + hi); T = t["t_total"]
ONSETS = np.arange(60, 460, 25)          # t_start spans 50-250; go well beyond it
NS = 10


def evaluate(path):
    with open(path, "rb") as f:
        b = pickle.load(f)
    if isinstance(b, dict) and "params" in b:
        params, config = b["params"], dict(b["config"])
    else:
        params = b
        _, config = c.init_params(jr.PRNGKey(4), n_input=1)
        config = dict(config)
    config["readout_source"] = "thalamus"
    rows, seps, n_already = [], [], 0
    for on in ONSETS:
        inp = np.zeros((NS, T, 1), np.float32)
        inp[:, on:on + t["t_cue"], 0] = 1.0
        ys, _ = c.batched_rnn(params, config, jnp.asarray(inp), jnp.zeros((NS, T, 20)),
                              jr.split(jr.PRNGKey(int(on)), NS))
        y = np.asarray(ys)[..., 0]
        w0, w1 = on + sup["delay"], on + sup["delay"] + sup["hold"]
        if w1 < T:
            m = np.zeros(T, bool); m[w0:w1] = True
            seps.append(y[:, m].mean() - y[:, ~m].mean())
        for s in range(NS):
            # A response must be a genuine UPWARD crossing after the cue. Accepting any
            # `post.any()` counts trials that were already above threshold at cue onset as
            # latency 0, which drags the slope down and inflates the sd -- it made the
            # verified reference checkpoint read 0.477 instead of 0.991.
            if y[s, on] > THR:
                n_already += 1
                continue
            post = y[s, on:] > THR
            if post.any():
                rows.append((on, on + int(np.argmax(post))))
    rows = np.array(rows)
    out = dict(n=len(rows), cross_rate=len(rows) / (len(ONSETS) * NS),
               above_at_cue=n_already / (len(ONSETS) * NS),
               sep=float(np.mean(seps)) if seps else float("nan"))
    if len(rows) > 20:
        A = np.polyfit(rows[:, 0], rows[:, 1], 1)
        lat = rows[:, 1] - rows[:, 0]
        out.update(slope=float(A[0]), r=float(np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]),
                   lat=float(lat.mean()), lat_sd=float(lat.std()))
    else:
        out.update(slope=float("nan"), r=float("nan"), lat=float("nan"), lat_sd=float("nan"))
    return out


paths = sorted(glob.glob(str(_root / "cbt_loop_noSCnoSTN" / "params_supervised_thal_seed*.pkl")))
paths.append(str(_root / "cbt_loop_noSCnoSTN" / "params_supervised_thal.pkl"))   # positive control
print(f"threshold {THR}, cue onsets {ONSETS[0]}-{ONSETS[-1]} (t_start spans "
      f"{int(np.min(t['t_start']))}-{int(np.max(t['t_start']))}), {NS} seeds each\n")
print(f"{'run':<24} {'slope':>7} {'r':>7} {'latency':>9} {'sd':>6} {'cross%':>7} {'hi@cue':>7} "
      f"{'sep':>8} {'%band':>6}  verdict")
print("-" * 108)
for p in paths:
    if not pathlib.Path(p).exists():
        continue
    nm = pathlib.Path(p).stem.replace("params_supervised_thal", "").lstrip("_") or "REFERENCE(verified)"
    try:
        e = evaluate(p)
    except Exception as ex:
        print(f"{nm:<24} FAILED: {ex}")
        continue
    if e["n"] <= 20:
        v = "no crossings"
    elif e["slope"] > 0.8 and e["r"] > 0.9:
        v = "SELF-TIMED"
    elif e["slope"] > 0.4:
        v = "partial"
    else:
        v = "FIXED-TIME (degenerate)"
    print(f"{nm:<24} {e['slope']:7.3f} {e['r']:7.3f} {e['lat']:9.1f} {e['lat_sd']:6.1f} "
          f"{100*e['cross_rate']:6.0f}% {100*e['above_at_cue']:6.0f}% {e['sep']:8.4f} {100*e['sep']/(hi-lo):5.0f}%  {v}")
