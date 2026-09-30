"""Run the three-level ramping analysis (/ramping_metrics.py) on a trained bundle.

Levels: (1) raw increase pre-cue -> pre-movement, (2) ramping index = is the rise gradual
or a late jump, (3) temporal scaling = do cue->movement dynamics stretch with the interval.
See ramping_metrics.py for definitions and CLAUDE.md for why these criteria exist.

Movement is defined per CLAUDE.md: thalamic mode = first timestep the readout crosses 0.5.

JITTER PROTOCOL. Level 3 needs a spread of cue->movement intervals. The way jitter is
induced in this project is by raising noise_std from the training 0.05 to the TEST_CONFIG
0.10 -- that doubles latency sd (9.8 -> 20.8 steps, range 227-335) with no spurious
threshold crossings. Two things that do NOT work and are deliberately not done here:
  * sweeping CUE ONSET adds no interval spread at all -- it shifts cue and movement
    together, leaving (move - cue) unchanged. It only inflates n.
  * noise 0.15 looks like more spread (sd 42.8) but 1% of trials cross at t=0, i.e.
    detection failures, not timing jitter. Those inflate the spread and corrupt level 3.
A latency sanity filter drops obvious detection failures regardless.

Usage:  python run_ramping_report.py [params.pkl] [--single]
        default   noise 0.05 + 0.10 at one cue onset -- the canonical jitter manipulation
        --single  training noise only (level 3 will be flagged as unidentified)
"""
import sys, pickle, pathlib
_here = pathlib.Path(__file__).resolve().parent
_root = next(p for p in _here.parents if (p / "config_script.py").exists())
sys.path[:0] = [str(_root), str(_root / "cbt_loop_noSCnoSTN")]

import numpy as np, jax.numpy as jnp, jax.random as jr
import config_script as cs; cfg = cs.for_family("cbt_loop_noSCnoSTN")
import cbt_rnn as c, ramping_metrics as rm

src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
      else _root / "cbt_loop_noSCnoSTN" / "params_supervised_thal.pkl"
SINGLE = "--single" in sys.argv
with src.open("rb") as f: b = pickle.load(f)
params, config = (b["params"], dict(b["config"])) if isinstance(b, dict) and "params" in b else (b, None)
if config is None:
    _, config = c.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=1); config = dict(config)
config["readout_source"] = "thalamus"
print(f"bundle: {src.name}   readout_source={config['readout_source']}")

t = cfg.TASK_CONFIG; sup = cfg.SUPERVISED_THAL_CONFIG
THR = 0.5 * (sup["target_lo"] + sup["target_hi"])       # midband == 0.5 for 0.333/0.666
T = t["t_total"]
onsets = np.array([200])                      # onset sweep adds no interval spread
noises = [config["noise_std"]] if SINGLE else [config["noise_std"], cfg.TEST_CONFIG["noise_std"]]
NS = 120

cues, moves, chunks = [], [], []
for ns_ in noises:
    cf = dict(config); cf["noise_std"] = ns_
    for on in onsets:
        inp = np.zeros((NS, T, 1), np.float32); inp[:, on:on+t["t_cue"], 0] = 1.0
        ys, xs = c.batched_rnn(params, cf, jnp.asarray(inp), jnp.zeros((NS, T, 20)),
                               jr.split(jr.PRNGKey(int(on) + int(ns_*1000)), NS))
        y = np.asarray(ys)[..., 0]
        for s in range(NS):
            post = y[s, on:] > THR
            if not post.any(): continue
            cues.append(on); moves.append(on + int(np.argmax(post)))
            chunks.append([np.asarray(x)[s] for x in xs])

cues, moves = np.array(cues), np.array(moves)
lat = moves - cues
keep = (lat > 0.4 * np.median(lat)) & (lat < 2.5 * np.median(lat))   # drop detection failures
if (~keep).any():
    print(f"[filter] dropped {int((~keep).sum())}/{len(lat)} trials with latency outside "
          f"0.4-2.5x the median ({np.median(lat):.0f}) -- spurious threshold crossings")
cues, moves = cues[keep], moves[keep]
chunks = [ch for ch, k in zip(chunks, keep) if k]
if len(cues) < 5:
    raise SystemExit(f"only {len(cues)} usable trials")
names = list(c.STATE_AREA_ORDER)
areas = {}
for k, nm in enumerate(names):
    arr = np.stack([ch[k] for ch in chunks])
    areas[nm] = arr[..., None] if arr.ndim == 2 else arr      # DA/Adenosine are scalar states
print(f"trials: {len(cues)}   cue onsets {sorted(set(cues))}   noise levels {noises}\n")
print(rm.format_report(rm.ramping_report(areas, cues, moves)))
