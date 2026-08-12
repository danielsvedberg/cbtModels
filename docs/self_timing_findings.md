# Self-timing: the recipe, results, and what's fragile

Investigation log, 2026-07-27 → 2026-08-12. Measured with the curriculum
(`train_hybrid --init scratch` → `train_from_hybrid` → `corticothalamic/eval_selftimed.py`)
on the honest task (`t_wait=300`, movement window 100, `t_start∈[50,540)`; cue-ignoring
fixed-time ceiling ≈ 27%).

---

## TL;DR

**All three CBT families self-time** — the response time tracks the cue with slope ≈ 1,
not a fixed time. It required a specific model + a fragile, seed-gated training basin.

| family | config | best slope | in-window | seed |
|---|---|---|---|---|
| cbt_loop | τ=10, PKA=500 | 1.00 | 100% (peak) | 3 |
| cbt_loop_noSCnoSTN | τ=10, PKA=500 (static kₐ) | 1.00 | 100% (crosses 0.5) | (early) |
| cbt_loop_noSC | τ=7, PKA=900, dyn-DA | 0.91 | 82% | 0 |
| cbt_loop_noSCnoSTN | τ=7, PKA=900, dyn-DA | 0.87 | 41% | 4 |

---

## 1. What self-timing requires (the model)

Self-timing is measured by `eval_selftimed.py`: slope of response-time vs cue-time
(→1 = self-timed, →0 = fixed-time), plus `sd(latency) ≪ sd(response_time)`. The
degenerate solution (respond at a fixed time, ignore the cue) scores the 27% ceiling.

The enabling model pieces (see `neuromodulator_model.md`):
- **PKA is the excitability variable** fed directly into `bg_nln` (mass-action-bounded,
  no soft-threshold gate), so its slow trace gates the striatum.
- **A slow PKA clock** (`tau_pka_fall`) — the only variable with a delay-scale timescale.
- **A trainable biased-sigmoid readout** (nonzero resting prob for RL exploration).

## 2. Timing rides the PKA clock, not loop memory

The decisive experiment. Lowering the membrane τ from 10→7 (for biophysical ~20 ms
neurons) shortens the loop memory ~380→~290 ms and **loses self-timing** (slope 0.34→0.02).
But **lengthening the PKA clock recovers it**:

| config | loop memory | self-timed slope |
|---|---|---|
| τ=10, PKA=500 | ~380 ms | 0.34 (partial) |
| τ=7, PKA=500 | ~290 ms | 0.02 (lost) |
| τ=7, **PKA=900** | ~290 ms | **0.91 (seed 0)** |

So the loop (~300 ms) cannot bridge the 3 s delay — the **~9 s PKA molecular clock** does.
There is **no biophysics-vs-timing tension**: fast ~20 ms neurons + slow PKA clock → timing.

## 3. Two trade-offs found while tuning the PKA clock

1. **PKA clock length vs cue-signal strength.** Longer `tau_pka_fall` = longer memory
   *but* higher resting saturation → weaker cue-evoked PKA modulation. `tau_pka_fall=1440`
   (the biophysical 10 s half-life) made the hybrid **unlearnable** (stuck at the silent
   plateau 0.0535); **900** is the learnable middle ground.
2. **Hybrid over-commitment kills self-timing.** More hybrid iters (10k→20k) saturates the
   hybrid (0.86→0.998) but a fully go-cue-committed model can't be reorganized into timing
   (self-timed slope → 0). *Under*-cooked hybrid leaves the plasticity the self-timed stage
   needs. Seen for both cbt_loop and noSC.

## 4. Self-timing is a fragile, seed-gated basin (~1/4–2/5)

Whether a run lands the timing basin depends on the seed. Seed sweeps (τ=7, PKA=900):

| family | self-timing hit rate | hybrid breakout rate |
|---|---|---|
| noSC (dyn-DA) | 1/5 (seed 0) | 3/5 (2 stalled) |
| noSCnoSTN (dyn-DA) | **2/5 (seeds 1, 4)** | **5/5** |

The smaller architecture (no SC, no STN) + dynamic DA is **more robust** on both axes.

**What's "special" about a winning seed?** (measured at fresh init, noSC seed sweep.)
- **Hybrid breakout** correlates cleanly with **cue→cortex drive strength** — the seeds
  that stalled had the weakest cue-evoked cortex response (≈0.24 vs 0.25–0.30).
- **Self-timing** among breakout seeds is only *weakly* separable: the winner has the
  strongest overall cue→(cortex, PKA) drive, but the gap is small. No single init
  statistic predicts it — it's an emergent basin-of-attraction property. `rho_lin` is
  1.000 for all seeds (balanced_init), so criticality is *not* the differentiator.
- Lever this implies (untested): boost the cue→cortex→PKA drive at init (larger `B_cue_c*`
  or `da_pka_gain`) to raise the hit rate.

## 5. Honest caveats

- Best-checkpoint is saved by **lowest loss, not highest reward**; loss and task-reward
  can diverge, so a run can commit a *regressed* checkpoint (seen once). Reward-based
  early-stopping would help.
- The at-target self-timing models often peak *sub-threshold* (output ~0.4, doesn't cross
  0.5) — correctly *timed* but low-amplitude; `eval_selftimed` uses a peak-based fallback.
  noSCnoSTN at τ=10 was the exception (100/100 real threshold crossings).
- `train_seed` is a single shared knob but each family times at a different seed
  (cbt_loop@3, noSC@0, noSCnoSTN@1/4). Committed `.pkl`s are the artifacts; a per-family
  seed is the clean fix (deferred).
