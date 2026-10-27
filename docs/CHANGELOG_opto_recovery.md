# Change log: recovering D1/D2 opto sensitivity in cbt_loop_noSCnoSTN

**Goal.** Get `cbt_loop_noSCnoSTN` to show what the precursor repo `~/Documents/CodeVault/cbtModel`
(baseline commit `12a8ade`) showed:
- it solves the self-timed task through the circuit;
- dSPN stimulation speeds timing and iSPN stimulation slows it;
- inhibiting each population has the opposite effect.

Keep the more realistic circuit of `cbtModels` while doing it. Newest entries go first. Each entry
says what changed, why, and what has and hasn't been verified.

---

## 2026-10-27: discrepancy audit of cbtModel vs noSCnoSTN (no code change)

These are the candidate causes of the lost opto sensitivity, ranked by expected impact. They come
from reading the code only; none has been tested causally yet.

1. **Objective.**
   - cbtModel used a self-referential target: early moves are penalized and late moves are free.
   - noSCnoSTN used a fixed plateau at cue+300, weighted 19×. That trains timing to resist being
     shifted. *(Addressed below.)*
2. **No cortex→thalamus pathway in cbtModel.**
   - There, all output passes through D1/D2.
   - noSCnoSTN has `B_cU_t_exc` and `B_cU_t_inh`, plus a cortico-thalamic loop normalized to
     rho 1.107. That loop can time the task while bypassing the basal ganglia.
3. **Striatum→readout path.**
   - In cbtModel, D1 excites thalamus and D2 inhibits it directly (`B_tbg` sign-split).
   - In noSCnoSTN the path is D1⊣SNr⊣Th and D2⊣GPe⊣SNr⊣Th. With SNr mostly silent, D1 stimulation
     cannot disinhibit thalamus any further.
4. **The D2 gain sign was swapped in commit `403033c`**: `gain_ed2 = 1.5 − pka_d2`
   (`cbt_rnn.py:742-744`).
   - DA lowers `pka_d2`, so DA *raises* D2 excitability. That is the opposite of the hypothesis.
   - Check whether the swap was intended.
5. **D1/D2 asymmetry.**
   - D1 is updated with `x_d1 = nln(gain_ed1 * x_d1)`; D2 has no gain inside its nln.
   - Because the stored state is the rate, the gain compounds into D1's leak.
6. **Smaller differences:**
   - opto enters multiplied by PKA gain (about 0.74× at rest);
   - stimulation is weaker relative to the operating point;
   - time constants: 7/10 vs 15/40;
   - DA can rectify to exactly 0 here, whereas cbtModel's sigmoid DA never does;
   - training length: about 800 effective iterations in cbtModel vs 10k here.

**Documentation found stale during the audit:**
- `stmt.exc`/`inh` are now `±abs(w)`, not `clip(tanh)`.
- The thalamic readout *does* apply `out_gain`/`out_bias` (`cbt_rnn.py:794`), despite the config
  comments.
- `train_supervised_thal.py` *does* have a `--seed` flag.

The two `CLAUDE.md` claims were corrected the same day.

---

## 2026-10-27: self-referential supervised target (option 1)

**Change.**
- `self_timed_movement_task.supervised_loss` and `fit_rnn_supervised` take two new arguments:
  `self_referential` (default `False`) and `selfref_threshold` (0.5).
- `SUPERVISED_THAL_CONFIG` sets `self_referential: True`, `selfref_threshold: 0.5`.
- `train_supervised_thal.py` passes both through. While `self_referential` is on, it skips
  `in_window_weight`.

**What it does.** This is the cbtModel `batched_nm_rnn_loss` rule:
- the hi window (`hold` steps long) opens at the output's first crossing of the threshold;
- an earlier crossing, or no crossing at all, puts the window at `t_open` (cue+delay);
- everywhere else the target is `lo`.

`batch_targets` now only supplies `t_open`, the band and the window length. The placement is
computed on `stop_gradient(ys)`. `aux["resp_frac"]` gives the fraction of trials that cross at or
after `t_open`, and the training log prints it.

**Why.**
- The fixed plateau penalizes a late response as much as an early one. Training against it gives
  a precisely pinned latency (285.6 ± 8.3), and that same pinning resists the shift opto is
  supposed to cause.
- cbtModel's one-sided objective leaves response time free after cue+300. It is the only version
  in which D1/D2 opto moved timing in the right directions.
- The 19× weight was tuned to the fixed window, and keeping it would re-pin timing to cue+300, so
  it is off under this mode.

**Cost.** None measurable. One jitted gradient step at init took 535.3 ms with the fixed loss and
536.2 ms with the self-referential one (GPU, batch 100, T 1000). cbtModel's training felt slow
because it converges slowly: the loss bounces off the never-move plateau. Each step is not
expensive.

**Unchanged.**
- The default is `False`, so every other caller is bit-identical. A synthetic check gave the same
  fixed-mode loss as before.
- Accuracy and `in_window`/`off_window` are still scored against the *fixed* plateau, so separation
  numbers stay comparable with earlier runs.
  - Under this objective a late-but-valid response shows up as low separation. Judge runs by
    `resp_frac`, the latency distribution and opto, not by separation alone.
- The band stays at 0.333/0.666; cbtModel used 0.25/0.75.

**Verified.**
- A synthetic test confirmed window placement for four cases: an early crossing goes to `t_open`,
  a late crossing goes to the crossing, no crossing goes to `t_open`, an on-time crossing stays
  put. `resp_frac` came out at 0.75 as expected, and the gradient is finite.
- A 400-iteration smoke test of `train_supervised_thal.py` (seed 4, tag `selfref_smoke`) ran end to
  end.

**Not yet working.**
- In that smoke test `resp_frac` stayed at 0.000 and the output was flat at about 0.349. It sat on
  the never-move plateau (loss 0.0054).
- This was expected from cbtModel, whose best checkpoint came at about 800 iterations after
  bouncing on the same plateau. It still needs a full-length, multi-seed run.
- **Watch the band.** At 0.333/0.666 with no weighting, never moving costs only about
  (50/1000)·(0.666−0.349)² ≈ 0.005. With cbtModel's 0.25/0.75 band and T 900 the cost was about
  0.013, roughly 2.5× more pull out of the plateau.
  - If runs stay stuck, try `target_lo/hi = 0.25/0.75` first, before anything else.
