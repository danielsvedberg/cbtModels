# cbtModels

Cortico-basal-ganglia-thalamic (CBT) RNN models of self-timed movement (STMT). Five families
share one central config; **`cbt_loop_noSCnoSTN` is the active one** (no superior
colliculus, no STN).

The intent is to model the CBT circuit as a recurrent neural network (RNN) that learns to perform a self-timed movement task (Hamilos...Assad, eLife 2021). 
The RNN is trained to produce a brief action in response to a cue after a delay. Specifically, the model is designed to test how 
dopamine's theorized neuromodulation of D1 and D2 striatal neurons' excitability (Laheri, Bevan, Neuron 2024;
Bonnavion...Giros, Nat. Neurosci. 2024; Ma...Zhong, Nature 2022), affects circuit dynamics and movement timing in the STMT.
Specifically, we hypothesize that dopamine increases the excitability of D1 neurons and decreases the excitability of D2 neurons, 
and that this drives the direct pathway to produce movement at the correct time.

In order to be a valid model, it must behave in a biologically plausible manner that is concordant with prior literature:
neurons in cortex and striatum must ramp up before movement (Yang...Inagaki, Nature 2026), neurons in SNR must ramp up before movement (Hou...Assad, BioRxiv 2025),
and neurons in SNc (and dopamine concentrations) must ramp up before movement (Hamilos...Assad, eLife 2021).
The model must also be able to reproduce the effects of optogenetic inhibition of D1 and D2 neurons on movement timing:
inhibiting D1 neurons should delay movement, while inhibiting D2 neurons should advance movement (Yang...Inagaki, Nature 2026).

## Output modes, and what "movement" means

The model has two readout modes, selected by the runtime key `readout_source`:

- **Thalamic mode** (`"thalamus"`, `train_supervised_thal.py`). The readout is a
  positive-weight vector `C_thal` off the thalamic excitatory relay pool, trained to match
  a plateau target. **Movement is the moment the output crosses above 0.5.** This mode is
  *easier to train* because there are fewer steps between the striatum — the focus of the
  model — and the output.
- **Medulla mode** (`"medulla"`, the REINFORCE path). Movement is defined by the REINFORCE
  implementation (sampled action within the reward window). This mode is *more faithful to
  the biology*, at the cost of being harder to train.

Both are legitimate; pick per experiment and say which one a result came from. A number
from one mode is not comparable to a number from the other.

## Judging the biological criteria

**"Ramps up before movement" is deliberately open.** It means roughly: *the average
population activity increases between a short period after the cue and a short period
immediately before the movement.* Do not collapse it into one hard threshold — the criterion
is qualitative on purpose.

To make it reportable without over-formalizing it, `/ramping_metrics.py` computes **three
progressively stricter levels per area**. Report them together; they are meant to be able to
disagree, and where they disagree is the finding.

| level | what it asks | numbers |
|---|---|---|
| **1 — raw increase** | Is there simply more activity before movement than before the cue? | `delta` = mean over `[move−10, move)` minus mean over `[cue−10, cue)` |
| **2 — ramping index** | Is the rise **gradual** across cue→movement, or a **sharp jump** just before movement? | `linearity` = per-unit correlation with a linear template (1.0 = clean ramp); `earliness` = AUC of the min-max normalized trace (0.5 = linear, <0.5 = late/sharp, >0.5 = early-then-plateau); `frac_ramping` = fraction of individual units with linearity > 0.5, which catches a population average that looks like a ramp but is built from heterogeneous steps |
| **3 — temporal scaling** | Are the cue→movement dynamics **timing-invariant** (they stretch with the interval) or **timing-dependent** (a fixed absolute time course)? | trials aligned three ways — `scaled` ([cue, move] warped to [0,1]), `cue_locked`, `move_locked` — and scored by how much variance one common template explains. `scaling_index = EV(scaled) − max(EV(cue), EV(move))`; **> 0 = scales with the interval, < 0 = locked to absolute time** |

Run it with `cbt_loop_noSCnoSTN/tests/ramping_metrics/run_ramping_report.py`.

**Level 3 needs a range of cue→movement intervals to mean anything, and the way to induce
that jitter is the noise level** — training `noise_std = 0.05` plus `TEST_CONFIG` 0.10.
Measured latency at cue onset 200: 287.4 ± 9.8 steps at 0.05, 282.5 ± 20.8 at 0.10, with
**zero** spurious threshold crossings at either. That is the protocol the report runs by
default. Two things that look like they would widen the spread but must not be used:

- **Sweeping cue onset adds no interval spread at all** — it shifts cue and movement
  together, leaving `move − cue` unchanged. It only inflates `n`.
- **noise 0.15 corrupts the metric.** It looks like more spread (sd 42.8, range 0–421) but
  ~1% of trials cross threshold at t=0 — detection failures, not timing jitter. Including
  them once flipped every area's `scaling_index` from positive to negative, i.e. it inverted
  the conclusion. The report now drops trials outside 0.4–2.5× the median latency and says
  how many.

With the correct protocol (spread ~40% of mean) `scaled` beats both absolute alignments in
10 of 11 areas — the dynamics do stretch with the interval, modestly (+0.01 to +0.04). GPe is
the exception and is movement-locked.

Caveat that still stands: the model is trained on a **single** delay, so this tests whether
trial-to-trial timing jitter comes with proportionally stretched dynamics — not the full
multi-interval scaling experiment, which would need training on several delays.

### The SNr conundrum — do not "fix" it

Naive basal-ganglia circuitry says SNr should *pause* to release the thalamus for movement,
and since the thalamic readout literally reads thalamus, an SNr ramp looks like it should
suppress the very signal that marks movement. That reasoning is wrong about the biology.

The cited SNr work (Hou…Assad) finds that the **vast majority of SNr neurons increase their
firing in anticipation of, and during, specific movements**. Those modulations are also not
patterned the way the action-selection model of the basal ganglia predicts: the SNr neurons
activated peri-movement are **movement-specific and movement-locked**. How and why this
happens is a genuine open conundrum in the field — it runs against what the circuitry would
naively predict.

So: an SNr ramp-up before movement is a **target to reproduce, not a bug to remove**. If SNr
activity and the thalamic readout seem to be in tension, that tension is the scientific
point, not an error to engineer away.

## Config

`/config_script.py` is the single source for all families — `config_script.for_family(name)`
returns a namespace (`RNN_CONFIG`, `TASK_CONFIG`, `SUPERVISED_THAL_CONFIG`, `params_path()`,
`plots_folder`, …). There are **no per-family config files**. Family-specific values live in
that file's `extra_rnn` / `extra_runtime` override blocks, which win over the shared defaults.

`for_family()` returns a **fresh namespace per call**, so `testing_script`, `plotting_functions`,
`opto_script` and `test_pnr` each hold their own copy. Redirecting output means patching all of
them (and `test_pnr.PNR_DIR`, computed at import).

## Judging results — read this before reporting any run as working

This model has a degenerate solution for almost every metric. Several have been hit:

- **Loss is not evidence.** A constant output is near-optimal under the weighted level MSE.
  At the 0.333/0.666 band the trivial constant is 0.3497, only 0.0167 above `target_lo`, so a
  collapsed run reads as a clean baseline hold.
- **Reward is not evidence.** A cue-ignoring fixed response time scores ~87% on STMT.
- **Separation** (in-window mean − off-window mean) is the shape metric.
- **Slope** is the self-timing metric: regress absolute response time on cue onset over a
  range of onsets *wider than `TASK_CONFIG["t_start"]`*. Slope ≈ 1.0 is cue-locked; ≈ 0.0 is a
  fixed trial time. Verified self-timing (commit `403033c`): slope 0.991, r 0.997, latency
  285.6 ± 8.3 steps from cue.
- **Plot the waveform before believing any scalar.** A run once satisfied a rise-onset metric
  at 95% while producing no step at all — it had learned a free-running ~68-step oscillator.

## Verified traps

- **`bg_nln` is dead code** in noSCnoSTN — defined at `cbt_rnn.py:38`, never called. Comments
  at lines 370/552/573/743/748 still describe it. PKA reaches the striatum only as gains
  `gain_e = pka + 0.5` and `gain_i = 1.5 − pka` on the SPN input currents.
  **In this repo comments outlive the code they describe — check a function is called.**
- **PKA fixed point.** With mass-action and no per-step squash, `pka* = P/(K + P)` where
  `K = tau_rise/tau_fall × pka_max` (= 0.1 at 50/500). Half-max at `P = K`; `pka* = 0.5`
  ⟺ `P = 0.1`. A per-step squash destroys the `tau_fall` timescale.
- **`exc`/`inh` have hard dead zones.** Both are `±clip(tanh(w), 0, None)`: for `w ≤ 0` they
  return exactly 0 with exactly 0 gradient. Nothing is negative at init (0 of 3650 wrapped
  weights), so anything dead was put there by training.
- **`supervised_loss` clips `ys` to `[1e-7, 1−1e-7]`** before scoring. With the unbounded
  linear readout, any output above 1 is scored as 1 with **zero gradient** — a silent ceiling.
- **`balanced_target_rho = 1.107` is a DC fix, not a criticality target.** `normalize_loop`
  scales the 17 loop blocks but not `B_snr_t_exc`, so at rho 1.0 the thalamus rectifies to
  zero on ~95% of steps. The number that governs stability is the **effective** rho* from
  `J = diag(nln'(z))·M` (`nln'(z) = 1 − x²` for an active unit), ≈ 0.90 at init.
- **Training drives rho* upward** — 0.912 → 1.083 over 10k iters in one run, crossing into a
  Hopf bifurcation whose complex pair produced the oscillator above. Nothing in the loss
  constrains rho. Recompute it on *trained* params; the init value says nothing.
- **`prod_d1` is rectified to zero at many inits.** `x_ado` is pinned (0.256) so the A1R brake
  `m_a1·x_ado ≈ 0.0339` is seed-independent, while the DA term spans 0.024–0.057 across seeds
  because SNc's net drive is a small residual of large opposing terms. Whether D1 has any
  PKA drive at all is close to a coin flip over init seeds.
- **The REINFORCE path is currently broken** by the unbounded linear readout:
  `cbt_rnn.evaluate` does `jr.bernoulli(p=ys)` and `loss_type="bce"` takes `log(ys)`, neither
  valid for `ys > 1`. Only the mse/supervised-thal path is safe as written.

## Running experiments

- `train_supervised_thal.py` uses **fixed seeds every run** (`SEED_CONFIG = {task_seed: 13,
  train_seed: 4}`); repeated runs are the same run, not independent samples. There is no
  `--seed` flag.
- Runs are **chaotic** — a ~1e-9 perturbation flips outcomes — and GPU reductions are
  non-deterministic, so n=1 config comparisons are coin flips. Budget matters more than
  learning rate.
- **In any scratch training loop, split the rng INSIDE the loop.** A fixed key set outside it
  shows the network one frozen noise trace, which it memorizes: a run once hit separation
  0.328 on the training seed and 0.019–0.074 on every other seed at the same noise level.
  `fit_rnn_supervised` already does this correctly. Always re-evaluate on held-out seeds.

## Tests and plots

Each test is a **self-contained folder** `<family>/tests/<test_name>/` holding its script, its
plots, and a docstring explaining what it tests and what the answer was. Never write test plots
to the family root or a shared `plots/` directory.

## The `docs/` folder — read, but check currency first

`docs/` holds ~5.7k lines written mostly while the **medulla/REINFORCE** output was the
target. Much is still valuable; some describes code that no longer exists. Before relying on
any claim there, check it against the current code.

**Substantive model-findings (2026-07 → 08) — worth reading:**

| doc | still current | stale in these respects |
|---|---|---|
| `criticality_and_timescales.md` | membrane τ=7 ≈ 20 ms EPSP decay; the three nested timescales (neuron ~20 ms / loop ~300 ms / PKA clock); rho\* is what matters, not rho_lin | assumes `nln = sigmoid(4(x−0.5))`, so its gain formula `g = 4r(1−r)` is wrong for the current `max(0,tanh)` (use `g = 1−x²`); quotes `tau_pka_fall=900` (now 500); "optimal rho_lin=1.0" is the shared default — noSCnoSTN overrides to 1.107 for a DC reason |
| `self_timing_findings.md` | self-timing rides the slow PKA clock, not loop memory; it is a fragile seed-gated basin; judge by slope, not reward | the enabling pieces it lists (PKA→`bg_nln`, biased-sigmoid readout) are gone; its curriculum (`train_hybrid` → `train_from_hybrid`) is the medulla path, not `train_supervised_thal` |
| `neuromodulator_model.md` | the DA/adenosine ↔ D1/D2 opponent sign scheme; why the mass-action throttle beats a per-step squash | PKA is no longer fed to `bg_nln`; `m_a1_cap=0.08` and PKA init clamp [0.4,0.6] are now 0.25/0.75; `tau_da=20`/`tau_ado=200` are now 50/50; **adenosine is currently pinned (`pin_ado=0.256`), not dynamic** |
| `noise_and_criticality.md` | noise is a fluctuation-*amplitude* knob, not a criticality knob; critical slowing peaks at the operating point | measured on the corticothalamic testbed under the sigmoid `nln` |
| `parameter_findings.md` | why spectral normalization was needed at all; the `log_reward` objective fix | describes the `nln` that was current in 2026-07 |

`parameter_findings.md` §1 carries a standing warning that has since fired a second time:
*"Always check `def nln` before analysing — this changed once and may change again."* It did:
`max(0,tanh)` → `sigmoid(4(x−0.5))` → back to `max(0,tanh)`. Treat every nonlinearity claim in
`docs/` as needing a re-check.

**Historical only (2026-07-17 set):** `START_HERE`, `QUICK_REFERENCE`, `VISUAL_GUIDE`,
`TRAINING_GUIDE`, `DESIGN_NOTES`, `IMPLEMENTATION_*`, `README_*`, `MASTER_SUMMARY`,
`COMPLETION_SUMMARY`, `CHECKLIST_AND_SUMMARY`, `SELF_TIMED_TASK_LOSS_UPDATE`, `LOSS_*`,
`INDEX`. These document a removed API — `batched_rnn_loss` no longer exists — and a
two-component loss that is not the current objective. Useful for history, not for how the
code works now.
