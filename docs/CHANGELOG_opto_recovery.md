# Change log: recovering D1/D2 opto sensitivity in cbt_loop_noSCnoSTN

**Goal.** Get `cbt_loop_noSCnoSTN` to show what the precursor repo `~/Documents/CodeVault/cbtModel`
(baseline commit `12a8ade`) showed:
- it solves the self-timed task through the circuit;
- dSPN stimulation speeds timing and iSPN stimulation slows it;
- inhibiting each population has the opposite effect.

Keep the more realistic circuit of `cbtModels` while doing it. Newest entries go first. Each entry
says what changed, why, and what has and hasn't been verified.

---

## 2026-10-27: `.json` → `<family>/data/`, `.log` → `<family>/logs/`

**Change.** Only `cbt_loop_noSCnoSTN` had loose files: 9 `.json` and 13 `.log`, moved with
`git mv`.

- **`data/`:**
  - `seed_stability_*.json` (4 variants plus `seed_stability_data.json`)
  - `block_swap_data.json`
  - `det_rep{1,2}.json`
  - `seed_bal19.json`
- **`logs/`:** every training and testing stdout capture (`train_supervised_thal_*.log`,
  `train_desat*.log`, `more_seeds.log` and so on).
- **Not moved:** `.json` and `.log` files inside `tests/<name>/` and `runs/<name>/`, which are
  self-contained, and the editor settings in `.claude/` and `.vscode/`.
- **`config_script`:** gains `cfg.data_dir` and `cfg.logs_dir`.

**Code fixed.** Four root scripts read or write the JSON next to themselves; they now use
`data/`:
- `block_swap.py`
- `seed_stability.py`: the default `DATA`, and its `--data` alternate-file option, which is how
  `det_rep*`/`seed_bal19` were written;
- `analyze_init.py`
- `analyze_grad.py`

No code reads the logs; they were shell redirects.

**Verified.** All four scripts compile, and every JSON file they reference exists at its new
path.

---

## 2026-10-27: training and testing scripts moved into `<family>/{training,testing}/`

**Change.** 55 scripts across all five families were moved with `git mv`:
- **to `training/`:** `train_*.py`, `training_script.py`, `pretrain_*.py`, `long_run.py`, and
  corticothalamic's `pci_train.py`;
- **to `testing/`:** `testing_script.py`, `test_*.py`, `opto_script.py`, `lesion_tests.py`, and
  corticothalamic's `eval_*.py`.

**Stayed in the family root:**
- the model files and `plotting_functions.py`;
- orchestrators: `run_pipeline.py`, and noSC's `train_and_test.py`, which drives both training and
  testing;
- analysis, sweep and diagnostic scripts (`seed_stability`, `block_swap`, `analyze_*`,
  `diagnose_*`, the corticothalamic sweeps and so on).

**Import fixes.**
- Each moved script got a bootstrap after its docstring that puts the family root, `training/` and
  `testing/` on `sys.path`. Usage lines that ran the script by its bare name now say
  `training/…` or `testing/…`.
- File-relative paths that meant "the family root" were changed to `parents[1]`:
  - `HERE` in `train_ported_{curriculum,hybrid}.py` and `pci_train.py`;
  - `ROOT` in `eval_{hybrid,selftimed}.py`;
  - the params path in `test_vanilla.py`;
  - `test_run.py`'s path insert.
- Importers that stayed put:
  - `run_pipeline.py` (noSCnoSTN, noSC) and `train_and_test.py` got the same bootstrap;
  - `cbt_loop/run_pipeline.py`'s package-style imports now read `cbt_loop.training.*` and
    `cbt_loop.testing.*`;
  - three `tests/` scripts and both `run_testing_script.py` files add `training/` and `testing/`
    to their path;
  - `corticothalamic/rho_pci_sweep.py` finds `pci_train` in `training/`.

**Why.** At the user's request: keep the family folders readable.

**Verified.**
- A static import check over 125 scripts (every moved script, every root script, and the `tests/`
  scripts) rebuilt each script's runtime `sys.path` and resolved all its imports. Everything
  resolves except:
  - six corticothalamic scripts that add the target family's folder from a command-line argument
    at runtime, which a static check cannot follow;
  - two `import cbt_loop.config_script` lines in `cbt_loop/testing/{lesion_tests,test_pavlovian}.py`.
    These were **already broken**: that file was deleted when the configs were centralized, and
    the move did not cause it.
- Two real runs: `python training/train_supervised_thal.py --help` imports the model, config and
  task, and `testing/testing_script.py` loads with `test_pnr`, `opto_script` and
  `plotting_functions`.
- Not run: the training or testing scripts end to end.

---

## 2026-10-27: pkls moved into `<family>/params/`

**Change.** Every pkl that sat loose in a family root has moved into that family's `params/`
folder: 48 files across all five families, plus the three `pkl file save/` backup folders, now at
`params/pkl file save/`. Tracked files were moved with `git mv`, so history follows them.

- **Not moved:** pkls inside `tests/<name>/` and `runs/<name>/`. They follow the self-contained
  folder rule.
- **`config_script`:** gains `params_dir(family)`. `for_family()` creates the folder and exposes
  `cfg.params_dir`, and all `cfg.*_params_path()` helpers resolve there. Code that derives a
  sibling file via `.with_name()` or `.parent / name` therefore follows automatically: for
  example `train_supervised_thal.py`'s tagged outputs, `testing_script`'s bundle and
  `pd_exp_params.pkl`.
- **Hard-coded paths fixed in 17 scripts:**
  - `vanilla_rnn/{test,plot}_vanilla.py`
  - `corticothalamic/{pci_train,eval_hybrid,eval_selftimed,pka_timer_probe}.py`
  - in noSCnoSTN: `train_ported_{hybrid,curriculum}.py`, `port_pci_loop.py`,
    `diagnose_cue_drive.py`
  - tests: `output_sharpness`, `pka_stability`, `seed_and_adenosine/evaluate_runs`,
    `ramping_metrics/run_ramping_report`
  - `cbt_loop/tests/plot_{area_outputs,selftiming,dspn_compare}.py`

  Scripts that take a bundle *name* argument (`eval_*`, `pka_timer_probe`, `plot_area_outputs`,
  `plot_selftiming`) now look it up in `params/`.

**Why.** At the user's request: keep the family folders readable. noSCnoSTN alone had 25 loose
pkls.

**Verified.**
- For every family, each `cfg.*_path()` that had a file before the move finds it after.
- Every hard-coded target exists.
- All 18 edited files compile.
- The moved files are unchanged, since `git mv` only renames.

---

## 2026-10-27: family refactor, step 2 — noSCnoSTN rebuilt as the modular core

**Decisions since step 1.**
- **Init keys:** at the same seed the core weights start identical in every family.
  - Core blocks keep their existing index-based keys (`split(rng_key, 60)`), so this step can
    reproduce the old weights exactly.
  - STN and SC weights will get keys derived from their weight names, so adding an area cannot
    shift a core weight.
  - Per-step noise follows the same rule: the core keeps `split(step_rng, 11)` and modules will
    use separate derived keys.
- **Folders:** they will be renamed as well as the files (step 6).

**Change.** `init_params` and `multiregion_rnn` moved from `cbt_loop_noSCnoSTN/cbt_rnn.py` into
`/cbt_core.py`:
- **`init_core_params(rng_key, n_input, family)`:** the old body moved by text extraction, not
  retyped. The only changes are `_FAMILY` → `family` and the module aliases.
- **`multiregion_rnn`:** rewritten as
  - `prepare_core_weights` — effective weights, pacers and constants, built once per trial;
  - `core_initial_state`;
  - one function per area (`_cortex`, `_thalamus`, `_snc`, `_neuromodulators`, `_pka`,
    `_striatum`, `_gpe`, `_snr`, `_medulla`, `_readout`), run in `CORE_AREA_ORDER` on a named
    state dict.
- **`W["ext"]` hooks:** each area adds any registered module currents just before its
  nonlinearity. The medulla uses the `x_med_E` hook, for the E units only. All hooks are empty in
  the core; STN and SC will register here.

The family file is now about 100 lines: state orders, one-line wrappers, `rnn_func` and the
helper bindings.

**Kept exactly, and now documented in the code:**
- The update order is part of the model: thalamus reads this step's cortex, SNc reads last step's
  D1/D2, and adenosine release reads this step's SNc/cU but last step's D1/D2.
- D1, D2 and medulla recurrence read the state *after* this step's leak:
  `J @ ((1−1/τ)x)`, not `J @ x`.
- Every area's t=0 noise uses the same `init_key`, so same-shaped areas get identical jitter.
- GPe recurrence `J_gpe` is built but not used.
- D1's gain is also applied inside its nonlinearity; D2's is not.
- The D2 gain pair is swapped.

**Dropped:** the unused `k_a` computation and the unused `pka_clip_eps` read inside the forward
pass. The `k_a` parameter itself is still initialized.

**Verified (CPU, exact `array_equal`):**
- The step-1 snapshot (274 arrays, all three families): 0 differences.
- A direct comparison of the old (`HEAD`) and new noSCnoSTN modules over 8 configurations, each
  with and without random opto stimulation:
  - configurations: default, thalamic readout, noise 0.1, `pin_ado`, `pin_snc`+`pin_ctx`,
    `pin_pka`, `nt_mode` mass_action, `pka_saturation` linear;
  - init params and runtime config identical;
  - **forward pass identical in all 16 cases**;
  - **gradients identical in 14 of 16**.
- Under `pin_pka_d1/d2` the gradients differ at float32 rounding level only: at most about 1.3e-7
  relative, no NaNs, forward pass still exact. The likely cause is that pinning makes the PKA
  computation dead code, and XLA then orders some backward-pass sums differently.
  - Training is chaotic, so a `pin_pka` training run will not retrace a pre-refactor one.
  - Unpinned training, which is what `train_supervised_thal.py` does, is bit-identical.
- Every `cbtl.<attr>` used by noSCnoSTN scripts still exists.

---

## 2026-10-27: family refactor, step 1 — shared helpers moved to `/cbt_core.py`

**Plan agreed with the user.** The three CBT families become one shared core plus area
modules. `cbt_loop_noSCnoSTN` is the core and the reference for every shared definition.
- **New names:** `cbt_simple` (now noSCnoSTN), `cbt_STN` (now noSC), `cbt_SC_STN` (now cbt_loop).
- **Structure:** STN and SC are added to the core as modules. The core is not a full circuit with
  areas switched off.
- **Weights:** each family keeps its own weights, at init and after training, learned over its
  whole circuit.
- **Mechanisms:** `bg_nln` is kept. The PKA-gate path (`pka_gate_*`, `pka_thresh`) will be removed
  when noSC and cbt_loop are rebuilt on the core.
- **D2 gain sign:** stays as it is for now and gets its own fix later.
- **Still undecided:**
  - init keys per weight name, shared across families or salted by family;
  - whether to rename the family directories too.

Steps:
1. shared helpers (this entry);
2. rebuild noSCnoSTN as a modular core that reproduces it exactly;
3. STN module → `cbt_STN`;
4. SC module → `cbt_SC_STN`;
5. thin `cbt_rnn.py` re-export shims for the 93 importers;
6. renames.

**Change.** New `/cbt_core.py` holds the one definition of:
- `exc`, `inh`, `nln`, `bg_nln` (these still delegate to `self_timed_movement_task`, whose losses
  use them);
- `no_autapse`, `match_input_channels`;
- `init_state_mask`, `freeze_init_state_optimizer`;
- `evaluate`, `get_brain_area`, `get_response_times_opto`, `get_d1_d2_ratio`, `get_slope`.

Helpers that need family data take it as an argument: `batched_rnn`, `STATE_AREA_ORDER` or
`INIT_STATE_KEYS`. Each family's `cbt_rnn.py`:
- re-exports the wrappers, so `cbtl.exc` and similar still work;
- binds the analysis helpers in one-line wrappers with unchanged signatures;
- keeps its family-specific helpers (`ciel_floor` in cbt_loop, `bg_nln_inh` in noSC).

The three files went from 851/816/964 to 717/674/820 lines.

**Why the name is `cbt_core`, not a top-level `cbt_rnn.py`.** Each family directory keeps a
`cbt_rnn.py`, and scripts run from that directory put it first on `sys.path`. A top-level module
with the same name would be shadowed, and the family file could not import it.

**Behaviour.** No change intended. Every family's copies already matched noSCnoSTN apart from
docstrings and whitespace.

**Verified.** A CPU snapshot of all three families was compared before and after: 274 arrays,
0 differences, compared exactly with `array_equal`. They cover:
- init params at seed 0;
- the `rnn_func` forward pass, including every state trace;
- `evaluate` outputs and actions;
- `get_d1_d2_ratio`, `get_slope`, `get_response_times_opto`, `get_brain_area`;
- the wrappers themselves.

Snapshots must be taken on CPU (`JAX_PLATFORMS=cpu`). On GPU, two runs of *unchanged* code
differed in 3 noSC arrays, from non-deterministic reductions. `freeze_init_state_optimizer` was
also checked: the 15 init-state keys get zero updates and weights still update.

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
