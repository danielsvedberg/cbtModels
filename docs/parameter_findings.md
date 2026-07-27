# Parameter findings: why the CBT model couldn't learn, and what fixed it

Investigation log, 2026-07-22 → 2026-07-27. Everything below is measured on this
repo with the tools named in each section, not inferred.

---

## TL;DR

Two parameter-level problems, each independently fatal, each now fixed:

| # | Problem | Symptom | Fix | Result |
|---|---------|---------|-----|--------|
| 1 | Cortico-thalamic loop starts **strongly super-critical** (`rho ≈ 1.76`) | Cortex saturates at ~0.99, local gain collapses to ~0.2, **no task gradient** reaches the loop (~0.1% of `‖grad‖`) | Spectral-normalize the loop at init to `rho ≈ 1.0` (`loop_init.normalize_loop`) | Cortex 0.99 → 0.35, gradient to loop 0.1% → 46–83%, cue→output sensitivity ×1000 |
| 2 | Training objective was **dense BCE**, which is nearly blind to the hazard reward | Reward pinned at ~0 regardless of tuning | `objective_mode = "log_reward"` | Hybrid task from scratch: reward **6e-4 → 0.88** |

Both were necessary. Fix 1 supplies a gradient; fix 2 points it at the right target.

---

## 1. The nonlinearity changed underneath the old conclusions

`self_timed_movement_task.py` now uses

```python
nln(x)    = sigmoid(4*(x - 0.5))              # was max(0, tanh(x))
bg_nln(x, b) = sigmoid(c*(x - d)),  c = 3/(1-b),  d = (1/6)*((1-b)/b)
```

Consequences that invalidate earlier reasoning:

* `nln` no longer rectifies and has **no zero-gradient region**, so the old
  "silence trap" (permanent zero gradient once the net goes quiet) cannot occur.
* `nln(0) = 0.119`, so `x = 0` is **not** a fixed point of any area. The old
  "dead cortex" failure mode is replaced by its mirror image: **saturation**.

**Always check `def nln` before analysing** — this changed once and may change again.

---

## 2. Cortex saturates because the loop is super-critical

Tool: `corticothalamic/loop_criticality.py <family>`

The loop update map is `M = (1 - 1/tau) I + (1/tau) W`. Stability at the *actual*
operating point is governed by the Jacobian `J* = diag(g) M`, where
`g_i = nln'(r_i) = 4 r_i (1 - r_i)` (exact for the sigmoid `nln`, so the gain is
read straight off the resting rate).

Raw init, all three CBT families: **`rho_lin = 1.764`** — identical, because they
share the `g_bg/√n` init and homogenized sizes. Activity grows until the sigmoid
saturates; the resulting gain collapse is the *only* thing stabilizing it.

| family | cortex rest | mean gain | `rho*` | `tau_eff` |
|---|---|---|---|---|
| cbt_loop | 0.99 (saturated) | 0.20 | 0.54 | 1.6 steps |
| cbt_loop_noSC | 0.79–0.89 | 0.60 | 0.96 | ~23 steps |
| cbt_loop_noSCnoSTN | 0.99 (saturated) | 0.24 | 0.71 | 2.9 steps |

A linear eigenvalue analysis alone is **misleading here**: it reports "unstable"
while the network is in fact sitting in a strongly stable, signal-dead attractor.

### Why cortex saturates — the current budget

Tool: `corticothalamic/desaturate_sweep.py`

Resting drive into each cU unit (noSCnoSTN):

```
E self  +2.61   E cross +2.57   I inhib -0.82   E thal +0.73   →  net +1.82
```

Excitation outweighs inhibition **~7:1**. Since `nln` saturates for drive > ~1.5
and is maximally responsive at drive ≈ 0.5, a net drive of 1.82 pins cortex at 0.99.

---

## 3. Saturation destroys the task gradient

Tool: `corticothalamic/loop_gradient.py <family>`

At the from-scratch init, on the Pavlovian task:

| family | grad → TC recurrent | grad → cue weights | grad → readout | d(window out)/d(cue) |
|---|---|---|---|---|
| cbt_loop | 0.17% | 0.00% | 98.3% | 1.8e-5 |
| cbt_loop_noSC | 0.13% | 0.00% | 90.5% | 2.4e-4 |
| cbt_loop_noSCnoSTN | 0.15% | 0.00% | 99.2% | 8.6e-6 |

Turning the cue on/off changed the in-window output by **+0.00000**. The cue
cannot perturb a saturated cortex; the perturbation dies in a sub-critical loop
long before the reward window; and the gradient dies retracing the same path.
The optimizer could only tune a **cue-blind constant** — which is exactly the
"dead solution" plateau seen historically.

---

## 4. Fix 1 — spectral normalization at init

Module: `loop_init.py` (shared by all three CBT families).
Config: `balanced_init = True`, `balanced_target_rho = 1.0`.

`normalize_loop()` scales the 17 loop blocks by one factor so `rho(M) = target`.

> **`balanced_target_rho` is a tuning constant, not a trained parameter.** It is
> applied once inside `init_params`, never enters the `params` dict, so the
> optimizer never sees it and there is no gradient w.r.t. it. It sets the starting
> dynamical regime; training then moves the weights and the realized rho drifts.

Effect of `rho: 1.76 → 1.00`:

| quantity | before | after |
|---|---|---|
| cortex rest | 0.99 saturated | 0.34–0.60 |
| loop gain `g` | 0.20 | 0.75–0.92 |
| `rho*` | 0.54–0.71 | 0.89–0.97 |
| `tau_eff` | 1.6–2.9 steps | 9–36 steps |
| grad → loop | ~0.15% | **7–83%** |
| d(window out)/d(cue) | ~1e-5 | **1e-3 – 2e-2** |

**`rho = 1.0` is optimal and the response is non-monotonic** — pushing higher
back-fires, because the loop re-saturates:

| target rho | cortex r | `rho*` | `tau_eff` | cue grad |
|---|---|---|---|---|
| 0.90 | 0.24 | 0.67 | 2.5 | 1.2e-3 |
| **1.00** | **0.35** | **0.96** | **23.2** | **1.6e-2** |
| 1.02 | 0.55 | 0.90 | 9.3 | 2.9e-3 |
| 1.05 | 0.62 | 0.76 | 3.6 | 8.0e-4 |

### What did *not* work
* **Per-row E/I balance** (rescaling each cell's inhibition to exactly cancel its
  excitation). Written for the old rectifying `nln`; it zeroes the net drive and,
  combined with a rectifier, makes `x=0` the only attractor → **dead cortex**.
  Removed.
* **`persistent_self_gain`** (bistable self-recurrence diagonal). Latches
  indiscriminately from baseline noise, so cued and uncued states converge.
  Removed.

---

## 5. Fix 2 — the objective, not the penalty coefficients

The reward is a **hazard model**: the *first* response must land in the window, so
pre-window firing compounds over the ~684-step wait. Dense BCE scores each
timestep independently and cannot represent this:

| pre-window p | BCE contribution | survival → reward |
|---|---|---|
| 0.0163 | 0.0112 | 1.3e-05 |
| 0.0010 | 0.0007 | **0.50** |

Going 0.0163 → 0.001 improves BCE by **0.011** (noise on a ~0.5 loss) but improves
reward **~39,000×**. So BCE has almost no incentive to fix the one thing that
determines reward.

### `silence_coef` cannot substitute (falsified hypothesis)

Sweep on the from-scratch hybrid task (5000 iters; baseline 0.5 @ 10k):

| `silence_coef` | in/out ratio | ratio, cue ablated | pre-window p | reward |
|---|---|---|---|---|
| 0.5 | 51.1 | **12.3** | 0.016 | 6e-4 |
| 2.0 | 12.1 | **1.02** | 0.066 | 0 |
| 5.0 | 1.00 | 1.00 | 0.049 | 0 |

* `2.0` made the solution **genuinely cue-driven** (no cue → no selectivity) but
  did **not** reduce pre-window firing, and reward stayed 0.
* `5.0` **over-suppressed** into a flat cue-blind constant.

Tuning a penalty inside an objective that cannot see the target does not work.

### Result of switching to `log_reward`

From-scratch hybrid, 10k iters, `silence_coef` unchanged at 0.5:

| step | log_reward | reward | pre-window p |
|---|---|---|---|
| 1000 | −74.3 | 0 | 0.107 |
| 5000 | −2.18 | 0.114 | 0.0022 |
| **10000** | **−0.125** | **0.8823** | **0.0002** |

The learned strategy is **hazard-optimal**, and the arithmetic closes:

```
pre-window p = 2e-4   → survival (1−2e-4)^684 = 0.872
in-window  p = 0.054  → P(fire in 300-step window) ≈ 1.000
                        predicted 0.872   vs   observed 0.882
```

Cue ablation (`corticothalamic/eval_hybrid.py`): in/out ratio **278**; ablating the
go cue drops it to exactly **1.00**. The response is entirely cue-driven, with none
of the trial-onset "fire late" heuristic the BCE run had leaned on.

Central default is now `objective_mode = "log_reward"`.

---

## 6. Other parameter findings

**Per-term `nln` deletes inhibition** (was in `cbt_loop_noSCnoSTN`). Each projection
was individually wrapped: `x += nln(J_self @ x) + nln(B_d2d1 @ x_d2) + …`. Since
`nln` maps into (0,1), an inhibitory current became a small **positive** floor
instead of subtracting — `nln(−1) = 0.0025`, `nln(−3) = 4e-5`. Sweeping the
inhibitory current 0 → 3 moved the striatal fixed point by nothing (0.993 → 0.992),
versus 0.99 → 0.17 with correctly summed signed currents. **Fixed**: sum raw signed
currents, one nonlinearity per area.

**Config homogenization.** All five families now load from one root
`config_script.py` (`for_family(name)`); a missing key raises rather than silently
defaulting. Values are homogenized to `cbt_loop` as canonical. Note this changed
pool sizes for `noSC`/`noSCnoSTN`, so their pre-existing `.pkl` bundles are
shape-incompatible and need retraining.

---

## 7. Open items

* **Self-timed stage is the remaining hard problem.** It removes the go cue and
  requires holding a ~310-step interval, but the loop's `tau_eff` is only ~23–36
  steps — an order of magnitude short. The hybrid solution also **ignores the
  preparatory cue entirely** (ablation: identical output), so that pathway starts
  from nothing.
* `rho ≈ 1.0` maximizes gradient flow but not memory. Long intervals likely need a
  structural mechanism (learned attractor, slow variable such as the PKA trace with
  `tau_pka_fall = 1440`, or curriculum), not a larger `rho`.
* `norm_resp_time` drifts toward mid-window during training, trading against the
  brevity term.

---

## Tool index

| tool | question it answers |
|---|---|
| `corticothalamic/loop_criticality.py` | Is the loop sub/critical/super-critical **at its real operating point**? |
| `corticothalamic/loop_gradient.py` | Does the task loss actually deliver gradient to the loop? |
| `corticothalamic/desaturate_sweep.py` | What pins cortex high, and what scale de-saturates it? |
| `corticothalamic/eval_hybrid.py` | Did the trained net use the cue, or learn a cue-blind constant? |
| `corticothalamic/stability_analysis.py` | `nln`-aware fixed-point + Jacobian analysis (supersedes the old linear probe) |
