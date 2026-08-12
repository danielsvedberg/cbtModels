# Criticality & timescales of the thalamocortical loop

Investigation log, 2026-07-24 → 2026-08-12. Everything below is *measured* on this
repo with the named tools (`corticothalamic/{loop_criticality,rho_sweep,autapse_criticality,tanh_synapse_test,tau_memory_sweep}.py`, `cbt_loop/tests/rho_sweep_cbt.py`), not inferred.

---

## TL;DR

| Finding | Number | Tool |
|---|---|---|
| Optimal init loop spectral radius | `rho_lin = 1.0` (peaks `rho*≈0.98`) | rho_sweep |
| Raw (un-normalized) cbt_loop rho | 1.76 → useless ~2-step memory | loop_criticality |
| `balanced_init` normalizes to | `rho_lin = 1.000` every seed | loop_init |
| no_autapse effect on rho (free-sign) | −0.001 (negligible) | autapse_criticality |
| no_autapse effect on rho (Dale) | −0.006…−0.014 (real, small) | autapse_criticality |
| tanh-wrapping synapses | shortens memory ~20–25% | tanh_synapse_test |
| membrane τ for ~20 ms EPSP decay | **τ = 7 steps** (dt=10 ms) | (analytic + sim) |
| the three nested timescales | neuron ~20 ms / loop ~300 ms / PKA ~9 s | — |

---

## 1. Spectral normalization and the optimal init rho

The loop update map is `M = (1−1/τ)·I + (1/τ)·W`. Two spectral radii matter:
- `rho_lin` = max|eig(M)| — the linear, gain-1 structural radius.
- `rho*` = max|eig(diag(g)·M)| at the resting fixed point, where `g = nln'(rest) = 4r(1−r)` is the operating-point gain of `nln = sigmoid(4(x−0.5))`. **`rho*` is what sets the actual memory** (`tau_eff = −1/ln(rho*)`).

**`balanced_init` (`loop_init.normalize_loop`)** rescales the 17 loop blocks so `rho_lin` hits a target exactly, for every random seed (bisection). Default target = **1.0**.

**Why it matters:** cbt_loop's *raw* loop rho is 1.76 → `rho* = 0.63`, `tau_eff ≈ 2 steps` (near-useless). Normalizing to 1.0 lifts `tau_eff` to ~45 steps — a **20× longer loop timescale** — and de-saturates the cortex (rest 0.9 → 0.3–0.5).

**The optimum is `rho_lin = 1.0`, and it is non-monotonic** (rho_sweep, both corticothalamic and cbt_loop): pushing the target *higher* drives the rates into saturation, which lowers the nln gain `g`, which pulls `rho*` back *down*.

| target rho_lin | 1.0 | 1.1 | 1.25 | 1.5 | 1.76 |
|---|---|---|---|---|---|
| cbt_loop rho* | **0.977** | 0.897 | 0.841 | 0.727 | 0.633 |
| tau_eff (steps) | **45** | 9 | 6 | 3 | 2 |

`rho*` tops out ~0.98 (never 1.0) because pools rest off-0.5. Pushing `rho*→1` requires **E/I balance** (rates→0.5, where `g=1`), not more rho — see §5.

## 2. The no-autapse rule and criticality

`no_autapse` zeroes the diagonal of every square within-population recurrence (no self-synapses); applied to all areas **except the medulla**. Effect on criticality depends on the sign structure:

- **Free-sign (zero-mean) recurrence:** Δrho ≈ **−0.001** — negligible. Radius is set by the off-diagonal bulk (circular law `σ√N`); removing N of N² diagonal entries barely moves it. **Not compensable by, nor needing, pool-size changes.**
- **Sign-constrained (Dale, `|w|`) recurrence:** Δrho ≈ **−0.006** (fixed offset, constant in N) from removing the mean entry `μ` off the Perron eigenvalue. Real but small.

Consequence: `balanced_init` normalizes the *with-diagonal* loop, so a family that also applies no_autapse to loop recurrences lands ~0.005 below target — fixed by **pre-zeroing the loop self-diagonals before normalize_loop** (done in noSC / corticothalamic).

The old free-sign corticothalamic testbed *under-reported* the no_autapse effect; the Dale reconfig (cU/cL/cI + t_exc/t_inh, all sign-constrained) restored it.

## 3. Per-synapse tanh() shortens memory (recoverable with τ)

Wrapping each synaptic term in `tanh()` before summing (vs the default linear sum, `syn_nln` flag) **shortens the effective memory ~20–25%** — because `tanh'<1` reduces each synapse's gain even at fixed `rho_lin=1.0` (more dissipative). Both balanced_init'd to 1.0:

| synapse mode | perturbation τ | cue-response τ |
|---|---|---|
| linear sum | 47.5 | 52.8 |
| per-term tanh | 36.7 | 40.5 |

**Raising the membrane τ fully recovers it** (tau_memory_sweep): effective memory τ climbs monotonically with membrane τ for both modes; **tanh @ τ=40 (~53 steps) already beats linear @ τ=20 (~52)**. So doubling τ more than recovers the tanh loss. Plateaus past τ≈80 (network approaches frozen).

## 4. Membrane τ calibration → ~20 ms EPSP

An **isolated** neuron (no synaptic input, `x(t+1)=nln((1−1/τ)x)`) does **not** decay at `τ·dt`: the nln gain (~0.8 at the low resting point) speeds it ~5×.

| membrane τ (steps) | isolated decay (ms, dt=10) |
|---|---|
| 5 | 15.7 |
| **7** | **~20** |
| 10 | 25.3 |
| 20 | 39.9 |

To match the slice-study **~20 ms** EPSP/membrane decay of cortical pyramidal cells, set the cortex/thalamus membrane τ to **7 steps** (commit 7a033fb: `tau_c=tau_t=7`; other area τ left at 10). Near-criticality is preserved (`balanced_init` re-normalizes `rho_lin=1.0` at any τ): rho* stayed 0.966–0.971.

## 5. The three nested timescales

The model spans the behavioral 3 s delay with three decoupled timescales, controlled by different knobs:

| scale | timescale | mechanism | knob |
|---|---|---|---|
| single neuron | **~20 ms** | membrane leak + nln gain | `tau_c`/`tau_t` (=7) |
| loop / network | **~300 ms** | near-critical recurrence | `balanced_target_rho` (=1.0) |
| molecular clock | **~9 s** | PKA de-activation | `tau_pka_fall` (=900) |

Fast biophysical neurons + slow near-critical network + very-slow molecular clock. **Timing rides the PKA clock, not loop memory** (see `self_timing_findings.md`). The E/I-balance lever (`rho*→1`, rates→0.5) would extend the *loop* scale but a *uniform* tonic bias can't reach it — the pools rest at spread rates (0.23–0.44) and a uniform shift overshoots the high pools past 0.5; **per-pool balancing** is required (noSC `ei_tonic` knob, default 0).
