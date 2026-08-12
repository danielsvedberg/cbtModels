# Noise and criticality in the corticothalamic loop

Investigation log, 2026-08-12 →. Measured on the Dale corticothalamic testbed
(τ=7, `balanced_init` rho_lin=1.0) with `corticothalamic/noise_sweep.py` and
`corticothalamic/rho_noise_sweep.py`. `noise_std` is the OU stationary std added to
each population per step.

---

## TL;DR

At the near-critical operating point (`rho_lin=1.0`, `rho*≈0.97`), **noise is a
fluctuation-*amplitude* knob, not a *criticality* knob**: over `noise_std` 0.01→0.1
the fluctuation size scales ~linearly while the operating point, `rho*`, and the
fluctuation *timescale* (~40 steps / ~400 ms) are nearly invariant.

---

## 1. Noise sweep at rho=1.0

`noise_sweep.py`, 4 seeds, T=4000, dt=10 ms. Per noise level: mean rate (operating
point), fluctuation std of the population-mean, autocorrelation time `tau_ac` of the
spontaneous fluctuations (the noise-driven memory timescale), and `rho*`/`tau_eff` at
the noise-shifted mean operating point.

| noise_std | mean rate | fluct std | tau_ac (steps) | tau_ac (ms) | rho* | tau_eff |
|---|---|---|---|---|---|---|
| 0.010 | 0.337 | 0.0012 | 42 | 420 | 0.972 | 35.7 |
| 0.025 | 0.337 | 0.0029 | 42 | 420 | 0.973 | 36.1 |
| 0.050 | 0.339 | 0.0057 | 42 | 415 | 0.974 | 37.4 |
| 0.100 | 0.344 | 0.0108 | 38 | 375 | 0.977 | 42.4 |

**Findings:**
1. **Criticality/timescale is noise-invariant.** Autocorrelation curves nearly overlap;
   fluctuation memory stays ~40 steps (~400 ms), `rho*`~0.97, mean rate ~0.34 across the
   10× noise range. The near-critical loop is a stable attractor — noise doesn't push it
   off criticality.
2. **Fluctuation amplitude ∝ noise** (std 0.0012→0.0108, ≈10× for 10× noise) — the
   hallmark of a stable fixed point (`rho*<1`): amplitude tracks the drive, timescale is
   intrinsic.
3. **Subtle nonlinearity at high noise.** More noise nudges the mean rate up toward 0.5
   (nln rectification), mildly *raising* `rho*` (0.972→0.977, i.e. faintly *toward*
   criticality); yet empirical `tau_ac` slightly *drops* (42→38) where linear `tau_eff`
   says rise — large excursions sample the nonlinearity's curvature and de-correlate a
   touch faster. Both effects small.

**Interpretation:** in the near-critical regime the ~400 ms loop memory the network relies
on is robust to noise level — a desirable property.

---

## 2. rho × noise grid — critical slowing down

`rho_noise_sweep.py`, 3 seeds, T=3000, burn=600, dt=10 ms. Sweeps the init target
`balanced_target_rho ∈ {0.95, 1.0, 1.05, 1.1, 1.2, 1.35}` against
`noise_std ∈ {0.01, 0.025, 0.05, 0.1}`. Per cell: fluctuation std, autocorrelation
time `tau_ac`, and `rho*` at the operating point. This is the key experiment: §1 held
`rho=1.0` fixed and varied only noise; here we move the operating point relative to the
critical surface.

| target rho_lin | rho* | tau_ac (steps) | fluct_std @noise .05 | susceptibility |
|---|---|---|---|---|
| 0.95 | 0.784 | 4 | 0.0018 | 0.037 |
| **1.00** | **0.973** | **37** | **0.0050** | **0.100** |
| 1.05 | 0.900 | 8 | 0.0029 | 0.058 |
| 1.10 | 0.860 | 4 | 0.0019 | 0.038 |
| 1.20 | 0.860 | 3 | 0.0014 | 0.027 |
| 1.35 | 0.848 | 3 | 0.0010 | 0.019 |

(`tau_ac` and susceptibility = `fluct_std/noise` are noise-invariant to within a few
percent — the table row is representative across all four noise levels.)

**Findings:**
1. **Both fluctuation *timescale* and *amplitude* peak sharply at `rho_lin=1.0`** and
   fall off steeply in *both* directions. `tau_ac` goes 4 → **37** → 8 → 4 → 3 → 3;
   susceptibility goes 0.037 → **0.100** → 0.058 → 0.038 → 0.027 → 0.019. This is the
   textbook **critical slowing down** / susceptibility-divergence signature: correlations
   grow long-ranged and fluctuations amplify as you approach the critical point.
2. **The peak sits where `rho*` is maximal (0.973), not where `rho_lin` is largest.** The
   non-monotonic `rho*(rho_lin)` (from `criticality_and_timescales.md` §1 — pushing
   `rho_lin` past 1 saturates rates, drops the nln gain, pulls `rho*` back down) means the
   *closest-to-critical* operating point is `rho_lin=1.0`, and the fluctuation statistics
   track `rho*`, not the bare init radius. Above 1.0 the network is actually *further* from
   critical.
3. **The peak location is noise-independent** (`tau_ac`=37 at noise 0.01 vs 33 at noise 0.1;
   the four susceptibility curves overlap). Criticality is an intrinsic property of the
   *operating point*, set by `rho_lin`; noise only scales the amplitude that rides on top of
   it. This is exactly consistent with §1 (noise is an amplitude knob) — §1 was the vertical
   slice through the `rho=1.0` peak of this grid.

**Interpretation:** `balanced_target_rho=1.0` is the network's critical point in the
concrete, measurable sense — it maximizes both the memory timescale (~370 ms, `tau_ac≈37`)
and the responsiveness (susceptibility) to input. The self-timing / working-memory
functions the loop is asked to perform want exactly this near-critical operating point,
and it is chosen by the init radius, robustly to noise. Detuning `rho_lin` even to 1.05
roughly quarters the loop memory (37→8 steps).
