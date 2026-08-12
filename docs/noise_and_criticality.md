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
