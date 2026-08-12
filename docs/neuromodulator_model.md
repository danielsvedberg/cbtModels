# PKA / DA / adenosine neuromodulator model

Design log, 2026-07-27 → 2026-08-12. The redesign of how dopamine, adenosine, and PKA
gate the striatum, and the two flavors now in the codebase.

---

## TL;DR

PKA is a **mass-action-bounded pool in (0,1)** fed **directly** into `bg_nln` as the
excitability argument `b` (no separate soft-threshold gate, no per-step state squash).
Dopamine raises D1 PKA / brakes D2 PKA; adenosine does the inverse. Both PKA pools rest
~0.5, so `bg_nln ≈ nln` at rest. Two flavors of the DA/adenosine input:

| flavor | families | DA/adenosine | production |
|---|---|---|---|
| static `k_a` | cbt_loop, noSCnoSTN (orig) | `mean_snc`, scalar `k_a` | `max(g·m_d1·mean_snc − m_a1·k_a, 0)` |
| **dynamic concentrations** | noSC, noSCnoSTN (ported) | `x_da`, `x_ado` states | `max(g·m_d1·x_da − m_a1·x_ado, 0)` |

---

## 1. PKA as bg_nln's `b` directly (gate-free)

Earlier designs used PKA as an unbounded leaky integrator read out through a trainable
soft-threshold gate. **Replaced** by a mass-action-bounded pool:

```
prod   = max(DA_drive − adenosine_drive, 0)              # rectified (no negative cAMP)
prod  *= max(1 − pka/pka_max, 0)                         # mass-action substrate throttle
pka    = (1 − 1/tau_pka_fall)·pka + (1/tau_pka_rise)·prod  # leaky integrator, linear leak
b      = clip(pka, eps, 1−eps)                            # fed directly into bg_nln
x_d1   = bg_nln(x_d1, b)
```

Key points:
- The **mass-action throttle bounds the state to (0, pka_max=1)** while the leak stays
  linear — so `tau_pka_fall` really sets the (slow) timescale. This replaced the legacy
  `pka ← nln(pka)` per-step squash, which destroyed the slow integration (measured
  half-life collapsed ~3 steps despite `tau_pka_fall=1440`).
- `bg_nln(x,b)=sigmoid(c·(x−d))`, `c=3/(1−b)`, `d=(1/6)(1−b)/b` — diverges at b=0,1, hence
  the small `clip`. At b=0.25 it equals `nln`; at b≈0.5 it's ~nln (both pools rest ~0.5).

## 2. D1-preservation guards (why dSPNs were dying)

Symptom: D1 pinned dead (`x_d1≈0`, `pka_d1≈0.002`). Cause: training grew the A1R gain
`m_a1` until adenosine inhibition swamped the (small, low-SNc) DA drive → `prod_d1→0` →
PKA leaks to 0 → gate shut. Fixes (all still in place):
- **Balance adenosine:** `m_a1` init ≈ `m_a2` (0.06 / 0.07).
- **Cap the A1R gain** (`m_a1_cap=0.08`) so training can't regrow it.
- **Clamp PKA inits** to [0.4, 0.6] (`pka_init_floor/cap`) so they can't drift to extreme
  bg_nln shifts.

Result (verified fresh init): both PKA pools rest ~0.5; cue Δpka_d1 **positive** (DA
excites D1), Δpka_d2 **negative** (DA brakes D2); both slow.

## 3. Dynamic DA/adenosine concentration model (noSC, noSCnoSTN)

Keeps the neuromodulators as **dynamic states** co-released by SNc, with mass-action
kinetics — DA fast, adenosine slow:

```
release = mean_snc
x_da  = x_da  + (1/tau_da)  · (da_release·release·max(1 − x_da/da_max, 0)  − x_da)   # tau_da  = 20
x_ado = x_ado + (1/tau_ado) · (ado_release·release·max(1 − x_ado/ado_max, 0) − x_ado)  # tau_ado = 200
prod_d1 = max(g·m_d1·x_da − m_a1·x_ado, 0)      # dynamic DA excites D1, dynamic adenosine inhibits
prod_d2 = max(m_a2·x_ado − g·m_d2·x_da, 0)      # inverse for D2
```

The fast-DA / slow-adenosine separation shows up directly in the cue response: DA peaks
~t+20, adenosine builds over ~80–160 steps. Ported into noSCnoSTN (2026-08-12, commit
031a7c0) by threading `x_da`/`x_ado` through the scan carry; it self-times **more often**
than static-kₐ (see `self_timing_findings.md`).

## 4. `tau_pka_fall` — the slow clock (calibration)

`tau_pka_fall` sets the PKA memory timescale (decay τ ≈ 84 → 212 steps as `tau_pka_fall`
100 → 1440). This is the model's delay-scale clock. Calibration (noSC, τ=7):

| `tau_pka_fall` | decay τ (steps) | learnable hybrid? | note |
|---|---|---|---|
| 500 | 137 | yes | too short → no self-timing at τ=7 |
| **900** | ~170 | **yes** | self-times (seed-gated) |
| 1440 (biophysical 10 s) | 212 | **no** | too-saturated PKA → weak cue → hybrid stuck |

Trade-off: longer clock = longer memory *but* weaker cue signal. 900 is the window.

## 5. Sign structure (Dale)

DA→PKA and adenosine→PKA gains (`m_d1/m_d2/m_a1/m_a2`) are sign-constrained non-negative
(`exc = |w|`), so dopamine can only *excite* D1 / *brake* D2 and adenosine only the
inverse — the A1R/A2R ↔ D1R/D2R opponent scheme. Cortex→SNc is constrained excitatory
(glutamatergic drive onto dopamine neurons). The cue-evoked SNc *decrease* observed in
trained noSC is driven by direct-pathway D1→SNc inhibition outweighing the direct
cortical excitation — a real striatonigral motif.
