# init-config search report (objective: pka_d1 = pka_d2 = 0.25)

7-D sweep: the 6 gains PLUS `ado0`, with x_ado CLAMPED at ado0 (`pin_ado`) for the whole trial. P_req (raw production for pka*=0.25) = 0.0370; with x_da~0 the D2 solution locus is m_a2*ado0 = P_req.

168 configs scored on how well the fresh-init PKA traces HOLD 0.25 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the min over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.25, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.25.

## Best config

```
m_d1 = 0.643
m_d2 = 0.074
m_a1 = 0.024
m_a2 = 0.619
g_da_release = 0.546
ado0 = 0.062
score=1.652  track_pka=1.000  pkaD1=0.24+-0.01 pkaD2=0.24+-0.01  alive_both=1.000
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | ado0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.65 | 1.00 | 0.24 | 0.01 | 0.24 | 0.01 | 1.00 | - | 0.18 | 0.18 | 0.05 | 0.06 | 0.643 | 0.074 | 0.024 | 0.619 | 0.546 | 0.062 |
| 2 | 1.65 | 1.00 | 0.24 | 0.01 | 0.25 | 0.00 | 1.00 | - | 0.17 | 0.21 | 0.06 | 0.08 | 0.592 | 0.090 | 0.029 | 0.525 | 0.591 | 0.082 |
| 3 | 1.65 | 1.00 | 0.24 | 0.01 | 0.25 | 0.00 | 1.00 | - | 0.18 | 0.22 | 0.06 | 0.09 | 0.607 | 0.093 | 0.029 | 0.512 | 0.598 | 0.086 |
| 4 | 1.65 | 1.00 | 0.24 | 0.01 | 0.25 | 0.00 | 1.00 | - | 0.17 | 0.22 | 0.06 | 0.09 | 0.602 | 0.096 | 0.030 | 0.498 | 0.606 | 0.090 |
| 5 | 1.65 | 1.00 | 0.25 | 0.00 | 0.27 | 0.02 | 1.00 | - | 0.18 | 0.27 | 0.06 | 0.11 | 0.621 | 0.075 | 0.034 | 0.448 | 0.635 | 0.108 |
| 6 | 1.65 | 1.00 | 0.25 | 0.00 | 0.27 | 0.02 | 1.00 | - | 0.19 | 0.27 | 0.06 | 0.12 | 0.637 | 0.120 | 0.037 | 0.414 | 0.656 | 0.124 |

## What drives the score (corr with score)

```
m_d1           corr=-0.34
m_d2           corr=-0.39
m_a1           corr=+0.01
m_a2           corr=+0.07
g_da_release   corr=+0.11
ado0           corr=-0.32
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=+0.31
m_a1/m_a2                  corr=-0.04
m_d1/m_a1                  corr=-0.12
m_d2/m_a2                  corr=-0.32
m_a2/ado0                  corr=+0.30
m_a1/ado0                  corr=+0.26
```

## Plots

- `score_vs_param_ado0sweep_pinSNC_pinCTX.png` — pka-target score vs each of the 6 init params
- `release_plane_ado0sweep_pinSNC_pinCTX.png` — m_a2 x ado0 (the two factors of the D2 PKA drive once x_ado is pinned), colored by score, with the exact-solution hyperbola m_a2*ado0 = P_req = 0.0370
- `score_vs_ratio_ado0sweep_pinSNC_pinCTX.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, m_d1/m_a1, m_d2/m_a2, m_a2/ado0, m_a1/ado0), log x, binned-median trend
- `timecourses_ado0sweep_pinSNC_pinCTX.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
