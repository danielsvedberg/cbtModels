# init-config search report (objective: pka_d1 = pka_d2 = 0.25)

7-D sweep: the 6 gains PLUS `ado0`, with x_ado CLAMPED at ado0 (`pin_ado`) for the whole trial. P_req (raw production for pka*=0.25) = 0.0370; with x_da~0 the D2 solution locus is m_a2*ado0 = P_req.

168 configs scored on how well the fresh-init PKA traces HOLD 0.25 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the min over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.25, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.25.

## Best config

```
m_d1 = 0.980
m_d2 = 0.294
m_a1 = 0.025
m_a2 = 0.315
g_da_release = 0.788
ado0 = 0.020
score=0.954  track_pka=0.562  pkaD1=0.16+-0.09 pkaD2=0.17+-0.08  alive_both=0.411
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | ado0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.95 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.02 | 0.980 | 0.294 | 0.025 | 0.315 | 0.788 | 0.020 |
| 2 | 0.95 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.06 | 0.00 | 0.02 | 0.980 | 0.302 | 0.025 | 0.307 | 0.781 | 0.020 |
| 3 | 0.95 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.03 | 0.980 | 0.302 | 0.025 | 0.233 | 0.789 | 0.031 |
| 4 | 0.95 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.02 | 0.980 | 0.356 | 0.029 | 0.267 | 0.752 | 0.024 |
| 5 | 0.95 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.02 | 0.980 | 0.355 | 0.029 | 0.267 | 0.751 | 0.024 |
| 6 | 0.95 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.02 | 0.979 | 0.358 | 0.029 | 0.269 | 0.748 | 0.024 |

## What drives the score (corr with score)

```
m_d1           corr=+0.28
m_d2           corr=+0.11
m_a1           corr=+0.07
m_a2           corr=-0.42
g_da_release   corr=+0.33
ado0           corr=-0.79
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=-0.03
m_a1/m_a2                  corr=+0.33
m_d1/m_a1                  corr=+0.03
m_d2/m_a2                  corr=+0.35
m_a2/ado0                  corr=+0.56
m_a1/ado0                  corr=+0.71
```

## Plots

- `score_vs_param_ado0sweep_recttanh.png` — pka-target score vs each of the 6 init params
- `release_plane_ado0sweep_recttanh.png` — m_a2 x ado0 (the two factors of the D2 PKA drive once x_ado is pinned), colored by score, with the exact-solution hyperbola m_a2*ado0 = P_req = 0.0370
- `score_vs_ratio_ado0sweep_recttanh.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, m_d1/m_a1, m_d2/m_a2, m_a2/ado0, m_a1/ado0), log x, binned-median trend
- `timecourses_ado0sweep_recttanh.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
