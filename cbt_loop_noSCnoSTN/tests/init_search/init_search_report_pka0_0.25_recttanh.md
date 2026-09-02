# init-config search report (objective: pka_d1 = pka_d2 = 0.25)

168 configs scored on how well the fresh-init PKA traces HOLD 0.25 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the min over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.25, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.25.

## Best config

```
m_d1 = 0.980
m_d2 = 0.050
m_a1 = 0.010
m_a2 = 0.124
g_da_release = 0.667
g_ado_release = 0.289
score=1.002  track_pka=0.556  pkaD1=0.16+-0.09 pkaD2=0.17+-0.08  alive_both=0.407
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | g_ado_release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.03 | 0.980 | 0.050 | 0.010 | 0.124 | 0.667 | 0.289 |
| 2 | 1.00 | 0.56 | 0.16 | 0.09 | 0.17 | 0.08 | 0.41 | - | 0.07 | 0.07 | 0.00 | 0.03 | 0.980 | 0.050 | 0.010 | 0.122 | 0.663 | 0.293 |
| 3 | 1.00 | 0.56 | 0.16 | 0.09 | 0.16 | 0.09 | 0.41 | - | 0.07 | 0.06 | 0.00 | 0.04 | 0.980 | 0.050 | 0.010 | 0.100 | 0.666 | 0.371 |
| 4 | 1.00 | 0.56 | 0.16 | 0.09 | 0.16 | 0.09 | 0.41 | - | 0.07 | 0.06 | 0.00 | 0.04 | 0.980 | 0.062 | 0.014 | 0.100 | 0.666 | 0.372 |
| 5 | 1.00 | 0.55 | 0.16 | 0.09 | 0.16 | 0.09 | 0.41 | - | 0.07 | 0.06 | 0.00 | 0.04 | 0.980 | 0.051 | 0.011 | 0.102 | 0.629 | 0.334 |
| 6 | 1.00 | 0.55 | 0.16 | 0.09 | 0.16 | 0.09 | 0.41 | - | 0.07 | 0.06 | 0.00 | 0.04 | 0.980 | 0.051 | 0.011 | 0.102 | 0.630 | 0.334 |

## What drives the score (corr with score)

```
m_d1           corr=+0.59
m_d2           corr=-0.41
m_a1           corr=-0.46
m_a2           corr=-0.75
g_da_release   corr=+0.25
g_ado_release  corr=-0.50
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=+0.49
m_a1/m_a2                  corr=+0.23
g_da_release/g_ado_release corr=+0.48
m_d1/m_a1                  corr=+0.55
m_d2/m_a2                  corr=+0.23
```

## Plots

- `score_vs_param_pka0_0.25_recttanh.png` — pka-target score vs each of the 6 init params
- `release_plane_pka0_0.25_recttanh.png` — g_da x g_ado colored by pka-target score
- `score_vs_ratio_pka0_0.25_recttanh.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, g_da_release/g_ado_release, m_d1/m_a1, m_d2/m_a2), log x, binned-median trend
- `timecourses_pka0_0.25_recttanh.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
