# init-config search report (objective: pka_d1 = pka_d2 = 0.25)

168 configs scored on how well the fresh-init PKA traces HOLD 0.25 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the min over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.25, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.25.

## Best config

```
m_d1 = 0.806
m_d2 = 0.127
m_a1 = 0.047
m_a2 = 0.254
g_da_release = 0.483
g_ado_release = 0.696
score=1.709  track_pka=1.000  pkaD1=0.28+-0.03 pkaD2=0.24+-0.01  alive_both=1.000
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | g_ado_release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.71 | 1.00 | 0.28 | 0.03 | 0.24 | 0.01 | 1.00 | - | 0.50 | 0.19 | 0.07 | 0.18 | 0.806 | 0.127 | 0.047 | 0.254 | 0.483 | 0.696 |
| 2 | 1.71 | 1.00 | 0.28 | 0.03 | 0.24 | 0.01 | 1.00 | - | 0.51 | 0.19 | 0.07 | 0.18 | 0.809 | 0.126 | 0.048 | 0.251 | 0.485 | 0.699 |
| 3 | 1.71 | 1.00 | 0.28 | 0.03 | 0.25 | 0.01 | 1.00 | - | 0.49 | 0.23 | 0.07 | 0.17 | 0.787 | 0.116 | 0.051 | 0.275 | 0.504 | 0.674 |
| 4 | 1.71 | 1.00 | 0.27 | 0.02 | 0.25 | 0.01 | 1.00 | - | 0.41 | 0.25 | 0.07 | 0.16 | 0.781 | 0.105 | 0.041 | 0.293 | 0.444 | 0.657 |
| 5 | 1.71 | 1.00 | 0.26 | 0.01 | 0.26 | 0.01 | 1.00 | - | 0.32 | 0.30 | 0.07 | 0.17 | 0.742 | 0.088 | 0.069 | 0.288 | 0.462 | 0.750 |
| 6 | 1.71 | 1.00 | 0.26 | 0.01 | 0.25 | 0.01 | 1.00 | - | 0.33 | 0.28 | 0.07 | 0.17 | 0.748 | 0.091 | 0.071 | 0.280 | 0.468 | 0.758 |

## What drives the score (corr with score)

```
m_d1           corr=+0.17
m_d2           corr=-0.51
m_a1           corr=+0.16
m_a2           corr=+0.14
g_da_release   corr=-0.20
g_ado_release  corr=+0.33
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=+0.52
m_a1/m_a2                  corr=+0.04
g_da_release/g_ado_release corr=-0.33
m_d1/m_a1                  corr=-0.12
m_d2/m_a2                  corr=-0.50
```

## Plots

- `score_vs_param_pka0_0.25.png` — pka-target score vs each of the 6 init params
- `release_plane_pka0_0.25.png` — g_da x g_ado colored by pka-target score
- `score_vs_ratio_pka0_0.25.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, g_da_release/g_ado_release, m_d1/m_a1, m_d2/m_a2), log x, binned-median trend
- `timecourses_pka0_0.25.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
