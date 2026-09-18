# init-config search report (objective: pka_d1 = pka_d2 = 0.25)

7-D sweep: the 6 gains PLUS `ado0`, with x_ado CLAMPED at ado0 (`pin_ado`) for the whole trial. P_req (raw production for pka*=0.25) = 0.0333; with x_da~0 the D2 solution locus is m_a2*ado0 = P_req.

98 configs scored on how well the fresh-init PKA traces HOLD 0.25 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.3*track_worst + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the MEAN and `track_worst` the MIN over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.25, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.25.

## Best config

```
m_d1 = 0.905
m_d2 = 0.192
m_a1 = 0.022
m_a2 = 0.890
g_da_release = 0.390
ado0 = 0.040
score=1.541  track_pka=1.000  pkaD1=0.26+-0.01 pkaD2=0.23+-0.02  alive_both=0.156
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | ado0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.54 | 1.00 | 0.26 | 0.01 | 0.23 | 0.02 | 0.16 | - | 0.04 | 0.02 | 0.04 | 0.04 | 0.905 | 0.192 | 0.022 | 0.890 | 0.390 | 0.040 |
| 2 | 1.54 | 1.00 | 0.26 | 0.01 | 0.23 | 0.02 | 0.16 | - | 0.04 | 0.02 | 0.04 | 0.04 | 0.905 | 0.192 | 0.022 | 0.890 | 0.390 | 0.040 |
| 3 | 1.54 | 1.00 | 0.26 | 0.01 | 0.23 | 0.02 | 0.16 | - | 0.04 | 0.02 | 0.04 | 0.04 | 0.905 | 0.192 | 0.022 | 0.890 | 0.390 | 0.040 |
| 4 | 1.52 | 1.00 | 0.28 | 0.03 | 0.22 | 0.03 | 0.13 | - | 0.05 | 0.02 | 0.05 | 0.07 | 0.795 | 0.140 | 0.014 | 0.488 | 0.515 | 0.065 |
| 5 | 1.52 | 1.00 | 0.28 | 0.03 | 0.22 | 0.03 | 0.13 | - | 0.05 | 0.02 | 0.05 | 0.07 | 0.795 | 0.140 | 0.014 | 0.488 | 0.515 | 0.065 |
| 6 | 1.52 | 1.00 | 0.28 | 0.03 | 0.22 | 0.03 | 0.13 | - | 0.05 | 0.02 | 0.05 | 0.07 | 0.795 | 0.140 | 0.014 | 0.488 | 0.515 | 0.065 |

## What drives the score (corr with score)

```
m_d1           corr=+0.03
m_d2           corr=-0.40
m_a1           corr=-0.25
m_a2           corr=-0.29
g_da_release   corr=+0.03
ado0           corr=-0.30
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=+0.37
m_a1/m_a2                  corr=+0.01
m_d1/m_a1                  corr=+0.24
m_d2/m_a2                  corr=-0.11
m_a2/ado0                  corr=+0.12
m_a1/ado0                  corr=+0.11
```

## Plots

- `score_vs_param_ado0sweep_bgnln_recttanh.png` — pka-target score vs each of the 6 init params
- `release_plane_ado0sweep_bgnln_recttanh.png` — m_a2 x ado0 (the two factors of the D2 PKA drive once x_ado is pinned), colored by score, with the exact-solution hyperbola m_a2*ado0 = P_req = 0.0333
- `score_vs_ratio_ado0sweep_bgnln_recttanh.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, m_d1/m_a1, m_d2/m_a2, m_a2/ado0, m_a1/ado0), log x, binned-median trend
- `timecourses_ado0sweep_bgnln_recttanh.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
