# init-config search report (objective: pka_d1 = pka_d2 = 0.2)

7-D sweep: the 6 gains PLUS `ado0`, with x_ado CLAMPED at ado0 (`pin_ado`) for the whole trial. P_req (raw production for pka*=0.2) = 0.0250; with x_da~0 the D2 solution locus is m_a2*ado0 = P_req.

173 configs scored on how well the fresh-init PKA traces HOLD 0.2 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.3*track_worst + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the MEAN and `track_worst` the MIN over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.2, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.2.

## Best config

```
m_d1 = 0.610
m_d2 = 0.467
m_a1 = 0.013
m_a2 = 0.501
g_da_release = 0.313
ado0 = 0.089
cross = 0.232
tonic = 0.210
score=2.467  track_pka=1.000  pkaD1=0.18+-0.02 pkaD2=0.21+-0.02  alive_both=1.000
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | ado0 | cross | tonic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.47 | 1.00 | 0.18 | 0.02 | 0.21 | 0.02 | 1.00 | - | 0.18 | 0.21 | 0.03 | 0.09 | 0.610 | 0.467 | 0.013 | 0.501 | 0.313 | 0.089 | 0.232 | 0.210 |
| 2 | 2.47 | 1.00 | 0.21 | 0.01 | 0.19 | 0.01 | 1.00 | - | 0.17 | 0.15 | 0.03 | 0.06 | 0.831 | 0.192 | 0.024 | 0.508 | 0.310 | 0.059 | 0.292 | 0.104 |
| 3 | 2.47 | 1.00 | 0.18 | 0.02 | 0.21 | 0.02 | 1.00 | - | 0.19 | 0.22 | 0.03 | 0.10 | 0.602 | 0.493 | 0.013 | 0.479 | 0.325 | 0.096 | 0.252 | 0.215 |
| 4 | 2.46 | 1.00 | 0.19 | 0.02 | 0.21 | 0.01 | 1.00 | - | 0.20 | 0.22 | 0.04 | 0.16 | 0.551 | 0.694 | 0.010 | 0.360 | 0.403 | 0.158 | 0.382 | 0.248 |
| 5 | 2.46 | 1.00 | 0.21 | 0.01 | 0.19 | 0.01 | 1.00 | - | 0.20 | 0.17 | 0.04 | 0.05 | 0.675 | 0.315 | 0.010 | 0.697 | 0.403 | 0.050 | 0.382 | 0.173 |
| 6 | 2.46 | 1.00 | 0.22 | 0.02 | 0.22 | 0.02 | 1.00 | - | 0.20 | 0.20 | 0.04 | 0.05 | 0.699 | 0.184 | 0.032 | 0.822 | 0.410 | 0.047 | 0.522 | 0.160 |

## What drives the score (corr with score)

```
m_d1           corr=-0.13
m_d2           corr=+0.10
m_a1           corr=-0.15
m_a2           corr=-0.13
g_da_release   corr=-0.13
ado0           corr=-0.55
cross          corr=-0.19
tonic          corr=+0.29
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=-0.13
m_a1/m_a2                  corr=-0.02
m_d1/m_a1                  corr=+0.11
m_d2/m_a2                  corr=+0.22
m_a2/ado0                  corr=+0.41
m_a1/ado0                  corr=+0.41
```

## Plots

- `score_vs_param_ei8d_tanhclip_saddle.png` — pka-target score vs each of the 6 init params
- `release_plane_ei8d_tanhclip_saddle.png` — m_a2 x ado0 (the two factors of the D2 PKA drive once x_ado is pinned), colored by score, with the exact-solution hyperbola m_a2*ado0 = P_req = 0.0250
- `score_vs_ratio_ei8d_tanhclip_saddle.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, m_d1/m_a1, m_d2/m_a2, m_a2/ado0, m_a1/ado0), log x, binned-median trend
- `timecourses_ei8d_tanhclip_saddle.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
