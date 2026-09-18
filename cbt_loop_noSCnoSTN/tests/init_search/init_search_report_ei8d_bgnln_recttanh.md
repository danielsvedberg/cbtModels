# init-config search report (objective: pka_d1 = pka_d2 = 0.25)

7-D sweep: the 6 gains PLUS `ado0`, with x_ado CLAMPED at ado0 (`pin_ado`) for the whole trial. P_req (raw production for pka*=0.25) = 0.0333; with x_da~0 the D2 solution locus is m_a2*ado0 = P_req.

211 configs scored on how well the fresh-init PKA traces HOLD 0.25 for the whole trial (0 validated by full training).

`score = 1.0*track_pka + 0.3*track_worst + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the MEAN and `track_worst` the MIN over pkaD1/pkaD2 of the time-mean tent (1 within +-0.05 of 0.25, 0 beyond +-0.15). PKA init state forced to pka_d10=pka_d20=0.25.

## Best config

```
m_d1 = 0.671
m_d2 = 0.286
m_a1 = 0.048
m_a2 = 0.522
g_da_release = 0.466
ado0 = 0.087
cross = 0.000
tonic = 0.250
score=1.958  track_pka=1.000  pkaD1=0.23+-0.02 pkaD2=0.24+-0.01  alive_both=0.999
```

## Top configs

| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | D1 | D2 | x_da | x_ado | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | ado0 | cross | tonic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.96 | 1.00 | 0.23 | 0.02 | 0.24 | 0.01 | 1.00 | - | 0.17 | 0.18 | 0.05 | 0.09 | 0.671 | 0.286 | 0.048 | 0.522 | 0.466 | 0.087 | 0.000 | 0.250 |
| 2 | 1.96 | 1.00 | 0.25 | 0.01 | 0.23 | 0.02 | 1.00 | - | 0.19 | 0.16 | 0.05 | 0.06 | 0.708 | 0.230 | 0.040 | 0.624 | 0.515 | 0.064 | 0.063 | 0.245 |
| 3 | 1.95 | 1.00 | 0.27 | 0.02 | 0.25 | 0.00 | 1.00 | - | 0.18 | 0.15 | 0.05 | 0.07 | 0.804 | 0.140 | 0.039 | 0.612 | 0.521 | 0.066 | 0.072 | 0.198 |
| 4 | 1.95 | 1.00 | 0.28 | 0.03 | 0.26 | 0.01 | 1.00 | - | 0.19 | 0.17 | 0.06 | 0.09 | 0.757 | 0.177 | 0.032 | 0.502 | 0.574 | 0.093 | 0.162 | 0.220 |
| 5 | 1.95 | 1.00 | 0.25 | 0.01 | 0.22 | 0.03 | 0.99 | - | 0.18 | 0.15 | 0.05 | 0.06 | 0.697 | 0.245 | 0.042 | 0.656 | 0.501 | 0.058 | 0.040 | 0.250 |
| 6 | 1.95 | 1.00 | 0.28 | 0.03 | 0.27 | 0.02 | 1.00 | - | 0.20 | 0.18 | 0.06 | 0.12 | 0.727 | 0.208 | 0.028 | 0.440 | 0.611 | 0.117 | 0.222 | 0.235 |

## What drives the score (corr with score)

```
m_d1           corr=+0.02
m_d2           corr=+0.06
m_a1           corr=-0.08
m_a2           corr=+0.02
g_da_release   corr=-0.04
ado0           corr=-0.33
cross          corr=-0.30
tonic          corr=+0.43
-- gain-pair balance (corr of log ratio with score) --
m_d1/m_d2                  corr=-0.06
m_a1/m_a2                  corr=-0.06
m_d1/m_a1                  corr=+0.07
m_d2/m_a2                  corr=+0.04
m_a2/ado0                  corr=+0.27
m_a1/ado0                  corr=+0.23
```

## Plots

- `score_vs_param_ei8d_bgnln_recttanh.png` — pka-target score vs each of the 6 init params
- `release_plane_ei8d_bgnln_recttanh.png` — m_a2 x ado0 (the two factors of the D2 PKA drive once x_ado is pinned), colored by score, with the exact-solution hyperbola m_a2*ado0 = P_req = 0.0333
- `score_vs_ratio_ei8d_bgnln_recttanh.png` — pka-target score vs the ratio of each gain pair (m_d1/m_d2, m_a1/m_a2, m_d1/m_a1, m_d2/m_a2, m_a2/ado0, m_a1/ado0), log x, binned-median trend
- `timecourses_ei8d_bgnln_recttanh.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
