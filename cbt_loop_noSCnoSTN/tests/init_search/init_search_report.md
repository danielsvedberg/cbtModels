# init-config search report

142 configs scored on the fresh-init operating band (0 validated by full training).

## Best config (band score)

```
m_d1 = 0.705
m_d2 = 0.980
m_a1 = 0.226
m_a2 = 0.020
g_da_release = 0.020
g_ado_release = 0.355
score=1.553  alive_both=0.807  D1=0.49 D2=0.66
```

## Top configs

| rank | band | alive_both | reward | D1 | D2 | x_da | x_ado | pkaD1 | pkaD2 | m_d1 | m_d2 | m_a1 | m_a2 | g_da_release | g_ado_release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.55 | 0.81 | - | 0.49 | 0.66 | 0.00 | 0.17 | 0.31 | 0.33 | 0.705 | 0.980 | 0.226 | 0.020 | 0.020 | 0.355 |
| 2 | 1.54 | 0.81 | - | 0.47 | 0.57 | 0.04 | 0.29 | 0.30 | 0.31 | 0.134 | 0.894 | 0.409 | 0.067 | 0.279 | 0.849 |
| 3 | 1.54 | 0.80 | - | 0.47 | 0.58 | 0.04 | 0.23 | 0.30 | 0.31 | 0.325 | 0.247 | 0.216 | 0.027 | 0.257 | 0.571 |
| 4 | 1.53 | 0.80 | - | 0.47 | 0.58 | 0.04 | 0.09 | 0.31 | 0.32 | 0.158 | 0.528 | 0.133 | 0.127 | 0.268 | 0.170 |
| 5 | 1.53 | 0.79 | - | 0.56 | 0.50 | 0.01 | 0.18 | 0.32 | 0.30 | 0.775 | 0.980 | 0.156 | 0.020 | 0.090 | 0.425 |
| 6 | 1.53 | 0.78 | - | 0.48 | 0.53 | 0.01 | 0.26 | 0.30 | 0.30 | 0.743 | 0.897 | 0.578 | 0.020 | 0.085 | 0.743 |

## What drives the band score (corr with score)

```
m_d1           corr=-0.06
m_d2           corr=+0.31
m_a1           corr=+0.31
m_a2           corr=-0.77
g_da_release   corr=-0.07
g_ado_release  corr=-0.16
```

## Plots

- `score_vs_param.png` — band score vs each of the 6 init params
- `release_plane.png` — g_da x g_ado colored by band score
- `timecourses.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs
