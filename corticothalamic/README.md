# Corticothalamic stability analysis + two-node RNN

## Stability analysis (primary)

`stability_analysis.py` — spectral / fixed-point stability of the cortico-thalamic
loop and the striatal gate, using the **actual** nonlinearity from
`self_timed_movement_task.py` (`nln`, `bg_nln`). It consolidates the former
`cbt_loop/tests/eigen_ramp_probe.py`, whose limitation was that it took the
spectral radius of the *linear* update map `J = (1-1/tau)I + (1/tau)W` — i.e. it
assumed a nonlinearity gain of 1 (a linearization about `x=0`). That misses the
fact that `nln` is contractive and saturating, so the loop can settle at a high,
signal-dead fixed point whose local gain → 0 (a strongly stable attractor the
linear test never sees).

This script instead:
1. iterates the **nonlinear** map to its fixed point `x*`,
2. linearizes there — `J* = diag(nln'(pre*)) · [(1-1/tau)I + (1/tau)W]` — and
   reports the operating-point spectral radius `rho*` alongside the naive linear
   `rho` and the mean fixed-point activity,
3. runs cue-evoked persistence and a size sweep on `rho*`, and
4. shows why the striatum's **per-term** nonlinearity (each projection wrapped in
   `nln`/`bg_nln` separately) deletes inhibition — `nln(negative current)` is a
   small *positive* floor, so a growing inhibitory current never gates the gate
   down — pinning D1/D2 to saturation.

Run: `python corticothalamic/stability_analysis.py` (plots → `plots/`).

> Note: `stmt.nln` is currently `sigmoid(4*(x-0.5))` (not the older
> `max(0, tanh)`); `stmt.bg_nln` is `sigmoid(c*(x-d))`. The analysis mirrors and
> asserts these against the live definitions at import.

## Two-node RNN (legacy exploration)

This folder also contains a densely connected two-node RNN (JAX + Optax) trained
with the generalized RL objective in `self_timed_movement_task.py`.

- Node 1: cortex (`ctx`), 20 units
- Node 2: thalamus (`t`), 20 units
- Inputs are projected to cortex only.
- Outputs are read out from thalamus only.
- All weight signs are unconstrained (can become positive or negative).

## Files

- `corticothalamic_rnn.py` - model + RL-compatible `rnn_func` wrapper + evaluation helper.
- `corticothalamic_config.py` - shared task/model/optimizer/RL settings.
- `train_corticothalamic.py` - RL training runner that saves `params_corticothalamic.pkl`.
- `test_corticothalamic.py` - small inference smoke test.
- `requirements.txt` - minimal dependencies for this folder.

## Quick Start

```bash
python /home/dsvedberg/Documents/CodeVault/cbtModels/corticothalamic/train_corticothalamic.py
python /home/dsvedberg/Documents/CodeVault/cbtModels/corticothalamic/test_corticothalamic.py
```

## Notes

- Dynamics are split across `ctx` and `t` with dense within- and cross-area coupling.
- Training uses `stmt.fit_rnn_reinforce` with objective mode controlled by config
  (`log_reward`, `reward_prob`, or `loss`).
- Task generation is configurable via `task_mode` in `corticothalamic_config.py`
  (`self_timed`, `hybrid`, `pavlovian`).
- Evaluation returns sampled binary `actions` in addition to `ys`, `x_ctx`, and `x_t`.
- Optional optogenetic stimulation is supported as concatenated `[ctx, t]` channels.

