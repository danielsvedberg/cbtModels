# Corticothalamic Two-Node RNN

This folder contains a densely connected two-node RNN (JAX + Optax) trained with
the generalized RL objective in `self_timed_movement_task.py`.

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

