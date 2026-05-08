# Vanilla One-Area RNN

This folder contains a densely connected single-area RNN (JAX + Optax) trained on
`self_timed_movement_task.self_timed_movement_task`.

## Files

- `vanilla_rnn.py` - model, batched loss, train loop, evaluation helper.
- `vanilla_config.py` - shared task/model/training config for pretraining, training, and plotting.
- `pretrain_vanilla.py` - hybrid-STMT pretraining runner that saves warm-start weights.
- `train_vanilla.py` - training runner that saves `params_vanilla.pkl`.
- `test_vanilla.py` - small inference smoke test.
- `plot_vanilla.py` - plotting helpers for cue-aligned traces, response-time histogram, and per-trial traces/loss.
- `requirements.txt` - minimal dependencies for this folder.

## Quick Start

From the workspace root:

```bash
python /home/dsvedberg/Documents/CodeVault/cbtModels/vanilla_rnn/pretrain_vanilla.py
python /home/dsvedberg/Documents/CodeVault/cbtModels/vanilla_rnn/train_vanilla.py
python /home/dsvedberg/Documents/CodeVault/cbtModels/vanilla_rnn/test_vanilla.py
python /home/dsvedberg/Documents/CodeVault/cbtModels/vanilla_rnn/plot_vanilla.py
```

`train_vanilla.py` will automatically load `pretrain_params_vanilla.py` when present.

## Notes

- Inputs/targets come directly from `self_timed_movement_task.self_timed_movement_task`.
- The RNN state update uses a dense recurrent matrix (`w_rec`) plus dense input/output projections.
- The training objective is masked MSE over full trajectories.

