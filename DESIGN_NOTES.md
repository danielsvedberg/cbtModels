cations# Self-Timed Movement Task: Objective Function Design

## Task Description

The self-timed movement task trains a neural network to:
1. **Hear a "start timing cue"** delivered at a random time (`T_start`, 1-3 seconds after trial start)
2. **Wait a minimum duration** (`T_wait`, 3 seconds) after the cue before responding
3. **Respond within a time window** (between `T_start + T_cue + T_wait` and `T_start + T_cue + T_wait + T_movement`)
4. **Respond by producing y ≥ 0.5** (a "button press" - a brief output above threshold)
5. **Receive reward** for responses in the correct time window

## Temporal Windows

For a trial with timestep duration of 10ms:

```
Time (s)     Time (steps)
0.0          0           ... Trial start
1.0-3.0      100-300     ... Random cue onset (T_start)
1.1-3.1      110-310     ... Cue period (T_cue)
4.1-6.1      410-610     ... Valid response window starts after minimum wait (T_start + T_cue + T_wait)
5.1-8.1      510-810     ... Valid response window ends (T_start + T_cue + T_wait + T_movement)
```

## Loss Function Design Philosophy

The loss function has TWO complementary components:

### 1. **Target Loss** (During Valid Response Window)
- **Window:** `T_start + T_cue + T_wait` to `T_start + T_cue + T_wait + T_movement`
- **Goal:** Encourage network to produce y ≥ 0.5 (a response)
- **Formula:** `∑((y - 1.0)² × mask) / ∑(mask)`
- **Intuition:** Penalizes outputs below 1.0, encouraging y → 1.0

### 2. **Baseline Loss** (Outside Valid Response Window)
- **Window:** Pre-cue, during wait period, after valid window
- **Goal:** Encourage network to keep y < 0.5 (inhibition)
- **Formula:** `∑(y² × mask) / ∑(mask)`
- **Intuition:** Penalizes any activity, ensuring baseline suppression

### Combined Loss
```
Total Loss = Target Loss + Baseline Loss
```

## Why This Works

This design naturally implements:

1. **False Alarm Prevention:** Before the valid window, `y < 0.5` is rewarded (baseline loss keeps y low)
2. **Hit Encouragement:** During valid window, `y ≥ 0.5` is rewarded (target loss pulls y toward 1.0)
3. **Late Response Prevention:** After valid window, `y < 0.5` is rewarded (baseline loss keeps y low)
4. **No Post-hoc Masking:** Unlike adaptive loss functions, this approach doesn't change targets based on network behavior
5. **Biological Plausibility:** Encourages clear suppression (low baseline) interrupted by brief pulses (responses)

## Task Generation Function

The `self_timed_movement_task()` function creates training data:

```python
inputs, outputs, masks = self_timed_movement_task(
    T_start=jnp.arange(100, 300),      # Cue times: 1.0-3.0 seconds
    T_cue=10,                            # Cue duration: 100ms
    T_wait=300,                          # Min wait: 3 seconds
    T_movement=200,                      # Response window: 2 seconds
    T=1000,                              # Total trial: 10 seconds
    null_trial=False
)
```

**Returns:**
- `inputs`: (num_conditions, T, 1) - Binary cue signal
- `outputs`: (num_conditions, T, 1) - Target signal (1 during valid window, 0 elsewhere)
- `masks`: (num_conditions, T, 1) - Loss masking (all 1s for all timesteps)

## Implementation Tips

### 1. **Timestep Resolution**
If using 10ms timesteps:
- 1 second = 100 timesteps
- 3 seconds = 300 timesteps
- 5 seconds = 500 timesteps

### 2. **Loss Function Call**
```python
loss = batched_rnn_loss(
    rnn_func=my_rnn,           # Your RNN model
    params=params,              # Model parameters
    x0=x0,                      # Initial RNN state
    batch_inputs=inputs,        # (batch, T, 1)
    batch_targets=outputs,      # (batch, T, 1) - 1 in valid window
    batch_mask=masks,           # (batch, T, 1)
    rng_keys=keys              # Random keys
)
```

### 3. **Monitoring Network Behavior**
During training, you can monitor:
- **Hit rate:** Fraction of trials with y ≥ 0.5 during valid window
- **False alarm rate:** Fraction of trials with y ≥ 0.5 before valid window
- **Late response rate:** Fraction of trials with y ≥ 0.5 after valid window
- **Response latency:** Time from cue onset to first y ≥ 0.5

### 4. **Hyperparameter Tuning**

If the network struggles with timing, try:
- **Increasing T_wait:** Make it easier (more time before responding allowed)
- **Increasing T_movement:** Wider response window (more forgiving)
- **Decreasing T_cue:** Shorter cue (less sustained activity needed)
- **More training iterations:** Let optimization work longer

If the network produces sustained responses, try:
- **Decreasing T_movement:** Narrower window (forces briefer responses)
- **Increasing T_wait:** Makes early responses more costly
- **Adding regularization:** Penalize large outputs in baseline periods

## Example Usage

```python
import jax.numpy as jnp
from self_timed_movement_task import self_timed_movement_task, batched_rnn_loss, fit_rnn

# Generate training data
T_start_values = jnp.arange(100, 300)  # 1-3 second cue onsets
inputs, targets, masks = self_timed_movement_task(
    T_start=T_start_values,
    T_cue=10,                    # 100ms cue
    T_wait=300,                  # 3s minimum wait
    T_movement=200,              # Up to 5s response window (3+2s)
    T=1000                       # 10s total trial
)

# Train RNN
best_params, loss_history = fit_rnn(
    inputs=inputs,
    targets=targets,
    loss_masks=masks,
    params=initial_params,
    optimizer=optimizer,
    x0=initial_state,
    num_iters=10000,
    log_interval=100
)
```

## Advanced: Custom Loss Weighting

For more sophisticated training, you might weight the two loss components differently:

```python
# In batched_rnn_loss:
target_loss = jnp.sum(((ys - 1.0) ** 2) * target_window * batch_mask) / (jnp.sum(target_window * batch_mask) + 1e-8)
baseline_loss = jnp.sum((ys ** 2) * non_target * batch_mask) / (jnp.sum(non_target * batch_mask) + 1e-8)

# Weight them differently:
total_loss = 1.0 * target_loss + 2.0 * baseline_loss  # Penalize baseline violations more
```

Or add a **sparsity penalty** to encourage brief responses:

```python
# Penalize duration of responses
response_duration_loss = jnp.sum((ys ** 2) * (Tarray > t_move_start) * (Tarray < t_move_end + 100))

total_loss = target_loss + baseline_loss + 0.1 * response_duration_loss
```

## References

This loss design is inspired by:
- Supervised learning with implicit temporal structure
- Behavioral inhibition tasks in neuroscience
- Interval timing models that use dual-process theory (rapid timing vs. accumulation)

