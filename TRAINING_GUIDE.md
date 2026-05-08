# Self-Timed Movement Task: Training Guide

## Summary of Changes

I've redesigned the objective function and loss calculation for your self-timed movement task. The new approach is:
- **Principled**: Loss directly encodes task structure, not adaptive to network output
- **Interpretable**: Clear separation of hits, false alarms, and late responses
- **Biologically plausible**: Encourages low baseline activity with brief response bursts
- **Mathematically clean**: Simple two-component loss that's easy to debug and extend

## What Changed

### 1. **Task Generation** (`self_timed_movement_task`)
**Before:** Confusing parameter names and unclear timing semantics
**After:** 
- Clear docstring explaining each window
- Parameter naming that directly maps to task structure:
  - `T_start`: Cue onset time (1-3s = 100-300 timesteps)
  - `T_cue`: Cue duration (e.g., 100ms = 10 timesteps)
  - `T_wait`: Minimum wait after cue (3s = 300 timesteps) **← Key parameter**
  - `T_movement`: Response window duration (2s = 200 timesteps, allowing 3-5s total) **← Key parameter**

### 2. **Loss Function** (`batched_rnn_loss`)
**Before:** Ad-hoc masking based on network response (problematic!)
**After:** Two well-designed components:

```python
# Component 1: Target Loss (during valid window)
# Encourages y → 1.0 when response is appropriate
target_loss = ∑((y - 1.0)² × valid_window_mask)

# Component 2: Baseline Loss (everywhere else)
# Encourages y → 0.0 when response is not appropriate  
baseline_loss = ∑(y² × non_target_mask)

# Total Loss
total_loss = target_loss + baseline_loss
```

### 3. **New Helper Function** (`analyze_task_performance`)
Provides immediate feedback on:
- **Hit rate**: Responses in the valid window (want ↑)
- **False alarm rate**: Responses before valid window (want ↓)
- **Late response rate**: Responses after valid window (want ↓)
- **Response latency**: Mean time from cue to response
- **Accuracy**: Composite metric combining hits and false alarms

## How to Use

### Step 1: Generate Training Data
```python
import jax.numpy as jnp
from self_timed_movement_task import self_timed_movement_task

# Example: 1-3s cue onset, 3s min wait, 5s max wait
T_start_values = jnp.arange(100, 300, 10)  # 1.0-3.0s in 10ms steps

inputs, targets, masks = self_timed_movement_task(
    T_start=T_start_values,
    T_cue=10,           # 100ms cue duration
    T_wait=300,         # 3s minimum wait (key!)
    T_movement=200,     # 2s response window (so 3-5s total)
    T=1000              # 10s total trial duration
)

# Now you have:
# inputs: (30, 1000, 1) - cue time series
# targets: (30, 1000, 1) - 1 during valid window, 0 elsewhere
# masks: (30, 1000, 1) - all ones (include all timesteps)
```

### Step 2: Train Your RNN
```python
from self_timed_movement_task import fit_rnn

best_params, loss_history = fit_rnn(
    inputs=inputs,
    targets=targets,
    loss_masks=masks,
    params=initial_params,      # Your initial RNN parameters
    optimizer=optimizer,          # e.g., optax.adam(1e-3)
    x0=initial_state,            # Your initial RNN state
    num_iters=10000,
    log_interval=100
)
```

### Step 3: Evaluate Performance
```python
from self_timed_movement_task import analyze_task_performance

# Get network predictions
ys, _, _ = your_rnn_model(best_params, x0, inputs, None, rng_keys)

# Analyze performance
performance = analyze_task_performance(
    ys=ys,                    # Network outputs
    targets=targets,          # Target signals
    inputs=inputs,            # Input cues
    response_threshold=0.5,   # Your response threshold
    dt=0.01                   # 10ms timesteps
)

print(f"Hit Rate: {performance['hit_rate']:.2%}")
print(f"False Alarm Rate: {performance['false_alarm_rate']:.2%}")
print(f"Late Response Rate: {performance['late_response_rate']:.2%}")
print(f"Mean Response Latency: {performance['mean_response_latency']:.2f}s")
print(f"Accuracy: {performance['accuracy']:.2%}")
```

## Key Task Parameters Explained

### Understanding Your Timing

If using 10ms timesteps (dt=0.01):

| Parameter | Value | Duration |
|-----------|-------|----------|
| T_start | 100-300 | 1.0-3.0 seconds |
| T_cue | 10 | 0.1 seconds (100ms) |
| T_wait | 300 | 3.0 seconds |
| T_movement | 200 | 2.0 seconds |
| **Total valid window** | 300-500 | 3.0-5.0 seconds after cue |

### Trial Timeline (for one example trial)
```
Cue onset at t=150 (1.5s)
│
├─ t=0-150 (0-1.5s): Pre-cue, y should be low
├─ t=150-160 (1.5-1.6s): Cue stimulus
│
├─ t=160-450 (1.6-4.5s): Wait period, y must stay low (very important!)
│ └─ This is the forbidden zone - responses here are false alarms
│
├─ t=450-650 (4.5-6.5s): **VALID RESPONSE WINDOW** ← Encourage y ≥ 0.5 here
│
└─ t=650-1000 (6.5-10s): Post-window, y should be low again
```

## Troubleshooting

### Problem: Network produces valid hits but many false alarms

**Cause:** Baseline loss not strong enough during wait period

**Solutions:**
1. Increase relative weight of baseline loss:
   ```python
   # In batched_rnn_loss
   total_loss = target_loss + 2.0 * baseline_loss  # Weight baseline more
   ```

2. Increase T_wait (give network more time to learn)
   ```python
   T_wait=400  # 4 seconds instead of 3
   ```

3. Add RNN regularization to suppress spontaneous activity

### Problem: Network produces valid hits but many late responses

**Cause:** Response window too wide

**Solutions:**
1. Decrease T_movement:
   ```python
   T_movement=100  # Narrower window (2-4s instead of 3-5s)
   ```

2. Add penalty for sustained responses after valid window:
   ```python
   # In batched_rnn_loss, after baseline_loss:
   post_window = jnp.where(Tarray > valid_window_end, 1, 0)
   late_loss = jnp.sum((ys ** 2) * post_window * batch_mask) / jnp.sum(post_window * batch_mask)
   total_loss = target_loss + baseline_loss + 0.5 * late_loss
   ```

### Problem: Network fails to produce any responses

**Cause:** Could be many things - network capacity, learning rate, weight initialization

**Solutions:**
1. Increase network capacity (more neurons, layers)
2. Increase learning rate or use different optimizer
3. Increase target_loss weight:
   ```python
   total_loss = 2.0 * target_loss + baseline_loss
   ```
4. Check that RNN can produce output > 0.5 (verify activation function)

## Advanced Modifications

### Adding a Sparsity Penalty
```python
def batched_rnn_loss_with_sparsity(rnn_func, params, x0, batch_inputs, 
                                    batch_targets, batch_mask, rng_keys, 
                                    sparsity_weight=0.1):
    ys, _, _ = rnn_func(params, x0, batch_inputs, None, rng_keys)
    
    target_window = batch_targets > 0.5
    non_target = batch_targets <= 0.5
    
    target_loss = jnp.sum(((ys - 1.0) ** 2) * target_window * batch_mask) / (jnp.sum(target_window * batch_mask) + 1e-8)
    baseline_loss = jnp.sum((ys ** 2) * non_target * batch_mask) / (jnp.sum(non_target * batch_mask) + 1e-8)
    
    # Sparsity: penalize any activity (encourage sparse, brief responses)
    sparsity_loss = jnp.mean(ys ** 2)
    
    total_loss = target_loss + baseline_loss + sparsity_weight * sparsity_loss
    return total_loss
```

### Class-Conditional Loss (for multiple task conditions)
```python
# If you have multiple cue types (e.g., visual vs auditory)
def batched_rnn_loss_multimodal(rnn_func, params, x0, batch_inputs, 
                                 batch_targets, batch_mask, batch_cue_type, rng_keys):
    ys, _, _ = rnn_func(params, x0, batch_inputs, None, rng_keys)
    
    losses = []
    for cue_type in [0, 1]:  # Two cue types
        mask_type = batch_cue_type == cue_type
        
        target_window = batch_targets > 0.5
        non_target = batch_targets <= 0.5
        
        # Compute loss only for this cue type
        type_loss = (jnp.sum(((ys[mask_type] - 1.0) ** 2) * target_window[mask_type] * batch_mask[mask_type]) + 
                     jnp.sum((ys[mask_type] ** 2) * non_target[mask_type] * batch_mask[mask_type]))
        losses.append(type_loss)
    
    return jnp.mean(jnp.array(losses))
```

## References & Notes

- **Inspiration**: Signal detection theory, temporal production tasks, interval timing models
- **Related Work**: Peak Interval Procedure (PIP), Bisection Task, Temporal Generalization
- **Neural Correlates**: Striatal medium spiny neurons show ramping activity before timed responses
- **Computational Models**: Multiple timer models, accumulator models, state-dependent networks

## Files Modified

1. `self_timed_movement_task.py`:
   - Updated `self_timed_movement_task()` function with clearer semantics
   - Replaced `batched_rnn_loss()` with principled two-component loss
   - Added `analyze_task_performance()` helper function

2. `DESIGN_NOTES.md` (new):
   - Detailed explanation of loss function design philosophy
   - Task structure and timing explanations
   - Implementation examples and advanced topics

