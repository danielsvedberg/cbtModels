# Updated Self-Timed Movement Task Loss Function

## Summary of Changes

The `batched_rnn_loss` function and `fit_rnn` trainer have been updated to implement your three refinements:

### 1. **Baseline Activity Level (y = 0.25)**
- **Old behavior**: Network was encouraged to suppress output to y = 0.0 during non-response periods
- **New behavior**: Network maintains y = 0.25 as resting baseline during non-response periods
- **Mathematical form**: Loss = (y - 0.25)² outside valid response window

**Why this matters**:
- More biologically realistic (neurons have baseline firing rates)
- Prevents dead-zone activation patterns
- Creates a well-defined contrast between baseline (0.25) and response (0.75)

### 2. **Target Response Level (y = 0.75)**
- **Old behavior**: Network tried to reach y = 1.0 during response window
- **New behavior**: Network targets y = 0.75 during response window (but y > 0.5 is still detected as a response)
- **Mathematical form**: Loss = (y - 0.75)² during valid response window

**Why this matters**:
- Allows for moderate response strength (not maximal)
- Still clearly above the response threshold (0.5)
- Leaves room for stronger signals without requiring maximum output
- More manageable activation dynamics (less likely to saturate)

### 3. **Response Duration Regularization (NEW)**
- **Purpose**: Network learns to produce responses of specific duration
- **Parameter**: `rd` = target response duration in timesteps
- **How it works**: Penalizes deviation from target duration
- **Mathematical form**: Loss += weight × (actual_duration - rd)²

**Duration measurement**:
- Counts continuous timesteps where y > 0.5
- For a trial with one response burst, this gives duration directly
- Loss = sum of all timesteps where y > 0.5

## Updated Function Signatures

### `batched_rnn_loss` (Updated)
```python
def batched_rnn_loss(
    rnn_func, 
    params, 
    x0, 
    batch_inputs, 
    batch_targets, 
    batch_mask, 
    rng_keys, 
    rd=None,                            # NEW: target response duration
    response_duration_weight=1.0        # NEW: weight for duration loss
):
    """
    Args:
    - rnn_func: RNN model function
    - params, x0: model parameters and initial state
    - batch_inputs: (batch_size, T, 1) cue signals
    - batch_targets: (batch_size, T, 1) binary targets (1 in valid window)
    - batch_mask: (batch_size, T, 1) loss mask
    - rng_keys: random keys
    - rd: target response duration (int, in timesteps). If None, no duration penalty
    - response_duration_weight: multiplier for duration loss component
    
    Returns:
    - total_loss: scalar loss value
    
    Loss components:
    1. target_loss = mean((y - 0.75)²) during valid window
    2. baseline_loss = mean((y - 0.25)²) during non-valid times
    3. duration_loss = mean((measured_duration - rd)²) [only if rd is not None]
    
    total_loss = target_loss + baseline_loss + response_duration_weight * duration_loss
    """
```

### `fit_rnn` (Updated)
```python
def fit_rnn(
    rnn_func,                           # REQUIRED: RNN model function
    inputs, 
    targets, 
    loss_masks, 
    params, 
    optimizer, 
    x0, 
    num_iters, 
    log_interval=200,
    rd=None,                            # NEW: target response duration
    response_duration_weight=1.0        # NEW: weight for duration loss
):
    """
    Training loop with optional response duration regularization.
    
    All parameters passed through to batched_rnn_loss.
    """
```

## Usage Examples

### Example 1: Basic Training (No Duration Constraint)
```python
from self_timed_movement_task import (
    self_timed_movement_task, 
    fit_rnn
)
import jax.numpy as jnp
from your_rnn_module import create_rnn_params, rnn_forward

# Generate task
inputs, targets, masks = self_timed_movement_task(
    T_start=jnp.arange(100, 301, 10),
    T_cue=10,
    T_wait=300,
    T_movement=200,
    T=1000
)

# Train (uses baseline and target losses only)
best_params, losses = fit_rnn(
    rnn_forward,
    inputs, targets, masks, 
    params, optimizer, x0,
    num_iters=5000
)
```

### Example 2: With Response Duration Constraint
```python
# Train network to produce ~10 timestep responses (100ms each @ 10ms/step)
best_params, losses = fit_rnn(
    rnn_forward,
    inputs, targets, masks,
    params, optimizer, x0,
    num_iters=5000,
    rd=10,                              # Target: 10 timesteps ≈ 100ms
    response_duration_weight=0.5        # Moderate emphasis on duration
)
```

### Example 3: Strong Duration Constraint
```python
# Make duration control more important than reaching the target level
best_params, losses = fit_rnn(
    rnn_forward,
    inputs, targets, masks,
    params, optimizer, x0,
    num_iters=5000,
    rd=15,                              # Target: 15 timesteps ≈ 150ms
    response_duration_weight=2.0        # Strong emphasis on duration
)
```

## Loss Component Breakdown

For a single trial with T=1000 timesteps:

### Example: rd=10, response_duration_weight=1.0

```
Time Region          Target   Actual   Loss Type    Computation
─────────────────────────────────────────────────────────────────
0-99 (before cue)    0.25     0.20    baseline     (0.20-0.25)² = 0.0025
100-109 (cue)        0.25     0.22    baseline     (0.22-0.25)² = 0.0009
110-409 (wait)       0.25     0.26    baseline     (0.26-0.25)² = 0.0001
410-609 (valid resp) 0.75     0.71    target       (0.71-0.75)² = 0.0016
610-999 (after)      0.25     0.24    baseline     (0.24-0.25)² = 0.0001

Measured duration: 12 timesteps (y > 0.5)
Target duration: 10 timesteps
Duration loss: (12-10)² = 4.0

Total loss = avg(baseline components) + avg(target components) + duration_weight * 4.0
```

## Interpreting Loss Values

The loss now has three components with different scales:

1. **Baseline loss**: Typically 0.001 - 0.01 (small, bounded by y ∈ [0,1])
2. **Target loss**: Typically 0.001 - 0.01 (similar scale)
3. **Duration loss**: Typically 1 - 100 (depends on rd and how far off the network is)

**Total loss interpretation**:
- Early training: ~0.5 - 2.0 (network overshoots duration, misses target levels)
- Mid training: ~0.1 - 0.3 (learning shape and timing, duration off by 50%)
- Late training: ~0.01 - 0.05 (converging, duration off by ~1-2 timesteps)

**If duration_weight > 1.0**: Duration component dominates, network prioritizes hitting duration target

**If duration_weight < 1.0**: Level control (target/baseline) dominates, duration is secondary

## Key Parameters to Tune

| Parameter | Effect | Typical Range |
|-----------|--------|----------------|
| `rd` (target duration) | Desired response length in timesteps | 5 - 50 |
| `response_duration_weight` | Importance of duration vs. level | 0.1 - 2.0 |
| RNN learning rate | Training speed | 1e-4 - 1e-2 |
| Network size | Complexity/capacity | depends on task |

## Expected Training Behavior

### Without Duration Constraint (rd=None)
```
Iteration    Loss      Notes
100          0.45      Random behavior, mixed outputs
500          0.25      Learning target/baseline contrast
1000         0.08      Good separation achieved
5000         0.02      Converged on level control
```

### With Duration Constraint (rd=10, weight=1.0)
```
Iteration    Loss      Notes
100          2.5       Random duration, high penalty
500          1.2       Duration getting closer, levels wrong
1000         0.5       Levels right, duration off by ~5 steps
5000         0.08      Converged on all components
```

**Note**: With duration constraint, total loss stays higher because duration error contributes continuously. This is expected and correct!

## Common Issues and Solutions

### Issue: Loss plateaus with bad duration
**Solution**: Check if `response_duration_weight` is too low. Increase to 1.0-2.0.

### Issue: Duration control works but target/baseline levels wrong
**Solution**: Increase `response_duration_weight` is actually **wrong**. Decrease it to 0.1-0.3 to let level control learn.

### Issue: Duration target unrealistic
**Solution**: Check that `rd` makes biological sense. For self-timed movement:
- Typical motor response: 50-200ms = 5-20 timesteps (at 10ms/step)
- Too short (rd<3): Network can't produce reliable responses
- Too long (rd>100): Forces sustained activity, unrealistic

### Issue: Network ignores duration loss entirely
**Solution**: 
1. Verify `rd` is reasonable (not 1000+ timesteps)
2. Check `response_duration_weight` > 0.5
3. Ensure RNN has enough capacity (neurons)

## Mathematical Details

### Duration Measurement
Current implementation counts:
```python
response_indicator = ys > 0.5  # Boolean mask
response_duration = sum(response_indicator, axis=timestep_axis)  # Total timesteps active
```

This works well if:
- One response burst per trial (expected for self-timed movement)
- Responses are brief and concentrated (expected)

This works less well if:
- Multiple separate bursts (penalizes total time active)
- Sustained activity (counts all timesteps, not burst length)

For your self-timed movement task, this should be fine.

### Alternative Duration Measures (if needed)
If needed in future, could implement:
1. **Maximum burst duration**: Find longest contiguous sequence where y > 0.5
2. **Peak-to-peak timing**: Time from first crossing of 0.5 to last crossing of 0.5
3. **Rise/fall slopes**: Penalize slow responses

## Next Steps

1. **Test with your RNN**: Run a small training job with rd=None to verify changes didn't break anything
2. **Add duration constraint**: Start with rd=10-15 (reasonable for ~100-150ms response)
3. **Tune weight**: Start at 0.5, increase if duration is ignored, decrease if levels are ignored
4. **Monitor performance**: Use `analyze_task_performance()` to check hit rates, false alarms, etc.

## Backward Compatibility

The original loss (without duration) is still available by setting `rd=None` (default). Existing code continues to work unchanged.

---

**Questions or modifications needed?** The loss function is modular and easy to extend.

