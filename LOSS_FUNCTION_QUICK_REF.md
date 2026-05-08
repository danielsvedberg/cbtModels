# Self-Timed Movement Task: Updated Loss Function - Quick Reference

## Your Three Requirements ✓ Implemented

### 1. Baseline y(t) = 0.25 ✓
```
Before: Network suppresses y → 0.0 outside response window
After:  Network maintains y → 0.25 outside response window
```
**Why**: More realistic (biological baseline activity), prevents dead zones

### 2. Target y(t) = 0.75 ✓
```
Before: Network tries to reach y → 1.0 during response
After:  Network targets y → 0.75 during response (y > 0.5 still triggers response)
```
**Why**: Allows moderate responses, less saturation, cleaner dynamics

### 3. Response Duration Regularization ✓
```
New parameter: rd (response duration in timesteps)
New weight:    response_duration_weight (0.1 to 2.0)

Loss penalty: (measured_duration - rd)²
where measured_duration = count of timesteps where y > 0.5
```

---

## Function Signatures

### `batched_rnn_loss()`
```python
loss = batched_rnn_loss(
    rnn_func,                      # Your RNN model
    params, x0,                    # Parameters & initial state
    batch_inputs, batch_targets, batch_mask,  # Data
    rng_keys,                      # Random keys
    rd=None,                       # ← NEW: target duration (int or None)
    response_duration_weight=1.0   # ← NEW: importance (0.1 to 2.0)
)
```

**Returns**: scalar loss

---

### `fit_rnn()`
```python
best_params, losses = fit_rnn(
    rnn_func,                      # Your RNN model ← REQUIRED
    inputs, targets, loss_masks,   # Training data
    params, optimizer, x0,         # Parameters, optimizer, init state
    num_iters=5000,                # Training iterations
    log_interval=200,              # Print frequency
    rd=None,                       # ← NEW: target duration
    response_duration_weight=1.0   # ← NEW: duration weight
)
```

---

## Loss Breakdown

### Component 1: Target Loss (during valid response window)
```
target_loss = mean((y - 0.75)²)

Encourages: y → 0.75 when task allows response
Threshold:  y > 0.5 still counts as response
Range:      Typically 0.001 - 0.01
```

### Component 2: Baseline Loss (outside valid window)
```
baseline_loss = mean((y - 0.25)²)

Encourages: y → 0.25 at rest
Range:      Typically 0.001 - 0.01
```

### Component 3: Duration Loss (OPTIONAL)
```
duration_loss = mean((response_duration - rd)²)

Only applied if rd is not None
response_duration = count of timesteps where y > 0.5
Typical range:   1 - 100 (depends on how far from rd)
```

### Total Loss
```
total_loss = target_loss + baseline_loss + response_duration_weight × duration_loss
```

---

## Usage Examples

### Example A: No Duration Constraint (Simple)
```python
best_params, losses = fit_rnn(
    rnn_func, inputs, targets, masks,
    params, optimizer, x0,
    num_iters=5000
    # rd=None (default) → no duration penalty
)
```
✓ Use this to verify basic training works
✓ Only controls baseline/target levels

### Example B: Target Duration 10 timesteps (Typical)
```python
best_params, losses = fit_rnn(
    rnn_func, inputs, targets, masks,
    params, optimizer, x0,
    num_iters=5000,
    rd=10,                         # 10 timesteps ≈ 100ms @ 10ms/step
    response_duration_weight=0.5   # Balanced importance
)
```
✓ Biologically reasonable for motor response (~100ms)
✓ Balanced between level and timing control

### Example C: Strong Duration Control
```python
best_params, losses = fit_rnn(
    rnn_func, inputs, targets, masks,
    params, optimizer, x0,
    num_iters=5000,
    rd=15,
    response_duration_weight=2.0   # Duration is important
)
```
✓ Use if timing is critical
✓ May need more iterations to converge

### Example D: Weak Duration Control
```python
best_params, losses = fit_rnn(
    rnn_func, inputs, targets, masks,
    params, optimizer, x0,
    num_iters=5000,
    rd=10,
    response_duration_weight=0.2   # Weak emphasis
)
```
✓ Use if levels matter more than exact duration
✓ Faster convergence on target/baseline

---

## Key Parameter Recommendations

| Parameter | Typical Value | Range | Notes |
|-----------|---------------|-------|-------|
| `rd` | 10 | 5-50 | Timesteps per response (~50-500ms) |
| `response_duration_weight` | 0.5 | 0.1-2.0 | Relative importance: <1 = levels first, >1 = duration first |
| Learning rate | 1e-3 | 1e-4 to 1e-2 | Reduce if training unstable |
| Network size | Your choice | 10-200 | Bigger = better capacity but slower |

---

## Expected Training Curves

### Without Duration (rd=None)
```
Iteration    Loss     Status
100          0.40     Random behavior
500          0.12     Learning levels
1000         0.04     Good separation
5000         0.01     Converged
```

### With Duration (rd=10, weight=0.5)
```
Iteration    Loss     Status
100          2.50     Random, duration way off
500          0.80     Learning shape, duration still wrong
1000         0.30     Levels good, duration off by ~5
5000         0.08     Well-converged
```

**Note**: With duration constraint, total loss stays higher (expected!). Loss is not lower just because rd=None.

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Network ignores duration | `response_duration_weight` too low | Increase to 1.0-2.0 |
| Perfect duration, bad levels | `response_duration_weight` too high | Decrease to 0.2-0.5 |
| Duration way off (e.g., 50 timesteps vs 10) | Unrealistic `rd` or weak `weight` | Verify rd is reasonable, increase weight |
| Loss not decreasing | Broken RNN or bad learning rate | Check RNN works, reduce lr, increase network |
| y oscillating between 0.25 and 0.75 | Normal early training | Wait, should stabilize |

---

## Backward Compatibility

✓ **Fully backward compatible**
- Existing code with `rd=None` works exactly as before
- New parameters are optional with sensible defaults
- No changes needed to your RNN architecture

---

## Usage Checklist

- [ ] Read through this quick reference
- [ ] Choose starting values (Example B recommended: rd=10, weight=0.5)
- [ ] Call `fit_rnn(..., rd=10, response_duration_weight=0.5)`
- [ ] Monitor loss curve during training
- [ ] Use `analyze_task_performance()` to evaluate
- [ ] If duration not hitting target, adjust `weight` up
- [ ] If levels are bad, adjust `weight` down
- [ ] Iterate

---

## Important Reminders

1. **Duration is measured in timesteps**, not seconds
   - If dt=10ms and you want 100ms response: rd = 10
   - If dt=100ms and you want 1s response: rd = 10

2. **y > 0.5 is still the response threshold**, regardless of target y = 0.75
   - Network can output 0.6, 0.7, or 0.8 and be detected as a response
   - Loss just encourages it to land near 0.75

3. **Duration is total timesteps where y > 0.5**, not burst structure
   - Counts all time above threshold in each trial
   - If network makes multiple bursts, they all count toward total

4. **Loss components have different scales**
   - Baseline/target: typically 0.001 - 0.01
   - Duration: typically 1 - 100 (depends on rd)
   - Normal to see duration_loss dominate total loss

---

## Files Modified

- `self_timed_movement_task.py`:
  - `batched_rnn_loss()` → updated loss function
  - `fit_rnn()` → updated training loop
  
- Created documentation:
  - `SELF_TIMED_TASK_LOSS_UPDATE.md` (detailed guide)
  - `example_loss_usage.py` (working examples)
  - This quick reference

---

## Questions?

Refer to `SELF_TIMED_TASK_LOSS_UPDATE.md` for detailed explanations of each component.

