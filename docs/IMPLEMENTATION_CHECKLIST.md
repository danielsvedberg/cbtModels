if # Implementation Checklist

Use this checklist to verify your setup and identify any issues.

## Code Changes ✅

### Files Modified
- [x] `self_timed_movement_task.py`
  - [x] Updated `self_timed_movement_task()` function
  - [x] Replaced `batched_rnn_loss()` function
  - [x] Added `analyze_task_performance()` function

### Files Created (Documentation)
- [x] `DESIGN_NOTES.md` - Loss function design philosophy
- [x] `TRAINING_GUIDE.md` - Practical training guide
- [x] `VISUAL_GUIDE.md` - Visual explanations and examples
- [x] `README_IMPLEMENTATION.md` - This implementation summary

## Before Training: Task Design ✅

### Parameter Definition
- [ ] Decided on cue onset times (T_start range)
  - Typical: 1-3 seconds = T_start in [100, 300] with dt=0.01
- [ ] Decided on cue duration (T_cue)
  - Typical: 100-200ms = 10-20 timesteps
- [ ] Decided on minimum wait time (T_wait)
  - Typical: 3-5 seconds = 300-500 timesteps
- [ ] Decided on response window duration (T_movement)
  - Typical: 1-3 seconds = 100-300 timesteps
- [ ] Decided on total trial duration (T)
  - Must be > T_start[-1] + T_cue + T_wait + T_movement + margin

### Task Parameter Validation
```python
# Check: Total task fits in trial length
assert T_start[-1] + T_cue + T_wait + T_movement < T

# Example with typical values:
assert 300 + 10 + 300 + 200 < 1000  # 810 < 1000 ✓
```

### Data Generation
```python
import jax.numpy as jnp
from self_timed_movement_task import self_timed_movement_task

inputs, targets, masks = self_timed_movement_task(
    T_start=jnp.arange(...),    # 1D array of cue onsets
    T_cue=...,                   # Integer, cue duration
    T_wait=...,                  # Integer, minimum wait
    T_movement=...,              # Integer, response window
    T=...                        # Integer, total trial time
)

# Verify shapes
print(f"inputs shape: {inputs.shape}")      # Should be (num_conditions, T, 1)
print(f"targets shape: {targets.shape}")    # Should be (num_conditions, T, 1)
print(f"masks shape: {masks.shape}")        # Should be (num_conditions, T, 1)

# Verify target signals
print(f"Target min: {targets.min()}")       # Should be 0.0
print(f"Target max: {targets.max()}")       # Should be 1.0
print(f"Number of 1s per trial: {targets.sum(axis=1)[0]}") # Should be T_movement
```

- [ ] Generated training data correctly
- [ ] Verified shapes match (batch_size, T, 1)
- [ ] Verified target signals have correct structure

## Before Training: RNN Setup ✅

### RNN Architecture
- [ ] Defined RNN model (your existing architecture)
  - Should accept: (params, x0, inputs, targets, rng_keys)
  - Should return: (outputs, hidden_states, other)
  - Output shape should be (batch_size, T, 1)

### RNN Initialization
```python
# Your RNN initialization
# my_rnn = MyRNNClass(...)
# params = my_rnn.init(...)
# x0 = my_rnn.initial_state(batch_size)
```

- [ ] RNN initialized with proper parameters
- [ ] Initial state has correct shape
- [ ] RNN can forward pass without errors

### Optimizer Setup
```python
import optax

optimizer = optax.adam(learning_rate=1e-3)
# Or try: optax.sgd(1e-3) or optax.adamw(1e-3)
```

- [ ] Optimizer created
- [ ] Learning rate reasonable (typically 1e-4 to 1e-1)

## Training Phase ✅

### Loss Function Verification
```python
from self_timed_movement_task import batched_rnn_loss
import jax

# Test loss computation
test_loss = batched_rnn_loss(
    rnn_func=my_rnn.forward,     # Or however you call your RNN
    params=params,
    x0=x0_batch,
    batch_inputs=inputs,          # Your training inputs
    batch_targets=targets,        # Your training targets
    batch_mask=masks,             # Your training masks
    rng_keys=jax.random.split(jax.random.PRNGKey(0), inputs.shape[0])
)

print(f"Initial loss: {test_loss}")
# Should be a reasonable number (not NaN, not inf)
# Typically 0.1-1.0 for random networks
```

- [ ] Loss function runs without errors
- [ ] Loss is finite (not NaN, not inf)
- [ ] Loss is reasonable magnitude

### Training Loop
```python
from self_timed_movement_task import fit_rnn

best_params, loss_history = fit_rnn(
    inputs=inputs,
    targets=targets,
    loss_masks=masks,
    params=params,
    optimizer=optimizer,
    x0=x0,
    num_iters=10000,
    log_interval=100
)
```

- [ ] Training runs without errors
- [ ] Loss values print to console
- [ ] Loss decreases over time (or investigate if not)
- [ ] No NaN or inf appearing in loss

### Training Progress Monitoring
```python
# Typical loss progression (should show decreasing trend):
# step 100, loss: 0.487
# step 200, loss: 0.432
# step 300, loss: 0.378
# step 400, loss: 0.325
# ...
# step 10000, loss: 0.045

# Issues to watch for:
# 1. Loss staying constant → increase target_loss weight
# 2. Loss increasing → reduce learning rate
# 3. Loss = NaN → check data for NaN values
# 4. Very slow decrease → network too small, increase capacity
```

- [ ] Loss is decreasing
- [ ] No NaN or inf in loss values
- [ ] Training time reasonable (hours, not weeks)

## Post-Training: Evaluation ✅

### Performance Analysis
```python
from self_timed_movement_task import analyze_task_performance

# Get network outputs on training data
ys, _, _ = my_rnn.forward(best_params, x0, inputs, None, rng_keys)

# Analyze performance
performance = analyze_task_performance(
    ys=ys,                        # Network outputs
    targets=targets,              # Training targets
    inputs=inputs,                # Training inputs
    response_threshold=0.5,       # Your threshold
    dt=0.01                       # 10ms timesteps
)

print(f"Hit rate: {performance['hit_rate']:.1%}")
print(f"False alarm rate: {performance['false_alarm_rate']:.1%}")
print(f"Late response rate: {performance['late_response_rate']:.1%}")
print(f"Correct rejection rate: {performance['correct_rejection_rate']:.1%}")
print(f"Mean response latency: {performance['mean_response_latency']:.3f}s")
print(f"Accuracy: {performance['accuracy']:.3f}")
```

- [ ] Hit rate > 50% (ideally > 70%)
- [ ] False alarm rate < 30% (ideally < 10%)
- [ ] Late response rate < 30% (ideally < 10%)
- [ ] Response latencies make sense (see below)

### Performance Interpretation
```
Good performance (well-trained):
- Hit rate: > 80%
- False alarm rate: < 10%
- Late response rate: < 10%
- Response latency: 0.3-0.8s (depends on T_wait)
- Accuracy: > 0.6

Needs improvement:
- Hit rate: 30-50% → increase target_loss weight or relax timing
- False alarm rate: > 30% → increase baseline_loss weight or T_wait
- Late response rate: > 30% → decrease T_movement or add late penalty
- Response latency: varies widely → add regularization
```

- [ ] Verified performance metrics are reasonable
- [ ] Identified any performance issues

### Response Latency Validation
```
If T_wait = 3 seconds:
- Response latency should be roughly 0.25-1.0s after cue off
- (i.e., responses come 3.25-4.0 seconds after cue onset)
- This means first response time ≈ (cue_time + T_cue + T_wait_actual + 0.25-1.0s)

Example:
- Cue at T_start = 150 (1.5s) = t(1.5s)
- Cue duration = 10 (100ms) → cue off at t(1.6s)
- Expected response: t(1.6 + 0.3 to 1.0) = t(1.9 to 2.6s) ← means latency is 0.3-1.0s
```

- [ ] Response latencies fall in expected range
- [ ] If not, verify T_wait and T_movement are as intended

## Sanity Checks ✅

### Data Sanity
```python
# Check that inputs and targets don't overlap incorrectly
import matplotlib.pyplot as plt

trial_idx = 0
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(inputs[trial_idx])
plt.title(f"Trial {trial_idx}: Cue Input")
plt.ylabel("Input (0=off, 1=on)")

plt.subplot(2, 1, 2)
plt.plot(targets[trial_idx])
plt.title(f"Trial {trial_idx}: Valid Response Window")
plt.ylabel("Target (0=suppress, 1=respond)")
plt.xlabel("Timestep")
plt.tight_layout()
plt.show()

# Verify:
# 1. Cue is a brief pulse
# 2. Target window comes AFTER the cue
# 3. Target window is shorter than the total trial
```

- [ ] Input cues look like brief pulses
- [ ] Target windows appear after cues
- [ ] Target windows don't overlap with cues
- [ ] No unexpected patterns

### Network Output Sanity
```python
# Check that network outputs are reasonable
print(f"Output min: {ys.min()}")   # Should be ~ 0
print(f"Output max: {ys.max()}")   # Should be ~ 1
print(f"Output mean: {ys.mean()}")  # Should be ~ 0.1-0.2

# If outputs are weird:
# min = NaN → check for numerical issues
# max < 0.5 → network can't produce responses
# max > 1.5 → outputs are out of expected range
```

- [ ] Network outputs are in expected range [0, 1]
- [ ] No NaN or inf in network outputs
- [ ] Outputs show variation (not all same value)

### Loss Components Sanity
```python
# Manually compute loss components to verify
target_window = targets > 0.5
non_target = targets <= 0.5

target_loss = jnp.sum(((ys - 1.0) ** 2) * target_window * masks) / jnp.sum(target_window * masks)
baseline_loss = jnp.sum((ys ** 2) * non_target * masks) / jnp.sum(non_target * masks)

print(f"Target loss: {target_loss}")      # Should decrease during training
print(f"Baseline loss: {baseline_loss}")  # Should decrease during training
print(f"Ratio: {baseline_loss/target_loss if target_loss > 0 else 'inf'}")
```

- [ ] Target loss is reasonable magnitude
- [ ] Baseline loss is reasonable magnitude
- [ ] Both losses decrease during training

## Debugging Checklist ✅

### If training fails to start
- [ ] Check that RNN forward pass works: `my_rnn(params, x0, inputs, None, keys)`
- [ ] Check that loss function runs: test it separately (see above)
- [ ] Check that data shapes are correct: (batch, T, 1)
- [ ] Check for NaN in initial data

### If loss doesn't decrease
- [ ] Is learning rate too small? Try 1e-2 instead of 1e-3
- [ ] Is network too small? Add more hidden units
- [ ] Is task too hard? Increase T_wait, decrease T_movement
- [ ] Are gradients vanishing? Check gradient magnitudes: `jax.grad(loss)(params)`
- [ ] Is loss balanced? Try: `2.0 * target_loss + baseline_loss`

### If false alarm rate is high
- [ ] Is baseline_loss weight too low? Try: `target_loss + 2.0 * baseline_loss`
- [ ] Is T_wait too short? Make wait period longer
- [ ] Is cue too memorable? Try longer total trial duration
- [ ] Is network suppressing responses well? Check y values during wait period

### If hit rate is low
- [ ] Is target_loss weight too low? Try: `2.0 * target_loss + baseline_loss`
- [ ] Is T_movement too short? Give network more time to respond
- [ ] Is network capable? Add more capacity
- [ ] Can output reach > 0.5? Check activation function

### If response latencies are wrong
- [ ] Is T_wait correct? Verify in code matches intended time
- [ ] Is cue time in inputs correct? Check first nonzero index
- [ ] Are responses happening in valid window? Use visualization
- [ ] Is RNN processing time-dependent info? May need recurrent connections

## Final Verification Checklist ✅

- [ ] Code runs without errors
- [ ] Training loss decreases over iterations
- [ ] Hit rate > 50%
- [ ] False alarm rate < 30%
- [ ] Late response rate < 30%
- [ ] Response latencies in expected range
- [ ] Can generate new trials with different T_start values
- [ ] Performance generalizes to test set (if you create one)
- [ ] Documentation read and understood

## Success Criteria

Your training is **successful** when:

```
✅ Hit Rate: > 70%
✅ False Alarm Rate: < 15%
✅ Late Response Rate: < 15%
✅ Accuracy: > 0.5
✅ Response latencies: ±0.3s of expected time
✅ Loss: < 0.05 (less than ~2.5% MSE)
```

Your training is **good progress** when:

```
✅ Hit Rate: > 40%
✅ False Alarm Rate: < 35%
✅ Late Response Rate: < 35%
✅ Accuracy: > 0.0
✅ Loss: < 0.2 (decreasing trend)
```

Your training **needs work** when:

```
❌ Hit Rate: < 20%
❌ False Alarm Rate: > 50%
❌ Late Response Rate: > 50%
❌ Accuracy: < 0.0
❌ Loss: not decreasing
```

## Common Success Patterns

### Pattern 1: Smooth Learning
```
Iteration    Loss    Hit%   FA%   Late%
100          0.48    15%    45%   10%
500          0.35    35%    30%   5%
1000         0.25    55%    20%   3%
5000         0.10    78%    12%   5%
10000        0.05    85%    8%    2%

→ Network learning well, stable improvement
```

### Pattern 2: Slow Start, Then Fast
```
Iteration    Loss    Hit%   FA%   Late%
100          0.50    5%     48%   2%
500          0.48    8%     47%   1%
1000         0.40    20%    35%   5%
2000         0.20    65%    18%   8%
5000         0.08    82%    10%   3%

→ Takes time for network to learn timing, then improves quickly
```

### Pattern 3: Plateauing
```
Iteration    Loss    Hit%   FA%   Late%
100          0.40    40%    25%    5%
500          0.38    42%    23%    5%
1000         0.38    43%    24%    5%
5000         0.38    44%    24%    5%

→ Network stuck, probably needs:
   - Higher target_loss weight if hit rate too low
   - Higher baseline_loss weight if false alarms too high
   - Relax timing constraints if timing is issue
```

## Support Resources

If still having issues, refer to:
1. **VISUAL_GUIDE.md** - Visual explanations of task structure
2. **TRAINING_GUIDE.md** - Practical hyperparameter tuning
3. **DESIGN_NOTES.md** - Mathematical background on loss function
4. **README_IMPLEMENTATION.md** - Overview and troubleshooting

Good luck! 🚀

