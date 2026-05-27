# Self-Timed Movement Task: Implementation Summary

## Overview

I've redesigned your self-timed movement task's objective function and loss calculation to be principled, interpretable, and biologically plausible. The new approach directly encodes task structure in the loss function rather than adapting dynamically to network behavior.

## What Was Changed

### File: `self_timed_movement_task.py`

#### 1. **Function: `self_timed_movement_task()`**
- **What changed**: Clearer semantics and documentation
- **Key improvement**: Parameter names now directly map to task timing
  - `T_start`: When cue begins (1-3 seconds)
  - `T_cue`: How long cue lasts (~100ms)
  - `T_wait`: Minimum wait after cue before responding allowed (~3 seconds)
  - `T_movement`: Duration of allowed response window (~2 seconds)
- **Why**: Previous naming was confusing and didn't clearly map to task structure

#### 2. **Function: `batched_rnn_loss()`**
- **What changed**: Complete redesign from adaptive loss to two-component structured loss
- **Old approach**: Modified targets based on network's first response time (problematic!)
- **New approach**: Fixed target signals based on task structure with two loss components:
  ```python
  # During valid response window: encourage y → 1.0
  target_loss = ∑((y - 1.0)² × valid_window)
  
  # Outside valid window: encourage y → 0.0
  baseline_loss = ∑(y² × non_window)
  
  total_loss = target_loss + baseline_loss
  ```
- **Why**: 
  - Principled: directly encodes task objectives
  - Interpretable: understand what each loss component does
  - Stable: doesn't depend on network behavior for loss calculation
  - Biological: encourages low baseline with brief response bursts

#### 3. **New Function: `analyze_task_performance()`**
- **What it does**: Analyzes network performance on task
- **Returns**: Hit rate, false alarm rate, late response rate, response latency
- **Why**: Immediate feedback on what the network is learning

## Key Design Principles

### Principle 1: Task Structure in Loss
Instead of computing loss based on where the network responds, we:
1. Define valid and invalid regions *before* training
2. Apply different loss functions to each region
3. Let gradient descent learn to respond appropriately

### Principle 2: Two Complementary Objectives
- **Target Loss**: "Produce a response when appropriate"
  - Applies: During the valid response window
  - Encourages: y ≥ 0.5 (y → 1.0)
  - Loss: (y - 1.0)²

- **Baseline Loss**: "Stay quiet when not appropriate"
  - Applies: Pre-cue, during wait, post-response
  - Encourages: y < 0.5 (y → 0.0)
  - Loss: y²

### Principle 3: No Post-hoc Masking
The old loss function detected where the network first responded and then created a mask. This is problematic because:
- It changes the loss function per trial (hard to debug)
- It can't penalize early responses that stay suppressed through the valid window
- It couples training dynamics to network behavior

The new approach has fixed, pre-defined regions that don't change.

## Task Parameters Explained

For the self-timed movement task with these parameters:
```python
T_start = jnp.arange(100, 301, 10)      # Cue at 1.0-3.0 seconds
T_cue = 10                               # Cue duration: 100ms
T_wait = 300                             # Minimum 3s wait after cue
T_movement = 200                         # Response window: 2 seconds
T = 1000                                 # Total trial: 10 seconds
```

The network learns:
1. **Suppress responses** for at least 3 seconds after cue (300 timesteps)
2. **Produce a response** between 3-5 seconds after cue (during 200-timestep window)
3. **Suppress again** after the window closes

This implements interval timing with a biologically realistic structure:
- **Cue detection** (immediate)
- **Count to minimum time** (active waiting/timing)
- **Ready to respond** (at minimum time)
- **Response generation** (when threshold is passed)
- **Response termination** (suppression resumes)

## Quick Start

```python
from self_timed_movement_task import self_timed_movement_task, fit_rnn, analyze_task_performance

# 1. Generate training data
inputs, targets, masks = self_timed_movement_task(
    T_start=jnp.arange(100, 301, 10),  # 20 conditions: 1-3 second cue times
    T_cue=10,                           # 100ms cue
    T_wait=300,                         # 3s minimum wait
    T_movement=200,                     # 2s response window (3-5s total)
    T=1000                              # 10s trial
)

# 2. Train
best_params, losses = fit_rnn(
    inputs, targets, masks,
    params=initial_params,
    optimizer=optimizer,
    x0=initial_state,
    num_iters=10000,
    log_interval=100
)

# 3. Evaluate
ys, _, _ = my_rnn(best_params, x0, inputs, None, rng_keys)
performance = analyze_task_performance(ys, targets, inputs)
print(f"Hit rate: {performance['hit_rate']:.1%}")
print(f"False alarms: {performance['false_alarm_rate']:.1%}")
```

## Understanding the Loss Function

Think of it like this:

```
During VALID WINDOW (encourage response):
  y = 0.0 → loss = (0.0 - 1.0)² = 1.0 ✗ Bad
  y = 0.5 → loss = (0.5 - 1.0)² = 0.25
  y = 1.0 → loss = (1.0 - 1.0)² = 0.0 ✓ Good

During BASELINE (encourage suppression):
  y = 0.0 → loss = 0.0² = 0.0 ✓ Good
  y = 0.5 → loss = 0.5² = 0.25
  y = 1.0 → loss = 1.0² = 1.0 ✗ Bad
```

The network learns by gradient descent:
- Move y closer to 1.0 during valid window (reduce target_loss)
- Move y closer to 0.0 during baseline (reduce baseline_loss)
- Both losses are equally weighted, so both matter

## Troubleshooting Guide

### Problem: High false alarm rate
**Symptom**: Network responds during wait period (before valid window)
**Root cause**: Baseline loss not strong enough during wait
**Solution**: 
1. Increase baseline_loss weight: `total = target_loss + 2.0 * baseline_loss`
2. Increase T_wait to make timing easier to learn
3. Add RNN regularization to suppress spontaneous activity

### Problem: High late response rate  
**Symptom**: Network responds after valid window closes
**Root cause**: Window too wide or baseline loss not penalizing
**Solution**:
1. Decrease T_movement (narrow the window)
2. Verify baseline_loss is being applied post-window
3. Add explicit penalty for post-window responses

### Problem: No responses generated
**Symptom**: Network output stays near 0 even during valid window
**Root cause**: Target loss not strong enough, or network capacity too small
**Solution**:
1. Increase target_loss weight: `total = 2.0 * target_loss + baseline_loss`
2. Relax timing constraints (increase T_wait, T_movement)
3. Increase RNN size (more neurons)
4. Check that output activation can reach > 0.5

### Problem: Training loss plateaus
**Symptom**: Loss stops decreasing after some iterations
**Root cause**: Network has hit local minimum or learning rate too low
**Solution**:
1. Increase learning rate
2. Try different optimizer (e.g., SGD with momentum instead of Adam)
3. Reduce task difficulty initially, then increase
4. Check for gradient explosion/vanishing

## Documentation Files

I've created three documentation files to help:

1. **`DESIGN_NOTES.md`** - Deep dive into the loss function design philosophy
   - Explains why this approach works
   - Advanced modifications (sparsity, multi-class, etc.)
   - Related work and references

2. **`TRAINING_GUIDE.md`** - Practical training guide
   - How to use the functions
   - Concrete parameter examples
   - Hyperparameter tuning strategies
   - Advanced loss modifications

3. **`VISUAL_GUIDE.md`** - Visual explanations and examples
   - Timeline diagrams showing task structure
   - Loss surface visualizations
   - Concrete NumPy examples
   - Step-by-step training example
   - Hyperparameter tuning flowcharts

## Key Takeaways

✅ **Do this:**
- Use fixed, pre-defined target signals based on task structure
- Implement loss with separate components for different regions
- Monitor hit rate, false alarm rate, and response latency
- Adjust T_wait and T_movement to control task difficulty
- Start with relaxed timing, gradually increase difficulty

❌ **Don't do this:**
- Compute targets adaptively based on network response
- Use a single loss function for the whole trial
- Ignore false alarms - they reveal timing problems
- Use extremely tight timing windows initially
- Assume one set of hyperparameters works for all RNN architectures

## Questions to Ask During Training

1. **Is the network learning anything?** - Does loss decrease? Yes → proceed. No → increase target_loss weight
2. **Are hits increasing?** - Yes → good. No → check target_loss weight and network capacity
3. **Are false alarms decreasing?** - Yes → good. No → increase baseline_loss weight or T_wait
4. **Is response latency reasonable?** - Should be 0.3-0.5s from cue. If longer → increase T_wait
5. **Is anything weird happening?** - NaN loss? Exploding gradients? → reduce learning rate or check data

## References & Context

- **Behavioral task**: Classic temporal production / self-timed response task
- **Loss design inspired by**: Signal detection theory, temporal learning models
- **Biological basis**: Striatal neurons show ramping activity before timed responses
- **Related literature**: Peak Interval Procedure, Bisection Task, Interval Timing models

## Files Summary

```
cbtModels/
├── self_timed_movement_task.py    ← Code modifications (functions 1-3 above)
├── DESIGN_NOTES.md                ← Design philosophy and deep dives
├── TRAINING_GUIDE.md              ← Practical training guide  
└── VISUAL_GUIDE.md                ← Visual explanations and examples
```

All code is fully functional and ready to use. The helper function `analyze_task_performance()` is optional but recommended for monitoring training progress.

## Next Steps

1. **Read the guides** - Start with VISUAL_GUIDE.md for intuition, then TRAINING_GUIDE.md for practical details
2. **Generate data** - Use `self_timed_movement_task()` with your preferred parameters
3. **Train your RNN** - Use `fit_rnn()` with your model
4. **Monitor progress** - Use `analyze_task_performance()` to track metrics
5. **Tune hyperparameters** - Adjust T_wait, T_movement, and loss weights based on results
6. **Analyze results** - Use existing plotting functions to visualize neural activity patterns

Good luck with your training! The new objective function should make it much clearer what the network is learning and why.

