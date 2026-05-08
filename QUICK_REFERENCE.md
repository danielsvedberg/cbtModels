by# Quick Reference: Self-Timed Movement Task Loss Function

## One-Page Summary

### The Task
```
Network learns to:
1. Wait silently after hearing a cue (minimum 3s)
2. Then produce a brief response (2s window = 3-5s total)
3. Then suppress response again

Timeline:
Cue → [Wait Period: Stay quiet] → [Response Window: Respond!] → [Post-Window: Stay quiet]
```

### The Loss Function  
```python
# Two simple components:

# During response window: minimize (y - 1.0)²
#   → Encourages y to go to 1.0
target_loss = ∑((y - 1.0)² × valid_window_mask)

# Outside response window: minimize y²  
#   → Encourages y to stay at 0.0
baseline_loss = ∑(y² × non_window_mask)

# Total: both components equally important
total_loss = target_loss + baseline_loss
```

### Key Parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `T_start` | [100, 300] | When cue appears (1-3 seconds) |
| `T_cue` | 10 | How long cue lasts (100ms) |
| `T_wait` | 300 | Minimum wait after cue (3s) **← Most Important** |
| `T_movement` | 200 | How long network can respond (2s) |
| `T` | 1000 | Total trial length (10s) |

### Expected Performance
- **Good**: Hit rate >80%, False alarms <10%, Late responses <10%
- **Okay**: Hit rate >40%, False alarms <35%, Late responses <35%
- **Poor**: Hit rate <20%, False alarms >50%, Late responses >50%

## Visual Loss Landscape

```
VALID RESPONSE WINDOW
y should go to 1.0:

Loss = (y - 1.0)²
│
│ 1.0 ╱
│  0.5╱
│   0┴──────────
└─────x───────→ y
     0.5  1.0

BASELINE REGIONS  
y should go to 0.0:

Loss = y²
│
│   1.0╲
│  0.5 ╲
│   0──┴────────
└─────x───────→ y
     0.5  1.0
```

## Training Progression Expected

```
Early training:     Middle training:    Late training:
Loss: 0.45          Loss: 0.20          Loss: 0.05
Hit%: 20%           Hit%: 60%           Hit%: 85%
FA%: 30%            FA%: 15%            FA%: 8%
Late%: 20%          Late%: 15%          Late%: 5%
```

## Parameter Tuning Quick Guide

```
🔴 Too many FALSE ALARMS (early responses)?
   ↓
   Try INCREASE: T_wait (more time to suppress)
   Try INCREASE: baseline_loss weight (2.0 * baseline_loss)
   Try DECREASE: T_cue (shorter stimulus)

🔴 Too many LATE RESPONSES (after window)?
   ↓
   Try DECREASE: T_movement (narrower window)
   Try INCREASE: baseline_loss weight

🔴 NO RESPONSES AT ALL?
   ↓
   Try INCREASE: T_movement (wider window)
   Try DECREASE: T_wait (easier timing)
   Try INCREASE: target_loss weight (2.0 * target_loss)
   Try INCREASE: network size (more capacity)

🟢 EVERYTHING WORKING?
   ↓
   Make task harder!
   - Decrease T_wait (earlier responses required)
   - Decrease T_movement (narrower window)
   - Add more variability to T_start
```

## Code Quick Reference

### Import
```python
from self_timed_movement_task import (
    self_timed_movement_task,  # Generate data
    batched_rnn_loss,          # Compute loss
    fit_rnn,                   # Train
    analyze_task_performance   # Evaluate
)
```

### Generate Data
```python
import jax.numpy as jnp

inputs, targets, masks = self_timed_movement_task(
    T_start=jnp.arange(100, 301, 10),  # 20 conditions
    T_cue=10,
    T_wait=300,
    T_movement=200,
    T=1000
)
```

### Train
```python
import optax

best_params, losses = fit_rnn(
    inputs, targets, masks,
    params, optimizer, x0,
    num_iters=10000,
    log_interval=100
)
```

### Evaluate
```python
perf = analyze_task_performance(
    ys=network_output,
    targets=targets,
    inputs=inputs,
    response_threshold=0.5,
    dt=0.01  # 10ms timesteps
)

print(f"Hits: {perf['hit_rate']:.1%}")
print(f"FAs: {perf['false_alarm_rate']:.1%}")
```

## Loss Components Explained (Intuitive)

```
Think of it like ANIMAL TRAINING:

DURING VALID WINDOW:
  Network produces response (y ≥ 0.5)
  │
  └─→ REWARD (loss goes down!)
      Loss = (y - 1.0)² 
      
  Network stays quiet (y < 0.5)
  │
  └─→ PUNISHMENT (loss goes up!)

DURING BASELINE:
  Network stays quiet (y ≈ 0.0)
  │
  └─→ REWARD (loss goes down!)
      Loss = y²
      
  Network responds (y > 0.5)
  │
  └─→ PUNISHMENT (loss goes up!)
```

The network learns by minimizing loss—just like animals learn to maximize reward!

## What Changed from Old Version

```
OLD (Problematic):
- Loss target computed from network's first response
- Changes per trial based on network behavior ❌
- Hard to debug and interpret ❌
- Can't penalize early false alarms ❌

NEW (Principled):
- Loss target fixed based on task structure
- Same for all trials of same condition ✅
- Clear interpretation ✅
- Naturally penalizes false alarms ✅
```

## Checklist Before Training

```
Data:
  ☐ Generated inputs/targets with correct shapes
  ☐ Target signals show 1s only in response window
  ☐ All inputs < 1.001 and > -0.001
  
RNN:
  ☐ Can run forward pass
  ☐ Output shape is (batch, T, 1)
  ☐ Output can reach > 0.5
  
Training:
  ☐ Optimizer created
  ☐ Loss function runs without errors
  ☐ Initial loss is reasonable (0.1-1.0)
  
Monitoring:
  ☐ Loss values printing to console
  ☐ Loss seems to be decreasing (not constant)
  ☐ No NaN or inf appearing
```

## Common Mistakes & Fixes

```
❌ Loss is NaN
  ✅ Check for NaN in data: jnp.any(jnp.isnan(inputs))
  ✅ Reduce learning rate: 1e-4 instead of 1e-2

❌ Loss stays constant
  ✅ Increase target_loss or baseline_loss weight
  ✅ Increase learning rate
  ✅ Check network can produce outputs

❌ Wrong response timing
  ✅ Verify T_wait and T_movement values
  ✅ Check network outputs during expected periods
  ✅ Visualize: plot(targets[0]) and plot(ys[0])

❌ High false alarm rate despite low loss
  ✅ True or False? Check loss formula:
     - target_loss only counts during valid window
     - baseline_loss counts outside it
  ✅ If actually high: increase baseline_loss weight
  ✅ Verify baseline_loss is implemented correctly
```

## Math on One Page

### Loss Function
```
Let:
  y(t) = network output at time t
  target(t) = 1 if t in valid window, else 0
  T_w = set of timesteps in valid window
  
Then:
  L_target = E[(y - 1)²] for t in T_w
           = [∑_t (y_t - 1)² · target(t)] / |T_w|
  
  L_baseline = E[y²] for t not in T_w  
             = [∑_t y_t² · (1 - target(t))] / (T - |T_w|)
  
  L_total = L_target + L_baseline

Gradient:
  ∂L_target/∂y = 2(y - 1) during valid window
               = 0 elsewhere
               
  ∂L_baseline/∂y = 0 during valid window
                 = 2y elsewhere
```

### Learning Dynamics
```
During valid window:
  ∂L/∂y = 2(y - 1)
  
  If y = 0:   ∂L/∂y = -2  → gradient up, y increases ✓
  If y = 0.5: ∂L/∂y = -1  → gradient up, y increases ✓
  If y = 1.0: ∂L/∂y = 0   → optimal ✓

During baseline:
  ∂L/∂y = 2y
  
  If y = 0:   ∂L/∂y = 0   → optimal ✓
  If y = 0.5: ∂L/∂y = 1   → gradient down, y decreases ✓
  If y = 1.0: ∂L/∂y = 2   → gradient down, y decreases ✓
```

## Next Steps

1. **Start** with VISUAL_GUIDE.md (easy, visual intuition)
2. **Implement** with TRAINING_GUIDE.md (practical steps)
3. **Debug** with IMPLEMENTATION_CHECKLIST.md (verification)
4. **Understand** with DESIGN_NOTES.md (deep theory)
5. **Monitor** with the `analyze_task_performance()` function
6. **Tune** hyperparameters based on hit/FA/late rates

## Key Insight

The loss function naturally implements the task:
- **Valid window:** Pushing y toward 1.0 with (y-1)²
- **Baseline:** Pushing y toward 0.0 with y²
- **No explicit timing:** Timing emerges from RNN's internal state

This is much cleaner than trying to hand-craft timing signals!

---

**TL;DR**: Two loss components, one for responses (target_loss) and one for inhibition (baseline_loss). Together they teach the network when to respond and when to stay silent. Monitor hit rate, false alarms, and late responses to see if it's working.

