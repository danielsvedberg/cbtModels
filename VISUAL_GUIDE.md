# Self-Timed Movement Task: Visual Guide & Examples

## Visual Task Structure

```
TRIAL TIMELINE (10-second trial with 10ms timesteps)
═══════════════════════════════════════════════════════════════════════

Time (s)   0.0      1.0      2.0      3.0      4.0      5.0      6.0      7.0      8.0      9.0    10.0
Time (idx)  0      100      200      300      400      500      600      700      800      900   1000
Date       ├──────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────┤
           │                                                                                            │
EXAMPLE 1: Cue at 1.5s (T_start=150)
           │        ┌─ CUE (10 ts) ─┐                            ┌───── VALID WINDOW (200 ts) ─────┐
           │        │               │                            │                                  │
Cue IN  │        └─────────────────┘────────────────────────────────────────────────────────────────
           │
y TARGET│        ┌───────────────────────────────────────────────┬───── VALID WINDOW: y ≥ 0.5 ─────┬─────
        │        │ Forbidden Zone (y < 0.5)                   │ │ Response Window (y ≥ 0.5)      │
        │        │ t=150-450 (1.5-4.5s wait)                 │ │ t=450-650 (4.5-6.5s respons)   │
        │        │                                              │ │                                 │
y OUT   │ ────────┴──────────────────────────────────────────────┴─ ──────────────────────────────┴─ ──
        │        │                                                 │                                 │
        │        │← 300ts (3s) wait period minimum ──────────────→│← 200ts (2s) response window ─→│
        └────────┴────────────────────────────────────────────────┴─────────────────────────────────┘

LABELS
───────
Cue:       Input stimulus (binary: ON during t=150-160)
Wait Period: Forbidden zone - responses here are FALSE ALARMS (t=160-450)
Valid Window: Allowed response zone (t=450-650) - responses here are HITS
Late Period: After window - responses here are LATE RESPONSES (t=650+)

Key Distances:
• t=150: Cue onset
• t=160: Cue offset (cue duration = 10 ts = 100ms)
• t=450: Valid window opens (t=160 + 300 = t_start + T_cue + T_wait)
• t=650: Valid window closes (t=450 + 200 = t_move_start + T_movement)
```

## Parameter Settings Examples

### Example 1: Tight Timing (3-5s window, 100ms cue)
```python
T_start = jnp.arange(100, 301, 10)    # 1.0-3.0s cue onset, 10s spacing
T_cue = 10                             # 100ms cue
T_wait = 300                           # 3s minimum wait
T_movement = 200                       # 2s response window (total 3-5s)
T = 1000                               # 10s trial

# Timeline for cue at 1.5s (T_start[5]=150):
# t=150-160: Cue ON
# t=160-450: Wait (forbidden, false alarms if respond) - 290 timesteps
# t=450-650: Valid response zone - 200 timesteps
# Pattern: STRICT TIMING REQUIREMENT
```

### Example 2: Relaxed Timing (2-6s window, 200ms cue + 500ms reaction time)
```python
T_start = jnp.arange(100, 301, 10)    # 1.0-3.0s cue onset
T_cue = 20                             # 200ms cue (longer)
T_wait = 200                           # 2s minimum wait (shorter)
T_movement = 400                       # 4s response window (longer)
T = 1000                               # 10s trial

# Timeline for cue at 1.5s (T_start[5]=150):
# t=150-170: Cue ON
# t=170-370: Wait (forbidden) - 200 timesteps
# t=370-770: Valid response zone - 400 timesteps
# Pattern: MORE FORGIVING TIMING
```

### Example 3: Longer trial structure (for slower RNNs)
```python
T_start = jnp.arange(200, 501, 25)    # 2.0-5.0s cue onset
T_cue = 50                             # 500ms cue
T_wait = 500                           # 5s minimum wait
T_movement = 300                       # 3s response window
T = 2000                               # 20s trial (double length)

# Pattern: More time between events, better for learning
```

## Loss Function Behavior: Visual Explanation

```
LOSS SURFACE FOR SINGLE TIMESTEP DURING VALID WINDOW
════════════════════════════════════════════════════

Target Loss: L_target(y) = (y - 1.0)²
```
  │
  │       L(y)
  │        4 ┌─
  │          │
  │        3 │  ╱╲
  │          │ ╱  ╲
  │        2 │╱    ╲
  │          │      ╲
  │        1 │       ╲
  │          │        ╲
  │        0 └─────────┴───
  │          0  0.5   1.0
  │              y (network output)
  │
  │ When y < 0.5: Large loss → gradient pushes y UP
  │ When y = 1.0: Zero loss → optimal
  │
  └────────────────────────────→ y-output


LOSS SURFACE DURING BASELINE PERIOD (NON-TARGET)
════════════════════════════════════════════════

Baseline Loss: L_baseline(y) = y²
```
  │
  │       L(y)
  │        2 ┌─
  │          │
  │        1 │       ╱╲
  │          │      ╱  ╲
  │        0 └─────╱────┴───
  │          0   0.5   1.0
  │              y (network output)
  │
  │ When y > 0.0: Loss increases → gradient pushes y DOWN
  │ When y = 0.0: Zero loss → optimal
  │
  └────────────────────────────→ y-output


COMBINED EFFECT ACROSS A TRIAL
═════════════════════════════

Trial Structure (arbitrary 200-step trial)
───────────────────────────────────────────
[0-30]   : Pre-cue, use baseline loss
[30-40]  : Cue period, use baseline loss
[40-80]  : Wait period, use baseline loss ← FALSE ALARMS PENALIZED HERE
[80-150] : Valid window, use target loss ← HITS ENCOURAGED HERE
[150-200]: Post-window, use baseline loss ← LATE RESPONSES PENALIZED HERE

Network Behavior (ideally):
───────────────────────────
[0-30]:   y→0  (baseline loss penalizes y>0)
[30-40]:  y→0  (baseline loss penalizes y>0)
[40-80]:  y→0  (baseline loss penalizes y>0) ← CRITICAL INHIBITION
[80-150]: y→1  (target loss penalizes y<1)   ← RESPONSE GENERATION
[150-200]:y→0  (baseline loss penalizes y>0) ← RAPID SUPPRESSION
```

## Concrete NumPy Example

```python
import numpy as np

# Example: Single trial with T=500 timesteps
T = 500
t_cue_start = 50      # Cue at 0.5s
t_cue_end = 60        # Cue duration: 100ms
t_move_start = 350    # Movement window opens at 3.5s
t_move_end = 450      # Movement window closes at 4.5s

# Create target signal
target = np.zeros((T, 1))
target[t_move_start:t_move_end] = 1.0  # 1 during valid window

# Network outputs (hypothetical)
y_good_trial = np.zeros((T, 1))
y_good_trial[350:370] = 1.0  # Responds at right time

y_false_alarm = np.zeros((T, 1))
y_false_alarm[100:120] = 1.0  # Responds too early

y_late = np.zeros((T, 1))
y_late[470:490] = 1.0  # Responds too late

# Compute losses
def loss(y, target):
    target_window = target > 0.5
    non_target = target <= 0.5
    
    target_loss = np.sum((y - 1.0) ** 2 * target_window) / np.sum(target_window)
    baseline_loss = np.sum(y ** 2 * non_target) / np.sum(non_target)
    
    return target_loss + baseline_loss

print("Good trial (response in valid window):")
print(f"  Loss: {loss(y_good_trial, target):.4f}")
print(f"  → Low loss, network learns correct behavior")

print("\nFalse alarm (response before valid window):")
print(f"  Loss: {loss(y_false_alarm, target):.4f}")
print(f"  → Higher baseline_loss due to early response")

print("\nLate response (response after valid window):")
print(f"  Loss: {loss(y_late, target):.4f}")
print(f"  → Higher baseline_loss due to late response")

print("\nNo response:")
y_no_response = np.zeros((T, 1))
print(f"  Loss: {loss(y_no_response, target):.4f}")
print(f"  → High target_loss because window has loss even with y=0")
```

Expected output:
```
Good trial (response in valid window):
  Loss: [small number, ~0.01]
  → Low loss, network learns correct behavior

False alarm (response before valid window):
  Loss: [medium number, ~0.20]
  → Higher baseline_loss due to early response

Late response (response after valid window):
  Loss: [medium number, ~0.20]
  → Higher baseline_loss due to late response

No response:
  Loss: [large number, ~0.50]
  → High target_loss because window has loss even with y=0
```

## Step-by-Step Training Example

### Session 1: Generate data and train

```python
import jax.numpy as jnp
from self_timed_movement_task import self_timed_movement_task, fit_rnn
import optax

# STEP 1: Generate training data
print("Step 1: Generating training data...")
T_start = jnp.arange(100, 301, 10)  # 20 different cue onset times
inputs, targets, masks = self_timed_movement_task(
    T_start=T_start,
    T_cue=10,
    T_wait=300,
    T_movement=200,
    T=1000
)
print(f"  Generated {inputs.shape[0]} trials")
print(f"  Each trial: {inputs.shape[1]} timesteps")

# STEP 2: Initialize your RNN
print("\nStep 2: Initializing RNN...")
# Your RNN initialization code here
# my_rnn = MyRNN(...)
# params = my_rnn.init(...)
# x0 = my_rnn.initial_state()

# STEP 3: Train
print("\nStep 3: Training...")
optimizer = optax.adam(learning_rate=1e-3)
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
print("  Training complete!")

# STEP 4: Evaluate
print("\nStep 4: Evaluating performance...")
from self_timed_movement_task import analyze_task_performance

ys, _, _ = my_rnn.forward(best_params, x0, inputs, None, rng_keys)
perf = analyze_task_performance(ys, targets, inputs)

print(f"  Hit rate: {perf['hit_rate']:.1%}")
print(f"  False alarm rate: {perf['false_alarm_rate']:.1%}")
print(f"  Late response rate: {perf['late_response_rate']:.1%}")
print(f"  Response latency: {perf['mean_response_latency']:.3f}s")
```

## Common Patterns & Interpretations

### Pattern 1: Learning Progresses Well
```
Loss decreases smoothly over training:
Loss: 0.50 → 0.40 → 0.30 → 0.20 → 0.15 → 0.12
Hit rate increases: 10% → 30% → 60% → 80% → 90% → 95%
False alarm rate decreases: 40% → 35% → 20% → 10% → 5% → 3%

Interpretation: Network learning correctly
Next: Monitor for saturation, try harder tasks
```

### Pattern 2: Early Responses (False Alarms) Dominate
```
Loss plateaus: 0.30 → 0.29 → 0.29 → 0.29
Hit rate: 85% but False alarm rate: 70%→ 65% (stuck)

Interpretation: Network responds during wait period
Solution:
1. Increase baseline_loss weight
2. Increase T_wait (more time to learn inhibition)
3. Add RNN recurrent regularization
```

### Pattern 3: No Learning
```
Loss constant: 0.50 → 0.50 → 0.50
Network output is near 0.5 always

Interpretation: Network can't produce enough output
Solution:
1. Check output activation function
2. Increase learning rate
3. Check if network can produce y>0.5 at all
4. Increase target_loss weight
```

## Hyperparameter Tuning Strategies

### If Response Timing is Inaccurate
```
┌─────────────────────────────────────────┐
│ Problem: Too many early responses       │
├─────────────────────────────────────────┤
│ Try 1: Increase T_wait                  │
│   T_wait: 300 → 400  (3s → 4s)         │
│                                         │
│ Try 2: Increase baseline_loss weight    │
│   total_loss = target_loss +            │
│               2.0 * baseline_loss       │
│                                         │
│ Try 3: Strengthen wait period encoding  │
│   Add input that codes "wait period"    │
│   (e.g., input signal that's high=0-300│
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Problem: Too many late responses        │
├─────────────────────────────────────────┤
│ Try 1: Decrease T_movement              │
│   T_movement: 200 → 100 (narrower)     │
│                                         │
│ Try 2: Decrease T_wait slightly         │
│   (less time before responding)         │
│                                         │
│ Try 3: Add brevity penalty              │
│   Penalize sustained activity           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Problem: No responses at all            │
├─────────────────────────────────────────┤
│ Try 1: Increase target_loss weight      │
│   total_loss = 2.0 * target_loss +      │
│               baseline_loss             │
│                                         │
│ Try 2: Relax constraints                │
│   T_wait: 300 → 200 (easier)           │
│   T_movement: 200 → 300 (more time)    │
│                                         │
│ Try 3: Network capacity                 │
│   Increase RNN size                     │
│   More hidden units                     │
└─────────────────────────────────────────┘
```

## Summary Checklist

Before training, verify:
- ✅ T_start values make sense (max should be << T)
- ✅ T_wait > 0 (you want minimum wait time)
-  ✅ T_movement > 0 (you want finite response window)
- ✅ T_start[-1] + T_cue + T_wait + T_movement < T (fits in trial)
- ✅ Response threshold (0.5) matches your task definition
- ✅ dt (timestep size) is consistent with your RNN

During training, monitor:
- ✅ Loss decreasing over iterations
- ✅ Hit rate increasing
- ✅ False alarm rate decreasing
- ✅ No NaN or inf in loss
- ✅ Gradient magnitudes reasonable

After training, check:
- ✅ Hit rate > 50%
- ✅ False alarm rate < 20%
- ✅ Late response rate < 20%
- ✅ Response latencies consistent

