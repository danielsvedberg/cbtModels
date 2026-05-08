# Visual Guide: Updated Self-Timed Movement Task Loss

## Timeline View

```
                       TIMELINE OF A SINGLE TRIAL
        
Time:    0    100    110    410    610    1000
         |----|----|-----|-----|-----|------|
         ↓    ↓    ↓     ↓     ↓     ↓      ↓
       Start  CUE  Wait  VALID  POST  End
               |    END  RESPONSE  WINDOW
              (100ms)  (300ms)  (200ms)

Desired   0.25  0.25  0.25  0.75  0.75   0.25
Output Y:

Loss      BL    BL    BL    TG   TG     BL
Type:

BL  = Baseline Loss    = (y - 0.25)²
TG  = Target Loss      = (y - 0.75)²
       (only if y > 0.5 counts as response)
```

---

## Loss Components: Single Trial Example

### Without Duration Constraint (rd=None)

```
BASELINE PERIOD (t=0-409):
  Target: y = 0.25
  
  If network outputs y = 0.20:  baseline_loss += (0.20 - 0.25)² = 0.0025 ✓ small penalty
  If network outputs y = 0.25:  baseline_loss += (0.25 - 0.25)² = 0.0000 ✓ no penalty
  If network outputs y = 0.30:  baseline_loss += (0.30 - 0.25)² = 0.0025 ✓ small penalty
  If network outputs y = 0.50:  baseline_loss += (0.50 - 0.25)² = 0.0625 ✓ moderate penalty
  If network outputs y = 0.75:  baseline_loss += (0.75 - 0.25)² = 0.2500 ✗ large penalty

VALID RESPONSE PERIOD (t=410-609):
  Target: y = 0.75
  
  If network outputs y = 0.50:  target_loss += (0.50 - 0.75)² = 0.0625 ✓ penalty (not high enough)
  If network outputs y = 0.75:  target_loss += (0.75 - 0.75)² = 0.0000 ✓ no penalty (perfect)
  If network outputs y = 0.80:  target_loss += (0.80 - 0.75)² = 0.0025 ✓ tiny penalty (close)
  If network outputs y = 1.00:  target_loss += (1.00 - 0.75)² = 0.0625 ✓ penalty (too high)
  If network outputs y = 0.20:  target_loss += (0.20 - 0.75)² = 0.3025 ✗ large penalty

NOTE: y > 0.5 is still the response threshold!
      y = 0.6, 0.7, or 0.8 all count as valid responses.
      Loss just encourages aiming for 0.75.
```

### With Duration Constraint (rd=10)

```
ADDITIONAL DURATION LOSS COMPONENT:
  
  Measure: Count timesteps where y > 0.5
  Target:  rd = 10 timesteps
  
  If measured_duration = 5  timesteps:  (5 - 10)² = 25    (large penalty: too short)
  If measured_duration = 10 timesteps:  (10 - 10)² = 0    (no penalty: perfect)
  If measured_duration = 15 timesteps:  (15 - 10)² = 25   (large penalty: too long)
  If measured_duration = 25 timesteps:  (25 - 10)² = 225  (huge penalty: way too long)

TOTAL LOSS = baseline_loss + target_loss + weight × duration_loss

Example:
  baseline_loss    = 0.02 (good baseline maintenance)
  target_loss      = 0.03 (response near 0.75)
  duration_loss    = 4.0  (measured 12 vs target 10)
  weight           = 0.5
  
  total_loss = 0.02 + 0.03 + 0.5 × 4.0 = 2.05
```

---

## Visual: How Loss Changes with Network Output

### Baseline Loss: (y - 0.25)²

```
Loss
  |
  |                      ╱╲
  |                     ╱  ╲
  |                    ╱    ╲
  |                   ╱      ╲
  |   ╱╲             ╱        ╲
  |  ╱  ╲           ╱          ╲
  | ╱    ╲        ╱              
  |╱______╲______╱________________
  +-----------●------------------→ y
     0   0.25  0.5  0.75  1.0
        (lowest loss here - network aims for 0.25)
        
Behavior:
  • y < 0.25: Penalty grows (network too quiet)
  • y = 0.25: No penalty (perfect)
  • y > 0.25: Penalty grows (network too active)
```

### Target Loss: (y - 0.75)²

```
Loss
  |                              ╱╲
  |                             ╱  ╲
  |                            ╱    ╲
  |                           ╱      ╲
  |                          ╱        ╲
  |                    ╱╲              ╲
  |                   ╱  ╲              
  |                  ╱    ╲             
  |_________________╱______●____________→ y
  0   0.25  0.5  0.75  1.0
                (lowest loss here - network aims for 0.75)
                
Behavior:
  • y < 0.75: Penalty grows (response too weak)
  • y > 0.75: Still activates response (y > 0.5) but penalized slightly
  • y = 0.75: No penalty (perfect)
```

### Duration Loss: (measured - rd)²

```
Loss
  |
  |        ╱╲
  |       ╱  ╲
  |      ╱    ╲
  |     ╱      ╲
  |    ╱        ╲
  |   ╱          ╲        ╱
  |  ╱            ╲      ╱
  | ╱              ╲    ╱
  |╱________________●__╱_________→ measured_duration
  0    5   10  (rd=10)  15  20+
           (lowest loss here)

Behavior:
  • measured < 10: Penalty grows (response too short)
  • measured = 10: No penalty (perfect duration)
  • measured > 10: Penalty grows (response too long)
```

---

## Loss Surface: 2D Example

Imagine trial with baseline only (0-409 steps) vs response only (410-609 steps):

```
                     BASELINE PERIOD
                        (400 steps)
                             ↓
RESPONSE PERIOD  
(200 steps)     | Simple Network Output
     ↓          |
Actual Goal:    | y_baseline=0.25, y_response=0.75
                |
                |     ╱╲ (goal: 0.25, 0.75)
                |    ╱  ╲
                |   ╱    ╲ Total Loss
                |  ╱      ╲  Surface
                | ╱        ╲
                |___________●_________
                        (minimum)
                        
When training:
  • Start: Random y values → high loss
  • Early: Network learns to separate baseline from response
  • Mid: Fine-tuning y_baseline and y_response
  • Late: Gradient descent converges toward the goal (0.25, 0.75)
```

---

## Gradient Flow Diagram

### Without Duration Constraint

```
Loss Function
    |
    ├─→ baseline_loss  ─→ ∂L/∂w_baseline  ─→ Gradient updates
    |   (penalizes y ≠ 0.25)              
    |                                        RNN learns
    ├─→ target_loss    ─→ ∂L/∂w_target   ─→ to produce
    |   (penalizes y ≠ 0.75)              
    |                                        baseline activity
    y_out ←────────────────────────         and response
           (network output)               
```

### With Duration Constraint

```
Loss Function
    |
    ├─→ baseline_loss  ─────┐
    |                        ├─→ Total Loss ─→ ∂L/∂parameters ─→ Optimizer
    ├─→ target_loss    ─────┤               
    |                        │
    ├─→ weight × duration_loss  (new!)
    |   (penalizes |measured_duration - rd|²)
    |
    y_out ←─────────────────┐
                            └─→ RNN learns to produce responses
    y > 0.5 ←─────┐           of specific duration
                  └─→ measure duration

When rd=10, weight=0.5:
  Network receives signals:
  1. "Keep baseline at 0.25"      (from baseline_loss)
  2. "Peak response at 0.75"      (from target_loss)  
  3. "Make response ~10 steps"    (from duration_loss, 50% importance)
```

---

## Training Dynamics: Loss Over Time

### Scenario A: No Duration Control

```
Loss Value
  |     ╲
  |      ╲___                     Converges
  |          ╲___              quickly because
  |              ╲____         only 2 components
  |                   ╲___    
  |                       ╲___
  |_________________________●→ Iteration
  0      1000    5000    10000+
         (converged)
```

### Scenario B: With Duration Control (weight=0.5)

```
Loss Value
  |     ╲
  |      ╲                    
  |       ╲___                Takes longer because
  |           ╲               3 components must
  |            ╲___           all align
  |                ╲____    
  |                     ╲___
  |_________________________●→ Iteration
  0      1000    5000    10000+
         (converged)

Extra duration loss component:
  Initially: very high (network duration way off)
  Mid: decreases as duration approach rd
  Late: minimal (duration close to target)
```

---

## Example: Perfect Network

```
Trial Timeline:
Time:              0         100       110        410        610         1000
                   |-----------|---------|---------|---------|-----------|
Input Cue:         Silent      BEEP      Silent    (wait)    (response   Silent
                   (.25)       (.25)     (.25)     window)   window)     (.25)
                   
Expected Output:   0.25        0.25      0.25      0.75      0.75        0.25
Perfect Network:   0.25        0.25      0.25      0.75      0.75        0.25
                   |           |         |         |         |           |
Baseline Loss:     0            0         0         N/A       N/A         0
Target Loss:       N/A          N/A       N/A       0         0           N/A
Duration:          ─────────────────────────────────10────────────────────
                                                      (exactly 200 steps)
Duration Loss:     0 (measured=200, expected=10, but we'd set rd=200)
                   
TOTAL LOSS = 0 for all timesteps ✓ Perfect convergence
```

---

## Real Network Training Example

```
Early Training (Iteration 100):
Time:              0      100    110    410    610    1000
Expected:          0.25   0.25   0.25   0.75   0.75   0.25
Actual Network:    0.30   0.40   0.35   0.40   0.50   0.45
                   ✗      ✗      ✗      ✗      ✗      ✗
Loss at each step: 0.003  0.023  0.010  0.123  0.065  0.040
Duration: 500 steps (network firing everywhere!)
Duration Loss: (500 - 10)² = 240100 (HUGE!)

Mid Training (Iteration 1000):
Expected:          0.25   0.25   0.25   0.75   0.75   0.25
Actual Network:    0.24   0.26   0.25   0.73   0.74   0.26
                   ✓      ✓      ✓      ✓      ✓      ✓
Loss at each step: 0.0    0.001  0.0    0.0004 0.0001 0.0
Duration: 205 steps (getting closer!)
Duration Loss: (205 - 10)² = 38025 (better, but still needs work)

Late Training (Iteration 5000):
Expected:          0.25   0.25   0.25   0.75   0.75   0.25
Actual Network:    0.251  0.249  0.250  0.751  0.749  0.250
                   ✓✓     ✓✓     ✓✓     ✓✓     ✓✓     ✓✓
Loss at each step: ~0.0   ~0.0   ~0.0   ~0.0   ~0.0   ~0.0
Duration: 11 steps (very close!)
Duration Loss: (11 - 10)² = 1 (minimal)

Total Loss: 0 + 0 + 0.5×1 = 0.5 (converged) ✓
```

---

## Summary: What Networks Learn

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Component 1: baseline_loss (y-0.25)²                  │
│  ────────────────────────────────────                  │
│  Network learns:                                        │
│    "Outside response window, stay quiet at 0.25"       │
│    → Reduces spontaneous activity                      │
│    → Creates clean baseline                            │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Component 2: target_loss (y-0.75)²                    │
│  ──────────────────────────────────                    │
│  Network learns:                                        │
│    "During response window, reach y = 0.75"            │
│    → Produces clear, strong response                   │
│    → But y > 0.5 still counts as valid response        │
│    → Prevents saturation (not forcing y = 1.0)         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Component 3: duration_loss (duration-rd)²             │
│  ───────────────────────────────────────               │
│  Network learns:                                        │
│    "Make response last exactly rd timesteps"           │
│    → Controls response duration                        │
│    → Creates brief, time-locked bursts                 │
│    → Biologically realistic response timing            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Common Patterns

### Pattern 1: Network Learns Baseline First

```
Iteration 100  →  Baseline: 0.40, Response: 0.40, Duration: 500
(Bad baseline)     Loss high in both regions

Iteration 500  →  Baseline: 0.25, Response: 0.30, Duration: 300
(Learning!)        Baseline correct, response still wrong

Iteration 1000 →  Baseline: 0.25, Response: 0.72, Duration: 25
(Good!)            Almost there, just need to control duration

Iteration 5000 →  Baseline: 0.25, Response: 0.75, Duration: 10
(Converged)        Perfect!
```

### Pattern 2: With Strong Duration Constraint (weight=2.0)

```
Iteration 100  →  Duration: 500 (VERY bad)
                  Loss high, duration dominates

Iteration 500  →  Duration: 150 (getting better)
                  Network learns to shorten response

Iteration 1000 →  Duration: 25 (close!)
                  Fine-tuning

Iteration 5000 →  Duration: 10 (perfect!)
```

---

## Quick Mental Model

```
WITHOUT duration (rd=None):
┌──────────────────┐
│ Network learns:  │
│ - Quiet baseline │  Loss gradient helps learn
│ - Strong peak    │  2 behaviors
│ (duration random)│  
└──────────────────┘

WITH duration (rd=10, weight=0.5):
┌──────────────────┐
│ Network learns:  │
│ - Quiet baseline │  Loss gradient helps learn
│ - Strong peak    │  3 behaviors
│ - Brief burst    │  (shared equally)
│  (10 timesteps)  │
└──────────────────┘

WITH strong duration (rd=10, weight=2.0):
┌──────────────────┐
│ Network learns:  │
│ - Quiet baseline │  Loss gradient 
│ - Strong peak    │  emphasizes duration
│ - Brief burst    │  (duration dominates)
│  (10 timesteps)  │
└──────────────────┘
```

