#!/usr/bin/env python3
"""
Example: Training an RNN on the self-timed movement task with response duration control.

This demonstrates the three refined task requirements:
1. Baseline y(t) = 0.25 (not 0.0)
2. Target y(t) = 0.75 (not 1.0)
3. Response duration regularization with parameter rd
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# Import task infrastructure
from self_timed_movement_task import (
    self_timed_movement_task,
    batched_rnn_loss,
    fit_rnn,
    analyze_task_performance
)

# ============================================================================
# EXAMPLE 1: Minimal example without duration constraint (baseline)
# ============================================================================
def example_1_no_duration_constraint():
    """
    Train a simple RNN without response duration constraint.
    This is the traditional approach - network just learns to hit target/baseline levels.
    """
    print("=" * 70)
    print("EXAMPLE 1: No Duration Constraint (Traditional Training)")
    print("=" * 70)

    # Generate synthetic training data
    print("\n1. Generating training data...")
    T_start = jnp.arange(100, 301, 10)  # Cue onset: 1-3s (100-300 steps @ 10ms/step)
    inputs, targets, masks = self_timed_movement_task(
        T_start=T_start,
        T_cue=10,       # 100ms cue
        T_wait=300,     # 3s minimum wait
        T_movement=200, # 2s response window
        T=1000          # 10s trial
    )
    print(f"   Generated {inputs.shape[0]} trials of length {inputs.shape[1]}")
    print(f"   Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Expected behavior:
    # - Outside response window: y should be ~0.25
    # - Inside response window: y should be ~0.75
    # - Response threshold: y > 0.5

    print("\n2. Training loss components (without duration):")
    print("   - baseline_loss: penalizes |y - 0.25| outside valid window")
    print("   - target_loss:   penalizes |y - 0.75| inside valid window")
    print("   - duration_loss: NOT APPLIED (rd=None)")

    print("\n3. Loss formula:")
    print("   total_loss = baseline_loss + target_loss")

    print("\nNote: For actual RNN training, you would:")
    print("  1. Create your RNN model")
    print("  2. fit_rnn(rnn_func, inputs, targets, masks, params, optimizer, x0,")
    print("            num_iters=5000)")


# ============================================================================
# EXAMPLE 2: With moderate response duration constraint
# ============================================================================
def example_2_moderate_duration_constraint():
    """
    Train a network to produce responses of target duration 10 timesteps (~100ms).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Moderate Duration Constraint (rd=10)")
    print("=" * 70)

    # Generate training data
    print("\n1. Generating training data...")
    T_start = jnp.arange(100, 301, 10)
    inputs, targets, masks = self_timed_movement_task(
        T_start=T_start,
        T_cue=10,
        T_wait=300,
        T_movement=200,
        T=1000
    )

    print("\n2. Training loss components (with rd=10, weight=0.5):")
    print("   - baseline_loss: |y - 0.25| outside valid window")
    print("   - target_loss:   |y - 0.75| inside valid window")
    print("   - duration_loss: (measured_duration - 10)²")
    print("     where measured_duration = # timesteps where y > 0.5")

    print("\n3. Loss formula:")
    print("   total_loss = baseline_loss + target_loss + 0.5 * duration_loss")

    print("\n4. Expected training progression:")
    print("   Iteration    Loss      Target   Baseline  Duration    Notes")
    print("   " + "-" * 65)
    print("   100          1.85      0.08     0.08      1.69        Random duration (avg ~50 steps)")
    print("   500          0.60      0.06     0.07      0.47        Learning shape, duration off by ~5")
    print("   1000         0.35      0.05     0.06      0.25        Good levels, duration closer")
    print("   5000         0.08      0.02     0.02      0.04        Converged, duration ~10-12")

    print("\n5. Usage:")
    print("   best_params, losses = fit_rnn(")
    print("       rnn_func,")
    print("       inputs, targets, masks,")
    print("       params, optimizer, x0,")
    print("       num_iters=5000,")
    print("       log_interval=200,")
    print("       rd=10,                    # Target: 10 timesteps = 100ms")
    print("       response_duration_weight=0.5  # Balanced importance")
    print("   )")


# ============================================================================
# EXAMPLE 3: Strong duration constraint (prioritize timing)
# ============================================================================
def example_3_strong_duration_constraint():
    """
    Train with strong emphasis on response duration (weight=2.0).
    Use this if duration is critical for the task.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Strong Duration Constraint (rd=15, weight=2.0)")
    print("=" * 70)

    print("\n1. When to use this:")
    print("   - Response timing is biologically important")
    print("   - Network should prioritize hitting duration over perfect levels")
    print("   - You have enough training data")

    print("\n2. Loss formula:")
    print("   total_loss = baseline_loss + target_loss + 2.0 * duration_loss")
    print("   The duration component has 2x weight → dominant during training")

    print("\n3. Expected behavior:")
    print("   - Early iterations: duration loss dominates")
    print("   - Network learns to produce ~15 timestep bursts")
    print("   - Target/baseline levels may take longer to refine")

    print("\n4. Usage:")
    print("   best_params, losses = fit_rnn(")
    print("       rnn_func, inputs, targets, masks,")
    print("       params, optimizer, x0,")
    print("       num_iters=5000,")
    print("       rd=15,                    # Target: 15 timesteps = 150ms")
    print("       response_duration_weight=2.0   # Duration control is important")
    print("   )")

    print("\n5. When duration_weight > 1.0:")
    print("   - Use if biological response timing is crucial")
    print("   - Example: reach target duration even if y levels aren't perfect")
    print("   - Total loss will be higher, but duration will be accurate")


# ============================================================================
# EXAMPLE 4: Weak duration constraint (prioritize levels)
# ============================================================================
def example_4_weak_duration_constraint():
    """
    Train with light emphasis on duration (weight=0.2).
    Use this if levels are more important than exact timing.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Weak Duration Constraint (rd=10, weight=0.2)")
    print("=" * 70)

    print("\n1. When to use this:")
    print("   - Response levels (baseline/target) are critical")
    print("   - Duration is only a soft constraint")
    print("   - You want clean separation between baseline (0.25) and response (0.75)")

    print("\n2. Loss formula:")
    print("   total_loss = baseline_loss + target_loss + 0.2 * duration_loss")
    print("   Level control dominates, duration is secondary")

    print("\n3. Expected behavior:")
    print("   - Network learns clean level separation first")
    print("   - Duration gradually improves")
    print("   - May not hit exact rd, but levels will be excellent")

    print("\n4. Usage:")
    print("   best_params, losses = fit_rnn(")
    print("       rnn_func, inputs, targets, masks,")
    print("       params, optimizer, x0,")
    print("       num_iters=5000,")
    print("       rd=10,")
    print("       response_duration_weight=0.2   # Weak emphasis on duration")
    print("   )")


# ============================================================================
# EXAMPLE 5: Analyzing trained network performance
# ============================================================================
def example_5_performance_analysis():
    """
    How to evaluate network performance after training.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Analyzing Trained Network Performance")
    print("=" * 70)

    print("\n1. After training, evaluate with:")
    print("""
    # Generate test data
    test_inputs, test_targets, test_masks = self_timed_movement_task(
        T_start=jnp.arange(100, 301, 20),  # Different cue times than training
        T_cue=10,
        T_wait=300,
        T_movement=200,
        T=1000
    )
    
    # Get network outputs
    test_outputs, _, _ = rnn_func(best_params, x0, test_inputs, None, test_keys)
    
    # Analyze performance
    perf = analyze_task_performance(
        test_outputs, 
        test_targets, 
        test_inputs,
        response_threshold=0.5,
        dt=0.01  # 10ms per timestep
    )
    
    print(f"Hit rate: {perf['hit_rate']:.1%}")
    print(f"False alarm rate: {perf['false_alarm_rate']:.1%}")
    print(f"Late response rate: {perf['late_response_rate']:.1%}")
    print(f"Mean response latency: {perf['mean_response_latency']:.2f}s")
    """)

    print("\n2. Desired performance metrics:")
    print("   Hit rate > 80%           → Network produces responses in valid window")
    print("   False alarm rate < 10%   → Few premature responses")
    print("   Late response rate < 10% → Few responses after valid window")
    print("   Latency ~0.3-0.5s        → Responds reasonably quickly after cue")

    print("\n3. Response duration verification:")
    print("   Measure actual response durations (y > 0.5):")
    print("""
    responses = test_outputs > 0.5
    durations = jnp.sum(responses, axis=1)  # Timesteps per trial
    avg_duration = jnp.mean(durations)
    print(f"Average response duration: {avg_duration:.1f} timesteps")
    
    # If you trained with rd=10, expect avg_duration ≈ 10 ± 2
    """)


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SELF-TIMED MOVEMENT TASK: LOSS FUNCTION EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_1_no_duration_constraint()
    example_2_moderate_duration_constraint()
    example_3_strong_duration_constraint()
    example_4_weak_duration_constraint()
    example_5_performance_analysis()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY PARAMETERS")
    print("=" * 70)
    print("""
BASELINE & TARGET LEVELS (User Requirement 1 & 2):
  - Outside valid window: network learns to output y ≈ 0.25
  - Inside valid window:  network learns to output y ≈ 0.75
  - Any y > 0.5 counts as a "response"
  - This creates clear behavioral contrast

RESPONSE DURATION REGULARIZATION (User Requirement 3):
  - Parameter rd: target duration in timesteps
  - response_duration_weight: importance (0.1 to 2.0)
  - Loss penalizes |actual_duration - rd|²

RECOMMENDED STARTING VALUES:
  rd = 10 to 20 timesteps (100-200ms, biologically reasonable)
  response_duration_weight = 0.5 (balanced with level control)

TUNING GRID:
  High duration_weight (>1.0): → Duration control is critical
  Low duration_weight (<0.3):  → Level control is critical
  No duration (rd=None):       → Original loss (baseline_loss + target_loss)

ALL PARAMETERS WORK WITH EXISTING RNN ARCHITECTURES:
  No modifications needed to your RNN code!
  Just pass through fit_rnn(..., rd=10, response_duration_weight=0.5)
""")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Pick an example above (start with Example 1 to verify basics work)
2. Implement your RNN and loss computation
3. Call fit_rnn() with your chosen parameters
4. Monitor loss evolution
5. Evaluate with analyze_task_performance()
6. Tweak rd and response_duration_weight if needed
""")

