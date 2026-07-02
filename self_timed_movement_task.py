import jax
import jax.numpy as jnp
import jax.random as jr

from jax import vmap, jit
from jax import lax
import optax
from jax.nn import tanh


def exc(w):
    return jax.nn.sigmoid(w)
    #return jnp.maximum(0, jnp.tanh(w))
    #return jnp.abs(w)
def inh(w):
    return -exc(w)
    #return -jnp.abs(w)
    #return -jnp.maximum(0, w)


def nln(x):
    return jnp.maximum(0, jax.nn.tanh(x))
    #return jax.nn.sigmoid(4*(x-0.5))
    #return jnp.maximum(0, x**3/(x**3+0.5**3))
    # return jax.nn.softplus(x - 4.0)
    # Hill function: defined on x >= 0 (firing rates are non-negative). Rectify
    # first — a fractional power of a negative input is NaN, which otherwise
    # poisons the whole forward/backward pass.
    #x = jnp.maximum(0.0, x)
    #return x**jnp.exp(1) / (x**jnp.exp(1) + 0.05)


def bg_nln(x, b):
    # Same Hill form with a PKA-shifted half-max; rectify the input (see nln).
    # x = jnp.maximum(0.0, x)
    #return x**jnp.exp(1) / (x**jnp.exp(1) + (1-b))
    c = b/(1-b)
    return jnp.maximum(jax.nn.tanh(c*x), 0)

def bg_nln_inh(x, b):
    c = 1-(b/(1-b))
    return jnp.minimum(jax.nn.tanh(-x*c), 0)



# Default indices into the rnn_func state tuple (cbt_rnn.STATE_AREA_ORDER)
# excluded from the dead-area inactivity floor: the two PKA excitability traces
# (modulatory signals, not area output rates) and the Medulla (excluded so the
# motor-output region can go quiet between responses). This default matches the
# no-STN/no-SC ordering; families with different state orderings (extra areas,
# DA/adenosine concentration states) pass their own skip set as the optional 6th
# rnn_func return, which overrides this default.
_DEAD_AREA_SKIP_INDICES = (7, 8, 9)


def dead_area_floor_loss(all_xs, coef, floor_min, dtype, skip_indices=_DEAD_AREA_SKIP_INDICES):
    """Penalize any region whose mean *late-trial* activity falls below a floor.

    For each area trajectory in ``all_xs`` (the full state tuple returned by
    rnn_func, shaped ``(batch, T, n)``), add ``max(0, floor_min - mean_activity)^2``
    to the penalty, skipping the PKA excitability traces and the Medulla. Summing
    the per-area hinge penalties means a single silenced region is enough to incur
    loss, so training is pushed to keep every (non-excluded) area active.

    The mean is taken over only the **second half** of each time series, so the
    network can't cheat the floor by setting a high initial activity and then
    decaying the region to silence — the penalty only sees steady-state activity.
    Returns 0 when disabled or unavailable.
    """
    if coef == 0.0 or all_xs is None:
        return jnp.array(0.0, dtype=dtype)
    loss = jnp.array(0.0, dtype=dtype)
    for i, area_traj in enumerate(all_xs):
        if i in skip_indices:
            continue
        # area_traj: (batch, T, n) — average only the late (second-half) window.
        half = area_traj.shape[1] // 2
        area_mean = jnp.mean(area_traj[:, half:, :])
        loss = loss + jnp.square(jnp.maximum(0.0, floor_min - area_mean))
    return coef * loss


def dead_projection_loss(params, coef, floor, dtype):
    """Penalize synaptic projections that collapse toward zero.

    Every 2-D weight matrix in ``params`` is a synaptic projection (within- and
    between-area connectivity, plus the cue-input and readout matrices); 1-D
    gains/pacemakers/initial-states and scalars are skipped. A projection counts
    as *dead* when its mean absolute weight falls below ``floor / n_connections``
    (n_connections = number of entries) — equivalently when its total absolute
    weight ``sum|W|`` drops below ``floor``. For each projection we add
    ``max(0, floor - sum|W|)^2``: penalizing the L1 deficit (not the mean) keeps
    the same dead/alive boundary the caller asked for while giving the gradient a
    usable scale, so a collapsing projection is actively pushed back to life.
    Returns 0 when disabled.
    """
    if coef == 0.0:
        return jnp.array(0.0, dtype=dtype)
    loss = jnp.array(0.0, dtype=dtype)
    for w in params.values():
        w = jnp.asarray(w)
        if w.ndim != 2:
            continue
        l1 = jnp.sum(jnp.abs(w))  # = mean|W| * n_connections
        loss = loss + jnp.square(jnp.maximum(0.0, floor - l1))
    return coef * loss


def self_timed_movement_task(T_start, T_cue, T_wait, T_movement, T, null_trial=False):
    """
    Simulate all possible input/output pairs for the self-timed movement task.
    
    The task structure:
    - Cue starts at T_start and lasts T_cue
    - Movement window opens at T_start + T_cue + T_wait (minimum wait before movement allowed)
    - Movement window closes at T_start + T_cue + T_wait + T_movement (maximum time to respond)
    - Responses (y > 0.5) are rewarded only within the movement window

    Arguments:
    T_start: array of possible cue onset times (in timesteps)
    T_cue: duration of cue (in timesteps)
    T_wait: minimum wait time after cue before movement is allowed (in timesteps)
    T_movement: duration of valid response window (in timesteps)
    T: total trial time (in timesteps)
    null_trial: if True, don't include cue (for null trials)

    Returns:
    inputs: (num_starts, T, 1), binary time series representing the cue
    outputs: (num_starts, T, 1), target task structure (1 = valid response window, 0 elsewhere)
    mask: (num_starts, T, 1), loss mask (1 = timesteps to include in loss, 0 = exclude)
    """
    num_starts = T_start.shape[0]

    def _single(interval_ind):
        t_start = T_start[interval_ind]
        t_cue_end = t_start + T_cue
        t_move_start = t_cue_end + T_wait  # Movement window opens here (3s after cue in your case)
        t_move_end = t_move_start + T_movement  # Movement window closes here (5s after cue in your case)

        # Initialize arrays
        inputs = jnp.zeros((T, 1))
        # Target signal: 1 during valid response window, 0 elsewhere
        outputs = jnp.zeros((T, 1))
        # Mask: which timesteps to include in loss calculation
        mask = jnp.ones((T, 1))

        # Add cue to inputs (unless null trial)
        if not null_trial:
            inputs = jax.lax.dynamic_update_slice(inputs, jnp.ones((T_cue, 1)), (t_start, 0))
        
        # Set target output to 1 during valid response window
        outputs = jax.lax.dynamic_update_slice(outputs, jnp.ones((T_movement, 1)), (t_move_start, 0))

        return inputs, outputs, mask

    inputs, outputs, masks = vmap(_single)(jnp.arange(num_starts))

    return inputs, outputs, masks


def pavlovian_task(T_start, T_cue, T_response, T, null_trial=False):
    """
    Pavlovian conditioning task.

    A single cue is delivered to cortex at a random time within the trial
    (T_start is expected to be drawn within [50, T - 50]). The network is
    reinforced for responding immediately after cue offset.

    - Cue starts at T_start[i] and lasts T_cue.
    - Response window opens at cue offset and lasts T_response.
    - The brevity shaping term (applied in ``reinforce_loss``) pushes the
      response toward the very start of that window, i.e. immediately after
      the cue. The silence/tail terms penalize activity before the cue and
      after the window, respectively.

    Arguments:
    T_start:    array of cue onset times (timesteps), drawn within [50, T-50].
    T_cue:      cue duration (timesteps).
    T_response: duration of the rewarded response window (timesteps).
    T:          total trial length (timesteps).
    null_trial: if True, omit the cue (cue-absent control).

    Returns:
    inputs:  (num_starts, T, 1) binary cue time series.
    outputs: (num_starts, T, 1) target (1 within the response window).
    masks:   (num_starts, T, 1) loss mask (all ones — silence/tail penalized).
    """
    num_starts = T_start.shape[0]
    time_idx = jnp.arange(T)

    def _single(interval_ind):
        t_start = T_start[interval_ind]
        t_cue_end = t_start + T_cue
        t_resp_end = t_cue_end + T_response

        # Cue input to cortex.
        inputs = jnp.zeros((T, 1))
        if not null_trial:
            inputs = jax.lax.dynamic_update_slice(inputs, jnp.ones((T_cue, 1)), (t_start, 0))

        # Response window opens at cue offset. Built with a boolean mask so a
        # late cue simply yields a truncated (not shifted) window.
        in_window = (time_idx >= t_cue_end) & (time_idx < t_resp_end)
        outputs = in_window.astype(jnp.float32)[:, None]

        # Full mask: silence (pre-cue) and tail (post-window) activity penalized.
        masks = jnp.ones((T, 1))

        return inputs, outputs, masks

    inputs, outputs, masks = vmap(_single)(jnp.arange(num_starts))
    return inputs, outputs, masks


def get_response_times(all_ys, kwargs, exclude_nan=True):
    n_seeds = all_ys.shape[0]
    n_conditions = all_ys.shape[1]
    test_start_t = kwargs.get('test_start_t', None)
    t_cue = kwargs.get('T_cue', None)
    move_thresh = kwargs.get('move_thresh', 0.5)

    response_times = jnp.full((n_seeds, n_conditions), jnp.nan)  # Default to NaN if no response is detected

    for seed_idx in range(n_seeds):
        for condition_idx in range(n_conditions):
            cue_end = test_start_t[condition_idx] + t_cue
            post_cue_activity = all_ys[seed_idx, condition_idx, cue_end:]  # Activity after the cue
            response_idx = jnp.argmax(post_cue_activity[:, 0] > move_thresh)  # Find first timestep where y > 0.5
            if post_cue_activity[response_idx, 0] > move_thresh: #second filter in case no response
                response_times = response_times.at[seed_idx, condition_idx].set((response_idx) * 0.01)

    # Flatten the response_times array, excluding NaN values
    if exclude_nan:
        valid_response_times = response_times[~jnp.isnan(response_times)].flatten()
    else:
        #replace NaN with T
        valid_response_times = response_times  #.flatten()
    return valid_response_times


def sem(data, axis=0):
    return jnp.std(data, axis=axis) / jnp.sqrt(data.shape[axis] - 1)


# Helper function to calculate mean ± SEM
def compute_mean_sem(data):
    return jnp.mean(data, axis=0), sem(data, axis=0)


def align_to_cue(data, cue_start, new_T=50):
    """
    align data to the cue
    data: shape (n_conditions, T, N) or (n_conditions, T)
    cue_start: shape (n_conditions,)
    return: shape (n_conditions, new_T, N) or (n_conditions, new_T)
    """
    n_conditions = data.shape[0]
    time = data.shape[1]
    ind_range = jnp.arange(time)
    new_data = []
    if n_conditions != len(cue_start):
        raise ValueError('n_conditions should be equal to the length of cue_start')

    for i, t in enumerate(cue_start):
        mask = (ind_range >= t - 100) & (ind_range < t + new_T)
        new_data.append(data[i, mask])

    cue_aligned = jnp.stack(new_data)
    return cue_aligned


def remove_outliers_from_array(data, threshold=3):
    """
    data: shape (cs.n_conditions, T, N) or (cs.n_conditions, T)
    """
    #replace outliers with nan
    z = jnp.abs((data - jnp.mean(data, axis=1, keepdims=True))) / jnp.std(data, axis=1, keepdims=True)
    mask = z > threshold
    return jnp.where(mask, jnp.nan, data)


def analyze_task_performance(ys, targets, inputs, response_threshold=0.5, dt=0.01):
    """
    Analyze network performance on the self-timed movement task.
    
    Args:
    - ys: network outputs, shape (batch_size, T) or (batch_size, T, 1)
    - targets: target signals, shape (batch_size, T) or (batch_size, T, 1)
    - inputs: input cues, shape (batch_size, T) or (batch_size, T, 1)
    - response_threshold: threshold for considering y a response (default 0.5)
    - dt: timestep duration in seconds (default 0.01 = 10ms)
    
    Returns:
    Dictionary with performance metrics:
    - hit_rate: fraction of trials with response in valid window
    - false_alarm_rate: fraction of trials with response before valid window
    - late_response_rate: fraction of trials with response after valid window
    - mean_response_latency: mean time from cue to response (in seconds)
    - accuracy: (hits - false_alarms) / total_trials
    """
    # Handle different input shapes
    if len(ys.shape) == 3:
        ys = ys[..., 0]
    if len(targets.shape) == 3:
        targets = targets[..., 0]
    if len(inputs.shape) == 3:
        inputs = inputs[..., 0]
    
    batch_size, T = ys.shape
    
    # Identify responses (y > threshold)
    responses = ys > response_threshold
    has_response = jnp.any(responses, axis=1)  # (batch_size,)
    first_response_idx = jnp.argmax(responses, axis=1)  # (batch_size,)
    
    # Identify valid/invalid windows
    valid_window = targets > 0.5  # (batch_size, T)
    
    # For each trial, identify the valid window bounds
    valid_window_start = jnp.argmax(targets, axis=1)  # First timestep where target=1
    valid_window_end = T - jnp.argmax(jnp.flip(targets, axis=1), axis=1)  # Last timestep where target=1
    
    # Identify cue onset
    cue_onset = jnp.argmax(inputs, axis=1)  # (batch_size,)
    
    # Categorize responses
    hits = jnp.zeros(batch_size)
    false_alarms = jnp.zeros(batch_size)
    late_responses = jnp.zeros(batch_size)
    correct_rejections = jnp.zeros(batch_size)
    
    for i in range(batch_size):
        if has_response[i]:
            resp_time = first_response_idx[i]
            if resp_time >= valid_window_start[i] and resp_time < valid_window_end[i]:
                hits = hits.at[i].set(1)
            elif resp_time < valid_window_start[i]:
                false_alarms = false_alarms.at[i].set(1)
            else:
                late_responses = late_responses.at[i].set(1)
        else:
            # No response
            if jnp.all(targets[i] == 0):
                correct_rejections = correct_rejections.at[i].set(1)
    
    # Calculate metrics
    hit_rate = jnp.mean(hits)
    false_alarm_rate = jnp.mean(false_alarms)
    late_response_rate = jnp.mean(late_responses)
    correct_rejection_rate = jnp.mean(correct_rejections)
    
    # Response latency (time from cue to response)
    latencies = jnp.zeros(batch_size)
    for i in range(batch_size):
        if has_response[i]:
            latency_steps = first_response_idx[i] - cue_onset[i]
            latencies = latencies.at[i].set(latency_steps * dt)
        else:
            latencies = latencies.at[i].set(jnp.nan)
    
    mean_latency = jnp.nanmean(latencies)
    
    # Accuracy (hits - false alarms) / total
    accuracy = (hit_rate - false_alarm_rate) / 2.0  # Normalized to [0, 1]
    
    return {
        'hit_rate': float(hit_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'late_response_rate': float(late_response_rate),
        'correct_rejection_rate': float(correct_rejection_rate),
        'mean_response_latency': float(mean_latency),
        'accuracy': float(accuracy),
        'mean_latency_std': float(jnp.nanstd(latencies)),
    }


def _stmt_shaping_task(T_start, T_cue, T_wait, T_movement, T, penalize_early, T_cue2=None, null_trial=False):
    """Shared builder for STMT shaping variants with a second (go) cue."""
    if T_cue2 is None:
        T_cue2 = T_cue

    num_starts = T_start.shape[0]
    time_idx = jnp.arange(T)

    def _single(interval_ind):
        t_start = T_start[interval_ind]
        t_cue_end = t_start + T_cue

        # Canonical movement window used across variants.
        t_move_start = t_cue_end + T_wait

        # Second cue at movement-window midpoint.
        t_go = t_move_start + T_movement // 2

        # Two input channels: channel 0 = preparatory cue, channel 1 = go cue.
        ch0 = jnp.zeros((T, 1))
        ch1 = jnp.zeros((T, 1))

        if not null_trial:
            ch0 = jax.lax.dynamic_update_slice(ch0, jnp.ones((T_cue, 1)), (t_start, 0))
            ch1 = jax.lax.dynamic_update_slice(ch1, jnp.ones((T_cue2, 1)), (t_go, 0))

        inputs = jnp.concatenate([ch0, ch1], axis=-1)  # (T, 2)

        # Shape behavior to respond after the second cue.
        outputs = jnp.zeros((T, 1))
        outputs = jax.lax.dynamic_update_slice(
            outputs, jnp.ones((T_movement, 1)), (t_go, 0)
        )

        if penalize_early:
            masks = jnp.ones((T, 1))
        else:
            # Pavlovian variant: no penalty for responses before go cue.
            masks = (time_idx >= t_go).astype(jnp.float32)[:, None]

        return inputs, outputs, masks

    inputs, outputs, masks = vmap(_single)(jnp.arange(num_starts))
    return inputs, outputs, masks


def pavlovian_stmt(T_start, T_cue, T_wait, T_movement, T, T_cue2=None, null_trial=False):
    """
    Pavlovian STMT shaping variant.

    - Adds a second cue at the midpoint of the movement window.
    - Encourages movement after second cue.
    - Early movements are not penalized.

    Returns:
        inputs, outputs, masks with shape (num_starts, T, 1)
    """
    return _stmt_shaping_task(
        T_start=T_start,
        T_cue=T_cue,
        T_wait=T_wait,
        T_movement=T_movement,
        T=T,
        penalize_early=False,
        T_cue2=T_cue2,
        null_trial=null_trial,
    )


def hybrid_stmt(T_start, T_cue, T_wait, T_movement, T, T_cue2=None, null_trial=False):
    """
    Hybrid STMT shaping variant.

    - Same second cue timing as pavlovian_stmt (mid-movement window).
    - Encourages movement after second cue.
    - Early movements are penalized.

    Returns:
        inputs, outputs, masks with shape (num_starts, T, 1)
    """
    return _stmt_shaping_task(
        T_start=T_start,
        T_cue=T_cue,
        T_wait=T_wait,
        T_movement=T_movement,
        T=T,
        penalize_early=True,
        T_cue2=T_cue2,
        null_trial=null_trial,
    )


def _binary_stmt_reward_from_actions(actions, batch_inputs):
    """Reward = 1 for correct STMT behavior, else 0, based on sampled binary actions."""
    # actions: (batch, T) with 0/1 values.
    T = actions.shape[1]
    cue_indicator = batch_inputs[..., 0] > 0.5
    cue_onsets = jnp.argmax(cue_indicator, axis=1)  # (batch,)

    time_idx = jnp.arange(T)[None, :]
    win_start = cue_onsets[:, None] + 300  # 3s after cue (dt=0.01)
    win_end = cue_onsets[:, None] + 600    # 6s after cue

    in_window = (time_idx >= win_start) & (time_idx < win_end)
    responds = actions > 0.5

    responds_in_window = jnp.any(responds & in_window, axis=1)
    responds_outside = jnp.any(responds & (~in_window), axis=1)

    success = responds_in_window & (~responds_outside)
    return success.astype(jnp.float32)


def _masked_target_reward_from_actions(actions, batch_targets, batch_mask):
    """Reward from task-defined target/mask regions (supports shaping variants)."""
    target_2d = batch_targets[..., 0] if batch_targets.ndim == 3 else batch_targets
    mask_2d = batch_mask[..., 0] if batch_mask.ndim == 3 else batch_mask

    valid = mask_2d > 0.5
    target_region = valid & (target_2d > 0.5)
    non_target_region = valid & (target_2d <= 0.5)

    responds = actions > 0.5
    responds_in_target = jnp.any(responds & target_region, axis=1)
    responds_in_non_target = jnp.any(responds & non_target_region, axis=1)

    success = responds_in_target & (~responds_in_non_target)
    return success.astype(jnp.float32)


def reinforce_loss(
    rnn_func,
    params,
    config,
    batch_inputs,
    batch_targets,
    batch_mask,
    rng_keys,
    entropy_coef=0.0,
    objective_mode="log_reward",
    brevity_coef=0.0,
    silence_coef=0.0,
    tail_coef=0.0,
    asym_coef=0.0,
    asym_margin=1.0,
    rest_pka_coef=0.0,
    rest_pka_margin=1.0,
    pathway_floor_coef=0.0,
    pathway_floor_min=1.0,
    c_snc_floor_coef=0.0,
    c_snc_floor_min=0.0,
    gpe_floor_coef=0.0,
    gpe_floor_min=0.0,
    dead_area_coef=0.0,
    dead_area_min=0.0,
    dead_proj_coef=0.0,
    dead_proj_floor=0.1,
):
    """
    Direct objective options for STMT optimization.

    Modes:
    - ``log_reward``: minimize -log P(reward)
    - ``reward_prob``: maximize reward probability directly
    - ``loss_min`` / ``loss``: minimize masked BCE against target trajectories

    A brevity bonus penalizes late responses within the window by adding
    ``brevity_coef * E[normalized_response_time]`` to the loss, where
    normalized_response_time = 0 at window start, 1 at window end.  This
    encourages the network to respond as early as possible.

    ``silence_coef`` penalizes off-window activity, and ``tail_coef``
    penalizes activity after the response window ends.

    ``asym_coef`` enforces biological asymmetry of striatonigral projections:
    dSPN→SNc should be stronger than iSPN→SNc. Penalizes
    ``max(0, ||inh(B_d2_snc)|| - asym_margin * ||inh(B_d1_snc)||)^2`` only
    when both keys exist in ``params``.

    ``pathway_floor_coef`` keeps the direct (D1→SNr) and indirect (D2→GPe)
    pathway projections from degenerating during training. Penalizes
    ``max(0, pathway_floor_min - ||inh(B_d1_snr)||)^2`` and the same for
    ``B_d2_gpe``, summed. Active only when each key is present.

    ``c_snc_floor_coef`` keeps the cortex→SNc excitatory projection from
    collapsing. Penalizes ``max(0, c_snc_floor_min - ||exc(B_c_snc)||)^2``.

    ``rest_pka_coef`` enforces D1 < D2 PKA asymmetry on the *runtime*
    pre-cue PKA trajectories (averaged over t < cue_onset and over neurons),
    not just the initial scalars. Penalizes
    ``max(0, mean_pre(pka_d1) - rest_pka_margin * mean_pre(pka_d2))^2`` when
    the rnn_func returns the PKA traces.

    ``rng_keys`` are kept for API compatibility.
    """
    _rnn_out = rnn_func(params, config, batch_inputs, None, rng_keys)
    ys, pkad1_traj, pkad2_traj = _rnn_out[0], _rnn_out[1], _rnn_out[2]
    # Optional GPe trajectory (exposed by rnn_func when available) for the GPe
    # activity floor; None for families whose rnn_func returns only PKA traces.
    gpe_traj = _rnn_out[3] if len(_rnn_out) > 3 else None
    # Optional full state tuple (exposed last by rnn_func) for the dead-area
    # inactivity floor; None for families whose rnn_func returns only PKA/GPe.
    all_xs = _rnn_out[4] if len(_rnn_out) > 4 else None
    # Optional 6th return: per-family dead-area skip indices (modulatory / quiet
    # states to exclude from the inactivity floor). Falls back to the default.
    dead_skip = _rnn_out[5] if len(_rnn_out) > 5 else _DEAD_AREA_SKIP_INDICES
    probs = jnp.clip(ys[..., 0], 1e-6, 1.0 - 1e-6)  # (batch, T)
    batch_size, T = probs.shape
    eps = 1e-7

    if batch_mask is None:
        mask_2d = jnp.ones((batch_size, T), dtype=probs.dtype)
    else:
        mask_2d = batch_mask[..., 0] if batch_mask.ndim == 3 else batch_mask

    # Reward window: use task-defined target/mask when available, otherwise 3-6s post cue.
    if batch_targets is None:
        cue_indicator = batch_inputs[..., 0] > 0.5
        cue_onsets = jnp.argmax(cue_indicator, axis=1)
        time_idx_raw = jnp.arange(T)[None, :]
        in_window = (time_idx_raw >= cue_onsets[:, None] + 300) & (time_idx_raw < cue_onsets[:, None] + 600)
        target_2d = in_window.astype(probs.dtype)
    else:
        target_2d = batch_targets[..., 0] if batch_targets.ndim == 3 else batch_targets
        in_window = (mask_2d > 0.0) & (target_2d > 0.0)

    # Hazard-model quantities (used for log_reward/reward_prob objectives and brevity bonus).
    # log p(first response at t) = log p_t + sum_{s<t} log(1 - p_s)
    log_surv = jnp.log1p(-probs)
    cum_log_surv = jnp.concatenate(
        [jnp.zeros((batch_size, 1), dtype=probs.dtype), jnp.cumsum(log_surv[:, :-1], axis=1)],
        axis=1,
    )
    log_p_each = jnp.log(probs) + cum_log_surv  # (batch, T)

    neg_inf = jnp.full_like(log_p_each, -1e30)
    masked_log_p_each = jnp.where(in_window, log_p_each, neg_inf)
    log_p_reward = jnp.asarray(jax.scipy.special.logsumexp(masked_log_p_each, axis=1))
    has_window = jnp.any(in_window, axis=1)
    fallback_log_reward = jnp.full_like(log_p_reward, jnp.log(eps))
    log_reward = jnp.where(has_window, log_p_reward, fallback_log_reward)
    reward_probs = jnp.clip(jnp.exp(log_reward), eps, 1.0)

    if objective_mode in ("loss", "loss_min"):
        target_2d_f = target_2d.astype(probs.dtype)
        bce_t = -(
            target_2d_f * jnp.log(probs + eps)
            + (1.0 - target_2d_f) * jnp.log(1.0 - probs + eps)
        )
        pg_loss = jnp.sum(bce_t * mask_2d) / (jnp.sum(mask_2d) + 1e-8)
    elif objective_mode == "log_reward":
        pg_loss = -jnp.mean(log_reward)
    elif objective_mode == "reward_prob":
        pg_loss = -jnp.mean(reward_probs)
    else:
        raise ValueError(
            f"Unsupported objective_mode='{objective_mode}'. "
            "Use 'log_reward', 'reward_prob', 'loss', or 'loss_min'."
        )

    mean_p_reward = jnp.mean(reward_probs)
    mean_logp_reward = jnp.mean(log_reward)

    # Brevity bonus: penalize late responses within the reward window.
    # Uses the hazard model to compute E[normalized_response_time | in_window].
    # normalized_response_time = 0 at window start, 1 at window end.
    in_window_float = in_window.astype(probs.dtype)
    time_idx = jnp.arange(T, dtype=probs.dtype)[None, :]                  # (1, T)
    window_start_t = jnp.argmax(in_window_float, axis=1, keepdims=True).astype(probs.dtype)
    window_lengths = jnp.sum(in_window_float, axis=1, keepdims=True)       # (batch, 1)
    norm_time = jnp.where(
        in_window,
        (time_idx - window_start_t) / (window_lengths + 1e-8),
        0.0,
    )  # (batch, T), in [0, 1] within window
    # p(first response at t) within window, normalized to a proper distribution.
    p_first_in_window = jnp.exp(log_p_each) * in_window_float
    p_first_norm = p_first_in_window / (jnp.sum(p_first_in_window, axis=1, keepdims=True) + 1e-8)
    expected_norm_time = jnp.sum(norm_time * p_first_norm, axis=1)         # (batch,) in [0, 1]
    brevity_term = brevity_coef * jnp.mean(expected_norm_time)

    # Silence/tail shaping terms for "brief pulse then return to zero" behavior.
    valid_mask = mask_2d > 0.0
    has_window_f = has_window.astype(probs.dtype)[:, None]
    window_end_t = window_start_t + window_lengths
    pre_or_non_window = valid_mask & (~in_window) & (time_idx < window_end_t)
    tail_mask = valid_mask & (time_idx >= window_end_t)

    pre_or_non_window_f = pre_or_non_window.astype(probs.dtype) * has_window_f
    tail_mask_f = tail_mask.astype(probs.dtype) * has_window_f

    silence_loss = silence_coef * (
        jnp.sum(probs * pre_or_non_window_f) / (jnp.sum(pre_or_non_window_f) + 1e-8)
    )
    tail_loss = tail_coef * (
        jnp.sum(probs * tail_mask_f) / (jnp.sum(tail_mask_f) + 1e-8)
    )

    # Optional entropy bonus.
    entropy_mask = mask_2d
    entropy_t = -(probs * jnp.log(probs) + (1.0 - probs) * jnp.log(1.0 - probs))
    entropy = jnp.sum(entropy_t * entropy_mask) / (jnp.sum(entropy_mask) + 1e-8)

    # dSPN→SNc should be stronger than iSPN→SNc (biological asymmetry).
    if "B_d1_snc" in params and "B_d2_snc" in params:
        d1_norm = jnp.linalg.norm(inh(params["B_d1_snc"]))
        d2_norm = jnp.linalg.norm(inh(params["B_d2_snc"]))
        asym_loss = asym_coef * jnp.square(jnp.maximum(0.0, d2_norm - asym_margin * d1_norm))
    else:
        asym_loss = jnp.array(0.0, dtype=probs.dtype)

    # Runtime resting PKA: average pkad1/pkad2 over the pre-cue window per trial,
    # then enforce D1 < margin * D2 across the batch. The cue is identified from
    # input channel 0 (consistent with the reward-window code above).
    if pkad1_traj is not None and pkad2_traj is not None:
        cue_indicator_for_rest = batch_inputs[..., 0] > 0.5
        cue_onsets_for_rest = jnp.argmax(cue_indicator_for_rest, axis=1)  # (batch,)
        t_idx_rest = jnp.arange(T)[None, :]                                # (1, T)
        pre_cue_mask = (t_idx_rest < cue_onsets_for_rest[:, None]).astype(probs.dtype)
        pre_cue_count = jnp.sum(pre_cue_mask, axis=1) + 1e-8                # (batch,)
        # pkad1_traj / pkad2_traj: (batch, T, n) — average over neurons → (batch, T).
        pkad1_pop = jnp.mean(pkad1_traj, axis=-1)
        pkad2_pop = jnp.mean(pkad2_traj, axis=-1)
        pre_d1 = jnp.sum(pkad1_pop * pre_cue_mask, axis=1) / pre_cue_count
        pre_d2 = jnp.sum(pkad2_pop * pre_cue_mask, axis=1) / pre_cue_count
        mean_pre_d1 = jnp.mean(pre_d1)
        mean_pre_d2 = jnp.mean(pre_d2)
        rest_pka_loss = rest_pka_coef * jnp.square(
            jnp.maximum(0.0, mean_pre_d1 - rest_pka_margin * mean_pre_d2)
        )
    else:
        rest_pka_loss = jnp.array(0.0, dtype=probs.dtype)

    # Pathway floor: prevent D1→SNr and D2→GPe from being degraded by training.
    pathway_floor_loss = jnp.array(0.0, dtype=probs.dtype)
    if "B_d1_snr" in params:
        d1_snr_norm = jnp.linalg.norm(inh(params["B_d1_snr"]))
        pathway_floor_loss = pathway_floor_loss + jnp.square(
            jnp.maximum(0.0, pathway_floor_min - d1_snr_norm)
        )
    if "B_d2_gpe" in params:
        d2_gpe_norm = jnp.linalg.norm(inh(params["B_d2_gpe"]))
        pathway_floor_loss = pathway_floor_loss + jnp.square(
            jnp.maximum(0.0, pathway_floor_min - d2_gpe_norm)
        )
    pathway_floor_loss = pathway_floor_coef * pathway_floor_loss

    # Cortex→SNc floor: prevent excitatory drive onto SNc from collapsing.
    if "B_c_snc" in params:
        c_snc_norm = jnp.linalg.norm(exc(params["B_c_snc"]))
        c_snc_floor_loss = c_snc_floor_coef * jnp.square(
            jnp.maximum(0.0, c_snc_floor_min - c_snc_norm)
        )
    else:
        c_snc_floor_loss = jnp.array(0.0, dtype=probs.dtype)

    # GPe activity floor: GPe is tonically active in vivo and should not go
    # silent. Penalize the mean GPe activity falling below gpe_floor_min so the
    # runaway D2->GPe inhibition can't train GPe to zero.
    if gpe_floor_coef != 0.0 and gpe_traj is not None:
        gpe_mean = jnp.mean(gpe_traj)
        gpe_floor_loss = gpe_floor_coef * jnp.square(jnp.maximum(0.0, gpe_floor_min - gpe_mean))
    else:
        gpe_floor_loss = jnp.array(0.0, dtype=probs.dtype)

    # Dead-area inactivity floor: require every region to stay active over the
    # latter half of each trial (see dead_area_floor_loss).
    dead_area_loss = dead_area_floor_loss(all_xs, dead_area_coef, dead_area_min, probs.dtype, dead_skip)

    # Dead-projection floor: keep every synaptic projection from collapsing to
    # zero (mean |weight| < dead_proj_floor / n_connections).
    dead_proj_loss = dead_projection_loss(params, dead_proj_coef, dead_proj_floor, probs.dtype)

    total_loss = (
        pg_loss - entropy_coef * entropy + brevity_term + silence_loss + tail_loss
        + asym_loss + rest_pka_loss + pathway_floor_loss + c_snc_floor_loss
        + gpe_floor_loss + dead_area_loss + dead_proj_loss
    )
    aux = {
        "success_rate": mean_p_reward,
        "reward_mean": mean_p_reward,
        "log_reward_mean": mean_logp_reward,
        "pg_loss": pg_loss,
        "entropy": entropy,
        "brevity_loss": brevity_term,
        "expected_norm_time": jnp.mean(expected_norm_time),
        "silence_loss": silence_loss,
        "tail_loss": tail_loss,
        "asym_loss": asym_loss,
        "rest_pka_loss": rest_pka_loss,
        "pathway_floor_loss": pathway_floor_loss,
        "c_snc_floor_loss": c_snc_floor_loss,
        "gpe_floor_loss": gpe_floor_loss,
        "dead_area_loss": dead_area_loss,
        "dead_proj_loss": dead_proj_loss,
    }
    return total_loss, aux


def fit_rnn_reinforce(
    rnn_func,
    params,
    config,
    inputs,
    loss_masks,
    optimizer,
    num_iters,
    log_interval=200,
    seed=0,
    baseline_momentum=0.9,
    entropy_coef=0.0,
    batch_targets=None,
    objective_mode="log_reward",
    brevity_coef=0.0,
    silence_coef=0.0,
    tail_coef=0.0,
    asym_coef=0.0,
    asym_margin=1.0,
    rest_pka_coef=0.0,
    rest_pka_margin=1.0,
    pathway_floor_coef=0.0,
    pathway_floor_min=1.0,
    c_snc_floor_coef=0.0,
    c_snc_floor_min=0.0,
    gpe_floor_coef=0.0,
    gpe_floor_min=0.0,
    dead_area_coef=0.0,
    dead_area_min=0.0,
    dead_proj_coef=0.0,
    dead_proj_floor=0.1,
):
    """
    Train an RNN policy with REINFORCE on the binary STMT objective.

    Returns:
    - best_params: parameters with the lowest tracked policy loss
    - losses: list of scalar policy losses logged every log_interval
    - reward_means: list of mean batch rewards logged every log_interval
    """
    opt_state = optimizer.init(params)
    n_data = inputs.shape[0]
    rng_key = jr.PRNGKey(seed)
    baseline = jnp.array(0.0, dtype=jnp.float32)

    @jit
    def _step(carry, _):
        cur_params, cur_opt_state, cur_rng_key, cur_baseline = carry

        cur_rng_key, subkey = jr.split(cur_rng_key)
        batch_rng_keys = jr.split(subkey, n_data)

        (loss_value, aux), grads = jax.value_and_grad(
            lambda prms: reinforce_loss(
                rnn_func,
                prms,
                config,
                inputs,
                batch_targets,
                loss_masks,
                batch_rng_keys,
                entropy_coef,
                objective_mode,
                brevity_coef,
                silence_coef,
                tail_coef,
                asym_coef,
                asym_margin,
                rest_pka_coef,
                rest_pka_margin,
                pathway_floor_coef,
                pathway_floor_min,
                c_snc_floor_coef,
                c_snc_floor_min,
                gpe_floor_coef,
                gpe_floor_min,
                dead_area_coef,
                dead_area_min,
                dead_proj_coef,
                dead_proj_floor,
            ),
            has_aux=True,
        )(cur_params)

        updates, cur_opt_state = optimizer.update(grads, cur_opt_state, cur_params)
        cur_params = optax.apply_updates(cur_params, updates)

        new_baseline = baseline_momentum * cur_baseline + (1.0 - baseline_momentum) * aux["reward_mean"]

        return (cur_params, cur_opt_state, cur_rng_key, new_baseline), (
            loss_value,
            aux["reward_mean"],
            aux["success_rate"],
            aux["log_reward_mean"],
            aux["entropy"],
            aux["brevity_loss"],
            aux["expected_norm_time"],
            aux["silence_loss"],
            aux["tail_loss"],
            aux["asym_loss"],
            aux["rest_pka_loss"],
            aux["pathway_floor_loss"],
            aux["c_snc_floor_loss"],
            aux["gpe_floor_loss"],
            aux["dead_area_loss"],
            aux["dead_proj_loss"],
        )

    losses = []
    reward_means = []
    best_loss = float("inf")
    best_params = params

    for n in range(num_iters // log_interval):
        (params, opt_state, rng_key, baseline), (
            loss_vals,
            reward_vals,
            success_vals,
            log_reward_vals,
            entropy_vals,
            brevity_vals,
            norm_time_vals,
            silence_vals,
            tail_vals,
            asym_vals,
            rest_pka_vals,
            pathway_floor_vals,
            c_snc_floor_vals,
            gpe_floor_vals,
            dead_area_vals,
            dead_proj_vals,
        ) = lax.scan(
            _step,
            (params, opt_state, rng_key, baseline),
            None,
            length=log_interval,
        )

        last_loss = float(loss_vals[-1])
        last_reward = float(reward_vals[-1])
        last_success = float(success_vals[-1])
        last_log_reward = float(log_reward_vals[-1])
        last_entropy = float(entropy_vals[-1])
        last_brevity = float(brevity_vals[-1])
        last_norm_time = float(norm_time_vals[-1])
        last_silence = float(silence_vals[-1])
        last_tail = float(tail_vals[-1])
        last_asym = float(asym_vals[-1])
        last_rest_pka = float(rest_pka_vals[-1])
        last_pathway_floor = float(pathway_floor_vals[-1])
        last_c_snc_floor = float(c_snc_floor_vals[-1])
        last_gpe_floor = float(gpe_floor_vals[-1])
        last_dead_area = float(dead_area_vals[-1])
        last_dead_proj = float(dead_proj_vals[-1])

        losses.append(last_loss)
        reward_means.append(last_reward)

        print(
            f"step {(n + 1) * log_interval}, "
            f"loss: {last_loss:.6f}, reward: {last_reward:.4f}, "
            f"log_reward: {last_log_reward:.4f}, success: {last_success:.4f}, entropy: {last_entropy:.4f}, "
            f"brevity: {last_brevity:.4f}, norm_resp_time: {last_norm_time:.3f}, "
            f"silence: {last_silence:.4f}, tail: {last_tail:.4f}, asym: {last_asym:.4f}, "
            f"rest_pka: {last_rest_pka:.4f}, pathway_floor: {last_pathway_floor:.4f}, "
            f"c_snc_floor: {last_c_snc_floor:.4f}, gpe_floor: {last_gpe_floor:.4f}, "
            f"dead_area: {last_dead_area:.4f}, dead_proj: {last_dead_proj:.4f}"
        )

        if last_loss < best_loss:
            best_loss = last_loss
            best_params = params

    return best_params, losses, reward_means


def supervised_loss(
    rnn_func,
    params,
    config,
    batch_inputs,
    batch_targets,
    batch_mask,
    rng_keys,
    loss_type="bce",
    asym_coef=0.0,
    asym_margin=1.0,
    rest_pka_coef=0.0,
    rest_pka_margin=1.0,
    pathway_floor_coef=0.0,
    pathway_floor_min=1.0,
    c_snc_floor_coef=0.0,
    c_snc_floor_min=0.0,
    gpe_floor_coef=0.0,
    gpe_floor_min=0.0,
    dead_area_coef=0.0,
    dead_area_min=0.0,
    dead_proj_coef=0.0,
    dead_proj_floor=0.1,
):
    """Dense supervised loss: match the network output to the target trajectory.

    Unlike ``reinforce_loss`` (which optimizes a sampled-policy reward via a
    hazard model), this directly regresses the deterministic output ``ys``
    onto the task's target waveform at every masked timestep. The gradient is
    therefore dense and informative even when the policy earns no reward, which
    sidesteps the REINFORCE cold-start problem.

    Args:
        loss_type: ``"bce"`` for masked binary cross-entropy (output treated as
            a per-timestep probability) or ``"mse"`` for mean squared error.

    The optional ``asym_*``/``rest_pka_*``/``pathway_floor_*``/``c_snc_floor_*``
    terms are the same biological structural penalties used by
    ``reinforce_loss`` (all default to off), so the same priors can be carried
    into supervised training. ``rng_keys`` are forwarded to ``rnn_func`` so
    state noise still regularizes training.

    Returns ``(total_loss, aux)`` where ``aux`` exposes the supervised loss and
    output-matching diagnostics.
    """
    _rnn_out = rnn_func(params, config, batch_inputs, None, rng_keys)
    ys, pkad1_traj, pkad2_traj = _rnn_out[0], _rnn_out[1], _rnn_out[2]
    # Optional GPe trajectory (exposed by rnn_func when available) for the GPe
    # activity floor; None for families whose rnn_func returns only PKA traces.
    gpe_traj = _rnn_out[3] if len(_rnn_out) > 3 else None
    # Optional full state tuple (exposed last) for the dead-area inactivity floor.
    all_xs = _rnn_out[4] if len(_rnn_out) > 4 else None
    # Optional 6th return: per-family dead-area skip indices (modulatory / quiet
    # states to exclude from the inactivity floor). Falls back to the default.
    dead_skip = _rnn_out[5] if len(_rnn_out) > 5 else _DEAD_AREA_SKIP_INDICES
    eps = 1e-7
    probs = jnp.clip(ys[..., 0], eps, 1.0 - eps)  # (batch, T)
    batch_size, T = probs.shape

    if batch_mask is None:
        mask_2d = jnp.ones((batch_size, T), dtype=probs.dtype)
    else:
        mask_2d = batch_mask[..., 0] if batch_mask.ndim == 3 else batch_mask

    if batch_targets is None:
        raise ValueError("supervised_loss requires batch_targets (the target trajectory).")
    target_2d = batch_targets[..., 0] if batch_targets.ndim == 3 else batch_targets
    target_2d = target_2d.astype(probs.dtype)

    mask_sum = jnp.sum(mask_2d) + 1e-8
    if loss_type == "bce":
        per_t = -(
            target_2d * jnp.log(probs)
            + (1.0 - target_2d) * jnp.log(1.0 - probs)
        )
    elif loss_type == "mse":
        per_t = jnp.square(probs - target_2d)
    else:
        raise ValueError(f"Unsupported loss_type='{loss_type}'. Use 'bce' or 'mse'.")
    sup_loss = jnp.sum(per_t * mask_2d) / mask_sum

    # Output-matching diagnostics (no effect on the gradient).
    valid = mask_2d > 0.0
    accuracy = jnp.sum(((probs > 0.5) == (target_2d > 0.5)) * mask_2d) / mask_sum
    in_window = valid & (target_2d > 0.5)
    off_window = valid & (target_2d <= 0.5)
    in_rate = jnp.sum(probs * in_window) / (jnp.sum(in_window) + 1e-8)
    off_rate = jnp.sum(probs * off_window) / (jnp.sum(off_window) + 1e-8)

    # dSPN->SNc should be stronger than iSPN->SNc (biological asymmetry).
    if "B_d1_snc" in params and "B_d2_snc" in params:
        d1_norm = jnp.linalg.norm(inh(params["B_d1_snc"]))
        d2_norm = jnp.linalg.norm(inh(params["B_d2_snc"]))
        asym_loss = asym_coef * jnp.square(jnp.maximum(0.0, d2_norm - asym_margin * d1_norm))
    else:
        asym_loss = jnp.array(0.0, dtype=probs.dtype)

    # Runtime resting-PKA asymmetry: D1 < margin * D2 averaged over the pre-cue window.
    if rest_pka_coef != 0.0 and pkad1_traj is not None and pkad2_traj is not None:
        cue_indicator = batch_inputs[..., 0] > 0.5
        cue_onsets = jnp.argmax(cue_indicator, axis=1)
        t_idx = jnp.arange(T)[None, :]
        pre_cue_mask = (t_idx < cue_onsets[:, None]).astype(probs.dtype)
        pre_cue_count = jnp.sum(pre_cue_mask, axis=1) + 1e-8
        pre_d1 = jnp.sum(jnp.mean(pkad1_traj, axis=-1) * pre_cue_mask, axis=1) / pre_cue_count
        pre_d2 = jnp.sum(jnp.mean(pkad2_traj, axis=-1) * pre_cue_mask, axis=1) / pre_cue_count
        rest_pka_loss = rest_pka_coef * jnp.square(
            jnp.maximum(0.0, jnp.mean(pre_d1) - rest_pka_margin * jnp.mean(pre_d2))
        )
    else:
        rest_pka_loss = jnp.array(0.0, dtype=probs.dtype)

    # Pathway floors: keep D1->SNr and D2->GPe from degrading.
    pathway_floor_loss = jnp.array(0.0, dtype=probs.dtype)
    if "B_d1_snr" in params:
        pathway_floor_loss = pathway_floor_loss + jnp.square(
            jnp.maximum(0.0, pathway_floor_min - jnp.linalg.norm(inh(params["B_d1_snr"])))
        )
    if "B_d2_gpe" in params:
        pathway_floor_loss = pathway_floor_loss + jnp.square(
            jnp.maximum(0.0, pathway_floor_min - jnp.linalg.norm(inh(params["B_d2_gpe"])))
        )
    pathway_floor_loss = pathway_floor_coef * pathway_floor_loss

    # Cortex->SNc floor.
    if "B_c_snc" in params:
        c_snc_floor_loss = c_snc_floor_coef * jnp.square(
            jnp.maximum(0.0, c_snc_floor_min - jnp.linalg.norm(exc(params["B_c_snc"])))
        )
    else:
        c_snc_floor_loss = jnp.array(0.0, dtype=probs.dtype)

    # GPe activity floor (see reinforce_loss): keep GPe tonically active.
    if gpe_floor_coef != 0.0 and gpe_traj is not None:
        gpe_floor_loss = gpe_floor_coef * jnp.square(
            jnp.maximum(0.0, gpe_floor_min - jnp.mean(gpe_traj))
        )
    else:
        gpe_floor_loss = jnp.array(0.0, dtype=probs.dtype)

    # Dead-area inactivity floor (see reinforce_loss): keep every region active
    # over the latter half of each trial.
    dead_area_loss = dead_area_floor_loss(all_xs, dead_area_coef, dead_area_min, probs.dtype, dead_skip)

    # Dead-projection floor (see reinforce_loss): keep every synaptic projection
    # from collapsing toward zero.
    dead_proj_loss = dead_projection_loss(params, dead_proj_coef, dead_proj_floor, probs.dtype)

    total_loss = (sup_loss + asym_loss + rest_pka_loss + pathway_floor_loss
                  + c_snc_floor_loss + gpe_floor_loss + dead_area_loss + dead_proj_loss)
    aux = {
        "sup_loss": sup_loss,
        "accuracy": accuracy,
        "in_window_rate": in_rate,
        "off_window_rate": off_rate,
        "asym_loss": asym_loss,
        "rest_pka_loss": rest_pka_loss,
        "pathway_floor_loss": pathway_floor_loss,
        "c_snc_floor_loss": c_snc_floor_loss,
        "gpe_floor_loss": gpe_floor_loss,
        "dead_area_loss": dead_area_loss,
        "dead_proj_loss": dead_proj_loss,
    }
    return total_loss, aux


def fit_rnn_supervised(
    rnn_func,
    params,
    config,
    inputs,
    loss_masks,
    optimizer,
    num_iters,
    batch_targets,
    log_interval=200,
    seed=0,
    loss_type="bce",
    asym_coef=0.0,
    asym_margin=1.0,
    rest_pka_coef=0.0,
    rest_pka_margin=1.0,
    pathway_floor_coef=0.0,
    pathway_floor_min=1.0,
    c_snc_floor_coef=0.0,
    c_snc_floor_min=0.0,
    gpe_floor_coef=0.0,
    gpe_floor_min=0.0,
    dead_area_coef=0.0,
    dead_area_min=0.0,
    dead_proj_coef=0.0,
    dead_proj_floor=0.1,
):
    """Train an RNN by direct supervision against the task target trajectory.

    Mirrors ``fit_rnn_reinforce`` (same optimizer/scan/logging structure) but
    optimizes ``supervised_loss`` instead of the policy-gradient objective.
    ``batch_targets`` is required.

    Returns:
    - best_params: parameters with the lowest tracked supervised loss
    - losses: list of scalar losses logged every log_interval
    - accuracies: list of masked output-matching accuracies logged likewise
    """
    opt_state = optimizer.init(params)
    n_data = inputs.shape[0]
    rng_key = jr.PRNGKey(seed)

    @jit
    def _step(carry, _):
        cur_params, cur_opt_state, cur_rng_key = carry

        cur_rng_key, subkey = jr.split(cur_rng_key)
        batch_rng_keys = jr.split(subkey, n_data)

        (loss_value, aux), grads = jax.value_and_grad(
            lambda prms: supervised_loss(
                rnn_func,
                prms,
                config,
                inputs,
                batch_targets,
                loss_masks,
                batch_rng_keys,
                loss_type,
                asym_coef,
                asym_margin,
                rest_pka_coef,
                rest_pka_margin,
                pathway_floor_coef,
                pathway_floor_min,
                c_snc_floor_coef,
                c_snc_floor_min,
                gpe_floor_coef,
                gpe_floor_min,
                dead_area_coef,
                dead_area_min,
                dead_proj_coef,
                dead_proj_floor,
            ),
            has_aux=True,
        )(cur_params)

        updates, cur_opt_state = optimizer.update(grads, cur_opt_state, cur_params)
        cur_params = optax.apply_updates(cur_params, updates)

        return (cur_params, cur_opt_state, cur_rng_key), (
            loss_value,
            aux["sup_loss"],
            aux["accuracy"],
            aux["in_window_rate"],
            aux["off_window_rate"],
            aux["asym_loss"],
            aux["rest_pka_loss"],
            aux["pathway_floor_loss"],
            aux["c_snc_floor_loss"],
            aux["gpe_floor_loss"],
            aux["dead_area_loss"],
            aux["dead_proj_loss"],
        )

    losses = []
    accuracies = []
    best_loss = float("inf")
    best_params = params

    for n in range(num_iters // log_interval):
        (params, opt_state, rng_key), (
            loss_vals,
            sup_vals,
            acc_vals,
            in_rate_vals,
            off_rate_vals,
            asym_vals,
            rest_pka_vals,
            pathway_floor_vals,
            c_snc_floor_vals,
            gpe_floor_vals,
            dead_area_vals,
            dead_proj_vals,
        ) = lax.scan(
            _step,
            (params, opt_state, rng_key),
            None,
            length=log_interval,
        )

        last_loss = float(loss_vals[-1])
        last_acc = float(acc_vals[-1])
        losses.append(last_loss)
        accuracies.append(last_acc)

        print(
            f"step {(n + 1) * log_interval}, "
            f"loss: {last_loss:.6f}, sup_loss: {float(sup_vals[-1]):.6f}, "
            f"acc: {last_acc:.4f}, in_window: {float(in_rate_vals[-1]):.4f}, "
            f"off_window: {float(off_rate_vals[-1]):.4f}, asym: {float(asym_vals[-1]):.4f}, "
            f"rest_pka: {float(rest_pka_vals[-1]):.4f}, pathway_floor: {float(pathway_floor_vals[-1]):.4f}, "
            f"c_snc_floor: {float(c_snc_floor_vals[-1]):.4f}, gpe_floor: {float(gpe_floor_vals[-1]):.4f}, "
            f"dead_area: {float(dead_area_vals[-1]):.4f}, dead_proj: {float(dead_proj_vals[-1]):.4f}"
        )

        if last_loss < best_loss:
            best_loss = last_loss
            best_params = params

    return best_params, losses, accuracies


def evaluate(rnn, params, config, all_inputs, noise_std=None, n_seeds=8):
    all_ys = []
    all_xs = []
    all_as = []
    eval_config = dict(config)
    if noise_std is not None:
        eval_config["noise_std"] = noise_std

    for seed in range(n_seeds):
        rng_key = jr.PRNGKey(seed)
        rng_key, action_key = jr.split(rng_key)
        batch_rng_keys = jr.split(rng_key, all_inputs.shape[0])
        n_hidden = eval_config["x0"].shape[0]
        batch_stim = jnp.zeros((all_inputs.shape[0], all_inputs.shape[1], n_hidden))
        ys, xs = rnn(params, eval_config, all_inputs, batch_stim, batch_rng_keys)
        actions = jr.bernoulli(action_key, p=ys).astype(ys.dtype)
        all_ys.append(ys)
        all_xs.append(xs)
        all_as.append(actions)

    return jnp.stack(all_ys), jnp.stack(all_xs), jnp.stack(all_as)