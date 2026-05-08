import jax
import jax.numpy as jnp
import jax.random as jr

from jax import vmap, jit
from jax import lax
import optax


def exc(w):
    #return jnp.maximum(0, w)
    return jnp.abs(w)


def inh(w):
    return -jnp.abs(w)
    #return -jnp.maximum(0, w)


def nln(x):
    return jnp.maximum(0, jax.nn.tanh(x))
    #return jax.nn.sigmoid(4*(x-0.5))
    #return jnp.maximum(0, x**3/(x**3+0.5**3))
    # return jax.nn.softplus(x - 4.0)


def bg_nln(x):
    #return jax.nn.sigmoid(4*(x-0.5))
    return jnp.maximum(0, jax.nn.tanh(x))


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

    ``rng_keys`` are kept for API compatibility.
    """
    ys, _, _ = rnn_func(params, config, batch_inputs, None, rng_keys)
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

    total_loss = pg_loss - entropy_coef * entropy + brevity_term + silence_loss + tail_loss
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

        losses.append(last_loss)
        reward_means.append(last_reward)

        print(
            f"step {(n + 1) * log_interval}, "
            f"loss: {last_loss:.6f}, reward: {last_reward:.4f}, "
            f"log_reward: {last_log_reward:.4f}, success: {last_success:.4f}, entropy: {last_entropy:.4f}, "
            f"brevity: {last_brevity:.4f}, norm_resp_time: {last_norm_time:.3f}, "
            f"silence: {last_silence:.4f}, tail: {last_tail:.4f}"
        )

        if last_loss < best_loss:
            best_loss = last_loss
            best_params = params

    return best_params, losses, reward_means


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