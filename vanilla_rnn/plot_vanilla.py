from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

import vanilla_rnn as vr
import vanilla_config as cfg

# Allow importing from repository root when running scripts from vanilla_rnn/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt


def _to_numpy(x):
    return np.asarray(x)


def _safe_nanmean(arr, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _ensure_ys_4d(ys):
    ys_np = _to_numpy(ys)
    if ys_np.ndim == 4:
        return ys_np
    if ys_np.ndim == 3:
        # (trials, time, 1) -> (1, trials, time, 1)
        return ys_np[None, ...]
    raise ValueError(f"Expected ys with 3 or 4 dims, got shape {ys_np.shape}")


def _ensure_xs_4d(xs):
    xs_np = _to_numpy(xs)
    if xs_np.ndim == 4:
        return xs_np
    if xs_np.ndim == 3:
        # (trials, time, n_hidden) -> (1, trials, time, n_hidden)
        return xs_np[None, ...]
    raise ValueError(f"Expected xs with 3 or 4 dims, got shape {xs_np.shape}")


def _squeeze_last_if_one(arr):
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _resolve_alignment_window(starts, t_total, pre_steps, post_steps, full_timeseries=False):
    starts_np = _to_numpy(starts).astype(int)
    if starts_np.ndim != 1:
        raise ValueError(f"starts must be 1D, got shape {starts_np.shape}")

    if full_timeseries:
        # Show complete trial extent in cue-aligned coordinates:
        # rel_t covers [ -max(start), t_total - min(start) ).
        resolved_pre = int(np.max(starts_np))
        resolved_post = int(t_total - np.min(starts_np))
        return max(resolved_pre, 1), max(resolved_post, 1)

    return int(pre_steps), int(post_steps)


def _cue_align(arr_4d, starts, pre_steps=100, post_steps=300):
    """
    Align (seeds, trials, time, features) to cue onset.

    Returns:
        aligned: (seeds, trials, pre_steps+post_steps, features) with NaN padding
        rel_t: relative time axis in timesteps
    """
    starts_np = _to_numpy(starts).astype(int)
    seeds, trials, t_total, n_feat = arr_4d.shape
    if trials != starts_np.shape[0]:
        raise ValueError(
            f"starts length ({starts_np.shape[0]}) must match trials ({trials})"
        )

    win = pre_steps + post_steps
    aligned = np.full((seeds, trials, win, n_feat), np.nan, dtype=float)

    for tr in range(trials):
        cue = int(starts_np[tr])
        src_start = max(0, cue - pre_steps)
        src_end = min(t_total, cue + post_steps)

        dst_start = src_start - (cue - pre_steps)
        dst_end = dst_start + (src_end - src_start)
        aligned[:, tr, dst_start:dst_end, :] = arr_4d[:, tr, src_start:src_end, :]

    rel_t = np.arange(-pre_steps, post_steps)
    return aligned, rel_t


def _plot_finite_segments(ax, t_vals, y_vals, **plot_kwargs):
    """Plot only contiguous finite regions to avoid NaN-padded rendering artifacts."""
    finite = np.isfinite(y_vals)
    if not np.any(finite):
        return

    idx = np.where(finite)[0]
    start = idx[0]
    prev = idx[0]
    for cur in idx[1:]:
        if cur != prev + 1:
            ax.plot(t_vals[start:prev + 1], y_vals[start:prev + 1], **plot_kwargs)
            start = cur
        prev = cur
    ax.plot(t_vals[start:prev + 1], y_vals[start:prev + 1], **plot_kwargs)


def compute_response_times(
    traces,
    starts,
    threshold=0.5,
    dt=0.01,
    from_cue=True,
    reduce_seeds="mean",
):
    """
    Compute first response time per trial where trace(t) > threshold.

    Args:
        traces: (seeds, trials, time, 1) or (trials, time, 1)
        starts: cue onset indices, shape (trials,)
        threshold: response threshold
        dt: seconds per timestep
        from_cue: if True, response time is relative to cue onset; else from trial start
        reduce_seeds: 'mean' or 'first'

    Returns:
        response_times_s: shape (trials,), np.nan for no response
    """
    traces_4d = _ensure_ys_4d(traces)
    traces_3d = _squeeze_last_if_one(traces_4d)  # (seeds, trials, time)

    if reduce_seeds == "mean":
        trial_traces = _safe_nanmean(traces_3d, axis=0)  # (trials, time)
    elif reduce_seeds == "first":
        trial_traces = traces_3d[0]
    else:
        raise ValueError("reduce_seeds must be 'mean' or 'first'")

    starts_np = _to_numpy(starts).astype(int)
    n_trials, t_total = trial_traces.shape
    if starts_np.shape[0] != n_trials:
        raise ValueError(
            f"starts length ({starts_np.shape[0]}) must match number of trials ({n_trials})"
        )

    response_times = np.full((n_trials,), np.nan, dtype=float)
    for tr in range(n_trials):
        trace = trial_traces[tr]
        above = np.where(trace > threshold)[0]
        if above.size == 0:
            continue
        first_idx = int(above[0])
        if from_cue:
            response_times[tr] = (first_idx - starts_np[tr]) * dt
        else:
            response_times[tr] = first_idx * dt

    return response_times


def plot_cue_aligned_summary(
    actions,
    xs,
    starts,
    pre_steps=100,
    post_steps=300,
    full_timeseries=True,
    dt=0.01,
    save_path=None,
):
    """
    Plot cue-aligned summary subplots:
    1) trial-averaged traces for xs and actions
    2) neuron-averaged, per-trial traces for xs and actions
    """
    actions_4d = _ensure_ys_4d(actions)
    xs_4d = _ensure_xs_4d(xs)

    resolved_pre, resolved_post = _resolve_alignment_window(
        starts,
        t_total=actions_4d.shape[2],
        pre_steps=pre_steps,
        post_steps=post_steps,
        full_timeseries=full_timeseries,
    )

    actions_aligned, rel_t = _cue_align(actions_4d, starts, pre_steps=resolved_pre, post_steps=resolved_post)
    xs_aligned, _ = _cue_align(xs_4d, starts, pre_steps=resolved_pre, post_steps=resolved_post)

    t_s = rel_t * dt
    actions_aligned = _squeeze_last_if_one(actions_aligned)  # (seeds, trials, win)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel 1: trial-averaged cue-aligned traces
    x_trial_avg = _safe_nanmean(xs_aligned, axis=(0, 1))  # (win, n_hidden)
    a_trial_avg = _safe_nanmean(actions_aligned, axis=(0, 1))  # (win,)

    for n in range(x_trial_avg.shape[1]):
        axes[0].plot(t_s, x_trial_avg[:, n], color="tab:blue", alpha=0.12, linewidth=1.0)
    axes[0].plot(t_s, _safe_nanmean(x_trial_avg, axis=1), color="tab:blue", linewidth=2.5, label="x mean")
    axes[0].plot(t_s, a_trial_avg, color="tab:orange", linewidth=2.5, label="a")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    axes[0].set_title("Trial-averaged, cue-aligned traces")
    axes[0].set_xlabel("Time from cue (s)")
    axes[0].set_ylabel("Activity")
    axes[0].legend(frameon=False)

    # Panel 2: neuron-averaged cue-aligned traces per trial
    x_neuron_avg = _safe_nanmean(xs_aligned, axis=-1)      # (seeds, trials, win)
    a_trial_traces = _safe_nanmean(actions_aligned, axis=0)     # (trials, win)
    x_trial_traces = _safe_nanmean(x_neuron_avg, axis=0)   # (trials, win)

    with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        for tr in range(x_trial_traces.shape[0]):
            _plot_finite_segments(
                axes[1], t_s, x_trial_traces[tr], color="tab:blue", alpha=0.18, linewidth=1.0
            )
            _plot_finite_segments(
                axes[1], t_s, a_trial_traces[tr], color="tab:orange", alpha=0.18, linewidth=1.0
            )

    axes[1].plot(t_s, _safe_nanmean(x_trial_traces, axis=0), color="tab:blue", linewidth=2.5, label="x (neuron-avg)")
    axes[1].plot(t_s, _safe_nanmean(a_trial_traces, axis=0), color="tab:orange", linewidth=2.5, label="a")
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    axes[1].set_title("Neuron-averaged cue-aligned traces")
    axes[1].set_xlabel("Time from cue (s)")
    axes[1].set_ylabel("Activity")
    axes[1].legend(frameon=False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_response_time_histogram(
    actions,
    starts,
    threshold=0.5,
    dt=0.01,
    bins=20,
    from_cue=True,
    save_path=None,
):
    """
    Extract and plot response-time histogram.
    """
    rts = compute_response_times(
        actions,
        starts,
        threshold=threshold,
        dt=dt,
        from_cue=from_cue,
        reduce_seeds="mean",
    )
    valid_rts = rts[np.isfinite(rts)]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if valid_rts.size > 0:
        ax.hist(valid_rts, bins=bins, color="tab:green", alpha=0.8, edgecolor="white")
    ax.set_title("Response time distribution")
    ax.set_xlabel("Response time (s from cue)" if from_cue else "Response time (s from trial start)")
    ax.set_ylabel("Count")

    n_total = rts.shape[0]
    n_resp = valid_rts.size
    ax.text(
        0.98,
        0.95,
        f"responses: {n_resp}/{n_total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax, rts


def _compute_loss_timeseries(y_trials, targets, masks=None):
    """
    Compute per-timestep loss for each trial.

    Returns:
        loss_ts: (n_trials, T) timeseries of loss values
    """
    targets_np = _to_numpy(targets)
    if targets_np.ndim == 3 and targets_np.shape[-1] == 1:
        targets_np = targets_np[..., 0]

    if masks is None:
        masks_np = np.ones_like(targets_np, dtype=float)
    else:
        masks_np = _to_numpy(masks)
        if masks_np.ndim == 3 and masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]

    if targets_np.shape != y_trials.shape:
        raise ValueError(
            f"targets shape {targets_np.shape} must match y_trials shape {y_trials.shape}"
        )
    if masks_np.shape != y_trials.shape:
        raise ValueError(
            f"masks shape {masks_np.shape} must match y_trials shape {y_trials.shape}"
        )

    loss_ts = ((y_trials - targets_np) ** 2) * masks_np
    return loss_ts


def plot_trial_traces_and_loss(
    actions,
    starts,
    targets=None,
    masks=None,
    threshold=0.5,
    dt=0.01,
    pre_steps=80,
    post_steps=250,
    full_timeseries=True,
    cmap="viridis",
    save_path=None,
):
    """
    Plot cue-aligned action traces and cue-aligned per-timestep loss, color-coded by response time.

    If targets/masks are provided, per-timestep loss is (a - target)^2 * mask.
    If not provided, per-timestep loss defaults to a^2.
    """
    actions_4d = _ensure_ys_4d(actions)
    action_trials = _safe_nanmean(_squeeze_last_if_one(actions_4d), axis=0)  # (trials, time)

    rts = compute_response_times(
        actions,
        starts,
        threshold=threshold,
        dt=dt,
        from_cue=True,
        reduce_seeds="mean",
    )

    if targets is None:
        loss_ts = action_trials ** 2
    else:
        loss_ts = _compute_loss_timeseries(action_trials, targets, masks=masks)

    # Cue-align both action and loss timeseries
    action_trials_4d = action_trials[None, :, :, None]
    loss_ts_4d = loss_ts[None, :, :, None]

    resolved_pre, resolved_post = _resolve_alignment_window(
        starts,
        t_total=action_trials.shape[1],
        pre_steps=pre_steps,
        post_steps=post_steps,
        full_timeseries=full_timeseries,
    )

    action_aligned, rel_t = _cue_align(action_trials_4d, starts, pre_steps=resolved_pre, post_steps=resolved_post)
    loss_aligned, _ = _cue_align(loss_ts_4d, starts, pre_steps=resolved_pre, post_steps=resolved_post)

    action_aligned = _squeeze_last_if_one(action_aligned[0])     # (trials, win)
    loss_aligned = _squeeze_last_if_one(loss_aligned[0])  # (trials, win)

    t_s = rel_t * dt

    finite_rts = rts[np.isfinite(rts)]
    if finite_rts.size > 0:
        vmin, vmax = np.nanmin(finite_rts), np.nanmax(finite_rts)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    # Panel 1: cue-aligned action traces with response-time color coding
    with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        for tr in range(action_aligned.shape[0]):
            color = "lightgray" if not np.isfinite(rts[tr]) else scalar_map.to_rgba(rts[tr])
            _plot_finite_segments(axes[0], t_s, action_aligned[tr], color=color, linewidth=1.2, alpha=0.9)

    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    axes[0].axhline(threshold, color="k", linestyle=":", linewidth=1.0, alpha=0.5)
    axes[0].set_title("Cue-aligned action traces (color = response time)")
    axes[0].set_xlabel("Time from cue (s)")
    axes[0].set_ylabel("a")
    axes[0].legend(frameon=False)

    # Panel 2: cue-aligned loss timeseries with same color coding by response time
    with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        for tr in range(loss_aligned.shape[0]):
            color = "lightgray" if not np.isfinite(rts[tr]) else scalar_map.to_rgba(rts[tr])
            _plot_finite_segments(axes[1], t_s, loss_aligned[tr], color=color, linewidth=1.2, alpha=0.9)

    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    axes[1].set_title("Cue-aligned loss timeseries (color = response time)")
    axes[1].set_xlabel("Time from cue (s)")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)

    cbar = fig.colorbar(scalar_map, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Response time from cue (s)")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, axes, rts, (action_aligned, loss_aligned)


def load_default_eval(n_seeds=4, noise_std=0.0):
    """
    Convenience loader using the same task config as train_vanilla.py.
    """
    params_path = Path(__file__).resolve().parent / "params_vanilla.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}. Run train_vanilla.py first.")

    import pickle as pkl
    import jax.numpy as jnp

    with params_path.open("rb") as f:
        params = pkl.load(f)

    # Reconstruct runtime config from loaded params + cfg defaults.
    n_hidden = params["b_h"].shape[0]
    config = {
        "noise_std": cfg.RNN_CONFIG["noise_std"],
        "x0": jnp.ones((n_hidden,)) * 0.1,
        "tau": 20.0,
    }

    starts = cfg.TASK_CONFIG["t_start"]
    inputs, targets, masks = stmt.self_timed_movement_task(
        starts,
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )
    ys, xs, actions = vr.evaluate(params, config, inputs, noise_std=noise_std, n_seeds=n_seeds)
    return ys, xs, actions, starts, targets, masks


def main():
    ys, xs, actions, starts, targets, masks = load_default_eval(n_seeds=4, noise_std=0.0)

    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cue_aligned_summary(
        actions,
        xs,
        starts,
        full_timeseries=True,
        dt=0.01,
        save_path=out_dir / "vanilla_cue_aligned_summary.png",
    )

    plot_response_time_histogram(
        actions,
        starts,
        threshold=0.5,
        dt=0.01,
        bins=12,
        save_path=out_dir / "vanilla_response_time_hist.png",
    )

    plot_trial_traces_and_loss(
        actions,
        starts,
        targets=targets,
        masks=masks,
        threshold=0.5,
        dt=0.01,
        full_timeseries=True,
        save_path=out_dir / "vanilla_trial_traces_and_loss.png",
    )

    print(f"Saved plots to: {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()

