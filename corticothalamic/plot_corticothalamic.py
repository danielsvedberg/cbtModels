"""
Plotting utilities for the corticothalamic two-area RNN.

All behavioral plots use sampled binary *actions* (not output probabilities)
for response-time extraction and trace display, matching the vanilla_rnn
convention in plot_vanilla.py.  `ys` (raw hazard probabilities) are kept
available for internal-state overlays when needed.
"""
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

import corticothalamic_rnn as ctrnn
import corticothalamic_config as cfg

# Allow importing from repository root when running scripts from corticothalamic/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import self_timed_movement_task as stmt


# ---------------------------------------------------------------------------
# Internal helpers (shared with vanilla plot utilities)
# ---------------------------------------------------------------------------

def _to_numpy(x):
    return np.asarray(x)


def _safe_nanmean(arr, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _ensure_4d(arr):
    """Ensure array is (seeds, trials, time, features)."""
    arr_np = _to_numpy(arr)
    if arr_np.ndim == 4:
        return arr_np
    if arr_np.ndim == 3:
        return arr_np[None, ...]
    raise ValueError(f"Expected 3 or 4-D array, got shape {arr_np.shape}")


def _squeeze_last_if_one(arr):
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _resolve_alignment_window(starts, t_total, pre_steps, post_steps, full_timeseries=False):
    starts_np = _to_numpy(starts).astype(int)
    if full_timeseries:
        resolved_pre = int(np.max(starts_np))
        resolved_post = int(t_total - np.min(starts_np))
        return max(resolved_pre, 1), max(resolved_post, 1)
    return int(pre_steps), int(post_steps)


def _cue_align(arr_4d, starts, pre_steps=100, post_steps=300):
    """
    Align (seeds, trials, time, features) to cue onset.

    Returns:
        aligned: (seeds, trials, pre_steps+post_steps, features) with NaN padding
        rel_t:   1-D relative time axis in timesteps
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


def _plot_finite_segments(ax, t_vals, y_vals, **kwargs):
    """Plot only contiguous finite regions to avoid NaN-padded rendering artifacts."""
    finite = np.isfinite(y_vals)
    if not np.any(finite):
        return
    idx = np.where(finite)[0]
    start = idx[0]
    prev = idx[0]
    for cur in idx[1:]:
        if cur != prev + 1:
            ax.plot(t_vals[start:prev + 1], y_vals[start:prev + 1], **kwargs)
            start = cur
        prev = cur
    ax.plot(t_vals[start:prev + 1], y_vals[start:prev + 1], **kwargs)


# ---------------------------------------------------------------------------
# Response-time computation
# ---------------------------------------------------------------------------

def compute_response_times(traces, starts, threshold=0.5, dt=0.01, from_cue=True, reduce_seeds="mean"):
    """
    Compute first response time per trial where trace(t) > threshold.

    Args:
        traces:        (seeds, trials, time, 1) or (trials, time, 1)
        starts:        cue onset indices, shape (trials,)
        threshold:     response threshold
        dt:            seconds per timestep
        from_cue:      if True, response time is relative to cue onset
        reduce_seeds:  'mean' or 'first'

    Returns:
        response_times_s: shape (trials,), np.nan for no response
    """
    traces_4d = _ensure_4d(traces)
    traces_3d = _squeeze_last_if_one(traces_4d)  # (seeds, trials, time)

    if reduce_seeds == "mean":
        trial_traces = _safe_nanmean(traces_3d, axis=0)
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
        response_times[tr] = (first_idx - starts_np[tr]) * dt if from_cue else first_idx * dt
    return response_times


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_cue_aligned_summary(
    actions,
    x_ctx,
    x_t,
    starts,
    pre_steps=100,
    post_steps=300,
    full_timeseries=True,
    dt=0.01,
    save_path=None,
):
    """
    Three-panel cue-aligned summary:
      1) Trial-averaged action trace + cortex mean + thalamus mean
      2) Neuron-averaged per-trial traces for cortex vs thalamus
      3) Action raster / PSTH: trial-averaged action probability over time
    """
    actions_4d = _ensure_4d(actions)
    x_ctx_4d = _ensure_4d(x_ctx)
    x_t_4d = _ensure_4d(x_t)

    t_total = actions_4d.shape[2]
    resolved_pre, resolved_post = _resolve_alignment_window(
        starts, t_total, pre_steps, post_steps, full_timeseries=full_timeseries
    )

    a_aligned, rel_t = _cue_align(actions_4d, starts, resolved_pre, resolved_post)
    ctx_aligned, _ = _cue_align(x_ctx_4d, starts, resolved_pre, resolved_post)
    t_aligned, _ = _cue_align(x_t_4d, starts, resolved_pre, resolved_post)

    t_s = rel_t * dt
    a_aligned_sq = _squeeze_last_if_one(a_aligned)  # (seeds, trials, win)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # ── Panel 1: trial-averaged traces ──────────────────────────────────────
    ax = axes[0]
    ctx_trial_avg = _safe_nanmean(ctx_aligned, axis=(0, 1))   # (win, n_ctx)
    t_trial_avg   = _safe_nanmean(t_aligned,   axis=(0, 1))   # (win, n_t)
    a_trial_avg   = _safe_nanmean(a_aligned_sq, axis=(0, 1))  # (win,)

    for n in range(ctx_trial_avg.shape[1]):
        ax.plot(t_s, ctx_trial_avg[:, n], color="tab:blue", alpha=0.10, linewidth=0.8)
    for n in range(t_trial_avg.shape[1]):
        ax.plot(t_s, t_trial_avg[:, n], color="tab:red", alpha=0.10, linewidth=0.8)

    ax.plot(t_s, _safe_nanmean(ctx_trial_avg, axis=1), color="tab:blue", linewidth=2.0, label="ctx mean")
    ax.plot(t_s, _safe_nanmean(t_trial_avg,   axis=1), color="tab:red",  linewidth=2.0, label="thal mean")
    ax.plot(t_s, a_trial_avg,                           color="tab:orange", linewidth=2.0, label="action")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    ax.set_title("Trial-averaged, cue-aligned")
    ax.set_xlabel("Time from cue (s)")
    ax.set_ylabel("Activity")
    ax.legend(frameon=False, fontsize=8)

    # ── Panel 2: neuron-averaged per-trial traces ────────────────────────────
    ax = axes[1]
    ctx_neuron_avg = _safe_nanmean(ctx_aligned, axis=-1)   # (seeds, trials, win)
    t_neuron_avg   = _safe_nanmean(t_aligned,   axis=-1)   # (seeds, trials, win)
    ctx_trial_tr   = _safe_nanmean(ctx_neuron_avg, axis=0)  # (trials, win)
    t_trial_tr     = _safe_nanmean(t_neuron_avg,   axis=0)  # (trials, win)
    a_trial_tr     = _safe_nanmean(a_aligned_sq,   axis=0)  # (trials, win)

    with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        for tr in range(ctx_trial_tr.shape[0]):
            _plot_finite_segments(ax, t_s, ctx_trial_tr[tr],   color="tab:blue",   alpha=0.18, linewidth=1.0)
            _plot_finite_segments(ax, t_s, t_trial_tr[tr],     color="tab:red",    alpha=0.18, linewidth=1.0)
            _plot_finite_segments(ax, t_s, a_trial_tr[tr],     color="tab:orange", alpha=0.18, linewidth=1.0)

    ax.plot(t_s, _safe_nanmean(ctx_trial_tr, axis=0), color="tab:blue",   linewidth=2.0, label="ctx")
    ax.plot(t_s, _safe_nanmean(t_trial_tr,   axis=0), color="tab:red",    linewidth=2.0, label="thal")
    ax.plot(t_s, _safe_nanmean(a_trial_tr,   axis=0), color="tab:orange", linewidth=2.0, label="action")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    ax.set_title("Per-trial, neuron-averaged")
    ax.set_xlabel("Time from cue (s)")
    ax.set_ylabel("Activity")
    ax.legend(frameon=False, fontsize=8)

    # ── Panel 3: action PSTH (mean across seeds + trials) ────────────────────
    ax = axes[2]
    psth = _safe_nanmean(a_aligned_sq, axis=(0, 1))  # (win,)
    ax.fill_between(t_s, psth, color="tab:orange", alpha=0.6)
    ax.plot(t_s, psth, color="tab:orange", linewidth=1.5)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    ax.set_title("Action PSTH (trial + seed mean)")
    ax.set_xlabel("Time from cue (s)")
    ax.set_ylabel("P(action)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False)

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
    """Extract and plot response-time histogram from action arrays."""
    rts = compute_response_times(actions, starts, threshold=threshold, dt=dt, from_cue=from_cue)
    valid_rts = rts[np.isfinite(rts)]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if valid_rts.size > 0:
        ax.hist(valid_rts, bins=bins, color="tab:green", alpha=0.8, edgecolor="white")
    ax.set_title("Response time distribution")
    ax.set_xlabel("Response time (s from cue)" if from_cue else "Response time (s from trial start)")
    ax.set_ylabel("Count")

    n_total = rts.shape[0]
    ax.text(
        0.98, 0.95,
        f"responses: {valid_rts.size}/{n_total}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax, rts


def plot_area_comparison(
    x_ctx,
    x_t,
    actions,
    starts,
    pre_steps=100,
    post_steps=300,
    full_timeseries=True,
    dt=0.01,
    cmap="viridis",
    save_path=None,
):
    """
    Two-panel plot comparing cortex vs thalamus mean activity,
    each colour-coded by response time.  Action traces are overlaid.
    """
    rts = compute_response_times(actions, starts, dt=dt)
    actions_4d = _ensure_4d(actions)
    x_ctx_4d = _ensure_4d(x_ctx)
    x_t_4d = _ensure_4d(x_t)

    t_total = x_ctx_4d.shape[2]
    resolved_pre, resolved_post = _resolve_alignment_window(
        starts, t_total, pre_steps, post_steps, full_timeseries=full_timeseries
    )

    ctx_aligned, rel_t = _cue_align(x_ctx_4d, starts, resolved_pre, resolved_post)
    t_aligned, _ = _cue_align(x_t_4d, starts, resolved_pre, resolved_post)
    a_aligned, _ = _cue_align(actions_4d, starts, resolved_pre, resolved_post)

    t_s = rel_t * dt

    # Per-trial mean across seeds + neurons
    ctx_trial = _safe_nanmean(_safe_nanmean(ctx_aligned, axis=0), axis=-1)  # (trials, win)
    t_trial   = _safe_nanmean(_safe_nanmean(t_aligned,   axis=0), axis=-1)  # (trials, win)
    a_trial   = _safe_nanmean(_squeeze_last_if_one(a_aligned), axis=0)       # (trials, win)

    finite_rts = rts[np.isfinite(rts)]
    vmin = float(np.nanmin(finite_rts)) if finite_rts.size > 0 else 0.0
    vmax = float(np.nanmax(finite_rts)) if finite_rts.size > 0 else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for area_name, area_data, ax in (
        ("Cortex (mean)", ctx_trial, axes[0]),
        ("Thalamus (mean)", t_trial, axes[1]),
    ):
        with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
            for tr in range(area_data.shape[0]):
                color = "lightgray" if not np.isfinite(rts[tr]) else sm.to_rgba(rts[tr])
                _plot_finite_segments(ax, t_s, area_data[tr], color=color, linewidth=1.2, alpha=0.9)
                _plot_finite_segments(ax, t_s, a_trial[tr],   color=color, linewidth=0.6, alpha=0.4, linestyle="--")

        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
        ax.set_title(f"{area_name}  (color = response time)")
        ax.set_xlabel("Time from cue (s)")
        ax.set_ylabel("Activity")
        ax.legend(frameon=False)

    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Response time from cue (s)")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, axes, rts


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
    Cue-aligned action traces (top) and per-timestep loss (bottom),
    both colour-coded by response time.
    """
    actions_4d = _ensure_4d(actions)
    action_trials = _safe_nanmean(_squeeze_last_if_one(actions_4d), axis=0)  # (trials, time)

    rts = compute_response_times(actions, starts, threshold=threshold, dt=dt, from_cue=True)

    if targets is None:
        loss_ts = action_trials ** 2
    else:
        targets_np = _to_numpy(targets)
        if targets_np.ndim == 3 and targets_np.shape[-1] == 1:
            targets_np = targets_np[..., 0]
        masks_np = np.ones_like(targets_np) if masks is None else _to_numpy(masks)
        if masks_np.ndim == 3 and masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]
        loss_ts = ((action_trials - targets_np) ** 2) * masks_np

    t_total = action_trials.shape[1]
    resolved_pre, resolved_post = _resolve_alignment_window(
        starts, t_total, pre_steps, post_steps, full_timeseries=full_timeseries
    )

    a_4d = action_trials[None, :, :, None]
    l_4d = loss_ts[None, :, :, None]
    a_aligned, rel_t = _cue_align(a_4d, starts, resolved_pre, resolved_post)
    l_aligned, _ = _cue_align(l_4d, starts, resolved_pre, resolved_post)

    a_aligned = _squeeze_last_if_one(a_aligned[0])  # (trials, win)
    l_aligned = _squeeze_last_if_one(l_aligned[0])  # (trials, win)
    t_s = rel_t * dt

    finite_rts = rts[np.isfinite(rts)]
    vmin = float(np.nanmin(finite_rts)) if finite_rts.size > 0 else 0.0
    vmax = float(np.nanmax(finite_rts)) if finite_rts.size > 0 else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        for tr in range(a_aligned.shape[0]):
            color = "lightgray" if not np.isfinite(rts[tr]) else sm.to_rgba(rts[tr])
            _plot_finite_segments(axes[0], t_s, a_aligned[tr], color=color, linewidth=1.2, alpha=0.9)

    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    axes[0].axhline(threshold, color="k", linestyle=":", linewidth=1.0, alpha=0.5)
    axes[0].set_title("Cue-aligned action traces (color = response time)")
    axes[0].set_xlabel("Time from cue (s)")
    axes[0].set_ylabel("a")
    axes[0].legend(frameon=False)

    with plt.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        for tr in range(l_aligned.shape[0]):
            color = "lightgray" if not np.isfinite(rts[tr]) else sm.to_rgba(rts[tr])
            _plot_finite_segments(axes[1], t_s, l_aligned[tr], color=color, linewidth=1.2, alpha=0.9)

    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="cue")
    axes[1].set_title("Cue-aligned loss timeseries (color = response time)")
    axes[1].set_xlabel("Time from cue (s)")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)

    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Response time from cue (s)")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, axes, rts, (a_aligned, l_aligned)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_default_eval(n_seeds=4, noise_std=0.0):
    """
    Load saved params, reconstruct config, run evaluation.

    Returns:
        ys, x_ctx, x_t, actions, starts, targets, masks
    """
    import pickle as pkl
    import jax.numpy as jnp

    p_path = cfg.params_path()
    if not p_path.exists():
        raise FileNotFoundError(f"Missing {p_path}. Run train_corticothalamic.py first.")

    with p_path.open("rb") as f:
        bundle = pkl.load(f)

    params = bundle["params"]
    config = bundle["config"]

    task = cfg.TASK_CONFIG
    starts = task["t_start"]

    task_kwargs = dict(
        T_start=starts,
        T_cue=task["t_cue"],
        T_wait=task["t_wait"],
        T_movement=task["t_movement"],
        T=task["t_total"],
    )
    task_mode = task.get("task_mode", "self_timed")
    if task_mode == "hybrid":
        inputs, targets, masks = stmt.hybrid_stmt(**task_kwargs)
    elif task_mode == "pavlovian":
        inputs, targets, masks = stmt.pavlovian_stmt(**task_kwargs)
    else:
        inputs, targets, masks = stmt.self_timed_movement_task(**task_kwargs)

    ys, x_ctx, x_t, actions = ctrnn.evaluate(
        params, config, inputs, noise_std=noise_std, n_seeds=n_seeds
    )
    return ys, x_ctx, x_t, actions, starts, targets, masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ys, x_ctx, x_t, actions, starts, targets, masks = load_default_eval(n_seeds=4, noise_std=0.0)

    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cue_aligned_summary(
        actions,
        x_ctx,
        x_t,
        starts,
        full_timeseries=True,
        dt=0.01,
        save_path=out_dir / "ct_cue_aligned_summary.png",
    )

    plot_response_time_histogram(
        actions,
        starts,
        threshold=0.5,
        dt=0.01,
        bins=12,
        save_path=out_dir / "ct_response_time_hist.png",
    )

    plot_area_comparison(
        x_ctx,
        x_t,
        actions,
        starts,
        full_timeseries=True,
        dt=0.01,
        save_path=out_dir / "ct_area_comparison.png",
    )

    plot_trial_traces_and_loss(
        actions,
        starts,
        targets=targets,
        masks=masks,
        threshold=0.5,
        dt=0.01,
        full_timeseries=True,
        save_path=out_dir / "ct_trial_traces_and_loss.png",
    )

    print(f"Saved plots to: {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()

