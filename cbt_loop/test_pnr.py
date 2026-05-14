"""Point-of-no-return (PNR) probes for the CBT-loop model.

Sweeps cue magnitude, cue duration, and mid-trial inhibitory perturbations,
then summarizes response probability / magnitude and hands the trajectories to
the binned-PNR plotting routines.
"""
import os
import pickle as pkl

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt
import plotting_functions as pf


# Output directory for the scalar summary plots.
PNR_DIR = os.path.join(cfg.plots_folder, "pnr_test")
os.makedirs(PNR_DIR, exist_ok=True)


def load_bundle():
    """Load params + config, rebuilding config for legacy (params-only) bundles."""
    params_path = cfg.params_path()
    try:
        with params_path.open("rb") as f:
            bundle = pkl.load(f)
    except FileNotFoundError:
        print(f"Error: {params_path} not found. Run training_script.py first.")
        return None, None

    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        return bundle["params"], bundle["config"]

    params = bundle
    _, config = cbtl.init_params(
        jr.PRNGKey(0),
        n_c=params["J_c"].shape[0],
        n_d1=params["J_d1"].shape[0],
        n_d2=params["J_d2"].shape[0],
        n_snc=params["P_snc"].shape[0],
        n_snr=params["P_snr"].shape[0],
        n_gpe=params["J_gpe"].shape[0],
        n_stn=params["J_stn"].shape[0],
        n_t=params["J_t"].shape[0],
        n_med=params["J_med_w1"].shape[0] * 2,
        n_input=1,
        n_output=1,
        noise_std=cfg.RNN_CONFIG["noise_std"],
    )
    return params, config


def _n_bg(params):
    return params["J_d1"].shape[0] + params["J_d2"].shape[0]


def _run_batch(params, config, inputs, stim, n_seeds, seed_offset=0):
    """Run the model across n_seeds; return stacked (all_ys, all_xs).

    all_ys: (n_seeds, n_conditions, T, 1)
    all_xs: tuple of (n_seeds, n_conditions, T, N) arrays (one per state area)
    """
    ys_list, xs_list = [], []
    for s in range(n_seeds):
        rng_keys = jr.split(jr.PRNGKey(s + seed_offset), inputs.shape[0])
        ys, xs = cbtl.batched_rnn(params, config, inputs, stim, rng_keys)
        ys_list.append(ys)
        xs_list.append(xs)
    all_ys = jnp.stack(ys_list)
    all_xs = tuple(jnp.stack([xs[i] for xs in xs_list]) for i in range(len(xs_list[0])))
    return all_ys, all_xs


def _summarize(all_ys, response_start, n_seeds):
    """Response probability and peak magnitude after `response_start`."""
    post = all_ys[:, :, response_start:, 0]  # (n_seeds, n_conditions, T')
    responded = jnp.any(post > 0.5, axis=2)
    max_mag = jnp.max(post, axis=2)
    prob = jnp.mean(responded, axis=0)
    mag_mean = jnp.mean(max_mag, axis=0)
    mag_sem = jnp.std(max_mag, axis=0) / jnp.sqrt(n_seeds)
    return prob, mag_mean, mag_sem


def test_cue_magnitude(params, config, magnitudes, n_seeds=100):
    """Sweep a multiplicative gain on the cue input."""
    start_t = 300
    base_inputs, _, _ = stmt.self_timed_movement_task(
        jnp.array([start_t]),
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )
    scaled = base_inputs[0:1] * magnitudes[:, None, None]  # (n_mags, T, 1)
    stim = jnp.zeros((scaled.shape[0], scaled.shape[1], _n_bg(params)))
    all_ys, all_xs = _run_batch(params, config, scaled, stim, n_seeds)
    cue_end = start_t + cfg.TASK_CONFIG["t_cue"]
    prob, mag_mean, mag_sem = _summarize(all_ys, cue_end, n_seeds)
    return prob, mag_mean, mag_sem, all_ys, all_xs


def test_cue_duration(params, config, durations, n_seeds=100):
    """Sweep cue duration (0 = cue omitted)."""
    start_t = 300
    T = cfg.TASK_CONFIG["t_total"]
    n_dur = len(durations)
    inputs = jnp.zeros((n_dur, T, 1))
    for i, dur in enumerate(durations):
        d = int(dur)
        if d > 0:
            inputs = inputs.at[i, start_t:start_t + d, 0].set(1.0)
    stim = jnp.zeros((n_dur, T, _n_bg(params)))
    all_ys, all_xs = _run_batch(params, config, inputs, stim, n_seeds, seed_offset=100)
    prob, mag_mean, mag_sem = _summarize(all_ys, start_t, n_seeds)
    return prob, mag_mean, mag_sem, all_ys, all_xs


def test_dynamic_perturbation(params, config, perturbation_times,
                              pert_strength=-5.0, pert_duration=5, n_seeds=100):
    """Inject a brief inhibitory pulse into the striatum at varying delays.

    Condition 0 is a cue-omitted baseline; conditions 1.. correspond to
    `perturbation_times`.
    """
    start_t = 300
    T = cfg.TASK_CONFIG["t_total"]
    n_bg = _n_bg(params)
    n_pert = len(perturbation_times)
    n_conds = n_pert + 1

    stim = jnp.zeros((n_conds, T, n_bg))
    for i, p_time in enumerate(perturbation_times):
        pt = int(p_time)
        stim = stim.at[i + 1, pt:pt + pert_duration, :].set(pert_strength)

    base_inputs, _, _ = stmt.self_timed_movement_task(
        jnp.array([start_t]),
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )
    inputs = jnp.repeat(base_inputs, n_conds, axis=0)
    inputs = inputs.at[0].set(0.0)  # condition 0: cue omitted

    all_ys, all_xs = _run_batch(params, config, inputs, stim, n_seeds, seed_offset=200)
    prob, mag_mean, mag_sem = _summarize(all_ys, start_t, n_seeds)
    return prob, mag_mean, mag_sem, all_ys, all_xs


def plot_and_save(x_vals, prob, mag_mean, mag_sem, xlabel, title, filename):
    """Dual-axis summary: response probability and peak magnitude vs. condition."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Response Probability", color=color)
    ax1.plot(x_vals, prob, marker="o", color=color, linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Max Response Magnitude", color=color)
    ax2.plot(x_vals, mag_mean, marker="s", color=color, linewidth=2, linestyle="--")
    ax2.fill_between(x_vals, mag_mean - mag_sem, mag_mean + mag_sem, color=color, alpha=0.2)
    ax2.axhline(0.5, color="gray", linestyle=":", label="Threshold")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(PNR_DIR, filename), dpi=200)
    plt.close(fig)


def run_pnr_experiments(params, config, n_seeds=60):
    """Run all three PNR sweeps and emit summary + binned-trajectory plots."""
    dt = cfg.TASK_CONFIG["dt_ms"]

    print("  [PNR] 1/3 cue magnitude sensitivity...")
    mags = jnp.linspace(0.0, 2.0, 21)
    m_prob, m_mag, m_sem, m_ys, m_xs = test_cue_magnitude(params, config, mags, n_seeds)
    plot_and_save(mags, m_prob, m_mag, m_sem,
                  "Cue Magnitude Multiplier", "Sensitivity to Cue Input Gain", "cue_magnitude.png")
    pf.plot_binned_pnr(m_ys, m_xs, None, mags, "Cue Magnitude", "Cue Magnitude Responses", "pnr_mag")
    pf.plot_colorbar(0, len(mags) - 1, "plasma", "Cue Magnitude", "pnr_mag_colorbar",
                     [0, len(mags) - 1], [f"{mags[0]:.1f}", f"{mags[-1]:.1f}"])

    print("  [PNR] 2/3 cue duration sensitivity...")
    durs = jnp.concatenate((jnp.array([0]), jnp.arange(2, 32, 2)))
    d_prob, d_mag, d_sem, d_ys, d_xs = test_cue_duration(params, config, durs, n_seeds)
    plot_and_save(durs * dt, d_prob, d_mag, d_sem,
                  "Cue Duration (ms)", "Sensitivity to Cue Duration", "cue_duration.png")
    pf.plot_binned_pnr(d_ys, d_xs, None, durs * dt, "Cue Duration (ms)",
                       "Cue Duration Responses", "pnr_dur")
    pf.plot_colorbar(0, len(durs) - 1, "plasma", "Cue Duration (ms)", "pnr_dur_colorbar",
                     [0, len(durs) - 1], [str(0.0), f"{(durs[-1] * dt):.1f}"])

    print("  [PNR] 3/3 dynamic perturbation (stop-signal)...")
    pert_times = jnp.arange(320, 600, 20)
    pt_prob, pt_mag, pt_sem, pt_ys, pt_xs = test_dynamic_perturbation(
        params, config, pert_times, n_seeds=n_seeds)
    pt_conditions = jnp.concatenate((jnp.array([0.0]), (pert_times - 300) * dt))
    pert_boxes = {}
    for i, p_time in enumerate(pert_times):
        p_start = (p_time - 300) / 100
        p_end = p_start + (5 / 100)
        pert_boxes[i + 1] = (p_start, p_end)
    plot_and_save(pt_conditions, pt_prob, pt_mag, pt_sem,
                  "Perturbation Delay from Cue (ms)",
                  "Point-of-No-Return (Inhibitory Stop Signal)", "dynamic_pnr.png")
    pf.plot_binned_pnr(pt_ys, pt_xs, None, pt_conditions, "Perturbation Delay (ms)",
                       "Dynamic PNR", "pnr_dyn", perturbation_boxes=pert_boxes)
    pf.plot_colorbar(0, len(pt_conditions) - 1, "plasma", "Perturbation Delay (ms)",
                     "pnr_dyn_colorbar", [0, len(pt_conditions) - 1],
                     ["No Cue", f"{pt_conditions[-1]:.1f}"])

    print(f"  [PNR] done. Summary plots in {PNR_DIR}")


def main(params=None, config=None, n_seeds=60):
    if params is None or config is None:
        params, config = load_bundle()
        if params is None:
            raise SystemExit(1)
    print("Running Point-of-No-Return analyses...")
    run_pnr_experiments(params, config, n_seeds=n_seeds)


if __name__ == "__main__":
    main()
