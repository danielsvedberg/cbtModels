"""Optogenetic perturbation experiments for the CBT-loop model.

Runs spatially-targeted stimulation/suppression of dSPNs (D1) and iSPNs (D2)
during the wait period and hands the trajectories to the opto plotting
routines in plotting_functions.
"""
import pickle as pkl

import jax.numpy as jnp
import jax.random as jr

import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt
import plotting_functions as pf


def load_bundle():
    """Load params + config, rebuilding config for legacy (params-only) bundles."""
    with cfg.params_path().open("rb") as f:
        bundle = pkl.load(f)
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


def _stim_vector(params, strength, target):
    """Build a (n_d1 + n_d2,) spatial stim vector targeting 'd1' or 'd2' cells."""
    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    vec = jnp.zeros(n_d1 + n_d2)
    if target == "d1":
        return vec.at[:n_d1].set(strength)
    return vec.at[n_d1:].set(strength)


def _temporal_mask(T):
    """1.0 during the opto window [opto_start, opto_end), else 0.0."""
    t = jnp.arange(T)
    return ((t >= cfg.opto_start) & (t < cfg.opto_end)).astype(jnp.float32)


def _run_condition(params, config, inputs, stim_profile, n_seeds, seed=0):
    """Run one opto condition across n_seeds. Returns (ys, xs) stacked over seeds.

    ys: (n_seeds, T, 1)
    xs: tuple of (n_seeds, T, N) arrays (one per state area)
    """
    batch_inputs = jnp.repeat(inputs, n_seeds, axis=0)
    batch_stim = jnp.repeat(stim_profile[None, :, :], n_seeds, axis=0)
    rng_keys = jr.split(jr.PRNGKey(seed), n_seeds)
    ys, xs = cbtl.batched_rnn(params, config, batch_inputs, batch_stim, rng_keys)
    return ys, xs


def _base_inputs():
    """Single-trial cue input with cue onset at cfg.opto_tstart."""
    starts = jnp.array([cfg.opto_tstart])
    inputs, _, _ = stmt.self_timed_movement_task(
        starts,
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )
    return inputs  # (1, T, 1)


def run_opto_demo(params, config, n_seeds=200,
                  inh_d1=-1.2, inh_d2=-0.6, stim_d1=1.2, stim_d2=0.6):
    """Control + representative inh/stim of dSPN and iSPN for the demo plots.

    Returns (opto_ys, opto_xs) as 5-element lists ordered:
        [control, inh dSPN, inh iSPN, stim dSPN, stim iSPN]
    matching what plot_opto_inh / plot_opto_stim expect.
    """
    inputs = _base_inputs()
    T = cfg.TASK_CONFIG["t_total"]
    n_d1 = params["J_d1"].shape[0]
    n_d2 = params["J_d2"].shape[0]
    tmask = _temporal_mask(T)[:, None]

    conditions = [
        ("control", jnp.zeros(n_d1 + n_d2)),
        ("inh dSPN", _stim_vector(params, inh_d1, "d1")),
        ("inh iSPN", _stim_vector(params, inh_d2, "d2")),
        ("stim dSPN", _stim_vector(params, stim_d1, "d1")),
        ("stim iSPN", _stim_vector(params, stim_d2, "d2")),
    ]

    opto_ys, opto_xs = [], []
    for idx, (label, vec) in enumerate(conditions):
        stim_profile = tmask * vec[None, :]  # (T, n_bg)
        ys, xs = _run_condition(params, config, inputs, stim_profile, n_seeds, seed=idx)
        opto_ys.append(ys)
        opto_xs.append(xs)
        print(f"  [opto] {label:10s} done ({n_seeds} seeds)")
    return opto_ys, opto_xs


def run_opto_sweep(params, config, n_seeds=200):
    """Full strength sweep over cfg.spatial_stim_list.

    Returns (opto_ys, opto_xs) lists aligned with cfg.stim_labels /
    cfg.stim_strengths, as plot_opto expects.
    """
    inputs = _base_inputs()
    T = cfg.TASK_CONFIG["t_total"]
    tmask = _temporal_mask(T)[:, None]

    opto_ys, opto_xs = [], []
    for idx, (stim_vec, label, strength) in enumerate(
        zip(cfg.spatial_stim_list, cfg.stim_labels, cfg.stim_strengths)
    ):
        stim_profile = tmask * stim_vec[None, :]
        ys, xs = _run_condition(params, config, inputs, stim_profile, n_seeds, seed=idx)
        opto_ys.append(ys)
        opto_xs.append(xs)
        print(f"  [opto-sweep] {label:10s} strength={float(strength): .3f} done")
    return opto_ys, opto_xs


def make_all_opto_plots(params, config, demo_seeds=200, sweep_seeds=120):
    """Run both opto experiments and emit every opto plot."""
    print("Running opto demo conditions...")
    demo_ys, demo_xs = run_opto_demo(params, config, n_seeds=demo_seeds)
    pf.plot_opto_inh(demo_ys, demo_xs, None, newT=cfg.TASK_CONFIG["t_total"])
    pf.plot_opto_stim(demo_ys, demo_xs, None, newT=cfg.TASK_CONFIG["t_total"])

    print("Running opto strength sweep...")
    sweep_ys, sweep_xs = run_opto_sweep(params, config, n_seeds=sweep_seeds)
    pf.plot_opto(sweep_xs, None, sweep_ys, newT=cfg.TASK_CONFIG["t_total"])
    print("Opto plots complete.")


def main(params=None, config=None, demo_seeds=200, sweep_seeds=120):
    if params is None or config is None:
        params, config = load_bundle()
    make_all_opto_plots(params, config, demo_seeds=demo_seeds, sweep_seeds=sweep_seeds)


if __name__ == "__main__":
    main()
