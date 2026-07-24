"""Lesion experiments: run the non-perturbational test plots for each
lesion condition and save outputs into per-condition folders under plots/.

Conditions:
- ns_lesion: nigrostriatal projection (SNc → striatum) zeroed.
- tc_lesion: thalamocortical and corticothalamic pathways zeroed.
- pd_test:   PD-experiment params (SNc lesion retrained for N iterations).
"""

from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSCnoSTN')
import plotting_functions as pf
import testing_script as ts

# plotting_functions and the scripts all import the same `config_script`
# module, so a single patch target keeps `save_fig` in sync with our redirects.
_PATCH_TARGETS = [cfg]


@contextmanager
def _plots_folder(name):
    """Redirect plot saves to plots/<name>/{svg,png}/ during the block."""
    base = Path(cfg.plots_folder) / name
    svg_dir = base / "svg"
    png_dir = base / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    old = [(m.plots_folder, m.svg_folder, m.png_folder) for m in _PATCH_TARGETS]
    for m in _PATCH_TARGETS:
        m.plots_folder = str(base)
        m.svg_folder = str(svg_dir)
        m.png_folder = str(png_dir)
    try:
        yield base
    finally:
        for m, (p, s, png) in zip(_PATCH_TARGETS, old):
            m.plots_folder, m.svg_folder, m.png_folder = p, s, png


def _run_plots(params, config, label):
    """Run the non-perturbational plot suite (no opto / no PNR) for given params."""
    starts = cfg.TEST_CONFIG["start_t"]
    inputs, targets, masks = ts._build_test_inputs(cfg.TASK_CONFIG, starts)
    all_ys, all_xs, all_actions = cbtl.evaluate(
        params,
        config,
        inputs,
        noise_std=cfg.TEST_CONFIG["noise_std"],
        n_seeds=cfg.TEST_CONFIG["n_seeds"],
    )

    response_times = ts._response_times_from_actions(
        all_actions, starts, cfg.TASK_CONFIG["t_cue"]
    )
    valid_rts = response_times[~jnp.isnan(response_times)]
    d1d2_ratio = cbtl.get_d1_d2_ratio(all_xs, 100, 300, avg_time=True, remove_outliers=False)

    target_2d = targets[..., 0]
    action_2d = all_actions[..., 0]
    in_target = target_2d[None, ...] > 0.5
    success = jnp.any((action_2d > 0.5) & in_target, axis=2)

    print(f"\n=== [{label}] mean success = {float(jnp.mean(success)):.3f} ===")
    if valid_rts.size > 0:
        print(f"=== [{label}] mean response time (s) = {float(jnp.mean(valid_rts)):.3f} ===")
    else:
        print(f"=== [{label}] mean response time (s) = NaN (no responses) ===")

    ts._safe(f"[{label}] weight matrix", ts._save_weight_matrix, params, cfg.plots_folder)
    ts._safe(f"[{label}] output activity", pf.plot_output, all_ys)
    ts._safe(f"[{label}] activity by area", pf.plot_activity_by_area, all_xs)
    ts._safe(f"[{label}] cue-aligned activity", pf.plot_cue_algn_activity, all_xs, ys=all_ys)
    ts._safe(f"[{label}] binned responses",
             pf.plot_binned_responses, all_ys, all_xs, None, all_actions)
    if valid_rts.size > 0:
        ts._safe(f"[{label}] response times", pf.plot_response_times, valid_rts)
    ts._safe(f"[{label}] dSPN-iSPN vs SNc",
             pf.plot_d1d2ratio_SNc_correlogram, d1d2_ratio, all_xs, response_times)
    ts._safe(f"[{label}] dSPN-iSPN vs slope",
             pf.plot_d1d2ratio_slope_correlogram, all_xs, response_times)
    ts._safe(f"[{label}] dSPN-iSPN vs RT",
             pf.plot_ratio_rt_correlogram, d1d2_ratio, response_times)


def main(pd_num_iters=2000):
    base_params, base_config = ts._load_bundle()

    # 1) Nigrostriatal lesion (SNc → striatum projection zeroed)
    print("\n[1/3] Nigrostriatal lesion ...")
    ns_params = ts.lesion_nigrostriatal(base_params)
    with _plots_folder("ns_lesion"):
        _run_plots(ns_params, base_config, "ns_lesion")

    # 2) Thalamocortical/corticothalamic lesion (T↔C reciprocal pathways zeroed)
    print("\n[2/3] Thalamocortical lesion ...")
    tc_params = ts.lesion_thalamocortical(base_params)
    with _plots_folder("tc_lesion"):
        _run_plots(tc_params, base_config, "tc_lesion")

    # 3) PD experiment (SNc-lesioned then retrained for pd_num_iters iterations)
    print(f"\n[3/3] PD experiment (SNc-lesion + retrain, {pd_num_iters} iters) ...")
    pd_params, pd_config = ts.load_pd_bundle(train_if_missing=True, num_iters=pd_num_iters)
    with _plots_folder("pd_test"):
        _run_plots(pd_params, pd_config, "pd_test")


if __name__ == "__main__":
    main()
