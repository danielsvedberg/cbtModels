"""Evaluate the hybrid-trained CBT loop and plot its activity.

Loads params_hybrid.pkl, runs the network on hybrid STMT test trials (preparatory
cue at trial start on channel 0 + Pavlovian "go" cue during the movement window
on channel 1), and saves two figures into the hybrid_plots/ folder:
  - activity by area (aligned to trial start)
  - cue-aligned activity

Mirrors test_pavlovian.py.
"""

from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")

import cbt_rnn as cbtl
import config_script as cfg
import plotting_functions as pf
import self_timed_movement_task as stmt
import testing_script as ts

# Preparatory-cue onset times used for the test trials. The cue-aligned plot
# aligns every condition by these values, so the inputs and cs.test_start_t must
# agree.
TEST_START_T = jnp.arange(270, 330, 10)

# plotting_functions and the scripts all import the same `config_script`
# module, so a single patch target keeps test_start_t / save_fig in sync.
_PATCH_TARGETS = [cfg]


@contextmanager
def _hybrid_plot_context(starts):
    """Redirect plot saves to hybrid_plots/{svg,png}/ and set test_start_t."""
    base = Path(cfg.plots_folder).parent / "hybrid_plots"
    svg_dir = base / "svg"
    png_dir = base / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    old = [(m.plots_folder, m.svg_folder, m.png_folder, m.test_start_t)
           for m in _PATCH_TARGETS]
    for m in _PATCH_TARGETS:
        m.plots_folder = str(base)
        m.svg_folder = str(svg_dir)
        m.png_folder = str(png_dir)
        m.test_start_t = starts
    try:
        yield base
    finally:
        for m, (p, s, png, tst) in zip(_PATCH_TARGETS, old):
            m.plots_folder, m.svg_folder, m.png_folder, m.test_start_t = p, s, png, tst


def _load_hybrid_bundle():
    """Load params + config from params_hybrid.pkl."""
    import pickle as pkl

    hyb_path = cfg.hybrid_params_path()
    if not hyb_path.exists():
        raise FileNotFoundError(
            f"Missing {hyb_path}. Run train_hybrid.py first."
        )
    with hyb_path.open("rb") as f:
        bundle = pkl.load(f)

    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        return bundle["params"], bundle["config"]
    raise ValueError(f"Unexpected bundle format in {hyb_path}.")


def main():
    params, config = _load_hybrid_bundle()
    task_cfg = cfg.TASK_CONFIG

    inputs, _targets, _masks = stmt.hybrid_stmt(
        T_start=TEST_START_T,
        T_cue=task_cfg["t_cue"],
        T_wait=task_cfg["t_wait"],
        T_movement=task_cfg["t_movement"],
        T=task_cfg["t_total"],
    )

    all_ys, all_xs, _all_actions = cbtl.evaluate(
        params,
        config,
        inputs,
        noise_std=cfg.TEST_CONFIG["noise_std"],
        n_seeds=cfg.TEST_CONFIG["n_seeds"],
    )

    with _hybrid_plot_context(TEST_START_T) as base:
        ts._safe("activity by area", pf.plot_activity_by_area, all_xs)
        ts._safe("cue-aligned activity", pf.plot_cue_algn_activity, all_xs, ys=all_ys)
        print(f"\nHybrid plots saved under {base}")


if __name__ == "__main__":
    main()
