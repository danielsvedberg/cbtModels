import csv
import os
import traceback
import pickle as pkl
from pathlib import Path
import sys
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt
import plotting_functions as pf
import test_pnr
import opto_script


def _build_weight_matrix(params):
    """Assemble the full effective weight matrix including input and output lines."""
    n_c   = params["J_c"].shape[0]
    n_d1  = params["J_d1"].shape[0]
    n_d2  = params["J_d2"].shape[0]
    n_snc = params["P_snc"].shape[0]
    n_gpe = params["J_gpe"].shape[0]
    n_stn = params["J_stn"].shape[0]
    n_snr = params["P_snr"].shape[0]
    n_t   = params["J_t"].shape[0]
    n_med = params["J_med_w1"].shape[0] * 2  # 2 E + 2 I units
    n_in  = params["B_cue_c"].shape[1]   # number of input channels
    n_out = params["C_med"].shape[0]      # number of output channels

    # Input and Output are prepended/appended so they appear as thin border lines.
    areas = ["Input", "Cortex", "D1", "D2", "SNc", "GPe", "STN", "SNr", "Thalamus", "Medulla", "Output"]
    sizes = [n_in, n_c, n_d1, n_d2, n_snc, n_gpe, n_stn, n_snr, n_t, n_med, n_out]

    offsets = {}
    off = 0
    for area, sz in zip(areas, sizes):
        offsets[area] = (off, off + sz)
        off += sz
    N = off

    W = np.zeros((N, N))

    def place(tgt, src, block):
        r0, r1 = offsets[tgt]
        c0, c1 = offsets[src]
        W[r0:r1, c0:c1] = np.array(block)

    # Input → Cortex
    place("Cortex",   "Input",    params["B_cue_c"])
    # Recurrent and inter-area connections
    place("Cortex",   "Cortex",   params["J_c"])
    place("Cortex",   "Thalamus", params["B_t_c"])
    place("D1",       "Cortex",   params["B_c_d1"])
    place("D1",       "D1",       cbtl.inh(params["J_d1"]))
    place("D2",       "Cortex",   params["B_c_d2"])
    place("D2",       "D2",       cbtl.inh(params["J_d2"]))
    place("SNc",      "STN",      cbtl.exc(params["B_stn_snc"]))
    place("SNc",      "D1",       cbtl.inh(params["B_d1_snc"]))
    place("SNc",      "D2",       cbtl.inh(params["B_d2_snc"]))
    place("GPe",      "D2",       cbtl.inh(params["B_d2_gpe"]))
    place("GPe",      "STN",      cbtl.exc(params["B_stn_gpe"]))
    place("GPe",      "GPe",      cbtl.inh(params["J_gpe"]))
    place("STN",      "GPe",      cbtl.inh(params["B_gpe_stn"]))
    place("STN",      "Cortex",   cbtl.exc(params["B_c_stn"]))
    place("STN",      "STN",      cbtl.exc(params["J_stn"]))
    place("SNr",      "D1",       cbtl.inh(params["B_d1_snr"]))
    place("SNr",      "STN",      cbtl.exc(params["B_stn_snr"]))
    place("Thalamus", "Cortex",   params["B_c_t"])
    place("Thalamus", "SNr",      cbtl.inh(params["B_snr_t"]))
    place("Thalamus", "Thalamus", params["J_t"])
    def _med_block(raw):
        return np.concatenate([np.array(cbtl.exc(raw[:, :1])), np.array(cbtl.inh(raw[:, 1:]))], axis=1)

    j_w1 = _med_block(params["J_med_w1"])
    j_w2 = _med_block(params["J_med_w2"])
    j_x  = _med_block(params["J_med_x"])
    j_med = np.array([
        [j_w1[0, 0], j_x[0, 0],  j_w1[0, 1], j_x[0, 1]],
        [j_x[0, 0],  j_w2[0, 0], j_x[0, 1],  j_w2[0, 1]],
        [j_w1[1, 0], j_x[1, 0],  j_w1[1, 1], j_x[1, 1]],
        [j_x[1, 0],  j_w2[1, 0], j_x[1, 1],  j_w2[1, 1]],
    ])
    place("Medulla",  "Medulla",  j_med)
    # B_c_med is (2, n_c): cortex only projects to E units (first 2 rows of Medulla)
    r0 = offsets["Medulla"][0]
    c0, c1 = offsets["Cortex"]
    W[r0:r0 + 2, c0:c1] = np.array(params["B_c_med"])
    # Medulla → Output
    c_med = np.concatenate([
        np.array(cbtl.exc(params["C_med"][:, :2])),
        np.array(cbtl.inh(params["C_med"][:, 2:])),
    ], axis=1)
    place("Output",   "Medulla",  c_med)

    labels = [f"{a}_{i}" for a, sz in zip(areas, sizes) for i in range(sz)]
    return W, labels, areas, sizes, offsets


def _save_weight_matrix(params, plots_folder):
    W, labels, areas, sizes, offsets = _build_weight_matrix(params)
    plots_folder = Path(plots_folder)

    # CSV with row and column unit labels
    csv_path = plots_folder / "weight_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, W):
            writer.writerow([label] + [f"{v:.6f}" for v in row])
    print(f"Weight matrix CSV saved to {csv_path}")

    # Heatmap
    vmax = np.abs(W).max() or 1.0
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="weight")

    # area boundary lines and tick labels
    ticks, tick_labels = [], []
    for area, sz in zip(areas, sizes):
        r0, r1 = offsets[area]
        ticks.append((r0 + r1) / 2)
        tick_labels.append(area)
        if r0 > 0:
            ax.axhline(r0 - 0.5, color="k", linewidth=0.6)
            ax.axvline(r0 - 0.5, color="k", linewidth=0.6)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Source area")
    ax.set_ylabel("Target area")
    ax.set_title("Effective weight matrix")
    fig.tight_layout()

    png_path = plots_folder / "png" / "weight_matrix.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Weight matrix heatmap saved to {png_path}")


def _build_test_inputs(task_cfg, starts):
    kwargs = {
        "T_start": starts,
        "T_cue": task_cfg["t_cue"],
        "T_wait": task_cfg["t_wait"],
        "T_movement": task_cfg["t_movement"],
        "T": task_cfg["t_total"],
    }
    mode = task_cfg.get("task_mode", "self_timed")
    if mode == "hybrid":
        return stmt.hybrid_stmt(**kwargs)
    if mode == "pavlovian":
        return stmt.pavlovian_stmt(**kwargs)
    return stmt.self_timed_movement_task(**kwargs)


def _response_times_from_actions(actions, starts, t_cue, threshold=0.5):
    n_seeds, n_conditions = actions.shape[:2]
    response_times = jnp.full((n_seeds, n_conditions), jnp.nan)
    for seed_idx in range(n_seeds):
        for cond_idx in range(n_conditions):
            cue_end = starts[cond_idx] + t_cue
            post = actions[seed_idx, cond_idx, cue_end:, 0]
            response_idx = jnp.argmax(post > threshold)
            if post[response_idx] > threshold:
                response_times = response_times.at[seed_idx, cond_idx].set(response_idx * 0.01)
    return response_times


def _load_bundle():
    """Load params + config, rebuilding config for legacy (params-only) bundles."""
    params_path = cfg.params_path()
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}. Run training_script.py first.")

    with params_path.open("rb") as f:
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


def _safe(label, fn, *args, **kwargs):
    """Run a plotting/analysis call, logging failures instead of aborting the run."""
    print(f"-> {label} ...")
    try:
        fn(*args, **kwargs)
        print(f"   [ok] {label}")
    except Exception as exc:  # noqa: BLE001 - we want every plot attempted
        print(f"   [FAIL] {label}: {exc}")
        traceback.print_exc()


def main():
    params, config = _load_bundle()

    starts = cfg.TEST_CONFIG["start_t"]
    inputs, targets, masks = _build_test_inputs(cfg.TASK_CONFIG, starts)

    all_ys, all_xs, all_actions = cbtl.evaluate(
        params,
        config,
        inputs,
        noise_std=cfg.TEST_CONFIG["noise_std"],
        n_seeds=cfg.TEST_CONFIG["n_seeds"],
    )

    # Derived quantities shared by several plots.
    response_times = _response_times_from_actions(all_actions, starts, cfg.TASK_CONFIG["t_cue"])
    valid_rts = response_times[~jnp.isnan(response_times)]
    d1d2_ratio = cbtl.get_d1_d2_ratio(all_xs, 100, 300, avg_time=True, remove_outliers=False)

    target_2d = targets[..., 0]
    action_2d = all_actions[..., 0]
    in_target = target_2d[None, ...] > 0.5
    success = jnp.any((action_2d > 0.5) & in_target, axis=2)

    print("ys shape:", all_ys.shape)
    print("actions shape:", all_actions.shape)
    print("num state arrays:", len(all_xs))
    print("state[0] shape:", all_xs[0].shape)
    print("mean success:", float(jnp.mean(success)))
    if valid_rts.size > 0:
        print("mean response time (s):", float(jnp.mean(valid_rts)))
    else:
        print("mean response time (s): NaN (no responses)")

    # --- Weight matrix --------------------------------------------------
    _safe("weight matrix (csv + heatmap)", _save_weight_matrix, params, cfg.plots_folder)

    # --- Core activity plots -------------------------------------------
    _safe("output activity", pf.plot_output, all_ys)
    _safe("activity by area", pf.plot_activity_by_area, all_xs)
    _safe("cue-aligned activity", pf.plot_cue_algn_activity, all_xs, ys=all_ys)
    _safe("binned responses", pf.plot_binned_responses, all_ys, all_xs, None, all_actions)
    if valid_rts.size > 0:
        _safe("response time distributions", pf.plot_response_times, valid_rts)

    # --- Correlograms ---------------------------------------------------
    _safe("dSPN-iSPN vs SNc correlogram",
          pf.plot_d1d2ratio_SNc_correlogram, d1d2_ratio, all_xs, response_times)
    _safe("dSPN-iSPN vs ramp-slope correlogram",
          pf.plot_d1d2ratio_slope_correlogram, all_xs, response_times)
    _safe("dSPN-iSPN vs response-time correlogram",
          pf.plot_ratio_rt_correlogram, d1d2_ratio, response_times)

    # --- Static schematic plots ----------------------------------------
    _safe("loss function schematic", pf.plot_loss_function)
    _safe("adaptive loss function schematic", pf.plot_loss_function_adaptive)

    # --- Opto experiments (plot_opto_inh / plot_opto_stim / plot_opto) --
    _safe("opto experiments", opto_script.make_all_opto_plots, params, config,
          120, 60)

    # --- Point-of-no-return experiments (plot_binned_pnr / colorbars) ---
    _safe("point-of-no-return experiments", test_pnr.run_pnr_experiments, params, config,
          30)

    print("Done. Plots written under", cfg.plots_folder)


if __name__ == "__main__":
    main()
