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
import optax
import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSCnoSTN')
import self_timed_movement_task as stmt
import plotting_functions as pf
import test_pnr
import opto_script


def _build_weight_matrix(params):
    """Assemble the full effective weight matrix including input and output lines."""
    # Cortex split into cU / cL (excitatory PT-like) + c_inh; packed [cU, cL, c_inh].
    n_c_U   = params["J_cU"].shape[0]
    n_c_L   = params["J_cL"].shape[0]
    n_c_inh = params["J_c_ii"].shape[0]
    n_c     = n_c_U + n_c_L + n_c_inh
    n_d1  = params["J_d1"].shape[0]
    n_d2  = params["J_d2"].shape[0]
    n_snc = params["P_snc"].shape[0]
    n_gpe = params["J_gpe"].shape[0]
    n_snr = params["P_snr"].shape[0]
    n_t_exc = params["J_t_ee"].shape[0]
    n_t_inh = params["J_t_ii"].shape[0]
    n_t     = n_t_exc + n_t_inh
    n_med = params["J_med_w1"].shape[0] * 2  # 2 E + 2 I units
    n_in  = params["B_cue_cU"].shape[1]   # number of input channels
    n_out = params["C_med"].shape[0]      # number of output channels

    def _from_cU(block):
        # Project from cU only: pad zeros on the cL / c_inh columns.
        block = np.array(block)
        return np.concatenate([block, np.zeros((block.shape[0], n_c_L + n_c_inh))], axis=1)

    def _from_cL(block):
        # Project from cL only: zeros on cU columns, block on cL, zeros on c_inh.
        block = np.array(block)
        return np.concatenate([np.zeros((block.shape[0], n_c_U)), block,
                               np.zeros((block.shape[0], n_c_inh))], axis=1)

    def _to_t_pools(exc_block, inh_block):
        return np.concatenate([np.array(exc_block), np.array(inh_block)], axis=0)

    def _assemble_J(ee, ei, ie, ii):
        top = np.concatenate([np.array(ee), np.array(ei)], axis=1)
        bot = np.concatenate([np.array(ie), np.array(ii)], axis=1)
        return np.concatenate([top, bot], axis=0)

    # Cortex recurrence: 3x3 block matrix in [cU, cL, c_inh] order (rows=post).
    row_cU = np.concatenate([cbtl.exc(params["J_cU"]),    cbtl.exc(params["B_cL_cU"]), cbtl.inh(params["J_ci_cU"])], axis=1)
    row_cL = np.concatenate([cbtl.exc(params["B_cU_cL"]), cbtl.exc(params["J_cL"]),    cbtl.inh(params["J_ci_cL"])], axis=1)
    row_ci = np.concatenate([cbtl.exc(params["J_cU_ci"]), cbtl.exc(params["J_cL_ci"]), cbtl.inh(params["J_c_ii"])],  axis=1)
    j_c_full = np.concatenate([row_cU, row_cL, row_ci], axis=0)
    j_t_full = _assemble_J(
        cbtl.exc(params["J_t_ee"]), cbtl.inh(params["J_t_ei"]),
        cbtl.exc(params["J_t_ie"]), cbtl.inh(params["J_t_ii"]),
    )

    # Input/Adenosine/Output are border areas: Input and Adenosine are tonic
    # sources (thin source lines), Output is a thin readout line.
    areas = ["Input", "Adenosine", "Cortex", "D1", "D2", "SNc", "GPe", "SNr", "Thalamus", "Medulla", "Output"]
    sizes = [n_in, 1, n_c, n_d1, n_d2, n_snc, n_gpe, n_snr, n_t, n_med, n_out]

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

    # Input → Cortex (cue projects to cU, cL, and c_inh pools).
    place("Cortex",   "Input",    np.concatenate(
        [np.array(params["B_cue_cU"]), np.array(params["B_cue_cL"]), np.array(params["B_cue_c_inh"])], axis=0))
    # Recurrent and inter-area connections
    place("Cortex",   "Cortex",   j_c_full)
    # Thalamus → Cortex: thalamic exc → cU (reciprocal) and c_inh (feedforward); cL none.
    to_c_from_t_exc = np.concatenate(
        [cbtl.exc(params["B_t_cU"]), np.zeros((n_c_L, n_t_exc)), cbtl.exc(params["B_t_c_inh"])], axis=0)
    b_t_c_full = np.concatenate([to_c_from_t_exc, np.zeros((n_c, n_t_inh))], axis=1)
    place("Cortex",   "Thalamus", b_t_c_full)
    place("D1",       "Cortex",   _from_cU(cbtl.exc(params["B_cU_d1"])))
    place("D1",       "D1",       cbtl.inh(params["J_d1"]))
    # D1↔D2 lateral inhibition is optional (may be disabled in the model).
    if "B_d2_d1" in params:
        place("D1",       "D2",       cbtl.inh(params["B_d2_d1"]))
    place("D2",       "Cortex",   _from_cU(cbtl.exc(params["B_cU_d2"])))
    place("D2",       "D2",       cbtl.inh(params["J_d2"]))
    if "B_d1_d2" in params:
        place("D2",       "D1",       cbtl.inh(params["B_d1_d2"]))
    place("SNc",      "Cortex",   _from_cL(cbtl.exc(params["B_cL_snc"])))
    place("SNc",      "D1",       cbtl.inh(params["B_d1_snc"]))
    place("SNc",      "D2",       cbtl.inh(params["B_d2_snc"]))
    # --- Neuromodulatory weights (dopamine + adenosine) onto D1/D2 PKA -----
    # Dopamine: SNc is broadcast (mean-pooled) as a single scalar to every SPN;
    # each SPN has a 1-D per-neuron gain m_d1[i] / m_d2[i]. To depict that in
    # the full weight matrix, spread each row's gain evenly across the n_snc
    # source columns (row-sum = effective scalar gain).
    m_floor, m_cap = 0.1, 0.5
    if "m_d1" in params:
        g_d1 = m_floor + np.logaddexp(0.0, np.array(params["m_d1"]))  # (n_d1,)
        place("D1", "SNc",
              np.broadcast_to(g_d1[:, None], (n_d1, n_snc)) / n_snc)
    if "m_d2" in params:
        g_d2 = m_cap / (1.0 + np.exp(-np.array(params["m_d2"])))      # (n_d2,)
        place("D2", "SNc",
              -np.broadcast_to(g_d2[:, None], (n_d2, n_snc)) / n_snc)
    # Adenosine: one tonic level k_a drives every SPN through per-neuron
    # weights m_a1 (inhibitory on D1 PKA) and m_a2 (excitatory on D2 PKA).
    # The depicted edge is the effective scalar contribution m_a · k_a.
    m_floor = 0.1
    k_a = max(0.0, np.tanh(float(np.array(params.get("k_a", 1.0)))))
    if "m_a1" in params:
        g_a1 = m_floor + np.maximum(0.0, np.tanh(np.array(params["m_a1"])))  # (n_d1,)
        place("D1", "Adenosine", -(g_a1 * k_a).reshape(n_d1, 1))
    if "m_a2" in params:
        g_a2 = m_floor + np.maximum(0.0, np.tanh(np.array(params["m_a2"])))  # (n_d2,)
        place("D2", "Adenosine", (g_a2 * k_a).reshape(n_d2, 1))
    place("GPe",      "D2",       cbtl.inh(params["B_d2_gpe"]))
    place("GPe",      "Cortex",   _from_cU(cbtl.exc(params["B_cU_gpe"])))  # cU → GPe (exc)
    place("GPe",      "GPe",      cbtl.inh(params["J_gpe"]))
    place("SNr",      "D1",       cbtl.inh(params["B_d1_snr"]))
    place("SNr",      "GPe",      cbtl.inh(params["B_gpe_snr"]))
    # Cortex → Thalamus: only cU projects (to both thalamic pools).
    b_cU_t = _to_t_pools(cbtl.exc(params["B_cU_t_exc"]), cbtl.exc(params["B_cU_t_inh"]))  # (n_t, n_c_U)
    place("Thalamus", "Cortex",   _from_cU(b_cU_t))
    place("Thalamus", "SNr",      _to_t_pools(cbtl.inh(params["B_snr_t_exc"]),
                                              cbtl.inh(params["B_snr_t_inh"])))
    place("Thalamus", "Thalamus", j_t_full)
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
    # B_cL_med is (2, n_c_L): cL projects to medullary E units only (cL columns).
    r0 = offsets["Medulla"][0]
    c0, c1 = offsets["Cortex"]
    W[r0:r0 + 2, c0 + n_c_U:c0 + n_c_U + n_c_L] = np.array(params["B_cL_med"])
    # Medulla → Output: readout from E units only (first 2 columns of Medulla).
    r0, r1 = offsets["Output"]
    c0 = offsets["Medulla"][0]
    W[r0:r1, c0:c0 + 2] = np.array(cbtl.exc(params["C_med"]))

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
    mode = task_cfg["task_mode"]
    if mode == "hybrid":
        return stmt.hybrid_stmt(**kwargs)
    if mode == "pavlovian":
        return stmt.pavlovian_stmt(**kwargs)
    return stmt.self_timed_movement_task(**kwargs)


def _match_input_channels(inputs, params):
    """Pad/truncate test inputs to match the model's cue-channel count.

    The STMT task emits a single (timing) cue channel. Models trained with
    train_from_pavlovian expect an extra Pavlovian-cue channel; the timing cue
    stays on channel 0 and any extra channels are filled with zeros (cue
    absent), which is the correct pure-STMT test condition.
    """
    n_expected = params["B_cue_cU"].shape[1]
    n_have = inputs.shape[-1]
    if n_have == n_expected:
        return inputs
    if n_have < n_expected:
        pad = jnp.zeros(inputs.shape[:-1] + (n_expected - n_have,), dtype=inputs.dtype)
        return jnp.concatenate([inputs, pad], axis=-1)
    return inputs[..., :n_expected]


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
    _, config = cbtl.init_params(jr.PRNGKey(0), n_input=1)
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


# ---------------------------------------------------------------------------
# Lesion utilities
# ---------------------------------------------------------------------------
def lesion_nigrostriatal(params):
    """Zero out the SNc → striatum DA modulation (nigrostriatal projection)."""
    p = dict(params)
    p["m_d1"] = jnp.zeros_like(params["m_d1"])
    p["m_d2"] = jnp.zeros_like(params["m_d2"])
    return p


def lesion_thalamocortical(params):
    """Zero out the reciprocal thalamus↔cortex (T→C and C→T) pathways."""
    p = dict(params)
    p["B_t_cU"] = jnp.zeros_like(params["B_t_cU"])
    p["B_t_c_inh"] = jnp.zeros_like(params["B_t_c_inh"])
    p["B_cU_t_exc"] = jnp.zeros_like(params["B_cU_t_exc"])
    p["B_cU_t_inh"] = jnp.zeros_like(params["B_cU_t_inh"])
    return p


def _make_lesioned_rnn_func(rnn_func, lesion_fn):
    """Wrap rnn_func so `lesion_fn` is reapplied to params on every forward call.

    Because the lesioned weights are replaced by zeros inside the loss, the
    gradient w.r.t. those raw params is exactly 0, so they stay dead under
    Adam/AdamW even though they're still nominally trainable.
    """
    def lesioned(params, config, batch_inputs, opto_stim, rng_keys):
        return rnn_func(lesion_fn(params), config, batch_inputs, opto_stim, rng_keys)
    return lesioned


def pd_params_path() -> Path:
    """Path to the SNc-lesion-retrained parameter bundle."""
    return Path(cfg.params_path()).parent / "pd_exp_params.pkl"


def train_pd_experiment(num_iters=2000, save_path=None):
    """Lesion SNc→striatum from the current trained bundle and retrain.

    Loads params_nm.pkl, zeros m_d1/m_d2, trains for `num_iters` REINFORCE
    iterations (lesion reapplied each forward pass), saves the result to
    pd_exp_params.pkl (lesion still applied) and returns (params, config).
    """
    base_params, config = _load_bundle()
    base_params = lesion_nigrostriatal(base_params)

    task_cfg = cfg.TASK_CONFIG
    train_cfg = cfg.TRAINING_CONFIG
    rl_cfg = cfg.RL_CONFIG

    inputs, targets, masks = _build_test_inputs(task_cfg, task_cfg["t_start"])

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]),
    )
    lesioned_rnn = _make_lesioned_rnn_func(cbtl.rnn_func, lesion_nigrostriatal)

    best_params, losses, rewards = stmt.fit_rnn_reinforce(
        lesioned_rnn,
        base_params,
        config,
        inputs,
        masks,
        optimizer,
        num_iters,
        log_interval=train_cfg["log_interval"],
        seed=train_cfg["seed"],
        baseline_momentum=rl_cfg["baseline_momentum"],
        entropy_coef=rl_cfg["entropy_coef"],
        objective_mode=rl_cfg["objective_mode"],
        batch_targets=targets,
        brevity_coef=rl_cfg["brevity_coef"],
        silence_coef=rl_cfg["silence_coef"],
        tail_coef=rl_cfg["tail_coef"],
    )
    # Persist the lesion in the saved bundle (defense against optimizer drift).
    best_params = lesion_nigrostriatal(best_params)

    save_path = Path(save_path) if save_path else pd_params_path()
    with save_path.open("wb") as f:
        pkl.dump({"params": best_params, "config": config}, f)
    print(f"Saved PD-trained params to: {save_path}")
    return best_params, config


def load_pd_bundle(train_if_missing=True, num_iters=2000, save_path=None):
    """Load pd_exp_params.pkl, optionally training (with SNc lesion) if absent."""
    save_path = Path(save_path) if save_path else pd_params_path()
    if save_path.exists():
        with save_path.open("rb") as f:
            bundle = pkl.load(f)
        return bundle["params"], bundle["config"]
    if not train_if_missing:
        raise FileNotFoundError(
            f"Missing {save_path}; call train_pd_experiment() or pass train_if_missing=True."
        )
    return train_pd_experiment(num_iters=num_iters, save_path=save_path)


def main():
    params, config = _load_bundle()

    starts = cfg.TEST_CONFIG["start_t"]
    inputs, targets, masks = _build_test_inputs(cfg.TASK_CONFIG, starts)
    inputs = _match_input_channels(inputs, params)

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
