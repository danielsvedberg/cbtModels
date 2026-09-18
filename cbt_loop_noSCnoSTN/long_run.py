"""One long training run on the 300-step self-timed task, fully self-contained in a folder.

Uses the current pipeline config: thalamic readout, plateau at [cue+300, cue+350), and the
19x in-window loss weight that took the breakthrough rate from 40% to 90% (see
config_script SUPERVISED_THAL_CONFIG.in_window_weight).

Writes into --outdir:
    params.pkl    trained params + config (best_params, by tracked loss)
    summary.md    metrics: separation, MSE, cue-lock offsets, per-area health
    objective.png output vs target, per-timestep error, per-cue-onset MSE
    timing.png    cue-aligned traces per cue onset, with the target window marked
    metrics.json  the same numbers, machine-readable

Usage:  python long_run.py --seed 42 --iters 20000 --outdir runs/seed42
"""
import argparse, json, pathlib, pickle, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp, jax.random as jr, optax

_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as cs
import cbt_rnn as cbtl
import self_timed_movement_task as stmt

cfg = cs.for_family("cbt_loop_noSCnoSTN")
AREAS = ["Cortex", "D1", "D2", "SNc", "GPe", "SNr", "Thalamus", "pkaD1", "pkaD2", "Medulla", "DA", "Adenosine"]


def build(sup, t, starts=None):
    T_start = t["t_start"] if starts is None else jnp.asarray(starts)
    return stmt.selftimed_step_target(
        T_start=T_start, T_cue=t["t_cue"], T_wait=t["t_wait"], T_movement=t["t_movement"],
        T=t["t_total"], lo=sup["target_lo"], hi=sup["target_hi"],
        hold=sup["hold"], delay=sup["delay"])


def weighted_mask(targets, sup):
    w = float(sup.get("in_window_weight", 1.0))
    if w == 1.0:
        return None, w
    hi = np.asarray(targets)[..., 0] > (sup["target_lo"] + sup["target_hi"]) / 2
    return jnp.asarray(np.where(hi, w, 1.0)[..., None].astype(np.float32)), w


def main(seed, iters, outdir):
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    sup, t = dict(cfg.SUPERVISED_THAL_CONFIG), cfg.TASK_CONFIG
    inputs, targets, masks = build(sup, t)
    wmask, w = weighted_mask(targets, sup)
    if wmask is not None:
        masks = wmask
    params, config = cbtl.init_params(jr.PRNGKey(seed), n_input=inputs.shape[-1])
    config = dict(config); config["readout_source"] = "thalamus"
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(learning_rate=cfg.OPTIM_CONFIG["learning_rate"]))
    if sup.get("freeze_init_states", False):
        opt = cbtl.freeze_init_state_optimizer(opt, params)
        print(f"[freeze] init-state params frozen: {len(cbtl.INIT_STATE_KEYS)} keys")
    print(f"[run] seed={seed} iters={iters} in_window_weight={w} delay={sup['delay']} "
          f"hold={sup['hold']} -> {out}", flush=True)
    t0 = time.time()
    best, losses, accs = stmt.fit_rnn_supervised(
        cbtl.rnn_func, params, config, inputs, masks, opt, iters,
        batch_targets=targets, log_interval=sup["log_interval"], seed=seed,
        loss_type=sup["loss_type"])
    secs = time.time() - t0
    with (out / "params.pkl").open("wb") as f:
        pickle.dump({"params": best, "config": config, "seed": seed, "iters": iters}, f)

    # ---- evaluation on held-out cue onsets (deterministic forward) ----
    starts = np.arange(260, 400, 20)
    e_in, e_tg, _ = build(sup, t, starts)
    out_all = cbtl.rnn_func(best, config, e_in, None, jr.split(jr.PRNGKey(0), len(starts)))
    ys = np.asarray(out_all[0])[..., 0]; xs = out_all[4]
    tg = np.asarray(e_tg)[..., 0]
    hi = tg > (sup["target_lo"] + sup["target_hi"]) / 2
    sep = float(ys[hi].mean() - ys[~hi].mean())
    mse = float(((ys - tg) ** 2).mean())
    half = (sup["target_lo"] + sup["target_hi"]) / 2
    offs = []
    for i, s0 in enumerate(starts):
        above = np.where(ys[i] > half)[0]
        offs.append(int(above.min()) - int(s0) if above.size else None)
    good = [o for o in offs if o is not None]
    health = {}
    for i, nm in enumerate(AREAS):
        a = np.asarray(xs[i])
        health[nm] = dict(mean=float(a.mean()), dead=float((a < 0.01).mean()),
                          sat=float((a > 0.9).mean()))
    m = dict(seed=seed, iters=iters, in_window_weight=w, secs=round(secs, 1),
             final_loss=float(losses[-1]) if losses else None,
             min_loss=float(min(losses)) if losses else None,
             separation=sep, mse=mse, target_separation=sup["target_hi"] - sup["target_lo"],
             offsets=offs, offset_mean=float(np.mean(good)) if good else None,
             offset_std=float(np.std(good)) if good else None,
             expected_offset=sup["delay"], area_health=health,
             loss_trace=[float(x) for x in losses])
    (out / "metrics.json").write_text(json.dumps(m, indent=1))

    # ---- plots ----
    pre, post = 60, sup["delay"] + sup["hold"] + 120
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.8))
    al_y, al_t, al_e = [], [], []
    for i, s0 in enumerate(starts):
        a, b = int(s0) - pre, int(s0) + post
        if a < 0 or b > ys.shape[1]: continue
        al_y.append(ys[i, a:b]); al_t.append(tg[i, a:b]); al_e.append((ys[i, a:b] - tg[i, a:b]) ** 2)
    al_y, al_t, al_e = np.array(al_y), np.array(al_t), np.array(al_e)
    tt = np.arange(-pre, post)
    ax = axs[0]
    ax.axvspan(sup["delay"], sup["delay"] + sup["hold"], color="green", alpha=0.12)
    ax.axvline(0, color="red", lw=0.8)
    ax.plot(tt, al_t.mean(0), "k--", lw=1.2, label="target")
    ax.plot(tt, al_y.mean(0), c="darkgreen", lw=1.3, label="output")
    ax.fill_between(tt, al_y.mean(0) - al_y.std(0), al_y.mean(0) + al_y.std(0),
                    color="darkgreen", alpha=0.2, lw=0)
    ax.set_xlabel("timesteps from cue"); ax.set_ylabel("output"); ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    ax.axvspan(sup["delay"], sup["delay"] + sup["hold"], color="green", alpha=0.12)
    ax.plot(tt, al_e.mean(0), c="firebrick", lw=1.0)
    ax.set_xlabel("timesteps from cue"); ax.set_ylabel("squared error")
    ax = axs[2]
    per = ((ys - tg) ** 2).mean(axis=1)
    ax.bar(range(len(starts)), per, color="steelblue")
    ax.set_xticks(range(len(starts))); ax.set_xticklabels([str(s) for s in starts], fontsize=6)
    ax.set_xlabel("cue onset"); ax.set_ylabel("MSE")
    fig.suptitle(f"seed {seed} | sep {sep:.4f} | MSE {mse:.6f} | offset "
                 f"{m['offset_mean']:.1f}±{m['offset_std']:.2f}" if good else f"seed {seed}",
                 fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out / "objective.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    for i, s0 in enumerate(starts):
        ax.plot(np.arange(ys.shape[1]) - s0, ys[i], lw=0.9, label=f"cue {s0}")
    ax.axvspan(sup["delay"], sup["delay"] + sup["hold"], color="green", alpha=0.12)
    ax.axvline(0, color="red", lw=0.8)
    ax.set_xlim(-80, sup["delay"] + sup["hold"] + 150)
    ax.set_xlabel("timesteps from cue onset"); ax.set_ylabel("output")
    ax.set_title(f"seed {seed}: cue-aligned, all onsets (curves should overlap if cue-locked)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    plt.tight_layout(); fig.savefig(out / "timing.png", dpi=150); plt.close(fig)

    lines = [f"# long run — seed {seed}\n",
             f"- iters **{iters}**, in-window weight **{w}**, wall clock **{secs/60:.1f} min**",
             f"- final loss **{m['final_loss']:.6f}**, min loss **{m['min_loss']:.6f}**",
             f"- separation **{sep:.4f}** of a possible {m['target_separation']:.2f}",
             f"- MSE **{mse:.6f}** (constant-output solution = 0.011875)",
             f"- cue-lock offset **{m['offset_mean']:.1f} ± {m['offset_std']:.2f}** "
             f"(expected {sup['delay']})" if good else "- no threshold crossing",
             f"- per-onset offsets: {offs}\n", "## area health (trained)\n",
             "| area | mean | frac dead (<0.01) | frac sat (>0.9) |", "|---|---|---|---|"]
    for nm, h in health.items():
        lines.append(f"| {nm} | {h['mean']:.4f} | {h['dead']:.3f} | {h['sat']:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[run] seed={seed} DONE sep={sep:.4f} mse={mse:.6f} "
          f"offset={m['offset_mean'] if good else None} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--outdir", type=str, required=True)
    a = ap.parse_args()
    main(a.seed, a.iters, a.outdir)
