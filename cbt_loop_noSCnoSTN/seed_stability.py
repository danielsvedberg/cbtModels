"""Seed-stability study: how often does training break through, and can anything at init predict it?

Training on the 300-step target is stochastic -- the solved config came back 2/5 across seeds.
Every config comparison made at n=1 in this project is therefore uninterpretable. This
accumulates per-seed data so the breakthrough RATE (and any init-time predictor) can be
estimated with real power instead of guessed from single runs.

For each seed it records: init-time diagnostics (measured BEFORE training, so they are
legitimate predictors), then the trained outcome (separation + MSE of best_params under a
deterministic forward pass -- not the logged loss, which mixes in penalty terms).

Results append to seed_stability_data.json after EVERY seed, so an interrupted batch keeps
what it finished. Re-running a seed already present is skipped unless --force.

Usage:  python seed_stability.py --start 0 --n 10 [--iters 1400]
"""
import argparse, json, pathlib, sys, time
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, optax

_root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as cs

# The configuration under test: the one that trained to separation 0.486 (raw units;
# exc/inh = clip(tanh(w),0,None), so a negative raw means the gain is exactly 0).
GAINS = {"m_d1": 1.1363, "m_d2": -1.5368, "m_a1": -3.4095, "m_a2": 0.0080}
RUNTIME = {"da_release": 0.2982, "ado_release": 1.1363, "pin_ado": 0.093,
           "stri_cross_scale": 0.162, "stri_tonic": 0.220}
BREAKTHROUGH_SEP = 0.20      # winners land ~0.45, collapses at exactly 0.000 -- wide margin
DATA = pathlib.Path(__file__).with_name("seed_stability_data.json")


def _load():
    return json.loads(DATA.read_text()) if DATA.exists() else []


def _save(rows):
    DATA.write_text(json.dumps(rows, indent=1))


def init_metrics(cbtl, stmt, cfg, seed, inp, n=16):
    """Everything measurable BEFORE training -- candidate predictors of breakthrough."""
    p, c = cbtl.init_params(jr.PRNGKey(seed), n_input=1)
    c = dict(c); c["readout_source"] = "thalamus"
    xs = cbtl.rnn_func(p, c, inp[:n], None, jr.split(jr.PRNGKey(seed), n))[4]
    nte = p["J_t_ee"].shape[0]
    te = np.asarray(xs[6])[..., :nte]
    return dict(D1=float(np.asarray(xs[1]).mean()), D2=float(np.asarray(xs[2]).mean()),
                cortex=float(np.asarray(xs[0]).mean()), thal=float(te.mean()),
                thal_frac0=float((te == 0).mean()),
                pkaD1=float(np.asarray(xs[7]).mean()), pkaD2=float(np.asarray(xs[8]).mean()),
                SNr=float(np.asarray(xs[5]).mean()))


def grad_metrics(cbtl, stmt, cfg, seed, inp, tg, mk, n=16):
    p, c = cbtl.init_params(jr.PRNGKey(seed), n_input=1)
    c = dict(c); c["readout_source"] = "thalamus"
    keys = jr.split(jr.PRNGKey(seed), n)
    def loss(pp):
        return stmt.supervised_loss(cbtl.rnn_func, pp, c, inp[:n], tg[:n], mk[:n], keys, loss_type="mse")[0]
    g = jax.grad(loss)(p)
    gn = {k: float(jnp.linalg.norm(jnp.asarray(v))) for k, v in g.items()}
    tot = float(np.sqrt(sum(v ** 2 for v in gn.values())))
    return dict(grad_total=tot, grad_C_thal_pct=100 * gn["C_thal"] / tot,
                grad_out_bias_pct=100 * gn["out_bias"] / tot)


def outcome(cbtl, stmt, cfg, params, config, inp, tg):
    """Deterministic separation + MSE of the trained weights -- the unambiguous outcome."""
    ys = np.asarray(cbtl.rnn_func(params, config, inp, None, jr.split(jr.PRNGKey(0), inp.shape[0]))[0])[..., 0]
    t2 = np.asarray(tg)[..., 0]
    hi = t2 > 0.5
    return float(ys[hi].mean() - ys[~hi].mean()), float(((ys - t2) ** 2).mean())


def main(start, n, iters, force, lr=None, wt=1.0, target_lo=None):
    st = cs._CBT_FAMILY_STRUCTURE["cbt_loop_noSCnoSTN"]
    st["extra_weight_init"] = dict(GAINS)
    st["extra_runtime"] = {**st["extra_runtime"], **RUNTIME}
    import cbt_rnn as cbtl, self_timed_movement_task as stmt
    cfg = cs.for_family("cbt_loop_noSCnoSTN")
    sup, t = dict(cfg.SUPERVISED_THAL_CONFIG), cfg.TASK_CONFIG
    if target_lo is not None:
        sup["target_lo"] = float(target_lo)
        print(f"target_lo = {sup['target_lo']}", flush=True)
    inp, tg, mk = stmt.selftimed_step_target(
        T_start=t["t_start"], T_cue=t["t_cue"], T_wait=t["t_wait"], T_movement=t["t_movement"],
        T=t["t_total"], lo=sup["target_lo"], hi=sup["target_hi"], hold=sup["hold"], delay=sup["delay"])
    if wt != 1.0:
        # weight in-window timesteps; mk is (batch, T, 1) of ones
        hi = (np.asarray(tg)[..., 0] > 0.5)
        mk = jnp.asarray(np.where(hi, wt, 1.0)[..., None].astype(np.float32))
        print(f"in-window loss weight = {wt}", flush=True)
    rows = _load()
    done = {r["seed"] for r in rows}
    opt_lr = cfg.OPTIM_CONFIG["learning_rate"] if lr is None else float(lr)
    print(f"learning_rate = {opt_lr}", flush=True)
    for sd in range(start, start + n):
        if sd in done and not force:
            print(f"seed {sd}: already recorded, skipping", flush=True); continue
        t0 = time.time()
        rec = {"seed": sd, "iters": iters, "lr": opt_lr, "wt": wt,
               "target_lo": sup["target_lo"]}
        rec.update(init_metrics(cbtl, stmt, cfg, sd, inp))
        rec.update(grad_metrics(cbtl, stmt, cfg, sd, inp, tg, mk))
        params, config = cbtl.init_params(jr.PRNGKey(sd), n_input=inp.shape[-1])
        config = dict(config); config["readout_source"] = "thalamus"
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=opt_lr))
        best, losses, accs = stmt.fit_rnn_supervised(
            cbtl.rnn_func, params, config, inp, mk, opt, iters,
            # log_interval MUST divide num_iters: the training loop is
            # `range(num_iters // log_interval)`, so a log_interval larger than num_iters
            # runs ZERO iterations and silently returns the UNTRAINED params. Keep it at
            # 200 and filter the printing downstream instead.
            batch_targets=tg, log_interval=200,
            seed=sd, loss_type=sup["loss_type"])
        sep, mse = outcome(cbtl, stmt, cfg, best, config, inp[:32], tg[:32])
        # Track whether training raises the SNr pacer toward the ~0.70 it needs to clear
        # D1+GPe inhibition. Before the double-nonlinearity fix the ceiling was 0.648, so
        # this was unreachable however hard training pushed; now it should be able to climb.
        rc = cfg.RUNTIME_CONFIG
        sig = lambda z: 1/(1+np.exp(-np.asarray(z)))
        pac = lambda P, lo, hi: float((lo + sig(P)*(hi-lo)).mean())
        xs_tr = cbtl.rnn_func(best, config, inp[:16], None, jr.split(jr.PRNGKey(0), 16))[4]
        snr_tr = np.asarray(xs_tr[5])
        rec.update(
            P_snr_init=float(np.asarray(params['P_snr']).mean()),
            P_snr_trained=float(np.asarray(best['P_snr']).mean()),
            snr_pacer_init=pac(params['P_snr'], rc['snr_pacer_min'], rc['snr_pacer_max']),
            snr_pacer_trained=pac(best['P_snr'], rc['snr_pacer_min'], rc['snr_pacer_max']),
            SNr_mean_trained=float(snr_tr.mean()),
            SNr_dead_trained=float((snr_tr < 0.01).mean()))
        # Store the full traces (one point per log_interval=200 steps). Init-time variables
        # were ruled out as predictors in batch 1, so WHEN a run breaks through -- and whether
        # collapsed runs show any early signature -- is the remaining place to look.
        trace = [float(x) for x in losses]
        CONST = 0.011875
        broke_at = next((200 * (i + 1) for i, x in enumerate(trace) if x < 0.9 * CONST), None)
        rec.update(separation=sep, mse=mse, min_loss=float(min(losses)) if losses else None,
                   loss_trace=trace, acc_trace=[float(x) for x in accs],
                   broke_at_step=broke_at,
                   breakthrough=bool(sep > BREAKTHROUGH_SEP), secs=round(time.time() - t0, 1))
        # Save the INIT weights so any init-side feature can be computed retroactively
        # without retraining (seeds are deterministic, but regenerating costs a forward pass
        # and this keeps the dataset self-contained).
        wdir = pathlib.Path(__file__).with_name("init_weights"); wdir.mkdir(exist_ok=True)
        p0, _ = cbtl.init_params(jr.PRNGKey(sd), n_input=1)
        np.savez_compressed(wdir / f"seed_{sd:03d}.npz", **{k: np.asarray(v) for k, v in p0.items()})
        rows = [r for r in rows if r["seed"] != sd] + [rec]
        rows.sort(key=lambda r: r["seed"])
        _save(rows)
        print(f"seed {sd:3d}  D1={rec['D1']:.4f}  sep={sep:+.4f}  mse={mse:.6f}  "
              f"broke@{rec['broke_at_step']}  "
              f"{'BREAKTHROUGH' if rec['breakthrough'] else 'collapsed'}  [{rec['secs']:.0f}s]", flush=True)
    n_bt = sum(r["breakthrough"] for r in rows)
    print(f"\nCUMULATIVE: {n_bt}/{len(rows)} breakthroughs "
          f"({100*n_bt/max(len(rows),1):.0f}%) across seeds {min(r['seed'] for r in rows)}-{max(r['seed'] for r in rows)}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1400)
    ap.add_argument("--force", action="store_true")
    # Prediction from the gradient analysis: breakthrough correlates NEGATIVELY with init
    # gradient magnitude (grad_total r=-0.31, g.C_thal r=-0.33, 22/118 features over the
    # uncorrected threshold vs 5.9 expected). If large gradients drive big early steps into
    # the trivial attractor, a smaller step should raise the breakthrough rate.
    ap.add_argument("--lr", type=float, default=None, help="override OPTIM_CONFIG learning_rate")
    # The target is 0.25 for 95% of timesteps and 0.75 for 5%, so 95% of the loss says
    # "hold baseline" and the optimal CONSTANT is 0.2750 -- exactly where every collapse
    # lands. supervised_loss uses the mask as a per-timestep WEIGHT (sum(per_t*mask)/sum(mask)),
    # so upweighting in-window steps rebalances it; 19x makes the two classes equal.
    ap.add_argument("--wt", type=float, default=1.0, help="in-window loss weight (19 = balanced)")
    # y = sigmoid(out_gain*(c_thal@x_t_exc) + out_bias) with c_thal>0 and x_t_exc>=0, so
    # y >= sigmoid(out_bias) = 0.25. With target_lo = 0.25 the ONLY way to hit baseline is a
    # silent thalamus -- the loss actively rewards killing the readout's input. Raising
    # target_lo above the floor makes baseline an interior solution instead.
    ap.add_argument("--target-lo", type=float, default=None, help="override SUPERVISED_THAL target_lo")
    ap.add_argument("--data", type=str, default=None, help="write to an alternate data file")
    a = ap.parse_args()
    if a.data:
        DATA = pathlib.Path(__file__).with_name(a.data)
        globals()["DATA"] = DATA
    main(a.start, a.n, a.iters, a.force, a.lr, a.wt, a.target_lo)
