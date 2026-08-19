"""TEST: init_search -- search the 6 scalar DA/adenosine PKA gains for the init that best
recruits BOTH D1 and D2 in train_hybrid while keeping every level (x_d1,x_d2,x_da,x_ado,
pka_d1,pka_d2) in an operating band -- not dead, not saturated -- across the whole trial.

WHY TWO STAGES
--------------
Full 6-seed x 1000-iter train_hybrid per config is ~1.75 hr/config (=> ~90 hr for a real
MC+SPSA sweep), and reward crawls to ~0 at any affordable length, so it CANNOT rank inits.
But the discriminating, goal-aligned signal -- do D1 AND D2 stay in-band through the trial --
is measurable from a FRESH-INIT forward pass in seconds. So:

  STAGE 1 (search, fast):  for each candidate, fresh init x N seeds, override the 6 scalars,
    run the hybrid trials WITHOUT training, score the operating band of the timecourses.
    Monte-Carlo sample + SPSA local ascent over 100s of configs (~30-45 min).
  STAGE 2 (validate):  fully train (6 seeds x 1000 iters) only the top-k, to measure the
    actual learning gradient of the best-band inits. (`--validate-top K`, ~1.5 hr/config.)

BAND SCORE (stage 1)
--------------------
band(tc) = mean_t of a trapezoid that is 1 in [0.15,0.85] and ramps to 0 at 0 and 1 -- so a
signal is rewarded for sitting mid-range the whole trial and punished for dead(0)/sat(1).
  alive_both = min(band D1, band D2)          # BOTH pathways must stay in-band (recruit both)
  regime_pka = mean(band pkaD1, band pkaD2)    # excitabilities near 0.5
  regime_nm  = mean(nsr DA, nsr Adenosine)     # drivers: not saturated + have dynamic range
  score = 1.0*alive_both + 0.5*regime_pka + 0.25*regime_nm
(x_da rests low & phasic, so DA/Ado are scored on "not saturated + has range", not on band.)

Usage:  python search_init_configs.py                    # stage-1 fast search + report
        python search_init_configs.py --smoke            # quick pipeline check
        python search_init_configs.py --validate-top 5   # stage-2: fully train top-5
        python search_init_configs.py --report-only
"""
import argparse
import json
import sys
import pathlib
import time
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FAMILY_DIR = HERE.parents[1]
_root = next(p for p in HERE.parents if (p / "config_script.py").exists())
for _p in (str(FAMILY_DIR), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cbt_rnn as cbtl
import train_hybrid
import config_script as C
import self_timed_movement_task as stmt

cfg = C.for_family("cbt_loop_noSCnoSTN")
AREAS = list(cbtl.STATE_AREA_ORDER)
PARAM_NAMES = ["m_d1", "m_d2", "m_a1", "m_a2", "g_da_release", "g_ado_release"]
# The 6 gains are searched in EFFECTIVE space (0,1): each is exc=sigmoid-wrapped in the
# forward, so the stored param is RAW and effective = sigmoid(raw). We sample effective and
# override params[gain] = logit(effective) (see _override). Sampling raw directly would only
# span sigmoid([0.05,1.5])=[0.51,0.82] and could never reach the low m_a2 that lowers pka_d2.
RANGES = {"m_d1": (0.02, 0.98), "m_d2": (0.02, 0.98), "m_a1": (0.02, 0.98),
          "m_a2": (0.02, 0.98), "g_da_release": (0.02, 0.98), "g_ado_release": (0.02, 0.98)}
LOG_SAMPLE = {"m_d1", "m_d2", "m_a1", "m_a2"}     # effective gains sampled log-uniform
W_ALIVE, W_PKA, W_NM = 1.0, 0.5, 0.25
TC_AREAS = ("D1", "D2", "DA", "Adenosine", "pkaD1", "pkaD2")
RESULTS = HERE / "results.json"
_HYBRID = None


def hybrid_batch():
    global _HYBRID
    if _HYBRID is None:
        _HYBRID = train_hybrid._build_hybrid_batch()
    return _HYBRID


def band(tc, lo=0.15, hi=0.85):
    """mean over the trial of a trapezoid: 1 in [lo,hi], ramps to 0 at 0 and 1."""
    tc = np.asarray(tc)
    left = np.clip(tc / lo, 0.0, 1.0)
    right = np.clip((1.0 - tc) / (1.0 - hi), 0.0, 1.0)
    return float(np.minimum(left, right).mean())


def not_sat_range(tc, sat=0.85, span=0.1):
    """for phasic drivers (DA/Ado): reward NOT saturated (near 1) AND having dynamic range."""
    tc = np.asarray(tc)
    not_sat = float(np.clip((1.0 - tc) / (1.0 - sat), 0.0, 1.0).mean())
    rng = float(np.clip((tc.max() - tc.min()) / span, 0.0, 1.0))
    return 0.5 * not_sat + 0.5 * rng


def eval_activity(params, config, inputs):
    """Run model on the hybrid trials; per-area overall-mean + batch-mean time-course."""
    nd1 = params["J_d1"].shape[0]; nd2 = params["J_d2"].shape[0]
    B, T, _ = inputs.shape
    stim = jnp.zeros((B, T, nd1 + nd2))
    _, xs = cbtl.batched_rnn(params, config, inputs, stim, jr.split(jr.PRNGKey(0), B))
    means, tc = {}, {}
    for a in TC_AREAS:
        x = np.asarray(xs[AREAS.index(a)])
        xm = x.mean(-1) if x.ndim > 2 else x          # (B,T)
        means[a] = float(xm.mean())
        tc[a] = xm.mean(0)                             # (T,)
    return means, tc


def _override(seed, vals, n_input):
    p, c = cbtl.init_params(jr.PRNGKey(seed), n_input=n_input)
    p = dict(p)
    for name, v in zip(PARAM_NAMES, vals):
        e = float(np.clip(v, 1e-3, 1.0 - 1e-3))   # v is the EFFECTIVE gain in (0,1)
        p[name] = jnp.array(np.log(e / (1.0 - e)))  # store raw = logit(effective); fwd sigmoids it
    return p, c


def _score_from_tc(tc_avg, means):
    b = {a: band(tc_avg[a]) for a in ("D1", "D2", "pkaD1", "pkaD2")}
    nm = {a: not_sat_range(tc_avg[a]) for a in ("DA", "Adenosine")}
    alive_both = min(b["D1"], b["D2"])
    regime_pka = 0.5 * (b["pkaD1"] + b["pkaD2"])
    regime_nm = 0.5 * (nm["DA"] + nm["Adenosine"])
    score = W_ALIVE * alive_both + W_PKA * regime_pka + W_NM * regime_nm
    metrics = dict(score=score, alive_both=alive_both, regime_pka=regime_pka, regime_nm=regime_nm,
                   reward=None,
                   **{f"band_{a}": b[a] for a in b}, **{f"nm_{a}": nm[a] for a in nm},
                   **{f"mean_{a}": float(means[a]) for a in means})
    return score, metrics


def eval_config_fast(vals, n_seeds, n_batch):
    """STAGE 1: fresh-init forward on the hybrid trials (no training), band-scored."""
    inputs = hybrid_batch()[0][:n_batch]
    tcs, ms = [], []
    for seed in range(n_seeds):
        p, c = _override(seed, vals, inputs.shape[-1])
        m, tc = eval_activity(p, c, inputs)
        tcs.append(tc); ms.append(m)
    tc_avg = {a: np.mean([tc[a] for tc in tcs], 0) for a in TC_AREAS}
    means = {a: float(np.mean([m[a] for m in ms])) for a in TC_AREAS}
    score, metrics = _score_from_tc(tc_avg, means)
    return score, metrics, {a: tc_avg[a].tolist() for a in tc_avg}, None


def train_eval(vals, n_seeds, n_iters, log_interval, lr):
    """STAGE 2: fully train the config, return (curve, final_reward, post metrics, post tc)."""
    inputs, targets, masks = hybrid_batch()
    rl = dict(cfg.RL_CONFIG)
    curves, tcs, ms = [], [], []
    for seed in range(n_seeds):
        p, c = _override(seed, vals, inputs.shape[-1])
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))
        best, _losses, rewards = stmt.fit_rnn_reinforce(
            cbtl.rnn_func, p, c, inputs, masks, opt, n_iters,
            log_interval=log_interval, seed=seed, batch_targets=targets, **rl)
        curves.append([float(r) for r in rewards])
        m, tc = eval_activity(best, c, inputs)
        tcs.append(tc); ms.append(m)
    L = min(len(c) for c in curves)
    curve = np.mean([c[:L] for c in curves], 0).tolist()
    final_reward = float(np.mean([c[-1] for c in curves]))
    tc_avg = {a: np.mean([tc[a] for tc in tcs], 0) for a in TC_AREAS}
    means = {a: float(np.mean([m[a] for m in ms])) for a in TC_AREAS}
    _, post = _score_from_tc(tc_avg, means)
    return curve, final_reward, post, {a: tc_avg[a].tolist() for a in tc_avg}


def sample_config(rng):
    out = []
    for n in PARAM_NAMES:
        lo, hi = RANGES[n]
        out.append(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if n in LOG_SAMPLE
                   else float(rng.uniform(lo, hi)))
    return out


def _norm(vals):
    return np.array([(v - RANGES[n][0]) / (RANGES[n][1] - RANGES[n][0]) for v, n in zip(vals, PARAM_NAMES)])


def _denorm(u):
    return [float(np.clip(u[i], 0, 1) * (RANGES[n][1] - RANGES[n][0]) + RANGES[n][0])
            for i, n in enumerate(PARAM_NAMES)]


def save(results):
    RESULTS.write_text(json.dumps(results, indent=1))


def search(args):
    rng = np.random.default_rng(args.seed)
    results = json.loads(RESULTS.read_text()) if (args.resume and RESULTS.exists()) else []
    ev = lambda vals: eval_config_fast(vals, args.seeds, args.batch)
    t0 = time.time()

    def record(vals, tag):
        s, m, tc, _ = ev(vals)
        results.append({"vals": [float(v) for v in vals], "metrics": m, "tc": tc,
                        "curve": None, "tag": tag})
        save(results)
        print(f"[{tag}] score={m['score']:.3f} alive_both={m['alive_both']:.2f} "
              f"(D1band={m['band_D1']:.2f} D2band={m['band_D2']:.2f}) "
              f"pka={m['regime_pka']:.2f} nm={m['regime_nm']:.2f} | "
              f"D1={m['mean_D1']:.2f} D2={m['mean_D2']:.2f} da={m['mean_DA']:.2f} "
              f"ado={m['mean_Adenosine']:.2f}  vals={[round(v,3) for v in vals]}  [{time.time()-t0:.0f}s]")
        return s

    for i in range(args.n_random):
        record(sample_config(rng), f"mc{i}")

    if args.n_refine and results:
        theta = _norm(max(results, key=lambda r: r["metrics"]["score"])["vals"])
        for k in range(args.n_refine):
            ck = 0.15 / (k + 1) ** 0.3
            ak = 0.30 / (k + 1) ** 0.6
            delta = rng.choice([-1.0, 1.0], size=len(theta))
            fp = record(_denorm(theta + ck * delta), f"spsa{k}+")
            fm = record(_denorm(theta - ck * delta), f"spsa{k}-")
            theta = np.clip(theta + ak * (fp - fm) / (2 * ck * delta), 0, 1)
            record(_denorm(theta), f"spsa{k}=")

    build_report(results)


def validate(args):
    results = json.loads(RESULTS.read_text())
    top = sorted(results, key=lambda r: r["metrics"]["score"], reverse=True)[:args.validate_top]
    t0 = time.time()
    for r in top:
        curve, fr, post, vtc = train_eval(r["vals"], args.val_seeds, args.val_iters,
                                           args.log_interval, args.lr)
        r["curve"] = curve
        r["metrics"]["final_reward"] = fr
        r["metrics"]["post_train"] = post
        r["val_tc"] = vtc
        save(results)
        print(f"[validate] band_score={r['metrics']['score']:.3f} -> final_reward={fr:.3f} "
              f"(post D1={post['mean_D1']:.2f} D2={post['mean_D2']:.2f})  "
              f"vals={[round(v,3) for v in r['vals']]}  [{time.time()-t0:.0f}s]")
    build_report(results)


def build_report(results):
    results = sorted(results, key=lambda r: r["metrics"]["score"], reverse=True)
    top = results[:min(6, len(results))]
    validated = [r for r in results if r.get("curve")]
    V = np.array([r["vals"] for r in results])
    S = np.array([r["metrics"]["score"] for r in results])

    # 1) score vs each param
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for i, n in enumerate(PARAM_NAMES):
        a = ax[i // 3][i % 3]
        a.scatter(V[:, i], S, c=S, cmap="viridis", s=30)
        a.set_xlabel(n); a.set_ylabel("band score"); a.grid(alpha=0.3)
        if n in LOG_SAMPLE:
            a.set_xscale("log")
    fig.suptitle("init value vs operating-band score (each param)", y=1.0); fig.tight_layout()
    fig.savefig(HERE / "score_vs_param.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 2) g_da x g_ado plane
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(V[:, 4], V[:, 5], c=S, cmap="viridis", s=60)
    ax.set_xlabel("g_da_release"); ax.set_ylabel("g_ado_release")
    ax.set_title("release-gain plane, colored by band score"); plt.colorbar(sc, ax=ax)
    fig.tight_layout(); fig.savefig(HERE / "release_plane.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 3) timecourses of top configs
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for i, a_name in enumerate(TC_AREAS):
        a = ax[i // 3][i % 3]
        for r in top:
            a.plot(r["tc"][a_name], lw=1, label=f"s={r['metrics']['score']:.2f}")
        a.axhline(0.15, color="r", ls=":", lw=0.7); a.axhline(0.85, color="r", ls=":", lw=0.7)
        a.set_title(a_name); a.set_xlabel("t"); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)
    ax[0][0].legend(fontsize=7)
    fig.suptitle("fresh-init timecourses of top-band configs (red = out-of-band)", y=1.0)
    fig.tight_layout(); fig.savefig(HERE / "timecourses.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 4) validated learning curves (stage 2)
    if validated:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in sorted(validated, key=lambda r: r["metrics"]["score"], reverse=True):
            ax.plot(r["curve"], label=f"band={r['metrics']['score']:.2f} "
                    f"reward={r['metrics'].get('final_reward', 0):.2f}")
        ax.set_xlabel("log point"); ax.set_ylabel("reward"); ax.legend(fontsize=8)
        ax.set_title("STAGE 2: reward learning curves of top-band configs"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "validation_curves.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    corrs = {n: float(np.corrcoef(np.log(V[:, i]) if n in LOG_SAMPLE else V[:, i], S)[0, 1])
             for i, n in enumerate(PARAM_NAMES)}
    lines = ["# init-config search report\n",
             f"{len(results)} configs scored on the fresh-init operating band "
             f"({len(validated)} validated by full training).\n",
             "## Best config (band score)\n", "```",
             *[f"{n} = {top[0]['vals'][i]:.3f}" for i, n in enumerate(PARAM_NAMES)],
             f"score={top[0]['metrics']['score']:.3f}  alive_both={top[0]['metrics']['alive_both']:.3f}  "
             f"D1={top[0]['metrics']['mean_D1']:.2f} D2={top[0]['metrics']['mean_D2']:.2f}",
             "```\n", "## Top configs\n",
             "| rank | band | alive_both | reward | D1 | D2 | x_da | x_ado | pkaD1 | pkaD2 | "
             + " | ".join(PARAM_NAMES) + " |",
             "|" + "---|" * (10 + len(PARAM_NAMES))]
    for k, r in enumerate(top):
        m = r["metrics"]
        rew = f"{m['final_reward']:.2f}" if m.get("final_reward") is not None else "-"
        lines.append(f"| {k+1} | {m['score']:.2f} | {m['alive_both']:.2f} | {rew} | {m['mean_D1']:.2f} | "
                     f"{m['mean_D2']:.2f} | {m['mean_DA']:.2f} | {m['mean_Adenosine']:.2f} | "
                     f"{m['mean_pkaD1']:.2f} | {m['mean_pkaD2']:.2f} | "
                     + " | ".join(f"{v:.3f}" for v in r["vals"]) + " |")
    lines += ["\n## What drives the band score (corr with score)\n", "```",
              *[f"{n:14s} corr={corrs[n]:+.2f}" for n in PARAM_NAMES], "```\n",
              "## Plots\n", "- `score_vs_param.png` — band score vs each of the 6 init params",
              "- `release_plane.png` — g_da x g_ado colored by band score",
              "- `timecourses.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs"]
    if validated:
        lines.append("- `validation_curves.png` — STAGE 2 reward learning curves (full training)")
    (HERE / "init_search_report.md").write_text("\n".join(lines) + "\n")
    print(f"report -> {HERE/'init_search_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=120)
    ap.add_argument("--n-refine", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=6)          # stage-1 fresh-init seeds
    ap.add_argument("--batch", type=int, default=32)         # stage-1 trials per forward
    ap.add_argument("--validate-top", type=int, default=0)   # stage-2: fully train top-K
    ap.add_argument("--val-seeds", type=int, default=6)
    ap.add_argument("--val-iters", type=int, default=1000)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = cfg.OPTIM_CONFIG["learning_rate"]
    if args.smoke:
        args.n_random, args.n_refine, args.seeds, args.batch = 3, 1, 2, 16
        args.val_seeds, args.val_iters, args.log_interval = 1, 40, 20
    if args.report_only:
        build_report(json.loads(RESULTS.read_text()))
    elif args.validate_top:
        validate(args)
    else:
        search(args)
