"""TEST: init_search -- search the DA/adenosine PKA init for the config that HOLDS
pka_d1 AND pka_d2 at the target excitability (PKA_TARGET=0.25) across the whole trial, while
keeping D1/D2 activity in an operating band -- not dead, not saturated.

WHY TWO STAGES
--------------
Full 6-seed x 1000-iter train_hybrid per config is ~1.75 hr/config (=> ~90 hr for a real
MC+SPSA sweep), and reward crawls to ~0 at any affordable length, so it CANNOT rank inits.
But the discriminating, goal-aligned signal -- do BOTH PKAs sit at the target through the
trial -- is measurable from a FRESH-INIT forward pass in seconds. So:

SEARCH SPACE (6-D): m_d1, m_d2, m_a1, m_a2, g_da_release, and `ado0` -- which REPLACES
g_ado_release. ado0 is set as x_ado0 AND as config["pin_ado"], so x_ado is CLAMPED at that level
all trial instead of integrating mean_stri; g_ado_release then feeds nothing (verified: zero
gradient, bit-identical timecourses) and is not searched. Note the adenosine half of regime_nm
loses its dynamic-range term by construction (a pinned trace has zero range), so absolute
scores are NOT comparable to the pre-clamp runs -- only the ranking within this run is.
With x_da ~ 0 the exact D2 solution locus is the hyperbola m_a2 * ado0 = P_req.

  STAGE 1 (search, fast):  for each candidate, fresh init x N seeds, override the 6 values,
    run the hybrid trials WITHOUT training, score how well the timecourses hit the target.
    Monte-Carlo sample + SPSA local ascent over 100s of configs (~30-45 min).
  STAGE 2 (validate):  fully train (6 seeds x 1000 iters) only the top-k, to measure the
    actual learning gradient of the best inits. (`--validate-top K`, ~1.5 hr/config.)

SCORE (stage 1) -- objective: pka_d1 = pka_d2 = PKA_TARGET (0.25) for the whole trial
--------------------------------------------------------------------------------------
at_target(tc) = mean_t of a tent: 1 within +-PKA_TOL of PKA_TARGET, ramping to 0 at
+-PKA_WIDTH -- so a PKA trace is rewarded only for STAYING AT 0.25, and any drift up
(toward saturation) or down (toward dead) is punished symmetrically.
  track_pka  = min(at_target pkaD1, at_target pkaD2)  # BOTH PKAs must hold the target
  alive_both = min(band D1, band D2)          # guard: neither pathway dead/saturated
  regime_nm  = mean(nsr DA, nsr Adenosine)    # drivers: not saturated + have dynamic range
  score = 1.0*track_pka + 0.5*alive_both + 0.25*regime_nm
band(tc) = mean_t of a trapezoid that is 1 in [0.15,0.85] and ramps to 0 at 0 and 1.
(x_da rests low & phasic, so DA/Ado are scored on "not saturated + has range", not on band.)
Entries in an existing results file that were scored under an older objective are re-scored
from their stored timecourses on load, so old sweeps stay comparable.

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
PARAM_NAMES = ["m_d1", "m_d2", "m_a1", "m_a2", "g_da_release", "ado0"]
# ado0 REPLACES g_ado_release in the sweep. It is NOT a sigmoid-wrapped gain: it is the
# adenosine STATE level, set as x_ado0 AND as config["pin_ado"], so x_ado is CLAMPED there for
# the whole trial (see _override). Each config therefore asks "what tonic adenosine level, with
# what gains?", and prod_d2's adenosine drive is the PRODUCT m_a2 * ado0.
# g_ado_release is DROPPED because the clamp makes it provably inert: it feeds only the x_ado
# integrator, whose result pin_ado overwrites on the same step. Measured with ado pinned,
# d(loss)/d(g_ado_release) = 0.0 exactly and g_ado_release 0.20 vs 0.80 gives bit-identical
# timecourses (max |diff| 0.0 across Adenosine/pkaD2/D2/Medulla), in BOTH nt_mode branches.
# Searching it would have burned one MC dimension and one SPSA probe direction on noise.
# The 6 gains are searched in EFFECTIVE space (0,1): each is exc=sigmoid-wrapped in the
# forward, so the stored param is RAW and effective = sigmoid(raw). We sample effective and
# override params[gain] = logit(effective) (see _override). Sampling raw directly would only
# span sigmoid([0.05,1.5])=[0.51,0.82] and could never reach the low m_a2 that lowers pka_d2.
RANGES = {"m_d1": (0.5, 0.98), "m_d2": (0.05, 0.7), "m_a1": (0.01, 0.1),
          "m_a2": (0.1, 0.9), "g_da_release": (0.2, 0.8),
          # ado0 = the clamped tonic adenosine level, a STATE in [0,1]. Spans well below and
          # above the 0.1 currently in config_script. prod_d2 = m_a2*ado0 - G*m_d2*x_da, so
          # ado0 acts multiplicatively with m_a2 -> log-sampled like the gains.
          "ado0": (0.02, 0.9)}
LOG_SAMPLE = {"m_d1", "m_d2", "m_a1", "m_a2", "ado0"}   # sampled log-uniform
# What often matters is the BALANCE within a gain pair, not either absolute value. Two kinds:
# ACROSS pathways (D1-side vs D2-side partner of the same messenger) and WITHIN a pathway
# (its DA drive vs its opposing adenosine drive). Both are plotted vs score in the report.
RATIO_PAIRS = (("m_d1", "m_d2"), ("m_a1", "m_a2"),
               ("m_d1", "m_a1"), ("m_d2", "m_a2"),
               # ado0 vs each adenosine sensitivity: with x_ado pinned, the D2 drive is the
               # PRODUCT m_a2*ado0 and the D1 brake is m_a1*ado0, so the level and the gain
               # trade off directly against each other.
               ("m_a2", "ado0"), ("m_a1", "ado0"))
W_PKA, W_ALIVE, W_NM = 1.0, 0.5, 0.25   # holding PKA at target is now the primary objective
TC_AREAS = ("D1", "D2", "DA", "Adenosine", "pkaD1", "pkaD2")
PKA_TARGET = 0.25      # THE GOAL: pka_d1 and pka_d2 should sit here all trial
PKA_TOL = 0.05         # full credit within +-TOL of the target
PKA_WIDTH = 0.15       # zero credit beyond +-WIDTH
PKA0 = PKA_TARGET      # force the PKA init state (pka_d10/pka_d20) to start on target
PIN_SNC = 0.1          # clamp every SNc unit to this rate for the whole trial (None = free)
PIN_CTX = 0.1          # clamp every cortical unit (cU, cL, c_inh) likewise
OBJECTIVE = f"pka_target_{PKA_TARGET}"   # stamped on metrics; stale entries get re-scored
# Raw production needed to hold pka* at PKA_TARGET: pka* = 9P/(1+9P) for tau_f/tau_r = 9, so
# P_req = p / ((tau_f/tau_r)(1-p)). With x_ado pinned and x_da ~ 0, prod_d2 = m_a2*ado0, so the
# exact-solution locus in the (m_a2, ado0) plane is the hyperbola m_a2*ado0 = P_req.
P_REQ = PKA_TARGET / ((cfg.RUNTIME_CONFIG["tau_pka_fall"] /
                       cfg.RUNTIME_CONFIG["tau_pka_rise"]) * (1.0 - PKA_TARGET))
TAG = "ado0sweep_pinSNC_halfnormal"  # output suffix: 7-D sweep (6 gains + ado0) under nln=max(0,tanh(x)),
                            # with x_ado CLAMPED at the sampled ado0 (config["pin_ado"]).
                            # Earlier tags: pka0_0.25          -- nln=sigmoid, dynamic adenosine
                            #               pka0_0.25_recttanh -- nln=rect-tanh, dynamic adenosine
                            # Stored timecourses differ across all three, so each keeps its own
                            # results file rather than being merged with --resume.
RESULTS = HERE / f"results_{TAG}.json"
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


def at_target(tc, target=PKA_TARGET, tol=PKA_TOL, width=PKA_WIDTH):
    """mean over the trial of a tent: 1 within +-tol of target, ramping to 0 at +-width.
    Symmetric, so drifting up (toward saturation) and down (toward dead) cost the same."""
    d = np.abs(np.asarray(tc) - target)
    return float(np.clip((width - d) / (width - tol), 0.0, 1.0).mean())


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
    p = dict(p); c = dict(c)
    for name, v in zip(PARAM_NAMES, vals):
        if name == "ado0":
            # NOT a gain: the adenosine STATE level, used directly (no sigmoid wrapper).
            # Set as the initial state AND as pin_ado, so x_ado is held at exactly this value
            # for the whole trial instead of integrating mean_stri. The forward also clips
            # every initial state to [0,1], so keep ado0 in range here.
            a = float(np.clip(v, 0.0, 1.0))
            p["x_ado0"] = jnp.full_like(jnp.asarray(p["x_ado0"]), a)
            c["pin_ado"] = a
        else:
            e = float(np.clip(v, 1e-3, 1.0 - 1e-3))   # v is the EFFECTIVE gain in (0,1)
            p[name] = jnp.array(np.log(e / (1.0 - e)))  # raw = logit(effective); fwd sigmoids it
    # SNc and all three cortical pools CLAMPED (see PIN_SNC/PIN_CTX): under max(0,tanh(x))
    # both rectify to exactly 0 at rest (SNc's pacer is capped at 0.2 against ~0.5 of D2+GPe
    # inhibition; cortex has no pacer at all), which zeroes x_da and makes pka_d1 unreachable
    # at ANY gain. Clamping them supplies the DA drive the search needs to discriminate.
    if PIN_SNC is not None:
        c["pin_snc"] = PIN_SNC
    if PIN_CTX is not None:
        c["pin_ctx"] = PIN_CTX
    p["pka_d10"] = jnp.full_like(jnp.asarray(p["pka_d10"]), PKA0)  # force PKA init start
    p["pka_d20"] = jnp.full_like(jnp.asarray(p["pka_d20"]), PKA0)  # (fwd clamps to [floor,cap])
    return p, c


def _score_from_tc(tc_avg, means):
    b = {a: band(tc_avg[a]) for a in ("D1", "D2")}
    trk = {a: at_target(tc_avg[a]) for a in ("pkaD1", "pkaD2")}
    dev = {a: float(np.abs(np.asarray(tc_avg[a]) - PKA_TARGET).mean()) for a in ("pkaD1", "pkaD2")}
    nm = {a: not_sat_range(tc_avg[a]) for a in ("DA", "Adenosine")}
    track_pka = min(trk["pkaD1"], trk["pkaD2"])   # BOTH PKAs must hold PKA_TARGET
    alive_both = min(b["D1"], b["D2"])
    regime_nm = 0.5 * (nm["DA"] + nm["Adenosine"])
    score = W_PKA * track_pka + W_ALIVE * alive_both + W_NM * regime_nm
    metrics = dict(score=score, track_pka=track_pka, alive_both=alive_both, regime_nm=regime_nm,
                   objective=OBJECTIVE, reward=None,
                   **{f"band_{a}": b[a] for a in b}, **{f"track_{a}": trk[a] for a in trk},
                   **{f"dev_{a}": dev[a] for a in dev}, **{f"nm_{a}": nm[a] for a in nm},
                   **{f"mean_{a}": float(means[a]) for a in means})
    return score, metrics


def _rescored(r):
    """Re-score an entry saved under an older objective from its stored timecourses."""
    m = r.get("metrics", {})
    if m.get("objective") == OBJECTIVE or not r.get("tc"):
        return r
    tc = {a: np.asarray(v) for a, v in r["tc"].items()}
    means = {a: m.get(f"mean_{a}", float(tc[a].mean())) for a in tc}
    _, new = _score_from_tc(tc, means)
    for k in ("final_reward", "post_train"):     # keep any stage-2 results already measured
        if k in m:
            new[k] = m[k]
    r = dict(r); r["metrics"] = new
    return r


def load_results():
    return [_rescored(r) for r in json.loads(RESULTS.read_text())] if RESULTS.exists() else []


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
    """gain values -> SPSA's [0,1]^6 search coords. LOG_SAMPLE params use LOG coordinates,
    matching how they are sampled, so an SPSA step is MULTIPLICATIVE there. Linearly, one
    step (ck=0.15 => ~0.14 in gain units) is a ~20% probe for m_d1~0.66 but a 6x jump for
    m_a1~0.02 that just clips to the floor -- which is exactly where the best configs sit."""
    out = []
    for v, n in zip(vals, PARAM_NAMES):
        lo, hi = RANGES[n]
        out.append((np.log(max(float(v), 1e-12)) - np.log(lo)) / (np.log(hi) - np.log(lo))
                   if n in LOG_SAMPLE else (float(v) - lo) / (hi - lo))
    return np.clip(np.array(out), 0.0, 1.0)


def _denorm(u):
    out = []
    for i, n in enumerate(PARAM_NAMES):
        lo, hi = RANGES[n]
        t = float(np.clip(u[i], 0.0, 1.0))
        out.append(float(np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo)))) if n in LOG_SAMPLE
                   else float(lo + t * (hi - lo)))
    return out


def save(results):
    RESULTS.write_text(json.dumps(results, indent=1))


def search(args):
    rng = np.random.default_rng(args.seed)
    results = load_results() if args.resume else []
    ev = lambda vals: eval_config_fast(vals, args.seeds, args.batch)
    t0 = time.time()

    def record(vals, tag):
        s, m, tc, _ = ev(vals)
        results.append({"vals": [float(v) for v in vals], "metrics": m, "tc": tc,
                        "curve": None, "tag": tag})
        save(results)
        print(f"[{tag}] score={m['score']:.3f} track_pka={m['track_pka']:.2f} "
              f"(pkaD1={m['mean_pkaD1']:.2f}+-{m['dev_pkaD1']:.2f} "
              f"pkaD2={m['mean_pkaD2']:.2f}+-{m['dev_pkaD2']:.2f}) "
              f"alive_both={m['alive_both']:.2f} nm={m['regime_nm']:.2f} | "
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
    results = load_results()
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
        print(f"[validate] score={r['metrics']['score']:.3f} -> final_reward={fr:.3f} "
              f"(post pkaD1={post['mean_pkaD1']:.2f} pkaD2={post['mean_pkaD2']:.2f} "
              f"D1={post['mean_D1']:.2f} D2={post['mean_D2']:.2f})  "
              f"vals={[round(v,3) for v in r['vals']]}  [{time.time()-t0:.0f}s]")
    build_report(results)


def build_report(results):
    results = sorted(results, key=lambda r: r["metrics"]["score"], reverse=True)
    top = results[:min(6, len(results))]
    validated = [r for r in results if r.get("curve")]
    V = np.array([r["vals"] for r in results])
    S = np.array([r["metrics"]["score"] for r in results])

    # 1) score vs each param (grid sized to len(PARAM_NAMES), not hardcoded)
    _nc = 3
    _nr = int(np.ceil(len(PARAM_NAMES) / _nc))
    fig, ax = plt.subplots(_nr, _nc, figsize=(5 * _nc, 4 * _nr), squeeze=False)
    for k in range(len(PARAM_NAMES), _nr * _nc):
        ax[k // _nc][k % _nc].axis("off")
    for i, n in enumerate(PARAM_NAMES):
        a = ax[i // _nc][i % _nc]
        a.scatter(V[:, i], S, c=S, cmap="viridis", s=30)
        a.set_xlabel(n); a.set_ylabel("pka-target score"); a.grid(alpha=0.3)
        if n in LOG_SAMPLE:
            a.set_xscale("log")
    fig.suptitle(f"init value vs pka-target({PKA_TARGET}) score (each param)", y=1.0)
    fig.tight_layout()
    fig.savefig(HERE / f"score_vs_param_{TAG}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 2) m_a2 x ado0 plane -- the two factors whose PRODUCT is the whole D2 PKA drive once
    # x_ado is pinned (prod_d2 = m_a2*ado0 - G*m_d2*x_da). Replaces the old g_da x g_ado plane,
    # which is uninformative here because g_ado_release is inert under the clamp.
    _ia2, _iad = PARAM_NAMES.index("m_a2"), PARAM_NAMES.index("ado0")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(V[:, _ia2], V[:, _iad], c=S, cmap="viridis", s=60)
    _p = np.logspace(np.log10(max(V[:, _ia2].min(), 1e-4)), np.log10(V[:, _ia2].max()), 50)
    ax.plot(_p, P_REQ / _p, "--", color="crimson", lw=1.6,
            label=f"m_a2*ado0 = P_req = {P_REQ:.4f}\n(exact pka_d2 = {PKA_TARGET} when x_da = 0)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(V[:, _ia2].min(), V[:, _ia2].max()); ax.set_ylim(V[:, _iad].min(), V[:, _iad].max())
    ax.set_xlabel("m_a2"); ax.set_ylabel("ado0 (clamped x_ado)"); ax.legend(fontsize=7)
    ax.set_title("D2 drive plane m_a2 x ado0, colored by pka-target score")
    plt.colorbar(sc, ax=ax)
    fig.tight_layout(); fig.savefig(HERE / f"release_plane_{TAG}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 3) score vs the ratio of each gain pair (balance, not absolute level)
    ratio_corrs = {}
    ncol = 3
    nrow = int(np.ceil(len(RATIO_PAIRS) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), squeeze=False)
    for k in range(len(RATIO_PAIRS), nrow * ncol):
        ax[k // ncol][k % ncol].axis("off")
    for k, (na, nb) in enumerate(RATIO_PAIRS):
        a = ax[k // ncol][k % ncol]
        ratio = V[:, PARAM_NAMES.index(na)] / V[:, PARAM_NAMES.index(nb)]
        rc = (float(np.corrcoef(np.log(ratio), S)[0, 1])
              if len(S) > 2 and ratio.max() > ratio.min() else float("nan"))
        ratio_corrs[f"{na}/{nb}"] = rc
        a.scatter(ratio, S, c=S, cmap="viridis", s=30)
        if ratio.max() / ratio.min() > 1.01:      # binned median: the trend through the cloud
            edges = np.logspace(np.log10(ratio.min()), np.log10(ratio.max()), 9)
            b_of = np.clip(np.digitize(ratio, edges[1:-1]), 0, 7)
            med = [np.median(S[b_of == b]) if (b_of == b).sum() >= 2 else np.nan for b in range(8)]
            a.plot(np.sqrt(edges[:-1] * edges[1:]), med, "o-", color="crimson", lw=1.5, ms=4,
                   label="binned median")
            a.legend(fontsize=7)
        a.axvline(1.0, color="k", ls="--", lw=0.8)   # balanced pair
        a.set_xscale("log"); a.set_xlabel(f"{na} / {nb}"); a.set_ylabel("pka-target score")
        a.set_title(f"{na} / {nb}   corr(log ratio, score)={rc:+.2f}", fontsize=10)
        a.grid(alpha=0.3)
    fig.suptitle(f"pka-target({PKA_TARGET}) score vs the balance of each gain pair "
                 f"(dashed = balanced; top row across pathways, bottom row DA vs adenosine "
                 f"within a pathway)", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / f"score_vs_ratio_{TAG}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 4) timecourses of top configs
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for i, a_name in enumerate(TC_AREAS):
        a = ax[i // 3][i % 3]
        for r in top:
            a.plot(r["tc"][a_name], lw=1, label=f"s={r['metrics']['score']:.2f}")
        if a_name in ("pkaD1", "pkaD2"):           # the objective: hold this line
            a.axhline(PKA_TARGET, color="k", ls="--", lw=1.0)
            a.axhline(PKA_TARGET - PKA_WIDTH, color="r", ls=":", lw=0.7)
            a.axhline(PKA_TARGET + PKA_WIDTH, color="r", ls=":", lw=0.7)
        else:
            a.axhline(0.15, color="r", ls=":", lw=0.7); a.axhline(0.85, color="r", ls=":", lw=0.7)
        a.set_title(a_name); a.set_xlabel("t"); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)
    ax[0][0].legend(fontsize=7)
    fig.suptitle(f"fresh-init timecourses of top configs "
                 f"(black = pka target {PKA_TARGET}, red = zero-credit edge)", y=1.0)
    fig.tight_layout(); fig.savefig(HERE / f"timecourses_{TAG}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 5) validated learning curves (stage 2)
    if validated:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in sorted(validated, key=lambda r: r["metrics"]["score"], reverse=True):
            ax.plot(r["curve"], label=f"score={r['metrics']['score']:.2f} "
                    f"reward={r['metrics'].get('final_reward', 0):.2f}")
        ax.set_xlabel("log point"); ax.set_ylabel("reward"); ax.legend(fontsize=8)
        ax.set_title("STAGE 2: reward learning curves of top pka-target configs"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / f"validation_curves_{TAG}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    corrs = {n: float(np.corrcoef(np.log(V[:, i]) if n in LOG_SAMPLE else V[:, i], S)[0, 1])
             for i, n in enumerate(PARAM_NAMES)}
    lines = [f"# init-config search report (objective: pka_d1 = pka_d2 = {PKA_TARGET})\n",
             f"7-D sweep: the 6 gains PLUS `ado0`, with x_ado CLAMPED at ado0 (`pin_ado`) "
             f"for the whole trial. P_req (raw production for pka*={PKA_TARGET}) = {P_REQ:.4f}; "
             f"with x_da~0 the D2 solution locus is m_a2*ado0 = P_req.\n",
             f"{len(results)} configs scored on how well the fresh-init PKA traces HOLD "
             f"{PKA_TARGET} for the whole trial ({len(validated)} validated by full training).\n",
             f"`score = 1.0*track_pka + 0.5*alive_both + 0.25*regime_nm`, where `track_pka` is the "
             f"min over pkaD1/pkaD2 of the time-mean tent (1 within +-{PKA_TOL} of {PKA_TARGET}, "
             f"0 beyond +-{PKA_WIDTH}). PKA init state forced to pka_d10=pka_d20={PKA0}.\n",
             "## Best config\n", "```",
             *[f"{n} = {top[0]['vals'][i]:.3f}" for i, n in enumerate(PARAM_NAMES)],
             f"score={top[0]['metrics']['score']:.3f}  track_pka={top[0]['metrics']['track_pka']:.3f}  "
             f"pkaD1={top[0]['metrics']['mean_pkaD1']:.2f}+-{top[0]['metrics']['dev_pkaD1']:.2f} "
             f"pkaD2={top[0]['metrics']['mean_pkaD2']:.2f}+-{top[0]['metrics']['dev_pkaD2']:.2f}  "
             f"alive_both={top[0]['metrics']['alive_both']:.3f}",
             "```\n", "## Top configs\n",
             "| rank | score | track_pka | pkaD1 | dev1 | pkaD2 | dev2 | alive_both | reward | "
             "D1 | D2 | x_da | x_ado | " + " | ".join(PARAM_NAMES) + " |",
             "|" + "---|" * (14 + len(PARAM_NAMES))]
    for k, r in enumerate(top):
        m = r["metrics"]
        rew = f"{m['final_reward']:.2f}" if m.get("final_reward") is not None else "-"
        lines.append(f"| {k+1} | {m['score']:.2f} | {m['track_pka']:.2f} | "
                     f"{m['mean_pkaD1']:.2f} | {m['dev_pkaD1']:.2f} | "
                     f"{m['mean_pkaD2']:.2f} | {m['dev_pkaD2']:.2f} | {m['alive_both']:.2f} | {rew} | "
                     f"{m['mean_D1']:.2f} | {m['mean_D2']:.2f} | {m['mean_DA']:.2f} | "
                     f"{m['mean_Adenosine']:.2f} | "
                     + " | ".join(f"{v:.3f}" for v in r["vals"]) + " |")
    lines += ["\n## What drives the score (corr with score)\n", "```",
              *[f"{n:14s} corr={corrs[n]:+.2f}" for n in PARAM_NAMES],
              "-- gain-pair balance (corr of log ratio with score) --",
              *[f"{p:26s} corr={ratio_corrs[p]:+.2f}" for p in ratio_corrs], "```\n",
              "## Plots\n", f"- `score_vs_param_{TAG}.png` — pka-target score vs each of the 6 init params",
              f"- `release_plane_{TAG}.png` — m_a2 x ado0 (the two factors of the D2 PKA drive "
              f"once x_ado is pinned), colored by score, with the exact-solution hyperbola "
              f"m_a2*ado0 = P_req = {P_REQ:.4f}",
              f"- `score_vs_ratio_{TAG}.png` — pka-target score vs the ratio of each gain pair "
              f"({', '.join(f'{a}/{b}' for a, b in RATIO_PAIRS)}), log x, binned-median trend",
              f"- `timecourses_{TAG}.png` — fresh-init x_da/x_ado/D1/D2/pkaD1/pkaD2 over the trial, top configs"]
    if validated:
        lines.append("- `validation_curves.png` — STAGE 2 reward learning curves (full training)")
    (HERE / f"init_search_report_{TAG}.md").write_text("\n".join(lines) + "\n")
    print(f"report -> {HERE / f'init_search_report_{TAG}.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=120)
    ap.add_argument("--n-refine", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8)          # stage-1 fresh-init seeds
    ap.add_argument("--batch", type=int, default=40)         # stage-1 trials per forward
    ap.add_argument("--validate-top", type=int, default=0)   # stage-2: fully train top-K
    ap.add_argument("--val-seeds", type=int, default=8)
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
        build_report(load_results())
    elif args.validate_top:
        validate(args)
    else:
        search(args)
