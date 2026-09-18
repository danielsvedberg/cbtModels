"""Do the INIT GRADIENTS predict breakthrough -- and which init weights shape them?

Init *weights* vs outcome came back null (CV AUC 0.42, p=0.73 at n=40). Gradients are the
mechanistic link between weights and learning: they are what the optimiser actually sees,
they are continuous rather than binary, and they cost one backward pass (no training), so
they can be computed for every labelled seed we already have.

Two questions, in order:
  Q1  does the init gradient STRUCTURE predict breakthrough?   (grad features -> outcome)
  Q2  which init WEIGHTS determine that structure?             (weights -> grad features)

Q1 is the useful one; Q2 only matters for whatever Q1 finds.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_script as cs
from analyze_init import cv_auc, auc

st = cs._CBT_FAMILY_STRUCTURE["cbt_loop_noSCnoSTN"]
st["extra_weight_init"] = {"m_d1": 1.1363, "m_d2": -1.5368, "m_a1": -3.4095, "m_a2": 0.0080}
st["extra_runtime"] = {**st["extra_runtime"], "da_release": 0.2982, "ado_release": 1.1363,
                       "pin_ado": 0.093, "stri_cross_scale": 0.162, "stri_tonic": 0.220}

import jax, jax.numpy as jnp, jax.random as jr
import cbt_rnn as cbtl, self_timed_movement_task as stmt

cfg = cs.for_family("cbt_loop_noSCnoSTN"); t = cfg.TASK_CONFIG; sup = cfg.SUPERVISED_THAL_CONFIG
inp, tg, mk = stmt.selftimed_step_target(
    T_start=t["t_start"][:16], T_cue=t["t_cue"], T_wait=t["t_wait"], T_movement=t["t_movement"],
    T=t["t_total"], lo=sup["target_lo"], hi=sup["target_hi"], hold=sup["hold"], delay=sup["delay"])


def grad_features(seed):
    p, c = cbtl.init_params(jr.PRNGKey(seed), n_input=1)
    c = dict(c); c["readout_source"] = "thalamus"
    keys = jr.split(jr.PRNGKey(seed), 16)
    def loss(pp):
        return stmt.supervised_loss(cbtl.rnn_func, pp, c, inp, tg, mk, keys, loss_type="mse")[0]
    g = jax.grad(loss)(p)
    gn = {k: float(jnp.linalg.norm(jnp.asarray(v))) for k, v in g.items()}
    tot = float(np.sqrt(sum(v ** 2 for v in gn.values())))
    feats, names = [tot, np.log10(tot + 1e-30)], ["grad_total", "log_grad_total"]
    for k in sorted(gn):                      # absolute norm AND share of total
        feats += [gn[k], gn[k] / (tot + 1e-30)]
        names += [f"g.{k}", f"gfrac.{k}"]
    return np.array(feats), names, p


def main():
    src = pathlib.Path(__file__).with_name("seed_stability_OLDPACER.json")
    rows = sorted(json.load(open(src)), key=lambda r: r["seed"])
    X, names, y, W = [], None, [], []
    eff = lambda w: np.clip(np.tanh(np.asarray(w)), 0, None)
    for r in rows:
        f, nm, p = grad_features(r["seed"])
        X.append(f); y.append(r["breakthrough"])
        W.append([float(eff(v).sum(1).mean()) for k, v in sorted(p.items())
                  if np.asarray(v).ndim >= 2])
        if names is None:
            names = nm
            wnames = [k for k, v in sorted(p.items()) if np.asarray(v).ndim >= 2]
    X = np.array(X); y = np.array(y, float); W = np.array(W)
    keep = X.std(0) > 1e-14
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    print(f"n={len(y)} seeds, {int(y.sum())} breakthroughs, {X.shape[1]} gradient features\n")

    print("Q1a  univariate screen (gradient feature -> breakthrough)")
    rs = []
    for i, n in enumerate(names):
        v = X[:, i]
        if v.std() > 0: rs.append((float(np.corrcoef(v, y)[0, 1]), n))
    rs.sort(key=lambda t: -abs(t[0]))
    rcrit = 1.96 / np.sqrt(len(y) - 1)
    import scipy.stats as ss
    bonf = ss.t.ppf(1 - 0.05 / (2 * len(rs)), len(y) - 2)
    rbonf = bonf / np.sqrt(bonf ** 2 + len(y) - 2)
    for r_, n in rs[:8]:
        print(f"   {n:26s} r={r_:+.3f} {'*' if abs(r_) > rcrit else ''}")
    print(f"   thresholds: uncorrected |r|>{rcrit:.3f} ; Bonferroni({len(rs)}) |r|>{rbonf:.3f}")
    print(f"   exceeding uncorrected: {sum(1 for r_,_ in rs if abs(r_)>rcrit)} of {len(rs)} "
          f"(expect ~{0.05*len(rs):.1f} by chance) ; exceeding Bonferroni: "
          f"{sum(1 for r_,_ in rs if abs(r_)>rbonf)}")

    print("\nQ1b  multivariate, cross-validated")
    obs = cv_auc(X, y)
    rng = np.random.default_rng(1)
    null = np.array([cv_auc(X, rng.permutation(y), rng=np.random.default_rng(200 + i)) for i in range(40)])
    null = null[~np.isnan(null)]
    print(f"   CV AUC = {obs:.3f}   permutation null mean {null.mean():.3f}, 95th {np.percentile(null,95):.3f}")
    print(f"   p = {(null >= obs).mean():.3f}  -> "
          f"{'SIGNIFICANT' if (null>=obs).mean()<0.05 else 'not distinguishable from chance'}")

    print("\nQ2  which init weights shape the gradient? (weights -> log grad_total, CV R^2)")
    tgt = X[:, names.index("log_grad_total")]
    idx = np.random.default_rng(0).permutation(len(y)); folds = np.array_split(idx, 5)
    pred = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        mu, sd = W[tr].mean(0), W[tr].std(0) + 1e-9
        A = (W[tr] - mu) / sd; B = (W[f] - mu) / sd
        lam = 1.0
        coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (tgt[tr] - tgt[tr].mean()))
        pred[f] = B @ coef + tgt[tr].mean()
    ss_res = ((tgt - pred) ** 2).sum(); ss_tot = ((tgt - tgt.mean()) ** 2).sum()
    print(f"   CV R^2 = {1 - ss_res/ss_tot:.3f}  (<=0 means weights do not predict gradient magnitude)")


if __name__ == "__main__":
    main()
