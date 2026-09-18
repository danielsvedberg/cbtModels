"""MULTIVARIATE analysis: do the initial weights predict breakthrough, across ALL areas?

Univariate screening (44 blocks, one at a time) found nothing surviving correction, but it
cannot see interactions -- e.g. "cortex saturated AND thalamus dead" being fatal while either
alone is benign. It was also thalamus-centric. This builds a full feature matrix over EVERY
area and fits a regularised multivariate classifier with honest cross-validation.

Features per seed, all measurable at INIT (no training):
  per area (cU, cL, c_inh, D1, D2, SNc, GPe, SNr, thal_exc, thal_inh, medulla, pkaD1, pkaD2):
      mean, p95, std, frac_dead (==0), frac_sat (>0.9 * area max)
  per weight block: mean fan-in row-sum of the effective weight
Both DEAD and SATURATED regimes are represented, per the hypothesis that either extreme
kills training.

Honesty controls, because this dataset is small and wide:
  * k-fold cross-validated AUC -- never in-sample fit
  * a permutation null (labels shuffled) to calibrate what AUC chance produces here
  * L2 regularisation, strength chosen INSIDE each training fold
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_script as cs

AREAS = ["Cortex", "D1", "D2", "SNc", "GPe", "SNr", "Thalamus", "pkaD1", "pkaD2", "Medulla", "DA", "Adenosine"]


def build_features():
    st = cs._CBT_FAMILY_STRUCTURE["cbt_loop_noSCnoSTN"]
    st["extra_weight_init"] = {"m_d1": 1.1363, "m_d2": -1.5368, "m_a1": -3.4095, "m_a2": 0.0080}
    st["extra_runtime"] = {**st["extra_runtime"], "da_release": 0.2982, "ado_release": 1.1363,
                           "pin_ado": 0.093, "stri_cross_scale": 0.162, "stri_tonic": 0.220}
    import jax.random as jr, cbt_rnn as cbtl, self_timed_movement_task as stmt
    cfg = cs.for_family("cbt_loop_noSCnoSTN"); t = cfg.TASK_CONFIG
    inp, _, _ = stmt.self_timed_movement_task(t["t_start"][:16], t["t_cue"], t["t_wait"],
                                              t["t_movement"], t["t_total"])
    rows = sorted(json.load(open(pathlib.Path(__file__).with_name("seed_stability_data.json"))),
                  key=lambda r: r["seed"])
    eff = lambda w: np.clip(np.tanh(np.asarray(w)), 0, None)
    X, names, y = [], None, []
    RC = cfg.RNN_CONFIG
    ncU, ncL = RC["n_c_U"], RC["n_c_L"]
    nte = RC["n_t_exc"]
    for r in rows:
        sd = r["seed"]
        p, c = cbtl.init_params(jr.PRNGKey(sd), n_input=1)
        c = dict(c); c["readout_source"] = "thalamus"
        xs = cbtl.rnn_func(p, c, inp, None, jr.split(jr.PRNGKey(sd), inp.shape[0]))[4]
        feats, fn = [], []
        def add(a, tag):
            a = np.asarray(a)
            mx = a.max() if a.size and a.max() > 0 else 1.0
            for lbl, v in (("mean", a.mean()), ("p95", np.percentile(a, 95)), ("std", a.std()),
                           ("dead", (a == 0).mean()), ("sat", (a > 0.9 * mx).mean())):
                feats.append(float(v)); fn.append(f"{tag}.{lbl}")
        # cortex split into its three pools -- they can fail independently
        xc = np.asarray(xs[0])
        add(xc[..., :ncU], "cU"); add(xc[..., ncU:ncU+ncL], "cL"); add(xc[..., ncU+ncL:], "c_inh")
        xt = np.asarray(xs[6])
        add(xt[..., :nte], "thal_exc"); add(xt[..., nte:], "thal_inh")
        for i, nm in ((1, "D1"), (2, "D2"), (3, "SNc"), (4, "GPe"), (5, "SNr"),
                      (7, "pkaD1"), (8, "pkaD2"), (9, "Medulla"), (10, "DA"), (11, "Adeno")):
            add(xs[i], nm)
        z = np.load(pathlib.Path(__file__).with_name("init_weights") / f"seed_{sd:03d}.npz")
        for k in sorted(z.files):
            if np.asarray(z[k]).ndim >= 2:
                feats.append(float(eff(z[k]).sum(1).mean())); fn.append(f"w.{k}")
        X.append(feats); y.append(r["breakthrough"])
        if names is None: names = fn
    return np.array(X, float), np.array(y, float), names


def fit_logreg(X, y, lam, iters=3000, lr=0.1):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        z = X @ w + b; pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (pr - y) / len(y) + lam * w
        gb = (pr - y).mean()
        w -= lr * g; b -= lr * gb
    return w, b


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg): return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cv_auc(X, y, k=8, lams=(0.01, 0.1, 1.0, 10.0), rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(len(y)); folds = np.array_split(idx, k)
    scores = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[f] - mu) / sd
        # pick lambda by an inner split
        best, bl = -1, lams[0]
        inner = rng.permutation(len(tr)); cut = max(2, len(tr) // 4)
        i_te, i_tr = inner[:cut], inner[cut:]
        for lam in lams:
            w, b = fit_logreg(Xtr[i_tr], y[tr][i_tr], lam)
            a = auc(y[tr][i_te], Xtr[i_te] @ w + b)
            if not np.isnan(a) and a > best: best, bl = a, lam
        w, b = fit_logreg(Xtr, y[tr], bl)
        scores[f] = Xte @ w + b
    return auc(y, scores)


if __name__ == "__main__":
    X, y, names = build_features()
    print(f"n={len(y)} seeds, {int(y.sum())} breakthroughs, {X.shape[1]} init features")
    keep = X.std(0) > 1e-12
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    print(f"{X.shape[1]} features after dropping constants")
    obs = cv_auc(X, y)
    print(f"\ncross-validated AUC = {obs:.3f}   (0.5 = chance)")
    rng = np.random.default_rng(1)
    null = [cv_auc(X, rng.permutation(y), rng=np.random.default_rng(100 + i)) for i in range(40)]
    null = np.array([v for v in null if not np.isnan(v)])
    pval = float((null >= obs).mean())
    print(f"permutation null: mean {null.mean():.3f}, 95th pct {np.percentile(null,95):.3f}")
    print(f"permutation p = {pval:.3f}  -> {'SIGNIFICANT' if pval < 0.05 else 'not distinguishable from chance'}")
    np.savez_compressed(pathlib.Path(__file__).with_name("init_features.npz"),
                        X=X, y=y, names=np.array(names))
    print("\nfeature matrix saved to init_features.npz")
