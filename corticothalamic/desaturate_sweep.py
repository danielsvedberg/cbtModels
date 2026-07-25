"""How to de-saturate cortex: diagnose what pins it high, then sweep the
cortico-thalamic recurrent gain to find where cortex sits near r~0.5 (max nln
gain) and a real cue->output gradient reappears. Run:

    python corticothalamic/desaturate_sweep.py cbt_loop_noSCnoSTN

nln = sigmoid(4(x-0.5)): r=nln(drive) saturates for drive > ~1.5, is maximally
responsive (gain nln'=1) at drive=0.5 (r=0.5). So cortex saturates iff its net
resting drive is large; de-saturating = shrinking that drive to ~0.5.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop_noSCnoSTN"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, FAM)); sys.path.insert(0, ROOT)

import cbt_rnn as cbtl          # noqa: E402
import config_script            # noqa: E402
import self_timed_movement_task as stmt  # noqa: E402

cfg = config_script.for_family(FAM)
AREAS = cbtl.STATE_AREA_ORDER
exc = lambda w: np.abs(np.asarray(w))
inh = lambda w: -np.abs(np.asarray(w))

TC_RECUR = ["J_cU", "J_cL", "B_cU_cL", "B_cL_cU", "J_cU_ci", "J_cL_ci", "J_ci_cU",
            "J_ci_cL", "J_c_ii", "J_t_ee", "J_t_ei", "J_t_ie", "J_t_ii",
            "B_t_cU", "B_t_c_inh", "B_cU_t_exc", "B_cU_t_inh"]
EDGES = [("cU", "cU", "J_cU", +1), ("cU", "cL", "B_cL_cU", +1), ("cU", "cI", "J_ci_cU", -1), ("cU", "tE", "B_t_cU", +1),
         ("cL", "cL", "J_cL", +1), ("cL", "cU", "B_cU_cL", +1), ("cL", "cI", "J_ci_cL", -1),
         ("cI", "cU", "J_cU_ci", +1), ("cI", "cL", "J_cL_ci", +1), ("cI", "cI", "J_c_ii", -1), ("cI", "tE", "B_t_c_inh", +1),
         ("tE", "tE", "J_t_ee", +1), ("tE", "tI", "J_t_ei", -1), ("tE", "cU", "B_cU_t_exc", +1),
         ("tI", "tE", "J_t_ie", +1), ("tI", "tI", "J_t_ii", -1), ("tI", "cU", "B_cU_t_inh", +1)]


def rest(params, config):
    c = dict(config); c["noise_std"] = 0.0
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    T = 800
    inp = jnp.zeros((1, T, params["B_cue_cU"].shape[1]))
    stim = jnp.zeros((1, T, n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), 1)
    _, xs = cbtl.batched_rnn(params, c, inp, stim, keys)
    cortex = np.asarray(xs[AREAS.index("Cortex")][0, -100:]).mean(0)
    thal = np.asarray(xs[AREAS.index("Thalamus")][0, -100:]).mean(0)
    return cortex, thal


def rho_star(params, config, cortex, thal):
    sizes = [("cU", params["J_cU"].shape[0]), ("cL", params["J_cL"].shape[0]),
             ("cI", params["J_c_ii"].shape[0]), ("tE", params["J_t_ee"].shape[0]),
             ("tI", params["J_t_ii"].shape[0])]
    idx, off = {}, 0
    for nm, sz in sizes:
        idx[nm] = slice(off, off + sz); off += sz
    W = np.zeros((off, off))
    for post, pre, key, s in EDGES:
        W[idx[post], idx[pre]] = (exc if s > 0 else inh)(params[key])
    r = np.concatenate([cortex, thal])
    g = 4.0 * r * (1.0 - r)
    tau = config["tau_c"]
    M = (1 - 1 / tau) * np.eye(off) + (1 / tau) * W
    return float(np.max(np.abs(np.linalg.eigvals(g[:, None] * M))))


def cue_sensitivity(params, config):
    B = 16
    tc = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=tc["t_start"], T_cue=tc["t_cue"], T_response=tc["t_response"], T=tc["t_total"])
    inp, tgt, msk = inputs[:B], targets[:B], masks[:B]
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((B, inp.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)
    win = ((tgt[..., 0] > 0) & (msk[..., 0] > 0)).astype(jnp.float32)

    def window_out(a):
        ys = cbtl.rnn_func(params, config, a, stim, keys)[0]
        ys = ys[..., 0] if ys.ndim == 3 else ys
        return jnp.sum(ys * win) / (jnp.sum(win) + 1e-8)
    return float(jnp.linalg.norm(jax.grad(window_out)(inp)))


def scaled(params, s):
    p = dict(params)
    for k in TC_RECUR:
        p[k] = jnp.asarray(params[k]) * s
    return p


def main():
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=1)
    tau = config["tau_c"]
    cortex, thal = rest(params, config)
    n_cU = params["J_cU"].shape[0]
    r_cU, r_cL = cortex[:n_cU], cortex[n_cU:2 * n_cU]
    r_cI = cortex[2 * n_cU:]
    r_tE = thal[:params["J_t_ee"].shape[0]]

    print("=" * 72)
    print(f"DE-SATURATING CORTEX — family: {FAM}  (tau_c={tau}, nln=sigmoid(4(x-0.5)))")
    print("=" * 72)
    # resting drive budget into cU (mean over cU units)
    E_self = (exc(params["J_cU"]) @ r_cU).mean()
    E_cross = (exc(params["B_cL_cU"]) @ r_cL).mean()
    I_inh = (inh(params["J_ci_cU"]) @ r_cI).mean()
    E_thal = (exc(params["B_t_cU"]) @ r_tE).mean()
    rec = E_self + E_cross + I_inh + E_thal
    pre = (1 - 1 / tau) * r_cU.mean() + (1 / tau) * rec
    print("resting drive budget into cU (mean/unit):")
    print(f"  E self  |J_cU|@r_cU     = {E_self:+.3f}")
    print(f"  E cross |B_cL_cU|@r_cL  = {E_cross:+.3f}")
    print(f"  I inhib -|J_ci_cU|@r_cI = {I_inh:+.3f}")
    print(f"  E thal  |B_t_cU|@r_tE   = {E_thal:+.3f}")
    print(f"  => recurrent sum = {rec:+.3f};  pre-activation drive = {pre:.3f} "
          f"-> nln = {1/(1+np.exp(-4*(pre-0.5))):.3f} (r_cU={r_cU.mean():.3f})")
    print(f"  E:I current ratio = {(E_self+E_cross+E_thal)/max(abs(I_inh),1e-9):.1f}  "
          f"(saturates because net drive {pre:.2f} >> 0.5)\n")

    print("SWEEP: scale the 17 cortico-thalamic recurrent blocks by s")
    print(f"{'s':>5}{'cortex r':>10}{'%>0.9':>8}{'rho*':>8}{'cue->out grad':>15}   note")
    for s in (1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
        p = scaled(params, s)
        cx, th = rest(p, config)
        rs = rho_star(p, config, cx, th)
        cs = cue_sensitivity(p, config)
        sat = 100.0 * (cx > 0.9).mean()
        note = ("SATURATED" if cx.mean() > 0.85 else
                "near r~0.5 (max gain)" if 0.4 <= cx.mean() <= 0.65 else
                "low" if cx.mean() < 0.4 else "")
        print(f"{s:>5.2f}{cx.mean():>10.3f}{sat:>7.0f}%{rs:>8.3f}{cs:>15.2e}   {note}")
    print("\nRead: find the s where cortex leaves saturation (r -> ~0.5, %>0.9 -> 0),")
    print("rho* climbs toward ~1 (long memory), and cue->out grad jumps off ~1e-5")
    print("(a real gradient appears). That scale is the de-saturation target; bake it")
    print("into the init (lower g_bg / add cortico-thalamic spectral normalization).")


if __name__ == "__main__":
    main()
