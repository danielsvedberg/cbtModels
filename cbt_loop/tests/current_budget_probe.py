"""Per-area CURRENT BUDGET at the resting fixed point.

For each area, decompose the total input current into its named source terms,
using the ACTUAL resting rates the network settles to. The fixed point of
    x <- (1-1/tau) x + (1/tau) * drive ;  x <- nln(x)
is x* = nln(drive), so `drive` is exactly what decides dead vs alive.

Also reports gradients on the neuromodulator params, i.e. whether training can
even reach the knobs that would revive the dead areas.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop')
import self_timed_movement_task as stmt

AREAS = cbtl.STATE_AREA_ORDER
exc = lambda w: np.abs(np.asarray(w))
inh = lambda w: -np.abs(np.asarray(w))


def build():
    rnn_cfg = cfg.RNN_CONFIG
    task_cfg = cfg.PAVLOVIAN_CONFIG
    inputs, targets, masks = stmt.pavlovian_task(
        T_start=task_cfg["t_start"], T_cue=task_cfg["t_cue"],
        T_response=task_cfg["t_response"], T=task_cfg["t_total"])
    params, config = cbtl.init_params(
        jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]),
        n_c_U=rnn_cfg["n_c_U"], n_c_L=rnn_cfg["n_c_L"], n_c_inh=rnn_cfg["n_c_inh"],
        n_d1=rnn_cfg["n_d1"], n_d2=rnn_cfg["n_d2"], n_snc=rnn_cfg["n_snc"],
        n_snr=rnn_cfg["n_snr"], n_gpe=rnn_cfg["n_gpe"], n_stn=rnn_cfg["n_stn"],
        n_t_exc=rnn_cfg["n_t_exc"], n_t_inh=rnn_cfg["n_t_inh"],
        n_input=inputs.shape[-1], n_output=1,
        g_bg=rnn_cfg["g_bg"], g_nm=rnn_cfg["g_nm"], noise_std=rnn_cfg["noise_std"],
        balanced_init=rnn_cfg.get("balanced_init", False))
    return params, config, inputs, targets, masks


def rest_rates(params, config, inputs):
    null = jnp.zeros_like(inputs[:8])
    n_d1 = params["J_d1"].shape[0]; n_d2 = params["J_d2"].shape[0]
    stim = jnp.zeros((8, null.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), 8)
    ys, xs = cbtl.batched_rnn(params, config, null, stim, keys)
    r = {}
    for name, x in zip(AREAS, xs):
        r[name] = np.asarray(x[:, -200:, :]).mean(axis=(0, 1))  # per-unit rest rate
    n_cU = params["J_cU"].shape[0]; n_cL = params["J_cL"].shape[0]
    r["cU"] = r["Cortex"][:n_cU]
    r["cL"] = r["Cortex"][n_cU:n_cU + n_cL]
    r["cI"] = r["Cortex"][n_cU + n_cL:]
    n_tE = params["J_t_ee"].shape[0]
    r["tE"] = r["Thalamus"][:n_tE]
    r["tI"] = r["Thalamus"][n_tE:]
    return r, np.asarray(ys[:, -200:]).mean()


def budget(title, terms, note=""):
    tot = sum(v for _, v in terms)
    print(f"\n--- {title} --- {note}")
    for nm, v in terms:
        bar = ("+" if v >= 0 else "-") * min(int(abs(v) * 40) + 1, 40)
        print(f"   {nm:<26} {v:+9.4f}  {bar}")
    print(f"   {'NET DRIVE':<26} {tot:+9.4f}   -> rest rate nln(drive) = "
          f"{max(0.0, np.tanh(tot)):.4f}"
          f"{'   <== DEAD (rectified off)' if tot <= 0 else ''}")
    return tot


def main():
    params, config, inputs, targets, masks = build()
    r, y = rest_rates(params, config, inputs)
    P = params
    n_cU = P["J_cU"].shape[0]; n_cL = P["J_cL"].shape[0]
    n_d1 = P["J_d1"].shape[0]; n_d2 = P["J_d2"].shape[0]
    n_snr = P["P_snr"].shape[0]; n_gpe = P["J_gpe"].shape[0]

    sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x)))
    snc_pacer = config["snc_pacer_min"] + sig(P["P_snc"]) * (
        config["snc_pacer_max"] - config["snc_pacer_min"])
    snr_pacer = config["snr_pacer_min"] + sig(P["P_snr"]) * (
        config["snr_pacer_max"] - config["snr_pacer_min"])
    gpe_pacer = config["gpe_pacer_min"] + sig(P.get("P_gpe", np.zeros(n_gpe))) * (
        config["gpe_pacer_max"] - config["gpe_pacer_min"])

    print("=" * 74)
    print("RESTING CURRENT BUDGET (mean over units; drive == the fixed point)")
    print("=" * 74)
    print("rest rates: " + "  ".join(
        f"{k}={np.mean(r[k]):.4f}" for k in
        ("cU", "cL", "cI", "tE", "tI", "D1", "D2", "SNc", "GPe", "STN", "SNr", "SC", "Medulla")))
    print(f"output = {y:.4f}")

    budget("GPe", [
        ("pacer", float(np.mean(gpe_pacer))),
        ("D2 -> GPe (inh)", float(np.mean(inh(P["B_d2_gpe"]) @ r["D2"] - 0.1 / n_d2))),
        ("cU -> GPe (exc)", float(np.mean((exc(P["B_cU_gpe"]) + 0.1 / n_cU) @ r["cU"]))),
        ("STN -> GPe (exc)", float(np.mean(exc(P["B_stn_gpe"]) @ r["STN"]))),
    ], "the one strongly-driven area: pacer floor >= 1")

    budget("SNc  (dopamine source)", [
        ("pacer", float(np.mean(snc_pacer))),
        ("STN -> SNc (exc)", float(np.mean(exc(P["B_stn_snc"]) @ r["STN"]))),
        ("cL -> SNc (exc)", float(np.mean((exc(P["B_cL_snc"]) + 0.1 / n_cL) @ r["cL"]))),
        ("D1 -> SNc (inh)", float(np.mean(inh(P["B_d1_snc"]) @ r["D1"]))),
        ("D2 -> SNc (inh)", float(np.mean(inh(P["B_d2_snc"]) @ r["D2"]))),
        ("GPe -> SNc (inh)", float(np.mean(inh(P["B_gpe_snc"]) @ r["GPe"]))),
    ], "GPe is the only sizeable term")

    budget("SNr  (motor gate)", [
        ("pacer", float(np.mean(snr_pacer))),
        ("D1 -> SNr (inh)", float(np.mean(inh(P["B_d1_snr"]) @ r["D1"] - 0.1 / n_d1))),
        ("GPe -> SNr (inh)", float(np.mean(inh(P["B_gpe_snr"]) @ r["GPe"] - 0.1 / n_gpe))),
        ("STN -> SNr (exc)", float(np.mean(exc(P["B_stn_snr"]) @ r["STN"]))),
    ], "D1's only lever on the gate is the tiny D1->SNr term")

    budget("Thalamus (exc)", [
        ("recurrent E (tE->tE)", float(np.mean(exc(P["J_t_ee"]) @ r["tE"]))),
        ("recurrent I (tI->tE)", float(np.mean(inh(P["J_t_ei"]) @ r["tI"]))),
        ("cU -> tE (exc)", float(np.mean(exc(P["B_cU_t_exc"]) @ r["cU"]))),
        ("SNr -> tE (inh)", float(np.mean(inh(P["B_snr_t_exc"]) @ r["SNr"] - 0.1 / n_snr))),
        ("SC -> tE (exc)", float(np.mean(exc(P["B_sc_t_exc"]) @ r["SC"]))),
    ], "tonic SNr closes the thalamic relay -> cortico-thalamic loop is severed")

    budget("Cortex cU", [
        ("recurrent E (cU->cU)", float(np.mean(exc(P["J_cU"]) @ r["cU"]))),
        ("cL -> cU (exc)", float(np.mean(exc(P["B_cL_cU"]) @ r["cL"]))),
        ("cI -> cU (inh)", float(np.mean(inh(P["J_ci_cU"]) @ r["cI"]))),
        ("tE -> cU (exc)", float(np.mean(exc(P["B_t_cU"]) @ r["tE"]))),
    ], "per-row balance makes the recurrent net drive ~0; no tonic input at all")

    budget("D1", [
        ("cU -> D1 (exc)", float(np.mean((exc(P["B_cU_d1"]) + 0.1 / n_cU) @ r["cU"]))),
        ("self (inh)", float(np.mean(inh(P["J_d1"]) @ r["D1"]))),
        ("D2 -> D1 (inh)", float(np.mean(inh(P.get("B_d2_d1", np.zeros((n_d1, n_d2)))) @ r["D2"]))),
    ], "then multiplied by the PKA gain below")

    budget("Medulla E", [
        ("cL -> med (exc)", float(np.mean(exc(P["B_cL_med"]) @ r["cL"]))),
        ("SC -> med (exc)", float(np.mean(exc(P["B_sc_med"]) @ r["SC"]))),
        ("SNr -> med (inh)", float(np.mean(-(exc(P["B_snr_med"]) + config["snr_med_floor"]) @ r["SNr"]))),
    ], "tonic SNr also directly clamps the motor output")

    # ---- PKA gate arithmetic -------------------------------------------
    print("\n" + "=" * 74)
    print("PKA GATE ARITHMETIC (what the floors actually control)")
    print("=" * 74)
    mf = config.get("m_floor", 0.1)
    fa1 = config.get("m_floor_a1", mf); fa2 = config.get("m_floor_a2", mf)
    cf = lambda w, f: f + np.abs(np.asarray(w)) * (1 - f)
    m_d1 = cf(P["m_d1"], mf); m_a1 = cf(P["m_a1"], fa1)
    m_d2 = cf(P["m_d2"], mf); m_a2 = cf(P["m_a2"], fa2)
    k_a = config["k_a_floor"] + float(sig(P["k_a"])) * (config["k_a_cap"] - config["k_a_floor"])
    snc = float(np.mean(r["SNc"]))
    g = config.get("da_pka_gain", 1.0)
    print(f"m_floor_a1 = {fa1}  but  m_a1 = floor + |w|*(1-floor) = {m_a1.mean():.4f}"
          f"   <-- the FLOOR is not what sets m_a1; the raw weight |w|={np.abs(np.asarray(P['m_a1'])).mean():.4f} is")
    print(f"k_a = {k_a:.4f}   (k_a_floor={config['k_a_floor']}, sigmoid(param)={float(sig(P['k_a'])):.3f})")
    print(f"m_d1 = {m_d1.mean():.4f}   mean_snc = {snc:.5f}   da_pka_gain = {g}")
    da, ad = g * m_d1.mean() * snc, m_a1.mean() * k_a
    print(f"\n  D1: DA {da:.5f}  vs  adenosine {ad:.5f}   -> net {max(da-ad,0):.5f}"
          f"  ({'CLAMPED, zero gradient' if da <= ad else 'alive'})")
    print(f"      pka_d1_ss = (tau_fall/tau_rise)*net = "
          f"{config['tau_pka_fall']/config['tau_pka_rise']*max(da-ad,0):.4f}"
          f"   -> bg_nln gain = "
          f"{(lambda b: b/max(1-b,1e-9))(config['tau_pka_fall']/config['tau_pka_rise']*max(da-ad,0)):.4f}")
    print(f"\n  what pka_d1 WOULD be if adenosine were zero: "
          f"{config['tau_pka_fall']/config['tau_pka_rise']*da:.4f}  <-- i.e. DA alone is *almost* enough;")
    print(f"  needed mean_snc to beat adenosine: {ad/(g*m_d1.mean()):.4f} "
          f"(currently {snc:.5f}, snc_pacer_max = {config['snc_pacer_max']})")

    # ---- gradients on the NM knobs -------------------------------------
    print("\n" + "=" * 74)
    print("GRADIENT ON THE KNOBS THAT WOULD REVIVE THE CASCADE")
    print("=" * 74)
    B = 16
    inp, tgt, msk = inputs[:B], targets[:B], masks[:B]
    stim = jnp.zeros((B, inp.shape[1], n_d1 + n_d2))
    keys = jr.split(jr.PRNGKey(0), B)

    def loss_fn(p):
        ys, _ = cbtl.batched_rnn(p, config, inp, stim, keys)
        e = 1e-6
        return jnp.mean((-(tgt * jnp.log(ys + e) + (1 - tgt) * jnp.log(1 - ys + e))) * msk)

    gr = jax.grad(loss_fn)(params)
    tot = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in gr.values())))
    print(f"||grad||_total = {tot:.4e}")
    for k in ("out_bias", "out_gain", "C_med", "P_snc", "P_snr", "P_gpe", "k_a",
              "m_d1", "m_a1", "m_d2", "m_a2", "B_d1_snr", "B_gpe_snc",
              "B_snr_t_exc", "B_cue_cU", "B_cue_cL", "J_cU", "B_cU_d1"):
        if k in gr:
            v = float(jnp.linalg.norm(gr[k]))
            print(f"  {k:<14} {v:.4e}   {100*v/tot:6.2f}% of total"
                  f"{'   <-- ZERO' if v < 1e-10 else ''}")


if __name__ == "__main__":
    main()
