"""Empirical criticality diagnostics for the CBT loop (no linearization needed).

These three measures estimate how close the recurrent loop sits to criticality
(spectral radius -> 1 / line-attractor regime) using only forward passes of the
trained model, so no refactor of ``cbt_rnn.multiregion_rnn`` is required:

1. ``impulse_response`` — perturb the loop with a brief cue pulse and measure how
   long the perturbation takes to decay back to baseline. A decay timescale far
   longer than the single-unit membrane leak (tau_c * dt ~= 50 ms) means the
   recurrent loop itself is adding slow dynamics, i.e. it is near-critical. A
   *growing* perturbation means the operating point is locally unstable.

2. ``intrinsic_timescale`` — with the cue absent and only state noise driving the
   network, measure the autocorrelation timescale of each area's population
   activity. This is the model analogue of the experimental "intrinsic timescale"
   measure; it diverges as the loop approaches criticality.

3. ``susceptibility`` — apply a small *sustained* cue step and measure the
   steady-state gain (delta activity / input amplitude). Steady-state gain scales
   like 1 / (1 - loop_gain), so a large susceptibility means loop_gain -> 1.

All three are evaluated at the trained operating point. To turn any of them into a
"distance to criticality" curve, sweep a recurrent-gain knob (that needs the
single-step refactor discussed separately) and watch the timescale / gain diverge.

Run as a script to load the default bundle and print a summary for every area:

    python criticality_diagnostics.py [path/to/params.pkl]
"""

import sys
import pickle as pkl
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSC')


# Firing-rate areas that make up the recurrent loop. The modulatory states
# (pkaD1/pkaD2/DA/Adenosine) are excluded from the loop-criticality measures
# because they are *intended* to be slow; they're reported separately by the
# intrinsic-timescale pass so you can see how much slowness is loop dynamics vs.
# neuromodulator leak.
RATE_AREAS = ["Cortex", "D1", "D2", "SNc", "GPe", "STN", "SNr", "Thalamus", "Medulla"]
MODULATOR_AREAS = ["pkaD1", "pkaD2", "DA", "Adenosine"]


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------
def load_bundle(path=None):
    """Load a (params, config) bundle, rebuilding config for legacy dumps.

    Mirrors testing_script._load_bundle but without pulling in the heavy plotting
    imports, so this module stays cheap to import.
    """
    params_path = Path(path) if path else cfg.params_path()
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}.")
    with params_path.open("rb") as f:
        bundle = pkl.load(f)
    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        return bundle["params"], bundle["config"]
    # Legacy params-only dump: rebuild a config from the param shapes.
    params = bundle
    _, config = cbtl.init_params(jr.PRNGKey(0), n_input=params["B_cue_cU"].shape[1])
    return params, config


def _run_trial(params, config, inputs, noise_std, seed=0):
    """One unbatched forward pass at a chosen noise level; returns (ys, xs)."""
    run_cfg = dict(config)
    run_cfg["noise_std"] = float(noise_std)
    return cbtl.multiregion_rnn(params, run_cfg, inputs, None, jr.PRNGKey(int(seed)))


def _n_input(params):
    return params["B_cue_cU"].shape[1]


# ---------------------------------------------------------------------------
# Fit helpers (numpy)
# ---------------------------------------------------------------------------
def _fit_decay(delta, t_off, dt_ms, floor_frac=0.02):
    """Estimate the decay timescale of a perturbation trace ``delta`` after t_off.

    Log-linear fit of the post-pulse envelope: slope < 0 -> decaying, tau_ms > 0;
    slope > 0 -> the perturbation *grows* (locally unstable), reported as a
    negative tau_ms with growing=True. Also returns robust, fit-free summaries
    (half-life and end/peak ratio) that survive a poor exponential fit.
    """
    d = np.asarray(delta[t_off:], dtype=float)
    out = {"tau_ms": np.nan, "growing": False, "half_life_ms": np.nan,
           "end_over_peak": np.nan, "peak": np.nan}
    if d.size < 5:
        return out
    peak = float(d.max())
    out["peak"] = peak
    if peak <= 0:
        return out
    out["end_over_peak"] = float(d[-1] / peak)
    # Robust half-life: first crossing below half the peak, measured from the peak.
    peak_idx = int(np.argmax(d))
    after = d[peak_idx:]
    half_hits = np.where(after < 0.5 * peak)[0]
    if half_hits.size:
        out["half_life_ms"] = float(half_hits[0] * dt_ms)
    # Log-linear envelope fit above a small floor to avoid fitting numerical noise.
    thr = floor_frac * peak
    idx = np.where(d > thr)[0]
    if idx.size >= 3:
        slope = np.polyfit(idx.astype(float), np.log(d[idx]), 1)[0]
        if slope < -1e-9:
            out["tau_ms"] = float(-dt_ms / slope)
        elif slope > 1e-9:
            out["growing"] = True
            out["tau_ms"] = float(-dt_ms / slope)  # negative => growth timescale
        else:
            out["tau_ms"] = np.inf
    return out


def _autocorr(sig, max_lag):
    """Normalized autocorrelation (lag 0..max_lag-1) of a 1-D signal."""
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 0:
        return np.zeros(max_lag)
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    return ac[:max_lag] / var


def _ac_timescale(ac, dt_ms):
    """1/e crossing of an autocorrelation curve, in ms (inf if never crosses)."""
    below = np.where(ac < 1.0 / np.e)[0]
    if below.size == 0:
        return np.inf
    return float(below[0] * dt_ms)


# ---------------------------------------------------------------------------
# 3a. Impulse response
# ---------------------------------------------------------------------------
def impulse_response(params, config, T=600, t_pert=100, pulse_dur=5,
                     amp=0.2, dt_ms=10):
    """Loop free-decay timescale from a brief deterministic cue pulse.

    Runs a baseline (no cue) and a perturbed (brief cue pulse) trial with noise
    off, so the only difference is the pulse. For each rate area we track the
    L2 distance between perturbed and baseline trajectories and fit its decay
    *after the pulse ends* — that post-pulse segment is the loop's free response.

    Returns a dict: {area: {tau_ms, half_life_ms, end_over_peak, growing, peak},
    plus a "_global" entry over all rate units stacked together}. Compare tau_ms
    to the single-unit leak tau_c*dt (~50 ms): tau_ms >> leak => the loop adds
    slow dynamics (near-critical); tau_ms ~ leak => over-damped.
    """
    n_in = _n_input(params)
    t_off = t_pert + pulse_dur

    base_in = jnp.zeros((T, n_in))
    pert_in = base_in.at[t_pert:t_off, 0].set(amp)

    _, xs_base = _run_trial(params, config, base_in, noise_std=0.0)
    _, xs_pert = _run_trial(params, config, pert_in, noise_std=0.0)

    results = {}
    stacked_base, stacked_pert = [], []
    for area in RATE_AREAS:
        xb = np.asarray(cbtl.get_brain_area(area, xs_base))  # (T, n)
        xp = np.asarray(cbtl.get_brain_area(area, xs_pert))
        stacked_base.append(xb)
        stacked_pert.append(xp)
        delta = np.linalg.norm(xp - xb, axis=-1)  # (T,)
        results[area] = _fit_decay(delta, t_off, dt_ms)

    gb = np.concatenate(stacked_base, axis=-1)
    gp = np.concatenate(stacked_pert, axis=-1)
    results["_global"] = _fit_decay(np.linalg.norm(gp - gb, axis=-1), t_off, dt_ms)
    return results


# ---------------------------------------------------------------------------
# 3b. Intrinsic timescale (autocorrelation)
# ---------------------------------------------------------------------------
def intrinsic_timescale(params, config, T=2000, burn=300, noise_std=None,
                        n_seeds=5, max_lag=None, dt_ms=10, include_modulators=True):
    """Autocorrelation timescale of each area's population activity (cue absent).

    Drives the network with state noise only (no cue), discards a ``burn`` window,
    then averages the population-mean autocorrelation over ``n_seeds`` and reads
    off the 1/e crossing. Larger tau => closer to criticality.

    Returns {area: {tau_ms, ac}} where ``ac`` is the mean autocorrelation curve.
    """
    n_in = _n_input(params)
    inputs = jnp.zeros((T, n_in))
    ns = config.get("noise_std", 0.01) if noise_std is None else noise_std
    if max_lag is None:
        max_lag = (T - burn) // 2

    areas = list(RATE_AREAS) + (MODULATOR_AREAS if include_modulators else [])
    acc = {a: np.zeros(max_lag) for a in areas}
    for seed in range(n_seeds):
        _, xs = _run_trial(params, config, inputs, noise_std=ns, seed=seed)
        for a in areas:
            x = np.asarray(cbtl.get_brain_area(a, xs))[burn:]  # (T', n) or (T',)
            # Rate areas are (T', n) -> population mean; scalar modulator states
            # (DA / Adenosine) are already 1-D and used as-is.
            sig = x.mean(axis=-1) if x.ndim > 1 else x
            acc[a] += _autocorr(sig, max_lag)
    results = {}
    for a in areas:
        ac = acc[a] / n_seeds
        results[a] = {"tau_ms": _ac_timescale(ac, dt_ms), "ac": ac}
    return results


# ---------------------------------------------------------------------------
# 3c. Steady-state susceptibility (gain)
# ---------------------------------------------------------------------------
def susceptibility(params, config, T=800, settle_frac=0.75,
                   amps=(0.05, 0.1, 0.2), dt_ms=10):
    """Steady-state gain to a sustained cue step: delta_activity / amplitude.

    Applies a constant cue for the whole (noise-off) trial and compares the
    late-trial steady state to the zero-input steady state. Gain ~ 1/(1-loop_gain),
    so a large susceptibility => loop_gain near 1. Sweeping several amplitudes
    checks linearity: a gain that grows with amplitude means you're riding into
    the saturating regime rather than a clean near-critical response.

    Returns {area: {amp: gain}, plus "_global": {amp: gain}} where gain is the
    per-unit-RMS activity change divided by the input amplitude.
    """
    n_in = _n_input(params)
    t0 = int(T * settle_frac)

    def _steady(inp):
        _, xs = _run_trial(params, config, inp, noise_std=0.0)
        return {a: np.asarray(cbtl.get_brain_area(a, xs))[t0:].mean(axis=0)
                for a in RATE_AREAS}

    base = _steady(jnp.zeros((T, n_in)))
    results = {a: {} for a in RATE_AREAS}
    results["_global"] = {}
    for amp in amps:
        step_in = jnp.zeros((T, n_in)).at[:, 0].set(amp)
        ss = _steady(step_in)
        g_all = []
        for a in RATE_AREAS:
            d = ss[a] - base[a]
            gain = float(np.sqrt(np.mean(d ** 2)) / amp)  # per-unit RMS gain
            results[a][amp] = gain
            g_all.append(d)
        results["_global"][amp] = float(
            np.sqrt(np.mean(np.concatenate(g_all) ** 2)) / amp
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(v, unit=""):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "   n/a"
    if np.isinf(v):
        return "   inf"
    return f"{v:7.1f}{unit}"


def report(params, config, dt_ms=10):
    """Run all three diagnostics and print a summary table."""
    tau_leak = float(config.get("tau_c", 5.0)) * dt_ms
    print("=" * 68)
    print("CBT loop criticality diagnostics")
    print(f"single-unit membrane leak  tau_c*dt = {tau_leak:.0f} ms  (over-damped baseline)")
    print("=" * 68)

    print("\n[1] IMPULSE RESPONSE  (loop free-decay after a brief cue pulse)")
    print("    tau >> leak => loop is near-critical;  growing => locally unstable")
    ir = impulse_response(params, config, dt_ms=dt_ms)
    print(f"    {'area':<10}{'tau(ms)':>10}{'half-life':>11}{'end/peak':>10}  flag")
    for area in ["_global"] + RATE_AREAS:
        r = ir[area]
        flag = "GROWING/unstable" if r["growing"] else ""
        print(f"    {area:<10}{_fmt(r['tau_ms']):>10}{_fmt(r['half_life_ms']):>11}"
              f"{r['end_over_peak']:>10.2f}  {flag}")

    print("\n[2] INTRINSIC TIMESCALE  (autocorr 1/e time, cue absent, noise-driven)")
    print("    larger tau => closer to criticality")
    it = intrinsic_timescale(params, config, dt_ms=dt_ms)
    print(f"    {'area':<12}{'tau(ms)':>10}")
    for area in RATE_AREAS:
        print(f"    {area:<12}{_fmt(it[area]['tau_ms']):>10}")
    print("    -- modulatory states (expected slow) --")
    for area in MODULATOR_AREAS:
        if area in it:
            print(f"    {area:<12}{_fmt(it[area]['tau_ms']):>10}")

    print("\n[3] SUSCEPTIBILITY  (steady-state gain to a sustained cue step)")
    print("    large gain => loop_gain near 1; gain rising with amp => saturating")
    su = susceptibility(params, config, dt_ms=dt_ms)
    amps = sorted(su["_global"].keys())
    header = "".join(f"amp={a:<7}" for a in amps)
    print(f"    {'area':<10}{header}")
    for area in ["_global"] + RATE_AREAS:
        row = "".join(f"{su[area][a]:<11.3f}" for a in amps)
        print(f"    {area:<10}{row}")

    print("\n" + "=" * 68)
    print("Read-out: if [1] tau and [2] tau are ~= the leak time and [3] gains are")
    print("small (<~1) and flat, the loop is over-damped (sub-critical). Push g_bg /")
    print("recurrent gain up and re-run; the timescales and gains should climb.")
    print("=" * 68)
    return {"impulse": ir, "intrinsic": it, "susceptibility": su}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    params, config = load_bundle(path)
    dt_ms = int(config.get("dt_ms", cfg.TASK_CONFIG["dt_ms"]))
    report(params, config, dt_ms=dt_ms)


if __name__ == "__main__":
    main()
