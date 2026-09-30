"""Three-level ramping analysis, per brain area.

The CLAUDE.md criterion "neurons must ramp up before movement" is deliberately loose.
These are three progressively stricter readings of it, meant to be reported together —
an area can pass level 1 and fail level 2, or pass both and fail level 3.

    LEVEL 1  raw increase      mean over [move-w, move) minus mean over [cue-w, cue).
                               Just: is there more activity before movement than before
                               the cue? Says nothing about the shape in between.

    LEVEL 2  ramping index     Is the rise GRADUAL across cue->movement, or a sharp jump
                               just before movement? Two numbers:
                                 linearity  - per-unit correlation with a linear template
                                              over [cue, move), averaged over units/trials.
                                              1.0 = clean monotone ramp.
                                 earliness  - area under the min-max normalized trace.
                                              0.5 = linear ramp, <0.5 = late/sharp rise,
                                              >0.5 = early rise then plateau.
                               `frac_ramping` guards against a population average that
                               looks like a ramp but is built from heterogeneous steps:
                               the fraction of individual units with linearity > 0.5.

    LEVEL 3  temporal scaling  Are the cue->movement dynamics TIMING-INVARIANT (they
                               stretch with the interval) or TIMING-DEPENDENT (a fixed
                               absolute time course)? Trials are aligned three ways and
                               we ask which alignment lets a single template explain the
                               most variance:
                                 scaled     - [cue, move] linearly warped to [0, 1]
                                 cue_locked - [cue, cue + L_min], absolute time from cue
                                 move_locked- [move - L_min, move], absolute time to move
                               scaling_index = EV(scaled) - max(EV(cue), EV(move)).
                               > 0  => dynamics scale with the interval (timing-invariant)
                               < 0  => dynamics are locked to absolute time
                               Level 3 needs a RANGE of cue->movement intervals to mean
                               anything. `interval_spread` is reported so a degenerate
                               near-constant-latency case is visible rather than silent.

All functions take `a` of shape (n_trials, T, n_units), plus per-trial `cue` and `move`
index arrays. Trials whose window is out of range are dropped.
"""
import numpy as np

__all__ = ["level1_raw_increase", "level2_ramp_index", "level3_temporal_scaling",
           "ramping_report", "format_report"]


def _valid(a, cue, move, pre):
    T = a.shape[1]
    ok = (cue - pre >= 0) & (move > cue) & (move <= T)
    return np.asarray(ok)


def level1_raw_increase(a, cue, move, w=10):
    """mean over [move-w, move) minus mean over [cue-w, cue). Per-trial, then averaged."""
    ok = _valid(a, cue, move, w)
    if not ok.any():
        return dict(delta=np.nan, pre_cue=np.nan, pre_move=np.nan, n=0)
    pre_c, pre_m = [], []
    for i in np.where(ok)[0]:
        pre_c.append(a[i, cue[i]-w:cue[i], :].mean())
        pre_m.append(a[i, max(move[i]-w, 0):move[i], :].mean())
    pre_c, pre_m = np.array(pre_c), np.array(pre_m)
    return dict(delta=float((pre_m - pre_c).mean()), pre_cue=float(pre_c.mean()),
                pre_move=float(pre_m.mean()), n=int(ok.sum()))


def _unit_linearity(seg):
    """Correlation of each unit's trace with a linear template. seg: (L, n_units)."""
    L = seg.shape[0]
    if L < 5:
        return np.full(seg.shape[1], np.nan)
    ramp = np.linspace(0.0, 1.0, L)
    r = ramp - ramp.mean()
    s = seg - seg.mean(0, keepdims=True)
    den = np.sqrt((r**2).sum() * (s**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-12, (r[:, None] * s).sum(0) / np.maximum(den, 1e-12), np.nan)


def _earliness(seg):
    """AUC of the min-max normalized population mean. 0.5 = linear ramp."""
    m = seg.mean(-1)
    lo, hi = m.min(), m.max()
    if hi - lo < 1e-12:
        return np.nan
    return float(((m - lo) / (hi - lo)).mean())


def level2_ramp_index(a, cue, move, min_len=20):
    """Gradual ramp vs sharp late jump, over [cue, move)."""
    ok = _valid(a, cue, move, 0) & ((move - cue) >= min_len)
    if not ok.any():
        return dict(linearity=np.nan, earliness=np.nan, frac_ramping=np.nan, n=0)
    lin, ear, frac = [], [], []
    for i in np.where(ok)[0]:
        seg = a[i, cue[i]:move[i], :]
        u = _unit_linearity(seg)
        u = u[~np.isnan(u)]
        if u.size:
            lin.append(u.mean()); frac.append((u > 0.5).mean())
        e = _earliness(seg)
        if not np.isnan(e):
            ear.append(e)
    return dict(linearity=float(np.mean(lin)) if lin else np.nan,
                earliness=float(np.mean(ear)) if ear else np.nan,
                frac_ramping=float(np.mean(frac)) if frac else np.nan,
                n=int(ok.sum()))


def _resample(seg, n):
    """Linearly resample (L, n_units) onto n points."""
    L = seg.shape[0]
    src = np.linspace(0.0, 1.0, L)
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, seg[:, u]) for u in range(seg.shape[1])], axis=1)


def _explained(stack):
    """Fraction of variance a single common template explains. stack: (n_trials, n, u)."""
    if stack.shape[0] < 3:
        return np.nan
    T = stack.mean(0, keepdims=True)
    resid = ((stack - T) ** 2).sum()
    total = ((stack - stack.mean()) ** 2).sum()
    return float(1.0 - resid / total) if total > 1e-12 else np.nan


def level3_temporal_scaling(a, cue, move, n_pts=60, min_len=20):
    """Do the cue->movement dynamics stretch with the interval, or track absolute time?"""
    ok = _valid(a, cue, move, 0) & ((move - cue) >= min_len)
    idx = np.where(ok)[0]
    out = dict(ev_scaled=np.nan, ev_cue_locked=np.nan, ev_move_locked=np.nan,
               scaling_index=np.nan, interval_mean=np.nan, interval_sd=np.nan,
               interval_spread=np.nan, n=int(ok.sum()))
    if idx.size < 3:
        return out
    L = (move - cue)[idx]
    Lmin = int(L.min())
    out.update(interval_mean=float(L.mean()), interval_sd=float(L.std()),
               interval_spread=float((L.max() - L.min()) / max(L.mean(), 1e-9)))
    scaled = np.stack([_resample(a[i, cue[i]:move[i], :], n_pts) for i in idx])
    cue_lk = np.stack([_resample(a[i, cue[i]:cue[i]+Lmin, :], n_pts) for i in idx])
    mv_lk = np.stack([_resample(a[i, move[i]-Lmin:move[i], :], n_pts) for i in idx])
    out.update(ev_scaled=_explained(scaled), ev_cue_locked=_explained(cue_lk),
               ev_move_locked=_explained(mv_lk))
    if not np.isnan(out["ev_scaled"]):
        out["scaling_index"] = float(out["ev_scaled"] -
                                     max(out["ev_cue_locked"], out["ev_move_locked"]))
    return out


def ramping_report(areas, cue, move, w=10, n_pts=60, min_len=20):
    """Run all three levels over a dict {area_name: (n_trials, T, n_units)}."""
    cue, move = np.asarray(cue, int), np.asarray(move, int)
    return {nm: dict(level1=level1_raw_increase(np.asarray(a), cue, move, w),
                     level2=level2_ramp_index(np.asarray(a), cue, move, min_len),
                     level3=level3_temporal_scaling(np.asarray(a), cue, move, n_pts, min_len))
            for nm, a in areas.items()}


def format_report(rep):
    """Human-readable table."""
    L = []
    any3 = next(iter(rep.values()))["level3"]
    L.append(f"cue->movement interval: {any3['interval_mean']:.0f} +- {any3['interval_sd']:.1f} steps "
             f"(spread {100*any3['interval_spread']:.1f}% of mean, n={any3['n']})")
    if any3["interval_spread"] < 0.15:
        L.append("  WARNING: interval spread < 15% -- level 3 is weakly identified here. "
                 "Widen it (vary the delay, or perturb timing) before trusting scaling_index.")
    L.append("")
    L.append(f"{'area':<11}| {'L1 delta':>9} {'pre-cue':>8} {'pre-mv':>8} "
             f"| {'L2 linty':>8} {'early':>6} {'%ramp':>6} "
             f"| {'L3 scaled':>9} {'cue-lk':>7} {'mv-lk':>7} {'index':>7}")
    L.append("-"*104)
    for nm, r in rep.items():
        a, b, c = r["level1"], r["level2"], r["level3"]
        L.append(f"{nm:<11}| {a['delta']:+9.4f} {a['pre_cue']:8.4f} {a['pre_move']:8.4f} "
                 f"| {b['linearity']:8.3f} {b['earliness']:6.3f} {100*b['frac_ramping']:5.0f}% "
                 f"| {c['ev_scaled']:9.3f} {c['ev_cue_locked']:7.3f} {c['ev_move_locked']:7.3f} "
                 f"{c['scaling_index']:+7.3f}")
    return "\n".join(L)
