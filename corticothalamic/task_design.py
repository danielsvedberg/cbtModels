"""How should the self-timed task be parameterized so that reward actually
measures self-timing?

    python corticothalamic/task_design.py

The degenerate strategy is a single FIXED response time that ignores the cue.
It scores well whenever the per-trial reward windows overlap. For
t_start ~ U[a,b] (range R) and window width W, a fixed time T lands in-window iff
    t_start in (T - wait - W,  T - wait]
so the best fixed time achieves a fraction

    ceiling ~ min(W, R) / R          ( = W/R when W < R )

Current task: W = t_movement = 300, R = 400-50 = 350  ->  ceiling 0.857.
Measured empirically: 0.870. The metric is therefore nearly useless: a model with
zero timing ability scores 0.87.

Constraint: the whole trial must fit, i.e.
    t_start_max + t_cue + t_wait + W <= T_total
so with t_start_min fixed, R = (T_total - t_cue - t_wait - W) - t_start_min.
"""
import numpy as np

T_CUE, T_WAIT = 10, 300
T_START_MIN = 50


def ceiling_uniform(W, R):
    """Best achievable in-window fraction for a cue-ignoring fixed response."""
    return min(W, R) / R if R > 0 else 1.0


def ceiling_empirical(t_starts, W, wait=T_CUE + T_WAIT, T_total=None):
    """Exact ceiling for a concrete t_start sample (max over integer times)."""
    lo = t_starts + wait
    hi = lo + W
    hi_t = int(T_total if T_total else hi.max() + 1)
    return max(float(np.mean((T >= lo) & (T < hi))) for T in range(hi_t))


def main():
    print("=" * 74)
    print("CURRENT TASK")
    print("=" * 74)
    R = 400 - T_START_MIN
    print(f"  T_total=1000  t_wait=300  window W=300  t_start in [50,400) -> R={R}")
    print(f"  fixed-time ceiling = W/R = {ceiling_uniform(300, R):.3f}  (measured 0.870)")
    print("  => a model with NO timing ability scores ~0.87. Metric is degenerate.\n")

    print("=" * 74)
    print("OPTION A — keep T_total=1000, narrow the window")
    print("=" * 74)
    print(f"{'W':>6}{'t_start max':>13}{'R':>7}{'ceiling':>10}   note")
    for W in (300, 200, 150, 100, 50, 30):
        ts_max = 1000 - T_CUE - T_WAIT - W
        R = ts_max - T_START_MIN
        c = ceiling_uniform(W, R)
        note = "degenerate" if c > 0.5 else ("usable" if c > 0.15 else "good")
        print(f"{W:>6}{ts_max:>13}{R:>7}{c:>10.3f}   {note}")

    print("\n" + "=" * 74)
    print("OPTION B — lengthen the trial (more compute/trial), keep a wider window")
    print("=" * 74)
    print(f"{'T_total':>9}{'W':>6}{'t_start max':>13}{'R':>7}{'ceiling':>10}{'cost':>8}")
    for T_total in (1000, 1250, 1500, 2000):
        for W in (100, 150):
            ts_max = T_total - T_CUE - T_WAIT - W
            R = ts_max - T_START_MIN
            c = ceiling_uniform(W, R)
            print(f"{T_total:>9}{W:>6}{ts_max:>13}{R:>7}{c:>10.3f}"
                  f"{T_total/1000:>7.2f}x")

    print("\n" + "=" * 74)
    print("RECOMMENDATION (verified on the exact sampled t_start below)")
    print("=" * 74)
    rng = np.random.default_rng(13)
    for label, T_total, W in (("current", 1000, 300),
                              ("A: narrow window", 1000, 50),
                              ("B: longer trial", 1500, 100)):
        ts_max = T_total - T_CUE - T_WAIT - W
        ts = rng.integers(T_START_MIN, ts_max, size=100)
        c = ceiling_empirical(ts, W, T_total=T_total)
        print(f"  {label:<20} T_total={T_total:<5} W={W:<4} "
              f"t_start in [{T_START_MIN},{ts_max})  ->  ceiling {c:.3f}")
    print("\n  Both A and B drop the cue-ignoring ceiling from 0.87 to <0.15, so")
    print("  reward becomes a real measure of timing. A is free (same trial length);")
    print("  B keeps a more forgiving window but costs 1.5x compute per trial.")
    print("\n  NOTE: a narrower window also demands a higher in-window firing rate to")
    print("  guarantee a response: P(fire) = 1-(1-p)^W, so W=50 needs p>=0.088 for")
    print("  99% vs p>=0.015 at W=300. Keep that in mind when reading reward.")


if __name__ == "__main__":
    main()
