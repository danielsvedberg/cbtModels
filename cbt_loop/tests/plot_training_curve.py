"""Plot the training curve (reward / loss / entropy / norm_resp_time vs step) from
a train_hybrid.py stdout log.

The log prints one line per 200 steps like:
  step 6400, loss: 1.03, reward: 0.4079, ..., entropy: 0.0993, ..., norm_resp_time: 0.148, ...

Usage:  python cbt_loop/tests/plot_training_curve.py [log_path]
(default log_path is this session's scratchpad hybrid-scratch log.)
"""
import re, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_LOG = ("/tmp/claude-1000/-home-dsvedberg-Documents-CodeVault-cbtModels/"
               "02953a08-a57a-4eda-b9df-a53339ae0f03/scratchpad/train_hybrid_scratch2.log")
LOG = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG)
OUT = pathlib.Path(__file__).resolve().parent / "plots" / "train_hybrid_curve.png"

pat = re.compile(
    r"step\s+(\d+),.*?loss:\s*([-\d.]+).*?reward:\s*([-\d.]+).*?"
    r"entropy:\s*([-\d.]+).*?norm_resp_time:\s*([-\d.]+)")
step, loss, reward, entropy, nrt = [], [], [], [], []
for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if m:
        step.append(int(m.group(1))); loss.append(float(m.group(2)))
        reward.append(float(m.group(3))); entropy.append(float(m.group(4)))
        nrt.append(float(m.group(5)))
print(f"parsed {len(step)} points, last step {step[-1] if step else 'NA'}, "
      f"last reward {reward[-1] if reward else 'NA'}")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
panels = [("reward", reward, "#2a6fdb"), ("loss", loss, "#d1495b"),
          ("entropy", entropy, "#4c9a5a"), ("norm_resp_time", nrt, "#e8a33d")]
for a, (name, y, c) in zip(ax.flat, panels):
    a.plot(step, y, lw=1.8, color=c)
    a.set_title(name); a.set_xlabel("step"); a.set_ylabel(name); a.grid(alpha=.25)
ax[0, 0].axhline(0.88, ls="--", color="grey", alpha=.6)
ax[0, 0].text(step[0] if step else 0, 0.88, " last-session hybrid 0.88", fontsize=8, color="grey", va="bottom")

fig.suptitle(f"train_hybrid --init scratch (mass-action PKA-as-b) — {step[-1] if step else 0} steps",
             fontsize=12)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
