"""Parse a pci_train log and plot the PCI progression over generations.

    python pci_report.py <log_path> [--out pci_progression.png] [--title ...]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"

# gen   170  val_reward  780.90  post_PCI  825.42  spont  44.52  calm 0.0019  rho 1.102  *BEST[4941s]
LINE = re.compile(
    r"gen\s+(\d+)\s+val_reward\s+([\-\d.]+)\s+post_PCI\s+([\-\d.]+)\s+"
    r"spont\s+([\-\d.]+)\s+calm\s+([\-\d.]+)\s+rho\s+([\-\d.]+)")


def parse(log_path):
    rows = {k: [] for k in ("gen", "val", "post", "spont", "calm", "rho")}
    best_gen = None
    for line in Path(log_path).read_text().splitlines():
        m = LINE.search(line)
        if not m:
            continue
        g, v, p, s, c, r = m.groups()
        rows["gen"].append(int(g)); rows["val"].append(float(v))
        rows["post"].append(float(p)); rows["spont"].append(float(s))
        rows["calm"].append(float(c)); rows["rho"].append(float(r))
        if "*BEST" in line:
            best_gen = int(g)
    return rows, best_gen


def plot(rows, best_gen, out, title):
    g = rows["gen"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) the headline: post-PCI and net reward vs generation
    ax[0, 0].plot(g, rows["post"], "-", color="C3", label="post-PCI (stim)")
    ax[0, 0].plot(g, rows["val"], "-", color="C2", label="net reward (post − spont)")
    if best_gen is not None:
        bi = g.index(best_gen)
        ax[0, 0].scatter([best_gen], [rows["val"][bi]], color="k", zorder=5,
                         label=f"best reward (gen {best_gen})")
    ax[0, 0].set_xlabel("generation"); ax[0, 0].set_ylabel("PCI (zip bytes)")
    ax[0, 0].set_title("PCI progression"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    # (0,1) spontaneous (catch-trial) complexity
    ax[0, 1].plot(g, rows["spont"], "-", color="C0")
    ax[0, 1].set_xlabel("generation"); ax[0, 1].set_ylabel("spontaneous ΔPCI (bytes)")
    ax[0, 1].set_title("Spontaneous (catch-trial) complexity — want low"); ax[0, 1].grid(alpha=0.3)

    # (1,0) loop spectral radius (free rho)
    ax[1, 0].plot(g, rows["rho"], "-", color="C4")
    ax[1, 0].axhline(1.0, color="grey", ls=":", lw=1)
    ax[1, 0].set_xlabel("generation"); ax[1, 0].set_ylabel("rho_lin")
    ax[1, 0].set_title("Loop spectral radius (free)"); ax[1, 0].grid(alpha=0.3)

    # (1,1) baseline restlessness
    ax[1, 1].plot(g, rows["calm"], "-", color="C1")
    ax[1, 1].set_xlabel("generation"); ax[1, 1].set_ylabel("baseline activity std")
    ax[1, 1].set_title("Baseline restlessness (diagnostic)"); ax[1, 1].grid(alpha=0.3)

    last = g[-1]
    fig.suptitle(f"{title}  —  through generation {last}", y=1.0, fontsize=13)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    outp = PLOTS / out
    fig.savefig(outp, dpi=110, bbox_inches="tight")
    print(f"saved -> {outp}  ({len(g)} points, latest gen {last}, "
          f"post-PCI {rows['post'][-1]:.0f}, reward {rows['val'][-1]:.0f})")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="pci_progression.png")
    ap.add_argument("--title", default="Corticothalamic PCI training (pop=1000)")
    args = ap.parse_args()
    rows, best = parse(args.log)
    plot(rows, best, args.out, args.title)
