"""Run the family's full testing_script against one checkpoint from the 10k seed sweep.

testing_script (and plotting_functions / opto_script / test_pnr) each hold their own
config_script.for_family() namespace and write every figure into the SHARED family
folder cbt_loop_noSCnoSTN/plots/{svg,png}. That would scatter this run's output into the
family root and overwrite whatever is already there, so this runner redirects all of them
into a per-run folder here before calling main().

Two things also have to be fixed up for a params-only checkpoint:
  * testing_script._load_bundle rebuilds config via init_params for a bare params dict,
    which does NOT set readout_source="thalamus" -- the analysis would then read the
    MEDULLA and every plot would be of the wrong signal. We build the bundle explicitly.
  * PARAMS_FILENAME is resolved as cfg.params_path().with_name(...), so params_path is
    redirected here too rather than into the family directory.

Usage:  python run_testing_script.py <trained_NNNN.pkl>
Writes: testing_<stem>/{svg,png}/...  plus weight_matrix.csv and pnr_test/
"""
import sys, pickle, pathlib, shutil

_here = pathlib.Path(__file__).resolve().parent
_root = next(p for p in _here.parents if (p / "config_script.py").exists())
_fam = _root / "cbt_loop_noSCnoSTN"
sys.path[:0] = [str(_root), str(_fam)]

import jax.random as jr
import config_script as _cs
import cbt_rnn as cbtl

src = pathlib.Path(sys.argv[1])
if not src.is_absolute():
    src = _here / src
if not src.exists():                       # also look in the family dir
    alt = _fam / src.name
    if alt.exists(): src = alt
if not src.exists():
    raise SystemExit(f"no such params file: {src}")

out = _here / f"testing_{src.stem}"
for sub in ("svg", "png", "pnr_test"):
    (out / sub).mkdir(parents=True, exist_ok=True)

# Build a proper {params, config} bundle with the thalamic readout selected.
with src.open("rb") as f:
    raw = pickle.load(f)
params = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
_, config = cbtl.init_params(jr.PRNGKey(_cs.SEED_CONFIG["train_seed"]), n_input=1)
config = dict(config)
config["readout_source"] = "thalamus"
bundle_name = "params_supervised_thal.pkl"      # the name testing_script looks for
with (out / bundle_name).open("wb") as f:
    pickle.dump({"params": params, "config": config}, f)
print(f"[runner] bundle -> {out / bundle_name}  (readout_source=thalamus)")

import testing_script as ts
import plotting_functions as pf
import opto_script, test_pnr

def _redirect(ns):
    ns.plots_folder = str(out)
    ns.svg_folder = str(out / "svg")
    ns.png_folder = str(out / "png")
    ns.params_path = lambda _o=out: _o / "params.pkl"   # .with_name() lands in `out`
    return ns

for mod, attr in ((ts, "cfg"), (pf, "cs"), (opto_script, "cfg"), (test_pnr, "cfg")):
    if hasattr(mod, attr):
        _redirect(getattr(mod, attr))
        print(f"[runner] redirected {mod.__name__}.{attr}")
test_pnr.PNR_DIR = str(out / "pnr_test")        # computed at import from the old folder
ts.PARAMS_FILENAME = bundle_name

print(f"[runner] all figures -> {out}")
ts.main()

n_png = len(list((out / "png").glob("*.png")))
n_svg = len(list((out / "svg").glob("*.svg")))
n_pnr = len(list((out / "pnr_test").rglob("*")))
print(f"\n[runner] wrote {n_png} png, {n_svg} svg, {n_pnr} pnr files into {out}")
