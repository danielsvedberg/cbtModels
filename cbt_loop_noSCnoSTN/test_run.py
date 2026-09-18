"""Run the FULL testing_script suite against one long-run folder, writing every plot there.

testing_script/plotting_functions/opto_script/test_pnr all save through pf.save_fig, which
reads cs.svg_folder / cs.png_folder off the per-family config namespace. Those are plain
attributes on a namespace object, so pointing them at <run>/plots redirects the whole suite
without touching the analysis code. _load_bundle is redirected to the run's own params.pkl.

Usage:  python test_run.py --run-dir runs/seed042
"""
import argparse, pathlib, pickle, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def main(run_dir):
    run = pathlib.Path(run_dir).resolve()
    bundle_path = run / "params.pkl"
    if not bundle_path.exists():
        raise SystemExit(f"no params.pkl in {run}")
    plots = run / "plots"
    (plots / "svg").mkdir(parents=True, exist_ok=True)
    (plots / "png").mkdir(parents=True, exist_ok=True)

    import plotting_functions as pf
    import testing_script as ts
    import opto_script, test_pnr

    # redirect every save_fig consumer at this run's folder
    for mod in (pf, ts, opto_script, test_pnr):
        c = getattr(mod, "cs", None) or getattr(mod, "cfg", None)
        if c is not None:
            c.svg_folder = str(plots / "svg")
            c.png_folder = str(plots / "png")
            c.plots_folder = str(plots)

    with bundle_path.open("rb") as f:
        bundle = pickle.load(f)
    params, config = bundle["params"], bundle["config"]
    print(f"[test_run] {run.name}: seed={bundle.get('seed')} iters={bundle.get('iters')} "
          f"readout={config.get('readout_source')} -> {plots}", flush=True)

    ts._load_bundle = lambda: (params, config)     # suite loads THIS run's weights
    ts.main()

    made = sorted(p.name for p in (plots / "png").glob("*.png"))
    print(f"[test_run] {run.name}: {len(made)} plots -> {plots}/png", flush=True)
    for n in made:
        print("   ", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    main(a.run_dir)
