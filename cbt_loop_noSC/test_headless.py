import os
import pickle as pkl

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax.random as jr

import cbt_rnn as cbtl
import sys as _sys, pathlib as _pl
_root = next(p for p in _pl.Path(__file__).resolve().parents if (p / 'config_script.py').exists())
_sys.path.insert(0, str(_root)) if str(_root) not in _sys.path else None
import config_script as _config_script
cfg = _config_script.for_family('cbt_loop_noSC')
import self_timed_movement_task as stmt


def main():
    with cfg.params_path().open("rb") as f:
        bundle = pkl.load(f)
    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        params = bundle["params"]
        config = bundle["config"]
    else:
        params = bundle
        _, config = cbtl.init_params(jr.PRNGKey(0), n_input=1)

    starts = cfg.TEST_CONFIG["start_t"][:8]
    inputs, _, _ = stmt.self_timed_movement_task(
        starts,
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )

    ys, xs, actions = cbtl.evaluate(params, config, inputs, noise_std=cfg.TEST_CONFIG["noise_std"], n_seeds=8)
    print("ys:", ys.shape, "actions:", actions.shape, "states:", len(xs))
    print("response fraction:", float(jnp.mean(actions > 0.5)))
    print("SUCCESS")


if __name__ == "__main__":
    main()
