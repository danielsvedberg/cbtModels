"""Port the FULL max-PCI corticothalamic loop (all 17 blocks) into a fresh noSCnoSTN
init as the starting thalamocortical weights.

Now that corticothalamic and noSCnoSTN share pool sizes (cortex 10/10/10, thalamus
10/5) AND the same clip-based Dale (stmt.exc=relu / stmt.inh=min0) with the same 17-block
sign convention, the loop transplants 1:1: copying the raw weight arrays reproduces the
max-PCI loop's effective (clipped) weights exactly -- no sign-projection loss.

Only the 17 loop blocks are replaced; the basal-ganglia / medulla / neuromodulator
blocks keep noSCnoSTN's own (now log-normal, Dale-signed) init. The ported loop is left
AS-IS (not renormalized) so its trained operating point is preserved.

Output: a {params, config} bundle usable as a training init (e.g. via train_from_pkl).
"""
import pickle as pkl
import sys
import pathlib

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import cbt_rnn as cbtl
_root = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "config_script.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import config_script as _config_script
import loop_init

cfg = _config_script.for_family("cbt_loop_noSCnoSTN")

SRC = _root / "corticothalamic" / "params_pci.pkl"
OUT = pathlib.Path(__file__).resolve().parent / "params_pci_loop_init.pkl"

# self-recurrence blocks whose diagonal the forward zeros (no_autapse)
_DIAG = ("J_cU", "J_cL", "J_c_ii", "J_t_ee", "J_t_ii")


def _rho(params, config, clip):
    """Loop rho. clip=False -> sign*|w| (loop_init); clip=True -> the real forward
    effective weights (exc=relu, inh=min0)."""
    S = (config["n_c_U"], config["n_c_L"], config["n_c_inh"],
         config["n_t_exc"], config["n_t_inh"])
    tau = config.get("tau_c", config.get("tau_ctx", 7.0))
    p = dict(params)
    for k in _DIAG:  # match no_autapse
        m = np.asarray(p[k]); p[k] = m * (1.0 - np.eye(m.shape[0]))
    if not clip:
        return loop_init.spectral_radius(p, *S, tau)
    sizes = [("cU", S[0]), ("cL", S[1]), ("cI", S[2]), ("tE", S[3]), ("tI", S[4])]
    idx = {}; off = 0
    for n, s in sizes:
        idx[n] = slice(off, off + s); off += s
    W = np.zeros((off, off))
    for post, pre, key, sign in loop_init.LOOP_EDGES:
        w = np.asarray(p[key]); eff = np.clip(w, 0, None) if sign > 0 else np.clip(w, None, 0)
        W[idx[post], idx[pre]] = eff
    lam = np.linalg.eigvals(W)
    return float(np.max(np.abs((1.0 - 1.0 / tau) + lam / tau)))


def main():
    params, config = cbtl.init_params(jr.PRNGKey(cfg.TRAINING_CONFIG["seed"]), n_input=2)
    params = dict(params)

    with SRC.open("rb") as f:
        src = pkl.load(f)
    sp, sconf = src["params"], src["config"]
    print(f"source max-PCI loop: N_t={sconf['n_t_exc']}/{sconf['n_t_inh']}, "
          f"clip-aware rho={_rho(sp, sconf, clip=True):.3f} "
          f"(abs {_rho(sp, sconf, clip=False):.3f})")

    for key in loop_init.LOOP_BLOCKS:
        assert params[key].shape == np.asarray(sp[key]).shape, \
            (key, params[key].shape, np.asarray(sp[key]).shape)
        params[key] = jnp.asarray(np.asarray(sp[key]))
    print(f"ported {len(loop_init.LOOP_BLOCKS)}/17 loop blocks (raw copy; both models "
          f"clip-Dale, so effective weights identical)")

    print(f"noSCnoSTN loop after port: clip-aware rho={_rho(params, config, clip=True):.3f} "
          f"(abs {_rho(params, config, clip=False):.3f}); BG blocks = native log-normal init")

    with OUT.open("wb") as f:
        pkl.dump({"params": params, "config": config}, f)
    print(f"saved -> {OUT}")
    print("Train from it via train_from_pkl.py (point params_path at this bundle).")


if __name__ == "__main__":
    main()
