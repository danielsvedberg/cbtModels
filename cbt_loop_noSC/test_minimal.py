import pickle as pkl

import jax.random as jr

import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt


def main():
    print("Loading params bundle...")
    with cfg.params_path().open("rb") as f:
        bundle = pkl.load(f)

    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        params = bundle["params"]
        config = bundle["config"]
    else:
        params = bundle
        n_c = params["J_c"].shape[0]
        n_d1 = params["J_d1"].shape[0]
        n_d2 = params["J_d2"].shape[0]
        n_snc = params["P_snc"].shape[0]
        n_snr = params["P_snr"].shape[0]
        n_gpe = params["J_gpe"].shape[0]
        n_stn = params["J_stn"].shape[0]
        n_t = params["J_t"].shape[0]
        _, config = cbtl.init_params(
            jr.PRNGKey(0),
            n_c=n_c,
            n_d1=n_d1,
            n_d2=n_d2,
            n_snc=n_snc,
            n_snr=n_snr,
            n_gpe=n_gpe,
            n_stn=n_stn,
            n_t=n_t,
            n_input=1,
            n_output=1,
            noise_std=cfg.RNN_CONFIG["noise_std"],
        )

    starts = cfg.TEST_CONFIG["start_t"][:4]
    inputs, _, _ = stmt.self_timed_movement_task(
        starts,
        cfg.TASK_CONFIG["t_cue"],
        cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"],
        cfg.TASK_CONFIG["t_total"],
    )

    print("Testing model...")
    ys, xs, actions = cbtl.evaluate(params, config, inputs, noise_std=0.0, n_seeds=4)

    print("ys:", ys.shape)
    print("n state arrays:", len(xs))
    print("state[0]:", xs[0].shape)
    print("actions:", actions.shape)
    print("Test completed successfully.")


if __name__ == "__main__":
    main()
