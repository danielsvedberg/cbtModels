import pickle as pkl
import jax.random as jr
import jax.numpy as jnp
import cbt_rnn as cbtl
import config_script as cfg
import self_timed_movement_task as stmt


def main():
    with cfg.params_path().open("rb") as f:
        bundle = pkl.load(f)
    if isinstance(bundle, dict) and "params" in bundle and "config" in bundle:
        params, config = bundle["params"], bundle["config"]
    else:
        params = bundle
        _, config = cbtl.init_params(
            jr.PRNGKey(0),
            n_c_exc=params["J_c_ee"].shape[0], n_c_inh=params["J_c_ii"].shape[0],
            n_d1=params["J_d1"].shape[0], n_d2=params["J_d2"].shape[0],
            n_snc=params["P_snc"].shape[0], n_snr=params["P_snr"].shape[0],
            n_gpe=params["J_gpe"].shape[0],
            n_t_exc=params["J_t_ee"].shape[0], n_t_inh=params["J_t_ii"].shape[0],
            n_input=1, n_output=1, noise_std=cfg.RNN_CONFIG["noise_std"],
        )

    starts = cfg.TEST_CONFIG["start_t"][:4]
    inputs, _, _ = stmt.self_timed_movement_task(
        starts, cfg.TASK_CONFIG["t_cue"], cfg.TASK_CONFIG["t_wait"],
        cfg.TASK_CONFIG["t_movement"], cfg.TASK_CONFIG["t_total"],
    )
    inputs = cbtl.match_input_channels(inputs, params)

    def run(overrides):
        cfg_o = dict(config)
        cfg_o.update(overrides)
        ys, xs, actions = cbtl.evaluate(params, cfg_o, inputs, noise_std=0.0, n_seeds=4)
        # STATE_VAR_ORDER: x_c,x_d1,x_d2,x_snc,x_gpe,x_snr,x_t,pka_d1,pka_d2,x_med
        pka_d1 = jnp.asarray(xs[7])
        pka_d2 = jnp.asarray(xs[8])
        return pka_d1, pka_d2

    sweeps = [
        ("baseline",                         {}),
        ("m_floor_a1=0.0",                   {"m_floor_a1": 0.0}),
        ("da_gain=4",                        {"da_pka_gain": 4.0}),
        ("da_gain=8",                        {"da_pka_gain": 8.0}),
        ("k_a_cap=0.3",                      {"k_a_cap": 0.3}),
        ("a1=0 + da=4",                      {"m_floor_a1": 0.0, "da_pka_gain": 4.0}),
        ("a1=0 + da=8 + k_cap=0.3",          {"m_floor_a1": 0.0, "da_pka_gain": 8.0, "k_a_cap": 0.3}),
    ]

    print(f"{'config':30s} {'pka_d1 final':>12s} {'pka_d1 max':>11s} {'pka_d2 final':>12s} {'d1 frac>0.05':>13s}")
    for name, ov in sweeps:
        d1, d2 = run(ov)
        d1f = d1[..., -1, :].mean()
        d2f = d2[..., -1, :].mean()
        print(f"{name:30s} {d1f:12.4f} {d1.max():11.4f} {d2f:12.4f} {(d1>0.05).mean():13.3f}")


if __name__ == "__main__":
    main()