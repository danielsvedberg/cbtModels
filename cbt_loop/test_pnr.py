import os
import jax.numpy as jnp
import jax.random as jr
import pickle as pkl
import matplotlib.pyplot as plt
import model_functions_with_ffin as mf
import config_script_with_ffin as cs
import plotting_functions as pf


# Output directory for plots
PNR_DIR = os.path.join('plots', 'pnr_test')
os.makedirs(PNR_DIR, exist_ok=True)

def load_params():
    try:
        with open('params_nm_ffin.pkl', 'rb') as f:
            return pkl.load(f)
    except FileNotFoundError:
        print("Error: params_nm.pkl not found.")
        return None

def test_cue_magnitude(params_nm, magnitudes, n_seeds=100):
    start_t = 300
    base_inputs, _, _ = mf.self_timed_movement_task(
        jnp.array([start_t]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], cs.config['T']
    )
    
    # Broadcast (n_seeds, num_magnitudes, T, 1)
    scaled_inputs = base_inputs[0, None, :, :] * magnitudes[:, None, None]
    scaled_inputs = jnp.repeat(scaled_inputs[None, ...], n_seeds, axis=0) # (seeds, mags, T, 1)
    
    all_resp, all_mag = [], []
    all_ys_list, all_xs_list, all_zs_list = [], [], []
    for s in range(n_seeds):
        rng_keys = jr.split(jr.PRNGKey(s), len(magnitudes))
        ys, xs, zs = mf.batched_nm_rnn(
            params_nm, cs.x0, cs.z0, scaled_inputs[s], cs.config['tau_x'], cs.config['tau_z'], 
            True, None, cs.test_noise_std, rng_keys
        )
        # ys shape: (mags, T, 1)
        post_cue_ys = ys[:, start_t+cs.config['T_cue']:, 0]
        
        responded = jnp.any(post_cue_ys > 0.5, axis=1)
        max_mag = jnp.max(post_cue_ys, axis=1)
        
        all_resp.append(responded)
        all_mag.append(max_mag)
        
        all_ys_list.append(ys)
        all_xs_list.append(xs)
        all_zs_list.append(zs)
        
    prob = jnp.mean(jnp.stack(all_resp), axis=0)
    mag_mean = jnp.mean(jnp.stack(all_mag), axis=0)
    mag_sem = jnp.std(jnp.stack(all_mag), axis=0) / jnp.sqrt(n_seeds)
    
    all_ys_stacked = jnp.stack(all_ys_list)
    all_xs_stacked = tuple(jnp.stack([x[i] for x in all_xs_list]) for i in range(len(all_xs_list[0])))
    all_zs_stacked = jnp.stack(all_zs_list)
    
    return prob, mag_mean, mag_sem, all_ys_stacked, all_xs_stacked, all_zs_stacked

def test_cue_duration(params_nm, durations, n_seeds=100):
    start_t = 300
    T = cs.config['T']
    num_durations = len(durations)
    
    inputs = jnp.zeros((num_durations, T, 1))
    for i, dur in enumerate(durations):
        if dur > 0:
            inputs = inputs.at[i, start_t:start_t+int(dur), 0].set(1.0)
        
    all_resp, all_mag = [], []
    all_ys_list, all_xs_list, all_zs_list = [], [], []
    for s in range(n_seeds):
        rng_keys = jr.split(jr.PRNGKey(s + 100), num_durations)
        ys, xs, zs = mf.batched_nm_rnn(
            params_nm, cs.x0, cs.z0, inputs, cs.config['tau_x'], cs.config['tau_z'], 
            True, None, cs.test_noise_std, rng_keys
        )
        
        # Max over everything after cue start
        post_cue_ys = ys[:, start_t:, 0]
        responded = jnp.any(post_cue_ys > 0.5, axis=1)
        max_mag = jnp.max(post_cue_ys, axis=1)
        
        all_resp.append(responded)
        all_mag.append(max_mag)
        
        all_ys_list.append(ys)
        all_xs_list.append(xs)
        all_zs_list.append(zs)
        
    prob = jnp.mean(jnp.stack(all_resp), axis=0)
    mag_mean = jnp.mean(jnp.stack(all_mag), axis=0)
    mag_sem = jnp.std(jnp.stack(all_mag), axis=0) / jnp.sqrt(n_seeds)
    
    all_ys_stacked = jnp.stack(all_ys_list)
    all_xs_stacked = tuple(jnp.stack([x[i] for x in all_xs_list]) for i in range(len(all_xs_list[0])))
    all_zs_stacked = jnp.stack(all_zs_list)
    
    return prob, mag_mean, mag_sem, all_ys_stacked, all_xs_stacked, all_zs_stacked

def test_dynamic_perturbation(params_nm, perturbation_times, pert_strength=-5.0, pert_duration=5, n_seeds=100):
    start_t = 300
    base_inputs, _, _ = mf.self_timed_movement_task(
        jnp.array([start_t]), cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], cs.config['T']
    )
    T = cs.config['T']
    num_certs = len(perturbation_times)
    n_conds = num_certs + 1 # Add baseline no-cue condition at index 0
    
    stim = jnp.zeros((n_conds, T, cs.config['n_bg']))
    for i, p_time in enumerate(perturbation_times):
        stim = stim.at[i+1, p_time:p_time+pert_duration, :].set(pert_strength)
    # batched_nm_rnn expects (batch, T, cells), which is what we built.
        
    inputs = jnp.repeat(base_inputs, n_conds, axis=0) # (n_conds, T, 1)
    inputs = inputs.at[0].set(0.0) # Condition 0 is cue omitted
    
    all_resp, all_mag = [], []
    all_ys_list, all_xs_list, all_zs_list = [], [], []
    for s in range(n_seeds):
        rng_keys = jr.split(jr.PRNGKey(s + 200), n_conds)
        ys, xs, zs = mf.batched_nm_rnn(
            params_nm, cs.x0, cs.z0, inputs, cs.config['tau_x'], cs.config['tau_z'], 
            True, stim, cs.test_noise_std, rng_keys
        )
        
        post_cue_ys = ys[:, start_t:, 0]
        responded = jnp.any(post_cue_ys > 0.5, axis=1)
        # For perturbation, we want to see if the peak STILL crosses threshold
        max_mag = jnp.max(post_cue_ys, axis=1)
        
        all_resp.append(responded)
        all_mag.append(max_mag)
        
        all_ys_list.append(ys)
        all_xs_list.append(xs)
        all_zs_list.append(zs)
        
    prob = jnp.mean(jnp.stack(all_resp), axis=0)
    mag_mean = jnp.mean(jnp.stack(all_mag), axis=0)
    mag_sem = jnp.std(jnp.stack(all_mag), axis=0) / jnp.sqrt(n_seeds)
    
    all_ys_stacked = jnp.stack(all_ys_list)
    all_xs_stacked = tuple(jnp.stack([x[i] for x in all_xs_list]) for i in range(len(all_xs_list[0])))
    all_zs_stacked = jnp.stack(all_zs_list)
    
    return prob, mag_mean, mag_sem, all_ys_stacked, all_xs_stacked, all_zs_stacked

def plot_and_save(x_vals, prob, mag_mean, mag_sem, xlabel, title, filename):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Response Probability', color=color)
    ax1.plot(x_vals, prob, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Max Response Magnitude', color=color)
    ax2.plot(x_vals, mag_mean, marker='s', color=color, linewidth=2, linestyle='--')
    ax2.fill_between(x_vals, mag_mean - mag_sem, mag_mean + mag_sem, color=color, alpha=0.2)
    ax2.axhline(0.5, color='gray', linestyle=':', label='Threshold')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(PNR_DIR, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    params = load_params()
    if params is not None:
        print("Running Point-of-No-Return analyses across multiple seeds...")
        n_seeds = 150
        
        print("1. Cue Magnitude Sensitivity...")
        mags = jnp.linspace(0.0, 2.0, 21)
        m_prob, m_mag, m_sem, m_ys, m_xs, m_zs = test_cue_magnitude(params, mags, n_seeds)
        plot_and_save(mags, m_prob, m_mag, m_sem, 
                      'Cue Magnitude Multiplier', 'Sensitivity to Cue Input Gain', 'cue_magnitude.png')
        pf.plot_binned_pnr(m_ys, m_xs, m_zs, mags, 'Cue Magnitude', 'Cue Magnitude Responses', 'pnr_mag')
        pf.plot_colorbar(0, len(mags)-1, 'plasma', 'Cue Magnitude', 'pnr_mag_colorbar', [0, len(mags)-1], [f"{mags[0]:.1f}", f"{mags[-1]:.1f}"])
        
        print("2. Cue Duration Sensitivity...")
        # Include 0 duration (cue omitted) in the tests
        durs = jnp.concatenate((jnp.array([0]), jnp.arange(2, 32, 2)))
        d_prob, d_mag, d_sem, d_ys, d_xs, d_zs = test_cue_duration(params, durs, n_seeds)
        plot_and_save(durs * cs.config['dt'], d_prob, d_mag, d_sem, 
                      'Cue Duration (ms)', 'Sensitivity to Cue Duration', 'cue_duration.png')
        pf.plot_binned_pnr(d_ys, d_xs, d_zs, durs * cs.config['dt'], 'Cue Duration (ms)', 'Cue Duration Responses', 'pnr_dur')
        pf.plot_colorbar(0, len(durs)-1, 'plasma', 'Cue Duration (ms)', 'pnr_dur_colorbar', [0, len(durs)-1], [str(0.0), f"{(durs[-1]*cs.config['dt']):.1f}"])
        
        print("3. Dynamic PNR (Stop Signal Perturbation)...")
        # Start perturbing 10 steps (100ms) after cue ends (310)
        pert_times = jnp.arange(320, 600, 20) 
        pt_prob, pt_mag, pt_sem, pt_ys, pt_xs, pt_zs = test_dynamic_perturbation(params, pert_times, n_seeds=n_seeds)
        
        # Format the conditions label array (index 0 is "No Cue")
        pt_conditions = jnp.concatenate((jnp.array([0.0]), (pert_times - 300) * cs.config['dt']))
        
        # Build the perturbation boxes dictionary for shading (skip index 0)
        pert_boxes = {}
        for i, p_time in enumerate(pert_times):
            p_start = (p_time - 300) / 100  # convert timestep offset to seconds
            p_end = p_start + (5 / 100) # pert_duration is 5 timesteps 
            pert_boxes[i+1] = (p_start, p_end)
        
        plot_and_save(pt_conditions, pt_prob, pt_mag, pt_sem, 
                      'Perturbation Delay from Cue (ms)', 'Point-of-No-Return (Inhibitory Stop Signal)', 'dynamic_pnr.png')
        pf.plot_binned_pnr(pt_ys, pt_xs, pt_zs, pt_conditions, 'Perturbation Delay (ms)', 'Dynamic PNR', 'pnr_dyn', perturbation_boxes=pert_boxes)
        pf.plot_colorbar(0, len(pt_conditions)-1, 'plasma', 'Perturbation Delay (ms)', 'pnr_dyn_colorbar', [0, len(pt_conditions)-1], ["No Cue", f"{pt_conditions[-1]:.1f}"])
        
        print(f"Done! Plots saved to {PNR_DIR}")
