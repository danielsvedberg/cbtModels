import matplotlib

# Prefer the interactive Qt backend, but fall back gracefully (e.g. when a
# headless caller such as testing_script has already forced the Agg backend).
try:
    matplotlib.use('Qt5Agg')
except Exception:
    pass
#set the font to arial
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams.update({'font.size': 8})  #use 8pt font everywhere
import matplotlib.pyplot as plt
import numpy as np
#import filter for gaussian smoothing
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import cbt_loop.cbt_rnn as cl
import cbt_loop.config_script as cs
import jax.numpy as jnp
import self_timed_movement_task as stmt

dt = cs.default_config['dt']

# Display labels for state areas (STATE_AREA_ORDER names -> human-readable).
_AREA_LABELS = {
    'D1': 'dSPN',
    'D2': 'iSPN',
    'pkaD1': 'dSPN excitability',
    'pkaD2': 'iSPN excitability',
}


def _area_label(name):
    return _AREA_LABELS.get(name, name)

def plot_loss(losses_nm):
    #if losses_nm are not in the right shape, then make it the right shape
    lc_dim = np.array(losses_nm).shape
    if len(lc_dim) == 1:
        loss_curve_nm = np.array(losses_nm)
    else:
        loss_curve_nm = [loss[-1] for loss in losses_nm]

    x_axis = np.arange(len(losses_nm)) * 200
    fig, ax = plt.subplots(figsize=(3, 2.2))
    ax.plot(x_axis, np.log10(loss_curve_nm), label='NM RNN')
    ax.set_ylabel('log10(error)')
    ax.set_xlabel('iteration')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'loss_curve')


def plot_output(all_ys):
    # Plot output activity (mean ± SEM)
    #fig = plt.figure(figsize=(4, 3))
    colors = plt.cm.berlin(jnp.linspace(0, 1, all_ys.shape[1]))
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    #plot each individual trace first
    ax = axs[0]
    for i in range(all_ys.shape[0]):
        for j in range(all_ys.shape[1]):
            ax.plot(all_ys[i, j, :, 0], c=colors[j], linewidth=0.1)

    #then plot the average
    ax = axs[1]
    mean_ys, sem_ys = stmt.compute_mean_sem(all_ys)
    for i in range(mean_ys.shape[0]):
        ax.plot(mean_ys[i, :, 0], c=colors[i])
        ax.fill_between(
            jnp.arange(mean_ys.shape[1]),
            mean_ys[i, :, 0] - sem_ys[i, :, 0],
            mean_ys[i, :, 0] + sem_ys[i, :, 0],
            color=colors[i],
            alpha=0.3,
        )
    plt.title(f'Output (mean ± SEM)')
    plt.tight_layout()
    save_fig(fig, 'output_activity')


def plot_activity_by_area(all_xs, all_zs=None):
    areas = ['Cortex', 'D1', 'D2', 'SNc', 'GPe', 'STN', 'SNr', 'Thalamus', 'Medulla']
    fig, axs = plt.subplots(len(areas), 2, figsize=(6, 1.5 * len(areas)), sharex=True)
    for idx, name in enumerate(areas):
        area_activity = cl.get_brain_area(name, all_xs)
        nrn_avg = jnp.mean(area_activity, axis=3)
        # Avg across neurons
        mean_act, sem_act = stmt.compute_mean_sem(nrn_avg)
        colors = plt.cm.berlin(jnp.linspace(0, 1, mean_act.shape[0]))

        #plot individual waves
        ax = axs[idx, 0]
        for i in range(area_activity.shape[1]):
            condition_activity = area_activity[:, i, :]
            for j in range(condition_activity.shape[0]):
                ax.plot(condition_activity[j], c=colors[i], label=f'Condition {i}', linewidth=0.1)

        #plot binned waves
        ax = axs[idx, 1]
        for i in range(mean_act.shape[0]):
            ax.plot(mean_act[i], c=colors[i], label=f'Condition {i}')
            ax.fill_between(
                jnp.arange(mean_act.shape[1]),
                mean_act[i] - sem_act[i],
                mean_act[i] + sem_act[i],
                color=colors[i],
                alpha=0.3,
            )
        ax.set_title(f'{name}')

    plt.suptitle('Aligned to trial start')
    plt.tight_layout()
    save_fig(fig, 'activity_by_area')


def plot_cue_algn_activity(all_xs, noiseless=False, ys=None):
    max_T_start = jnp.max(cs.test_start_t)
    new_T = cs.config['T'] - max_T_start
    # Plot activity aligned to cue (mean ± SEM)

    '''aligned_xs = []#[[] for n in range(3)]
    aligned_zs = []
    for i in range(3):
        for seed_idx in range(cs.n_seeds):

            aligned_xs.append(stmt.align_to_cue(all_xs[i][seed_idx], cs.test_start_t, new_T=new_T))
        aligned_zs.append(stmt.align_to_cue(all_zs[seed_idx], cs.test_start_t, new_T=new_T))
    aligned_xs = [jnp.array(aligned_x) for aligned_x in aligned_xs]
    aligned_zs = jnp.array(aligned_zs)'''
    x_axis = (jnp.arange(new_T + 100) - 100) / 100

    # Brain areas plus D1/D2 PKA excitability traces (pkaD1, pkaD2).
    area_list = ['Cortex', 'D1', 'D2', 'SNc', 'GPe', 'STN', 'SNr', 'Thalamus',
                 'Medulla', 'pkaD1', 'pkaD2']
    n_area_rows = len(area_list)
    n_rows = n_area_rows + 1 if ys is not None else n_area_rows
    fig_h = 1.5 * n_rows
    fig, axs = plt.subplots(n_rows, 4, figsize=(12, fig_h), sharex=True, sharey=False)
    for idx, name in enumerate(area_list):
        area_activity = cl.get_brain_area(name, all_xs)
        area_activity = jnp.stack(
            [stmt.align_to_cue(area_activity_seed, cs.test_start_t, new_T=new_T) for area_activity_seed in area_activity]
        )

        colors = plt.cm.coolwarm(jnp.linspace(0, 1, area_activity.shape[1]))

        #first column: plot individual traces
        ax = axs[idx, 0]
        for i in range(area_activity.shape[1]): #loop through start times
            for j in range(area_activity.shape[0]): #loop through trials
                for k in range(area_activity.shape[3]): #loop through neurons
                    subset = area_activity[j, i, :, k]
                    ax.plot(x_axis, subset, c=colors[i], label=f'Condition {i}', linewidth=0.05)
        if idx == 0:
            ax.set_title('Individual traces')

        #second column: plot each neuron's trial-average activity for each condition
        ax = axs[idx, 1]
        for i in range(area_activity.shape[1]):
            for j in range(area_activity.shape[3]):  # loop through neurons
                subset = area_activity[:, i, :, j]
                mean_act, sem_act = stmt.compute_mean_sem(subset)  # (n_conditions, T)
                ax.plot(x_axis, mean_act, c=colors[i], label=f'Condition {i}', linewidth=0.1)
                ax.fill_between(
                    x_axis,
                    mean_act - sem_act,
                    mean_act + sem_act,
                    color=colors[i],
                    alpha=0.1,
                )
        if idx == 0:
            ax.set_title('trial-averaged traces')

        #third column: plot neuron-average activity across neurons for each condition
        ax = axs[idx, 2]
        for i in range(area_activity.shape[1]):
            for j in range(area_activity.shape[0]):
                subset = area_activity[j, i, :, :]
                #swap around axes to have (n_neurons, T)
                subset = jnp.swapaxes(subset, 0, 1)  # (n_neurons, T)
                mean_act, sem_act = stmt.compute_mean_sem(subset)  # (T,)

                ax.plot(x_axis, mean_act, c=colors[i], linewidth=0.2, label=f'Condition {i}')
                ax.fill_between(
                    x_axis,
                    mean_act - sem_act,
                    mean_act + sem_act,
                    color=colors[i],
                    alpha=0.1,
                )
        if idx == 0:
            ax.set_title('neuron-averaged traces')

        #fourth column: average trials and neurons
        ax = axs[idx, 3]
        for i in range(area_activity.shape[1]):
            #average across neurons
            subset = area_activity[:, i, :, :].mean(axis=-1)  # (n_trials, T)
            mean_act, sem_act = stmt.compute_mean_sem(subset)  # (T,)
            ax.plot(x_axis, mean_act, c=colors[i], label=f'Condition {i}', linewidth=0.2)
            ax.fill_between(
                x_axis,
                mean_act - sem_act,
                mean_act + sem_act,
                color=colors[i],
                alpha=0.1,
            )
        if idx == 0:
            ax.set_title('trial & neuron-averaged traces')
        #create a second axis on left, label it with the brain area
        ax2 = ax.twinx()
        ax2.set_ylabel(f'{_area_label(name)}', rotation=270, labelpad=10)
        ax2.set_yticks([])

    if ys is not None:
        ys_aligned = jnp.stack(
            [stmt.align_to_cue(ys_seed, cs.test_start_t, new_T=new_T) for ys_seed in ys]
        )
        colors = plt.cm.coolwarm(jnp.linspace(0, 1, ys_aligned.shape[1]))
        out_row = n_rows - 1

        ax = axs[out_row, 0]
        for i in range(ys_aligned.shape[1]):
            for j in range(ys_aligned.shape[0]):
                subset = ys_aligned[j, i, :, 0]
                ax.plot(x_axis, subset, c=colors[i], linewidth=0.08)

        ax = axs[out_row, 1]
        for i in range(ys_aligned.shape[1]):
            subset = ys_aligned[:, i, :, 0]
            mean_act, sem_act = stmt.compute_mean_sem(subset)
            ax.plot(x_axis, mean_act, c=colors[i], linewidth=0.25)
            ax.fill_between(
                x_axis,
                mean_act - sem_act,
                mean_act + sem_act,
                color=colors[i],
                alpha=0.15,
            )

        ax = axs[out_row, 2]
        mean_cond = jnp.mean(ys_aligned[:, :, :, 0], axis=0)
        for i in range(mean_cond.shape[0]):
            ax.plot(x_axis, mean_cond[i], c=colors[i], linewidth=0.3)

        ax = axs[out_row, 3]
        subset = ys_aligned[:, :, :, 0].reshape(-1, ys_aligned.shape[2])
        mean_act, sem_act = stmt.compute_mean_sem(subset)
        ax.plot(x_axis, mean_act, c='black', linewidth=0.5)
        ax.fill_between(
            x_axis,
            mean_act - sem_act,
            mean_act + sem_act,
            color='black',
            alpha=0.2,
        )
        ax2 = ax.twinx()
        ax2.set_ylabel('Output', rotation=270, labelpad=10)
        ax2.set_yticks([])

    plt.suptitle('Aligned to cue')
    plt.tight_layout()
    save_fig(fig, 'cue_aligned_activity', dpi=150)


def plot_response_times(valid_response_times):
    max_response_time = jnp.max(valid_response_times)
    # Plot the distribution
    beh_start_t = 0
    x_axis = jnp.linspace(0, max_response_time, 100)

    cue_start_t = 0
    cue_end_t = (cue_start_t + cs.config['T_cue']) / 100

    fig = plt.figure(figsize=(1.8, 1.8))
    plt.hist(valid_response_times, bins=20, color='blue', alpha=0.7, edgecolor='black', density=True
             )
    plt.xlabel('time after cue (s)')
    plt.ylabel('Density')
    #plt.title('Distribution of Response Times')
    plt.tight_layout()
    #set x axis lower limit to zero
    plt.xlim(0, max_response_time)
    plt.xticks([0, 3])
    plt.axvspan(3, x_axis[-1], color='green', alpha=0.2)
    plt.show()
    save_fig(fig, 'response_times_hist')
    # Sort the response times
    sorted_response_times = jnp.sort(valid_response_times)

    # Compute the cumulative proportion of responses
    cumulative_proportion = jnp.arange(1, len(sorted_response_times) + 1) / len(sorted_response_times)

    # Plot the cumulative psychometric curve
    fig = plt.figure(figsize=(1.6, 1.6))
    plt.plot(sorted_response_times, cumulative_proportion, color='black', linewidth=2)
    plt.axvspan(3, 5, color='green', alpha=0.2)
    plt.axvspan(0, cue_end_t, color='red', alpha=0.2)
    plt.xlim(0, 5)
    plt.xticks([0, 3])
    plt.yticks([0, 0.5, 1])
    plt.xlabel('Response time\nafter cue (s)')
    plt.ylabel('Proportion (CDF)')
    plt.title(' ')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'response_times_cdf')


def truncate_colormap(cmap, minval=0.0, maxval=1.0, alpha=1, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def bin_normal_data(x, target_bins=10, min_per_bin=3,
                    pct_low=1, pct_high=99):

    x = np.asarray(x)

    # Filter out NaN values first
    x_valid = x[~np.isnan(x)]

    if len(x_valid) < min_per_bin:
        return None, None, f"Not enough valid data ({len(x_valid)} values, need at least {min_per_bin})."

    # 1. Determine central range, ignoring outliers
    low, high = np.percentile(x_valid, [pct_low, pct_high])
    central = x_valid[(x_valid >= low) & (x_valid <= high)]

    # 2. Determine number of bins allowed
    max_bins_possible = len(central) // min_per_bin
    if max_bins_possible == 0:
        return None, None, "Not enough data to make even 1 bin."

    nbins = min(target_bins, max_bins_possible)
    trls_per_bin = len(x_valid) // nbins
    #sort x_valid
    x_valid.sort()

    edges = []
    edges.append(x_valid[0])
    for i in range(nbins):
        edges.append(x_valid[(i*trls_per_bin) + trls_per_bin])
    edges[-1] = x_valid[-1]


    # 3. Construct bin edges
    #edges = np.linspace(low, high, nbins + 1)

    # 4. Digitize central values only
    bin_idx = np.digitize(central, edges) - 1
    # digitize returns indices 1..nbins; shift to 0..nbins-1
    #count how many response times fit in each bin:
    bin_counts = np.histogram(x_valid, bins=edges)[0]
    #if any bincounts are zero,throw an error
    if np.any(bin_counts == 0):
        raise ValueError('Bin edges resulted in empty bins. Consider adjusting pct_low and pct_high to include more data or reducing target_bins.')

    return bin_idx, edges, None


def plot_binned_responses(all_ys, all_xs, all_zs=None, all_actions=None):
    """Bin trials by response time and plot output, per-area activity, and the
    dSPN-iSPN activity difference. D1/D2 PKA excitability (pkaD1, pkaD2) are
    plotted as their own rows.

    Response times are detected from ``all_actions`` (sampled binary actions)
    when provided, otherwise from the raw output ``all_ys``. ``all_zs`` is
    accepted for backward compatibility and ignored (the model packs all
    states into ``all_xs``).
    """
    cue_start_t = 0
    cue_end_t = (cue_start_t + cs.config['T_cue']) / 100
    beh_start_t = cs.config['T_wait'] / 100
    beh_end_t = (cs.config['T_wait'] + cs.config['T_movement']) / 100
    n_conditions = all_ys.shape[1]
    n_seeds = all_ys.shape[0]

    response_times = jnp.full((n_seeds, n_conditions), jnp.nan)  # Default to NaN if no response is detected

    # Sampled actions cross threshold; the raw probability output may not.
    detect = all_actions if all_actions is not None else all_ys

    for seed_idx in range(n_seeds):
        for condition_idx in range(n_conditions):
            cue_start = cs.test_start_t[condition_idx]# + cs.config['T_cue']
            post_cue_activity = detect[seed_idx, condition_idx, cue_start:]  # Activity after the cue start
            response_idx = jnp.argmax(post_cue_activity[:, 0] > 0.5)  # Find first timestep where response detected
            if post_cue_activity[response_idx, 0] > 0.5:
                response_times = response_times.at[seed_idx, condition_idx].set(response_idx * 0.01)

    #round the response times
    response_times = np.round(response_times, 2)
    # Define the response time bins (left closed, right open)
    _, bin_boundaries, error_msg = bin_normal_data(response_times)

    # Handle case where there's not enough data to bin
    if bin_boundaries is None:
        print(f"Warning: Cannot create binned response plot - {error_msg}")
        print(f"Valid response times: {np.sum(~np.isnan(response_times))}/{response_times.size}")
        raise Exception("Cannot create binned response plot - invalid response times.")

    bin_boundaries = np.round(bin_boundaries,2)

    bin_labels = [f'{bin_boundaries[i]}-{bin_boundaries[i + 1]}' for i in range(len(bin_boundaries) - 1)]

    # Initialize lists for binning the xs, ys data
    binned_xs = [[[] for n in range(len(all_xs))] for _ in bin_labels]
    binned_ys = [[] for _ in bin_labels]
    binned_response_times = [[] for _ in bin_labels]

    # Assign each trial to a bin based on its response time
    for seed_idx in range(n_seeds):
        aligned_xs = [stmt.align_to_cue(all_x[seed_idx], cs.test_start_t, new_T=500) for all_x in all_xs]
        aligned_ys = stmt.align_to_cue(all_ys[seed_idx], cs.test_start_t, new_T=500)

        for condition_idx in range(n_conditions):  #for all cue start times
            response_time = response_times[seed_idx, condition_idx]

            # Find the corresponding bin for the current response time
            for bin_idx, (lower, upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
                if lower <= response_time < upper:
                    for i in range(len(all_xs)):
                        binned_xs[bin_idx][i].append(aligned_xs[i][condition_idx])
                    binned_ys[bin_idx].append(aligned_ys[condition_idx])
                    binned_response_times[bin_idx].append(response_time)
                    break

    # Convert lists to arrays
    binned_xs = [[jnp.array(bin_data) for bin_data in bin_xs] for bin_xs in binned_xs]
    binned_ys = [jnp.array(bin_data) for bin_data in binned_ys]

    # Print the shapes
    print("Shape of binned_xs:", [bin_data.shape for bin_data in binned_xs[0]])
    print("Shape of binned_ys:", [bin_data.shape for bin_data in binned_ys])

    cmap = plt.cm.get_cmap('turbo')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(bin_labels) - 1)
    # Plot output activity (mean ± SEM) for each response time bin
    fig = plt.figure(figsize=(1.8, 1.5))
    # Plot the activity for each response time bin
    for bin_idx, bin_data in enumerate(binned_ys):
        if len(bin_data) == 0:  # Skip empty bins
            continue
        # Compute mean and SEM for the current bin
        mean_ys, sem_ys = stmt.compute_mean_sem(bin_data)  # Compute mean and SEM across trials

        # Plot each bin with mean ± SEM
        ax = plt.subplot(1, 1, 1)  # Plot on a single axis
        x_axis = (jnp.array(range(mean_ys.shape[0])) - 100) / 100
        color = cmap(norm(bin_idx))
        ax.plot(x_axis, mean_ys[:, 0], label=f'{bin_labels[bin_idx]}', c=color)

        # Plot the shaded region representing SEM
        ax.fill_between(
            x_axis,
            mean_ys[:, 0] - sem_ys[:, 0],
            mean_ys[:, 0] + sem_ys[:, 0],
            color=color,
            alpha=0.3,
        )
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Necessary for creating a color bar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Response time\nafter cue (s)", rotation=270, labelpad=5)
    cbar.set_ticks([0, len(bin_labels) - 1])
    cbar.set_ticklabels([str(bin_boundaries[0]), str(bin_boundaries[-1])])
    ax.set_xticks([0, 3, 6])

    ax.set_xlabel('time (s)')
    ax.set_ylabel('output activity')
    ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
    ax.axvspan(beh_start_t, x_axis[-1], color='green', alpha=0.2)
    plt.tight_layout()
    save_fig(fig, 'clue_aligned_binned_responses')

    # Plot activity in each brain area for different response time bins (mean ± SEM).
    # pkaD1 / pkaD2 are included as their own dSPN/iSPN excitability rows.
    brain_areas = ['Cortex', 'D1', 'D2', 'SNc', 'GPe', 'STN', 'SNr', 'Thalamus',
                   'Medulla', 'pkaD1', 'pkaD2']
    cmap = plt.cm.get_cmap('turbo')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(bin_labels) - 1)
    n_brain_areas = len(brain_areas)
    fig, axs = plt.subplots(n_brain_areas, 1, figsize=(1.8, 0.75 * n_brain_areas), sharex=True)
    for idx, name in enumerate(brain_areas):
        ax = axs[idx]

        # Collect the activity data from all bins and align to cue
        for bin_idx in range(len(binned_xs)):
            xs_bin = binned_xs[bin_idx]  # list of n_trials * T * n_neurons arrays
            if binned_ys[bin_idx].size == 0:  # skip empty bins
                continue

            # Get the brain area activity (already cue-aligned)
            area_activity = cl.get_brain_area(name, xs_bin)  # trials * T * N

            # Compute mean and SEM for the current bin
            mean_area_activity = jnp.mean(area_activity, axis=-1)  # trials * T
            mean_act, sem_act = stmt.compute_mean_sem(mean_area_activity)  # T
            x_axis = (jnp.array(range(mean_act.shape[0])) - 100) / 100
            mask = (x_axis > -0.5) & (x_axis < 5)
            x_axis = x_axis[mask]
            mean_act = mean_act[mask]
            sem_act = sem_act[mask]
            color = cmap(norm(bin_idx))
            ax.plot(x_axis, mean_act, label=f'{bin_labels[bin_idx]}', c=color)
            ax.fill_between(
                x_axis,
                mean_act - sem_act,
                mean_act + sem_act,
                alpha=0.3,
                color=color)

        ax.set_xticks([0, 3])
        # create a second axis on the right, label it with the brain area
        ax2 = ax.twinx()
        ax2.set_ylabel(f'{_area_label(name)}', rotation=270, labelpad=10)
        ax2.set_yticks([])

        # Add vertical lines for cue, wait, and movement phases
        ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
        ax.axvspan(beh_start_t, x_axis[-1], color='green', alpha=0.2)

        if idx == n_brain_areas - 1:
            ax.set_xlabel('time after cue (s)')
            ax.set_ylabel('activity (AU)')
    axs[0].set_title('Binned by response time')

    plt.tight_layout()
    save_fig(fig, 'binned_responses_by_area')

    #plot the ratio of d1/d2 activity
    fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.5), sharex=True)
    ax = axs

    # Collect the activity data from all bins and align to cue
    for bin_idx in range(len(binned_xs)):
        xs_bin = binned_xs[bin_idx]  # list of n_trials * T * n_neurons arrays
        if binned_ys[bin_idx].size == 0:  # skip empty bins
            continue

        # Get the brain area activity (already cue-aligned)
        d1_activity = cl.get_brain_area('D1', xs_bin)  # trials * T * N
        d2_activity = cl.get_brain_area('D2', xs_bin)  # trials * T * N

        #get rid of all negative values in mean_ratio
        mean_d1 = jnp.mean(d1_activity, axis=-1)  # trials * T
        mean_d2 = jnp.mean(d2_activity, axis=-1)  # trials * T

        ratio_activity = mean_d1 - mean_d2
        #replace all negative values with nan
        #ratio_activity = jnp.where(ratio_activity < 0, jnp.nan, ratio_activity)
        # Compute mean and SEM for the current bin
        mean_act = jnp.nanmean(ratio_activity, axis=0)  # T
        #log mean_act
        #mean_act = jnp.log(mean_act)
        #smooth w gaussian kernel
        mean_act = gaussian_filter1d(mean_act, 10)

        x_axis = (jnp.array(range(mean_act.shape[0])) - 100) / 100
        mask = (x_axis >= -0) & (x_axis < 2)
        x_axis = x_axis[mask]
        mean_act = mean_act[mask]
        #sem_act = sem_act[mask]
        color = cmap(norm(bin_idx))
        ax.plot(x_axis, mean_act, label=f'{bin_labels[bin_idx]}', c=color)
        '''ax.fill_between(
            x_axis,
            mean_act - sem_act,
            mean_act + sem_act,
            alpha=0.3,
            color=color)'''
    ax.set_xlabel('time after cue (s)')
    ax.set_xticks([0, 1, 2])
    # create a second axis on left, label it with the brain area

    ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
    ax.axvspan(0.25, 1.5, color='gray', alpha=0.2)
    #ax.axvspan(beh_start_t, x_axis[-1], color='green', alpha=0.2)
    ax.set_ylabel('dSPN - iSPN activity')
    #ax.set_yscale('symlog')
    plt.tight_layout()
    save_fig(fig, 'binned_responses_D1D2ratio')


def _plot_opto_demo(opto_ys, opto_xs, label_list, colors, title, cdf_name, area_name, newT):
    """Shared body for the opto inhibition / stimulation demo figures.

    opto_ys / opto_xs are 3-element lists [control, perturb dSPN, perturb iSPN].
    """
    brain_areas = ['Cortex', 'D1', 'D2', 'SNc', 'GPe', 'STN', 'SNr', 'Thalamus',
                   'Medulla', 'pkaD1', 'pkaD2']

    # CDF of response times.
    fig = plt.figure(figsize=(1.8, 1.8))
    plotted_any = False
    for stim_idx, label in enumerate(label_list):
        rts = cl.get_response_times_opto(opto_ys[stim_idx], cue_start=cs.opto_tstart, exclude_nan=True)
        sorted_rts = jnp.sort(rts)
        if len(sorted_rts) == 0:
            continue
        cdf = jnp.arange(1, len(sorted_rts) + 1) / len(sorted_rts)
        plt.plot(sorted_rts, cdf, marker='o', color=colors[stim_idx], alpha=0.7, label=label)
        plotted_any = True
    plt.axvspan(3, 6, color='green', alpha=0.2)
    plt.xlim(0, 6)
    plt.xticks([0, 3, 6])
    plt.yticks([0, 0.5, 1])
    plt.xlabel('Response time\nafter cue (s)')
    plt.ylabel('Proportion (CDF)')
    if plotted_any:
        plt.legend(fontsize=6)
    plt.tight_layout()
    save_fig(fig, cdf_name)

    # Per-area activity traces.
    cue_start_t = 0
    cue_end_t = (cue_start_t + cs.config['T_cue']) / 100
    beh_start_t = cs.config['T_wait'] / 100
    ops = (cs.opto_start - cs.opto_tstart) / 100
    ope = (cs.opto_end - cs.opto_tstart) / 100

    fig, axs = plt.subplots(len(brain_areas), 1, figsize=(2, 0.85 * len(brain_areas)), sharex=True)
    for idx, name in enumerate(brain_areas):
        ax = axs[idx]
        t_m = None
        for stim_idx, label in enumerate(label_list):
            area_activity = cl.get_brain_area(name, opto_xs[stim_idx]).mean(axis=-1)  # (n_seeds, T)
            mean_activity, sem_activity = stmt.compute_mean_sem(area_activity)
            nbin = len(mean_activity)
            t = jnp.linspace(-cs.opto_tstart / 100, (newT - cs.opto_tstart) / 100, nbin)
            tmask = (t > -0.5) & (t < 6)
            t_m = t[tmask]
            mean_activity = mean_activity[tmask]
            sem_activity = sem_activity[tmask]
            ax.plot(t_m, mean_activity, c=colors[stim_idx], label=label)
            ax.fill_between(t_m, mean_activity - sem_activity,
                            mean_activity + sem_activity, color=colors[stim_idx], alpha=0.3)
        ax.set_xticks([0, 3, 6])
        ax2 = ax.twinx()
        ax2.set_ylabel(_area_label(name), rotation=270, labelpad=10)
        ax2.set_yticks([])
        ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
        ax.axvspan(beh_start_t, t_m[-1], color='green', alpha=0.2)
        ax.axvspan(ops, ope, color='dodgerblue', alpha=0.2)
        if idx == 0:
            ax.set_title(title)
        if idx == len(brain_areas) - 1:
            ax.set_xlabel('time after cue (s)')
            ax.set_ylabel('activity (AU)')
    plt.tight_layout()
    save_fig(fig, area_name)


def plot_opto_inh(opto_ys, opto_xs, opto_zs=None, newT=900):
    """Control vs. inhibition of dSPN / iSPN.

    Expects opto_ys / opto_xs ordered
    [control, inh dSPN, inh iSPN, stim dSPN, stim iSPN]; only the first 3 are used.
    ``opto_zs`` is accepted for backward compatibility and ignored.
    """
    _plot_opto_demo(
        opto_ys[0:3], opto_xs[0:3],
        label_list=['control', 'inh. dSPN', 'inh. iSPN'],
        colors=['black', 'green', 'red'],
        title='Opto: dSPN / iSPN inhibition',
        cdf_name='response_times_cdf_opto_inh',
        area_name='opto_inh_demo',
        newT=newT,
    )


def plot_opto_stim(opto_ys, opto_xs, opto_zs=None, newT=900):
    """Control vs. stimulation of dSPN / iSPN.

    Expects opto_ys / opto_xs ordered
    [control, inh dSPN, inh iSPN, stim dSPN, stim iSPN]; uses control + last 2.
    ``opto_zs`` is accepted for backward compatibility and ignored.
    """
    _plot_opto_demo(
        [opto_ys[0]] + list(opto_ys[3:5]), [opto_xs[0]] + list(opto_xs[3:5]),
        label_list=['control', 'stim. dSPN', 'stim. iSPN'],
        colors=['black', 'green', 'red'],
        title='Opto: dSPN / iSPN stimulation',
        cdf_name='response_times_cdf_opto_stim',
        area_name='opto_stim_demo',
        newT=newT,
    )


def plot_opto(opto_xs, opto_zs, opto_ys, newT=900):
    """Full opto strength sweep: response-time CDFs and per-area activity.

    opto_xs / opto_ys are lists aligned with cs.stim_labels / cs.stim_strengths.
    ``opto_zs`` is accepted for backward compatibility and ignored.
    """
    sl = np.array(cs.stim_labels)
    cols = np.unique(sl)
    n_stims = len(cols)

    stim_labels = ['inh. dSPN', 'inh. iSPN', 'stim. dSPN', 'stim. iSPN']
    brain_areas = ['Cortex', 'D1', 'D2', 'SNc', 'GPe', 'STN', 'SNr', 'Thalamus',
                   'Medulla', 'pkaD1', 'pkaD2']

    cue_start_t = 0
    cue_end_t = (cue_start_t + cs.config['T_cue']) / 100
    beh_start_t = cs.config['T_wait'] / 100
    beh_end_t = cue_start_t + 4
    ops = (cs.opto_start - cs.opto_tstart) / 100
    ope = (cs.opto_end - cs.opto_tstart) / 100

    cmap = plt.cm.get_cmap('coolwarm')
    min_stim = float(np.min(cs.stim_strengths))
    max_stim = float(np.max(cs.stim_strengths))
    norm = matplotlib.colors.Normalize(vmin=min_stim, vmax=max_stim)

    # Response-time CDFs per stim family.
    fig, axs = plt.subplots(1, 4, figsize=(5.1, 1.5), sharex=True, sharey=True)
    for opidx, colname in enumerate(cols):
        opfilter = np.where(sl == colname)[0]
        ys_sub = [opto_ys[i] for i in opfilter]
        stim_mags = [cs.stim_strengths[i] for i in opfilter]
        ax = axs[opidx]
        for magidx, opmag in enumerate(stim_mags):
            rts = cl.get_response_times_opto(ys_sub[magidx], cue_start=cs.opto_tstart, exclude_nan=True)
            sorted_rts = jnp.sort(rts)
            if len(sorted_rts) == 0:
                continue
            cdf = jnp.arange(1, len(sorted_rts) + 1) / len(sorted_rts)
            ax.plot(sorted_rts, cdf, color=cmap(norm(opmag)))
        ax.axvspan(3, 6, color='green', alpha=0.2)
        ax.axvspan(ops, ope, color='blue', alpha=0.2)
        ax.axvspan(0, cue_end_t, color='red', alpha=0.2)
        ax.set_title(f'{stim_labels[opidx]}')
        if opidx == 0:
            ax.set_xlabel('Response time\nafter cue (s)')
            ax.set_ylabel('Proportion (CDF)')
        ax.set_xticks([0, 3, 6])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, 6)
    plt.tight_layout()
    save_fig(fig, 'response_times_cdf_opto')

    # Per-area activity sweep.
    fig, axs = plt.subplots(len(brain_areas), 4, figsize=(5.1, 0.9 * len(brain_areas)),
                            sharex=True, sharey='row')
    for opidx, colname in enumerate(cols):
        opfilter = np.where(sl == colname)[0]
        xs_sub = [opto_xs[i] for i in opfilter]
        stim_mags = [cs.stim_strengths[i] for i in opfilter]
        for idx, name in enumerate(brain_areas):
            ax = axs[idx, opidx]
            t_m = None
            for magidx, opmag in enumerate(stim_mags):
                area_activity = cl.get_brain_area(name, xs_sub[magidx]).mean(axis=-1)
                mean_activity, sem_activity = stmt.compute_mean_sem(area_activity)
                nbin = len(mean_activity)
                t = jnp.linspace(-cs.opto_tstart / 100, (newT - cs.opto_tstart) / 100, nbin)
                tmask = (t > -0.5) & (t < 4)
                t_m = t[tmask]
                mean_activity = mean_activity[tmask]
                sem_activity = sem_activity[tmask]
                ax.plot(t_m, mean_activity, color=cmap(norm(opmag)))
                ax.fill_between(t_m, mean_activity - sem_activity,
                                mean_activity + sem_activity, alpha=0.3,
                                color=cmap(norm(opmag)))
            if idx == 0:
                ax.set_title(f'{stim_labels[opidx]}')
            if opidx == n_stims - 1:
                ax2 = ax.twinx()
                ax2.set_ylabel(_area_label(name), rotation=270, labelpad=15)
                ax2.set_yticks([])
            ax.set_xticks([0, 3, 6])
            ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
            ax.axvspan(beh_start_t, beh_end_t, color='green', alpha=0.2)
            ax.axvspan(ops, ope, color='blue', alpha=0.1)

    ax = axs[-1, 0]
    ax.set_ylabel('activity (AU)')
    ax.set_xlabel('time (s)')
    plt.tight_layout()
    save_fig(fig, 'opto_demo')

    # Standalone colorbar for opto strength.
    fig, ax = plt.subplots(1, 1, figsize=(5, 0.675))
    colorbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    colorbar.set_label('Opto strength')
    plt.tight_layout()
    save_fig(fig, 'opto_strength_colorbar')


def plot_d1d2ratio_SNc_correlogram(d1d2_ratio, all_xs, response_times):
    # all_xs is the packed state list/tuple; SNc is pulled via get_brain_area.
    snc = cl.get_brain_area('SNc', all_xs)  # (n_seeds, n_conditions, T, N)

    # Ensure SNc has neuron axis for downstream averaging.
    if snc.ndim == 3:
        snc = snc[..., None]

    n_seeds = snc.shape[0]
    snc = jnp.stack(
        [stmt.align_to_cue(snc[seed], cs.test_start_t, new_T=500) for seed in range(n_seeds)]
    )
    snc = snc[:, :, 150:250, :].mean(axis=-1).mean(axis=-1)

    #verify d1d2_ratio, snc, and response_times all have the same shape, else throw exception
    if d1d2_ratio.shape != snc.shape or snc.shape != response_times.shape:
        raise Exception('d1d2_ratio, snc, and response_times must have the same shape')

    #get outliers in snc
    snc_z_scores = abs((snc - snc.mean()) / snc.std())
    snc_outliers = snc_z_scores > 2
    snc = jnp.where(snc_outliers, jnp.nan, snc)

    # mark response time outliers as nan in d1d2_ratio and snc
    d1d2_ratio = jnp.where(jnp.isnan(response_times), jnp.nan, d1d2_ratio)
    snc = jnp.where(jnp.isnan(d1d2_ratio), jnp.nan, snc)

    # mark snc outliers as nan in d1d2_ratio
    d1d2_ratio = jnp.where(jnp.isnan(snc), jnp.nan, d1d2_ratio)

    d1d2_ratio = d1d2_ratio[~jnp.isnan(d1d2_ratio)]
    snc = snc[~jnp.isnan(snc)]

    d1d2_ratio = np.array(d1d2_ratio)
    snc = np.array(snc)
    # verify that d1d2_ratio and snc are the same shape
    if d1d2_ratio.shape != snc.shape:
        raise Exception('d1d2_ratio and snc must have the same shape')

    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.6))
    ax.scatter(snc, d1d2_ratio, c='blue', alpha=0.1)
    # calculate and plot a linear regression line for the regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(snc, d1d2_ratio)
    # create a p value string: if p_value is 0, print p<0.001, else print p={p_value}
    if p_value < 0.001:
        p_val_str = 'p<0.001'
    else:
        p_value = round(p_value, 3)
        p_val_str = f'p={p_value}'

    print(p_value)
    # calculate a line of best fit
    # sort snc
    snc_sorted = np.sort(snc)
    line = slope * snc_sorted + intercept
    ax.plot(snc_sorted, line, c='black', label=f'y={slope:.2f}x+{intercept:.2f}')
    # above the line, plot the r2 and p value
    text_str = f'R^2={r_value:.2f}\n{p_val_str}'
    ax.text(0.9, 0.1, text_str, ha='right', va='bottom', transform=ax.transAxes)

    ax.set_ylabel('dSPN-iSPN activity')
    ax.set_xlabel('SNc activity')
    # ax.set_title('D1:D2 ratio vs. Response time')
    plt.tight_layout()
    save_fig(fig, 'd1d2_ratio_vs_SNc')


def plot_d1d2ratio_slope_correlogram(all_xs, response_times):
    t_start = 100
    t_end = 300
    d1d2_ratio = cl.get_d1_d2_ratio(all_xs, t_start, t_end, avg_time=True, remove_outliers=False)
    slopes = cl.get_slope(all_xs, t_start, t_end, avg_neurons=True, remove_outliers=False)
    # verify d1d2_ratio, slopes, and response_times all have the same shape, else throw exception
    if d1d2_ratio.shape != slopes.shape or slopes.shape != response_times.shape:
        raise Exception('d1d2_ratio, slopes, and response_times must have the same shape')

    # mark response time outliers as nan in d1d2_ratio and snc
    d1d2_ratio = jnp.where(jnp.isnan(response_times), jnp.nan, d1d2_ratio)
    slopes = jnp.where(jnp.isnan(response_times), jnp.nan, slopes)
    d1d2_ratio = d1d2_ratio[~jnp.isnan(d1d2_ratio)]
    slopes = slopes[~jnp.isnan(slopes)]
    # for i in [0]#,1]:
    #slopes = jnp.where(jnp.isnan(d1d2_ratio), jnp.nan, slopes)
    #d1d2_ratio = jnp.where(jnp.isnan(slopes), jnp.nan, d1d2_ratio)

    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.5), sharey=True)

    d1d2_copy = np.array(d1d2_ratio).flatten()
    slope = np.array(slopes).flatten()

    # ax = axs[i]
    ax.scatter(slope, d1d2_copy, c='blue', alpha=0.1)
    # calculate and plot a linear regression line for the regression
    sl, intercept, r_value, p_value, std_err = stats.linregress(slope, d1d2_copy)
    # create a p value string: if p_value is 0, print p<0.001, else print p={p_value}
    if p_value < 0.001:
        p_val_str = 'p<0.001'
    else:
        p_value = round(p_value, 3)
        p_val_str = f'p={p_value}'

    print(p_value)
    # calculate a line of best fit
    # sort snc
    slope_sorted = np.sort(slope)
    line = sl * slope_sorted + intercept
    ax.plot(slope_sorted, line, c='black', label=f'y={sl:.2f}x+{intercept:.2f}')
    # above the line, plot the r2 and p value
    text_str = f'R^2={r_value:.2f}\n{p_val_str}'
    ax.text(0.05, 0.95, text_str, ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('cortical ramp slope')
        # ax.set_title(f'{brain_areas[i]}')
    # ax.set_title('D1:D2 ratio vs. Response time')
    ax.set_ylabel('dSPN / iSPN activity')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'd1d2_ratio_vs_ramp_slopes')


def plot_ratio_rt_correlogram(d1d2_ratio, response_times):
    # plot a correlogram of the d1d2 ratio and response times,
    # assuming both are arrays of equal length
    # mark all entries in d1d2 ratio that are nan in response times as nan
    d1d2_ratio = jnp.where(jnp.isnan(response_times), jnp.nan, d1d2_ratio)
    # mark all entries in response times that are nan in d1d2 ratio as nan
    response_times = jnp.where(jnp.isnan(d1d2_ratio), jnp.nan, response_times)

    # remove all nan values from d1d2_ratio and response_times
    d1d2_ratio = d1d2_ratio[~jnp.isnan(d1d2_ratio)]
    response_times = response_times[~jnp.isnan(response_times)]
    # convert response times and d1d2_ratio to numpy arrays
    response_times = np.array(response_times)
    d1d2_ratio = np.array(d1d2_ratio)

    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.5))
    ax.scatter(response_times, d1d2_ratio, c='blue', alpha=0.1)
    # calculate and plot a linear regression line for the regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(response_times, d1d2_ratio)
    # create a p value string: if p_value is 0, print p<0.001, else print p={p_value}
    if p_value < 0.001:
        p_val_str = 'p<0.001'
    else:
        p_value = round(p_value, 3)
        p_val_str = f'p={p_value}'

    print(p_value)
    # calculate a line of best fit
    line = slope * response_times + intercept
    ax.plot(response_times, line, c='black', label=f'y={slope:.2f}x+{intercept:.2f}')
    # above the line, plot the r2 and p value
    text_str = f'R^2={r_value:.2f}\n{p_val_str}'
    ax.text(0.95, 0.95, text_str, ha='right', va='top', transform=ax.transAxes)
    # set the y limit from 0 to 5
    # ax.set_ylim(2, 5)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylabel('dSPN / iSPN activity')
    ax.set_xlabel('response time after cue (s)')
    # ax.set_title('D1:D2 ratio vs. Response time')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'd1d2_ratio_vs_response_time')


def plot_loss_function():
    """Illustrate the STMT target window: reward 1 inside [3, 6] s, 0 elsewhere."""
    fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
    x = np.linspace(-0.1, 6, 100)
    y = ((x >= 3) & (x <= 6)).astype(float)
    ax.plot(x, y, c='darkgreen')
    ax.axvspan(0, 0.1, color='red', alpha=0.2)
    ax.axvspan(3, 6, color='green', alpha=0.1)
    ax.set_xlabel('time after cue (s)')
    ax.set_ylabel('reward')
    ax.set_xticks([0, 3, 6])
    plt.tight_layout()
    save_fig(fig, 'loss_function')


def plot_loss_function_adaptive():
    fig, axs = plt.subplots(1, 2, figsize=(2.6, 1.4), sharey=True)
    # make an array of x values from -0.1
    x = np.linspace(-0.1, 6, 100)

    ax = axs[0]
    mask = (x >= 3) & (x <= 3.4)
    # each y value should be zero
    y = np.zeros(100)
    #set all values of y where mask is true to 1
    y[mask] = 1
    ax.plot(x, y, c='darkgreen')
    ax.axvspan(0, 0.1, color='red', alpha=0.2)
    ax.axvspan(3, 6, color='green', alpha=0.1)

    # plot a bell curve centered at 4.2
    x0 = np.linspace(0, 6, 100)
    y = np.exp(-(x0 - 2.2) ** 2 / 0.1)
    ax.plot(x0, y, c='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('time after cue (s)')
    ax.set_ylabel('output')
    ax.set_xticks([0, 3, 6])

    ax = axs[1]
    # get indices of x greater than 3 and less than 3.4
    mask = (x >= 4) & (x <= 4.4)
    #each y value should be zero
    y = np.zeros(100)
    #set all values of y where mask is true to 1
    y[mask] = 1
    ax.plot(x, y, c='darkgreen')
    ax.axvspan(0, 0.1, color='red', alpha=0.2)
    ax.axvspan(3, 6, color='green', alpha=0.1)

    #plot a bell curve centered at 4.2
    x0 = np.linspace(0, 6, 100)
    y = np.exp(-(x - 4.2) ** 2 / 0.1)
    ax.plot(x0, y, c='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('time after cue (s)')
    #ax.set_ylabel('output target')
    ax.set_xticks([0, 3, 6])
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'adaptive_loss_function')


def save_fig(fig, name, dpi=900):
    svgname = cs.svg_folder + '/' + name + '.svg'
    pngname = cs.png_folder + '/' + name + '.png'
    fig.savefig(svgname)
    fig.savefig(pngname, dpi=dpi)

def plot_binned_pnr(all_ys, all_xs, all_zs=None, conditions=None, xlabel='', title='',
                    filename='pnr', perturbation_boxes=None):
    """
    Similar to plot_binned_responses, but bins correspond to experimental conditions directly.
    all_ys: (n_seeds, n_conditions, T, 1)
    all_xs: tuple of (n_seeds, n_conditions, T, N) arrays
    all_zs: accepted for backward compatibility and ignored.
    """
    n_conditions = len(conditions)
    cmap = plt.cm.get_cmap('plasma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_conditions - 1)
    
    cue_start_t = 0
    cue_end_t = (cue_start_t + cs.config['T_cue']) / 100
    beh_start_t = cs.config['T_wait'] / 100
    
    # 1. Plot Output Activity (ys)
    fig_y, ax_y = plt.subplots(figsize=(2.5, 2))
    for c_idx in range(n_conditions):
        ys = all_ys[:, c_idx]
        mean_ys, sem_ys = stmt.compute_mean_sem(ys)
        x_axis = (jnp.arange(mean_ys.shape[0]) - 300) / 100
        color = cmap(norm(c_idx))
        
        ax_y.plot(x_axis, mean_ys[:, 0], color=color)
        ax_y.fill_between(x_axis, mean_ys[:, 0] - sem_ys[:, 0], mean_ys[:, 0] + sem_ys[:, 0], color=color, alpha=0.3)
        
        if perturbation_boxes is not None and c_idx in perturbation_boxes:
            p_start, p_end = perturbation_boxes[c_idx]
            ax_y.axvspan(p_start, p_end, color=color, alpha=0.15)
        
    ax_y.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
    ax_y.axvspan(beh_start_t, x_axis[-1], color='green', alpha=0.2)
    ax_y.set_xticks([0, 3, 6])
    ax_y.set_xlabel('time (s)')
    ax_y.set_ylabel('output activity')
    ax_y.set_title(title)
    
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_y)
    cbar.set_label(xlabel, rotation=270, labelpad=15)
    cbar.set_ticks([0, n_conditions - 1])
    # Format tick labels to 1 decimal place if floats, else string
    val_start = f"{conditions[0]:.1f}" if isinstance(conditions[0], (float, jnp.floating)) else str(conditions[0])
    val_end = f"{conditions[-1]:.1f}" if isinstance(conditions[-1], (float, jnp.floating)) else str(conditions[-1])
    cbar.set_ticklabels([val_start, val_end])
    
    plt.tight_layout()
    save_fig(fig_y, f'{filename}_outputs')
    
    # 2. Plot Brain Area Activities (incl. Medulla and D1/D2 PKA excitability rows)
    brain_areas = ['Cortex', 'D1', 'D2', 'SNc', 'GPe', 'STN', 'SNr', 'Thalamus',
                   'Medulla', 'pkaD1', 'pkaD2']
    fig_a, axs_a = plt.subplots(len(brain_areas), 1,
                                figsize=(1.8, 0.75 * len(brain_areas)), sharex=True)

    for idx, name in enumerate(brain_areas):
        ax = axs_a[idx]
        x_ax_filt = None
        for c_idx in range(n_conditions):
            xs_c = [x[:, c_idx] for x in all_xs]

            # get_brain_area returns (seeds, T, N)
            act = cl.get_brain_area(name, xs_c)  # trials * T * N

            # Compute mean and SEM for the current condition
            mean_act_n = jnp.mean(act, axis=-1) # average over neurons -> (seeds, T)
            mean_act, sem_act = stmt.compute_mean_sem(mean_act_n) # (T,)

            x_axis = (jnp.arange(mean_act.shape[0]) - 300) / 100
            mask = (x_axis > -0.5) & (x_axis < 5)
            x_ax_filt = x_axis[mask]
            color = cmap(norm(c_idx))

            ax.plot(x_ax_filt, mean_act[mask], color=color)
            ax.fill_between(x_ax_filt, mean_act[mask]-sem_act[mask], mean_act[mask]+sem_act[mask], color=color, alpha=0.3)

            if perturbation_boxes is not None and c_idx in perturbation_boxes:
                p_start, p_end = perturbation_boxes[c_idx]
                ax.axvspan(p_start, p_end, color=color, alpha=0.15)

        ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
        ax.axvspan(beh_start_t, x_ax_filt[-1], color='green', alpha=0.2)
        ax.set_xticks([0, 3])

        ax2 = ax.twinx()
        ax2.set_ylabel(_area_label(name), rotation=270, labelpad=10)
        ax2.set_yticks([])

        if idx == len(brain_areas) - 1:
            ax.set_xlabel('time after cue (s)')

    axs_a[0].set_title(title)

    plt.tight_layout()
    save_fig(fig_a, f'{filename}_areas')

    # 3. Plot D1/D2 Ratio Plot
    fig_ratio, ax_ratio = plt.subplots(figsize=(1.8, 1.5))
    for c_idx in range(n_conditions):
        xs_c = [x[:, c_idx] for x in all_xs]

        d1_act = cl.get_brain_area('D1', xs_c)
        d2_act = cl.get_brain_area('D2', xs_c)
        
        mean_d1 = jnp.mean(d1_act, axis=-1)
        mean_d2 = jnp.mean(d2_act, axis=-1)
        ratio_act = mean_d1 - mean_d2 # (seeds, T)
        
        mean_act = jnp.nanmean(ratio_act, axis=0)
        mean_act = gaussian_filter1d(mean_act, 10)
        
        x_axis = (jnp.arange(mean_act.shape[0]) - 300) / 100
        mask = (x_axis >= -0.1) & (x_axis < 2)
        x_ax_filt = x_axis[mask]
        color = cmap(norm(c_idx))
        
        ax_ratio.plot(x_ax_filt, mean_act[mask], color=color)
        
        if perturbation_boxes is not None and c_idx in perturbation_boxes:
            p_start, p_end = perturbation_boxes[c_idx]
            ax_ratio.axvspan(p_start, p_end, color=color, alpha=0.15)
            
    ax_ratio.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
    ax_ratio.axvspan(0.25, 1.5, color='gray', alpha=0.2)
    ax_ratio.set_xticks([0, 1, 2])
    plt.tight_layout()
    save_fig(fig_ratio, f'{filename}_ratio')

def plot_colorbar(vmin, vmax, cmap_name, label, filename, ticks=None, ticklabels=None):
    """
    Creates and saves a standalone colorbar.
    """
    fig, ax = plt.subplots(figsize=(1, 3))
    cmap = plt.cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_label(label, rotation=270, labelpad=15)
    
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
        
    plt.tight_layout()
    save_fig(fig, filename)
