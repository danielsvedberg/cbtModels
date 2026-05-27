# AGENTS.md

## Big Picture
- This repo models self-timed movement with a JAX/Optax multiregion RNN (cue -> wait -> movement), then analyzes response timing and circuit activity.
- The model is implemented in `cbt_rnn.py`, with `config_script.py` for configuration and `plotting_functions.py` for visualization.
- Entrypoints (no CLI): `training_script.py`, `testing_script.py`, `opto_script.py`, `test_pnr.py`, `train_from_pkl.py`. Headless/minimal test scripts: `test_headless.py`, `test_minimal.py`.
- Checkpoints are root pickles: `params_nm.pkl`.

## Model Intent
- The RNN models basal ganglia-thalamocortical loop with regions (cortex, striatum D1/D2, SNc, GPe, STN, SNr, thalamus)
- The RNN connectivity is as follows: 
- Cortex takes input from cue, as well as from thalamus. Cortex has recurrent connections. Cortex outputs to striatum (D1/D2 SPNs), STN, and thalamus. Cortex→STN is constrained excitatory (hyperdirect pathway). The behavioral readout from cortex must be excitatory. 
- Striatum (D1/D2) outputs connections to SNc, and striatum has recurrence. D1 neurons output to SNr. D2 Neurons output to GPe. All outputs from striatum are inhibitory
- GPe outputs to STN, and has recurrence. All output from GPe is inhbitory, including recurrence. 
- STN outputs to GPe, SNc, and SNr, and has recurrence. Connections are excitatory. 
- SNr outputs to thalamus, connections are inhibitory. 
- Thalamus outputs to cortex, and has recurrence. 
- The RNN ultimately is trained to perform a simulation of the "self-timed movement task", where mice hear a brief cue and must wait 3 seconds before performing a brief action to receieve reward
- An assumption we make is that cues are received through cortex as a pulse increase in activity, and a pulse increase in activity from certain thalamic cells signals movement; the network must learn to transform the cue into the correct temporal dynamics to reach the movement threshold at the right time.
- An assumption we make is that the action performed to receieve reward can be represented as a brief/binary signal read out from cortex. 
- An assumption we make is that no continuous external input is needed after the cue; the network must learn to generate the correct temporal dynamics internally to perform the response.
- The scientific (wet-lab) model/understanding we have is that when a brief cue for a delayed movement is given, the coordinated network dynamics evolve to reach a decision threshold at the correct time, and that neuromodulatory signals (dopamine) modulates this timing.
- Neuromodulatory signals regulate the activation of Protein Kinase A (PKA), which increases excitability of SPNs (both D1 and D2)--the excitability of SPNs is tied to the levels of activated PKA.
- In the absence of regulation, PKA activation "leaks" downwards in all SPNs. The half-life of "leaky" PKA de-activation is about 10 seconds.
- In D1SPNs, dopamine activates the D1 receptor, which activates PKA. This D1R-PKA activation pathway is inhibited by a tonic adenosine signal mediated by adenosine receptor 1 (A1R).
- In D2SPNs, a tonic adenosine signal activates PKA (via A2R). This A2R-PKA activation pathway is inhibited by dopaminergic activation of the D2 receptor.
- PKA is implemented as state variables `pka_d1` / `pka_d2` (per-cell, shape `(n_d1,)` / `(n_d2,)`). D1 target = `nln(k_d1r * DA - k_a1r)`; D2 target = `nln(k_a2r - k_d2r * DA)`. Dynamics are asymmetric: activation uses `tau_pka_rise = 20` steps (~200 ms) and deactivation uses `tau_pka_fall = 1440` steps (10 s half-life), selected per-cell by `jnp.where(target > pka, ...)`. PKA enters `bg_nln(x, pka)` as a rheobase shift: higher PKA → lower threshold → more excitable.
- The primary testable hypothesis is that the balance of d1/d2 activity (and their modulation by dopamine) is critical for timing and perhaps the generation of movement-driving ramping activity, and that optogenetic stimulation of these populations will shift timing in predictable ways.

## Architecture and Data Flow
- `config_script.py` defines global `config`, `optimizer`, and opto stim vectors; creates `plots/` at import time.
- Task builders in `self_timed_movement_task.py` (`self_timed_movement_task`, `pavlovian_stmt`, `hybrid_stmt`) emit `(inputs, targets, masks)` with shape `(n_conditions, T, 1)`; this is the shared dataset contract.
- `stmt.fit_rnn_reinforce()` trains with REINFORCE + `lax.scan` and returns `(best_params, losses, rewards)`.
- `cbtl.evaluate()` runs many random seeds and returns `(all_ys, all_xs, all_as)`.
- `opto_script.py` builds temporal stimulation windows (`opto_start/opto_end`) over `spatial_stim_list` and calls `cbtl.batched_rnn`.

## Integrations
- Environment is in `environment.yml` (notably JAX 0.4.35, Optax 0.2.4, Matplotlib, SciPy, PyQt5).
- Default plotting backend is interactive `Qt5Agg`; CI/headless runs must override to `Agg` as in `test_headless.py`.
- Randomness is explicitly seed-driven (`jr.PRNGKey(seed)` loops + split keys); preserve key shapes when modifying batched code.
- No prior AI instruction files were found by the requested glob search; use this file as the agent baseline.
