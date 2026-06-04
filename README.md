# cbtModels — cortico-basal-ganglia-thalamic (CBT) loop models

Multiregion RNN models of the cortico-basal-ganglia-thalamic loop, trained on
self-timed-movement and Pavlovian tasks. Three model families differ only by
which areas they include:

| family | extra areas |
|---|---|
| `cbt_loop_noSCnoSTN` | — (reference) |
| `cbt_loop_noSC` | + subthalamic nucleus (STN) |
| `cbt_loop` (full) | + STN + superior colliculus (SC) |

## Cortical architecture: split descending PT populations (cU / cL)

The cortical **excitatory** population is split into two distinct pyramidal-tract
(PT)-like projection populations — **cU** ("upper") and **cL** ("lower") — plus a
shared inhibitory interneuron pool **c_inh**. This is motivated by:

> Economo, Michael N., et al. "Distinct descending motor cortex pathways and
> their roles in movement." *Nature* 563.7729 (2018): 79–84.

Economo et al. show that motor cortex contains two intermingled but distinct
classes of layer-5 PT neurons with different long-range projection targets and
different roles in movement: one class projecting predominantly to **thalamus**
(and other telencephalic/forebrain targets), and a separate class projecting to
**medulla** to drive movement. Treating the cortical output as a single
homogeneous pool obscures this division of labor, so we model the two pathways
explicitly:

- **cU (upper / thalamus-projecting / basal-ganglia loop):**
  - reciprocal excitatory connections with **thalamus** (cU→T, T→cU)
  - excitatory projections to **striatum** (D1, D2) and **GPe**
  - (families with STN) sends the cortical hyperdirect projection to **STN**
  - (full family only) excitatory projection to **superior colliculus** — SC's
    cortical input comes from cU only
- **cL (lower / descending / motor output):**
  - excitatory projections to the **excitatory medulla units** and to **SNc**
  - (families with STN) also contributes to the **STN** hyperdirect projection
- **Both cU and cL:**
  - within-population excitatory recurrence
  - reciprocal excitatory connections to each other (cU↔cL)
  - inhibitory input from the shared **c_inh** pool (and they drive c_inh)
  - direct **cue** input

In the families with STN, the hyperdirect cortex→STN projection is driven by
**both** cU and cL.

Downstream analysis code still sees a single `Cortex` state array; internally it
is packed as `[cU..., cL..., c_inh...]`.
