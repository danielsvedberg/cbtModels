"""Compare D1 (dSPN) / D2 (iSPN) activity between the pka_d10=0.1 self-timing model
and the pka_d10=0.25 retrained model. Shows the 0.1 solution has D1 totally dead."""
import pickle as pkl, sys, pathlib
import numpy as np
import jax.numpy as jnp, jax.random as jr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm; from matplotlib.colors import Normalize
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"cbt_loop")); sys.path.insert(0, str(ROOT))
import cbt_rnn as cbtl, config_script as C, self_timed_movement_task as stmt
OUT = pathlib.Path(__file__).resolve().parent/"plots"/"dspn_compare.png"
cfg=C.for_family("cbt_loop"); t=cfg.TASK_CONFIG; A=cbtl.STATE_AREA_ORDER
inputs,_,_=stmt.self_timed_movement_task(T_start=t["t_start"],T_cue=t["t_cue"],T_wait=t["t_wait"],T_movement=t["t_movement"],T=t["t_total"])
models=[("pka_d10=0.1 (self-timing)","/tmp/claude-1000/-home-dsvedberg-Documents-CodeVault-cbtModels/02953a08-a57a-4eda-b9df-a53339ae0f03/scratchpad/params_shaped_pka01.pkl"),("pka_d10=0.25 (retrained)","cbt_loop/params_shaped.pkl")]
ts=np.asarray(t["t_start"]); order=np.argsort(ts); pick=order[np.linspace(0,len(ts)-1,6).astype(int)]
cmap=cm.viridis; norm=Normalize(ts.min(),ts.max())
fig,ax=plt.subplots(2,2,figsize=(13,8),sharex=True)
for col,(name,path) in enumerate(models):
    b=pkl.load(open(path,"rb")); p,conf=b["params"],b["config"]
    inp=cbtl.match_input_channels(inputs,p); B=inp.shape[0]
    nd1=p["J_d1"].shape[0]; nd2=p["J_d2"].shape[0]
    _,xs=cbtl.batched_rnn(p,dict(conf,noise_std=0.0),inp,jnp.zeros((B,inp.shape[1],nd1+nd2)),jr.split(jr.PRNGKey(0),B))
    d1=np.asarray(xs[A.index("D1")]).mean(-1); d2=np.asarray(xs[A.index("D2")]).mean(-1)
    for r,(sig,lbl) in enumerate([(d1,"D1 (dSPN)"),(d2,"D2 (iSPN)")]):
        a=ax[r,col]
        for i in pick: a.plot(sig[i],color=cmap(norm(ts[i])),lw=1.6)
        a.set_ylim(0,1.02); a.grid(alpha=.25)
        a.set_title(f"{lbl}  —  {name}\nmean {sig.mean():.3f}, max {sig.max():.3f}")
        if col==0: a.set_ylabel(f"{lbl} pop-mean")
        if r==1: a.set_xlabel("time step (color = cue time)")
fig.suptitle("dSPN/iSPN activity: pka_d10=0.1 (D1 dead) vs 0.25 (D1 alive)",fontsize=13)
plt.tight_layout(); plt.savefig(OUT,dpi=110); print("saved",OUT)
