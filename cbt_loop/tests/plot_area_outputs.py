"""Compact per-area activity for a trained CBT model: population-mean over time
for every area, one line per cue time (color). Small enough to view/send, unlike
testing_script's full activity_by_area.png (which draws every trace).

Usage:  python <family>/tests/plot_area_outputs.py [family] [bundle.pkl]
  family defaults to 'cbt_loop'; bundle defaults to 'params_shaped.pkl'.
  Works for any CBT family (e.g. cbt_loop_noSCnoSTN)."""
import sys, pathlib, pickle as pkl
import numpy as np
import jax.numpy as jnp, jax.random as jr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm; from matplotlib.colors import Normalize
ROOT = pathlib.Path(__file__).resolve().parents[2]
FAM = sys.argv[1] if len(sys.argv) > 1 else "cbt_loop"
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "params_shaped.pkl"
sys.path.insert(0, str(ROOT/FAM)); sys.path.insert(0, str(ROOT))
import cbt_rnn as cbtl, config_script as C, self_timed_movement_task as stmt
OUT = pathlib.Path(__file__).resolve().parent/"plots"/f"area_outputs_{FAM}.png"
cfg = C.for_family(FAM); t = cfg.TASK_CONFIG; A = cbtl.STATE_AREA_ORDER
inputs,_,_ = stmt.self_timed_movement_task(T_start=t["t_start"],T_cue=t["t_cue"],T_wait=t["t_wait"],T_movement=t["t_movement"],T=t["t_total"])
b = pkl.load(open(ROOT/FAM/BUNDLE,"rb")); p,conf = b["params"],b["config"]
inp = cbtl.match_input_channels(inputs,p); B=inp.shape[0]; nd1=p["J_d1"].shape[0]; nd2=p["J_d2"].shape[0]
ys,xs = cbtl.batched_rnn(p,dict(conf,noise_std=0.0),inp,jnp.zeros((B,inp.shape[1],nd1+nd2)),jr.split(jr.PRNGKey(0),B))
ys = np.asarray(ys[...,0]); ts = np.asarray(t["t_start"])
order = np.argsort(ts); pick = order[np.linspace(0,B-1,6).astype(int)]
cmap = cm.viridis; norm = Normalize(ts.min(),ts.max())
areas = list(A)
n = len(areas)+1; ncol=3; nrow=int(np.ceil(n/ncol))
fig,axs = plt.subplots(nrow,ncol,figsize=(13,2.0*nrow),sharex=True)
axs=axs.flatten()
for k,name in enumerate(areas):
    a=axs[k]; sig=np.asarray(xs[A.index(name)]).mean(-1)  # (B,T) pop-mean
    for i in pick: a.plot(sig[i],color=cmap(norm(ts[i])),lw=1.3)
    a.set_title(name,fontsize=9); a.grid(alpha=.2); a.tick_params(labelsize=7)
# output
a=axs[len(areas)]
for i in pick: a.plot(ys[i],color=cmap(norm(ts[i])),lw=1.3)
a.set_title("OUTPUT",fontsize=9); a.grid(alpha=.2); a.tick_params(labelsize=7)
for j in range(len(areas)+1,len(axs)): axs[j].axis("off")
sm=cm.ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
fig.colorbar(sm,ax=axs.tolist(),fraction=0.015,pad=0.01,label="cue time t_start")
fig.suptitle(f"{FAM}: per-area population-mean activity ({BUNDLE}, color = cue time)",fontsize=12)
plt.savefig(OUT,dpi=95); print("saved",OUT)
