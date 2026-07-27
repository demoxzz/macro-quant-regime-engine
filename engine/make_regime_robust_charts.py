#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures robustesse "facon paires" pour le regime engine : DSR/PBO + hold-out.
Lit /tmp/macro_quant_backtest.json (cle 'robustness'). Sortie -> Macro/Quant/analysis/."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME={"DGS10":"UST 10Y","DGS2":"UST 2Y","T10Y2Y":"2s10s","VIXCLS":"VIX",
      "DTWEXBGS":"USD","DEXUSEU":"EUR/USD","DCOILBRENTEU":"Brent","NASDAQCOM":"Nasdaq"}
FIG=os.path.join(os.path.dirname(__file__),"..","analysis","macro-quant"); os.makedirs(FIG,exist_ok=True)
R=json.load(open("/tmp/macro_quant_backtest.json"))["robustness"]

# ---- fig A : Deflated Sharpe (N=24) par asset ----
dsr=R["dsr"]; sids=list(dsr.keys())
vals=[dsr[s]["dsr"]["24"] for s in sids]; names=[NAME.get(s,s) for s in sids]
order=sorted(range(len(vals)),key=lambda i:vals[i])
vals=[vals[i] for i in order]; names=[names[i] for i in order]
fig,ax=plt.subplots(figsize=(8,4.2))
cols=["#2e7d32" if v>=0.95 else ("#f9a825" if v>=0.5 else "#c62828") for v in vals]
ax.barh(names,vals,color=cols)
ax.axvline(0.95,ls="--",c="k",lw=1,label="seuil robustesse 0.95")
ax.axvline(0.5,ls=":",c="grey",lw=1)
ax.set_xlim(0,1); ax.set_xlabel("Deflated Sharpe Ratio (N=24 essais)")
ax.set_title("DSR — aucun Sharpe de strategie ne survit au data-snooping")
ax.legend(loc="lower right",fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG,"fig18_regime_dsr.png"),dpi=110); plt.close(fig)

# ---- fig B : hold-out IC train vs test ----
ho=R["holdout"]; sids=list(ho.keys())
names=[NAME.get(s,s) for s in sids]
ic_tr=[ho[s]["ic_train"] for s in sids]; ic_te=[ho[s]["ic_test"] for s in sids]
import numpy as np
x=np.arange(len(sids)); w=0.38
fig,ax=plt.subplots(figsize=(9,4.2))
ax.bar(x-w/2,ic_tr,w,label=f"IC train (<{R['holdout_split']})",color="#90caf9")
ax.bar(x+w/2,ic_te,w,label=f"IC test (>={R['holdout_split']}, jamais vu)",color="#1565c0")
ax.axhline(0,c="k",lw=.8); ax.set_xticks(x); ax.set_xticklabels(names,rotation=30,ha="right")
ax.set_ylabel("Information Coefficient (10j)")
best=NAME.get(R["best_train"],R["best_train"])
ax.set_title(f"Hold-out : seul {best} garde son IC hors-echantillon (pick IS -> tient OOS)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG,"fig19_regime_holdout.png"),dpi=110); plt.close(fig)

print("PBO =",R["pbo"],"| best_IS =",NAME.get(R["best_is"],R["best_is"]),
      "| best_train =",NAME.get(R["best_train"],R["best_train"]))
print("OK -> fig18_regime_dsr.png, fig19_regime_holdout.png")
