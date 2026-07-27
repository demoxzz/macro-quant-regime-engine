#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION VIX — le "macro engine" bat-il des baselines de mean-reversion du VIX ?

Reprend EXACTEMENT la machinerie causale de macro_quant_backtest.py
(metrique PCA-whitening re-estimee <= t, purge gj+h<=gi, embargo 5, k=max(60,5%))
mais compare plusieurs jeux de features / baselines sur la MEME cible :
    R_{t,h} = VIXCLS[t+h] - VIXCLS[t]   (points de VIX)

Modeles :
  FULL      k-NN sur les 9 features du moteur
  VIXONLY   k-NN sur vix_lvl seul                     <- baseline 1
  NOVIX     k-NN sur les 8 autres (sans vix_lvl)      <- ablation decisive
  VIXTS     k-NN sur {vix_lvl, ts_slope}              <- baseline 4 (term structure)
  PCTL      -percentile expanding du VIX              <- baseline 2
  OLS       OLS causale expanding R ~ a + b*VIX       <- baseline 3 (AR-like)
  OLSLOG    OLS causale expanding R ~ a + b*log(VIX)
  VRP       OLS causale R ~ a + b*VIX + c*RV22(NDX)   <- baseline 5 (HAR/VRP-lite)

Sorties : IC Pearson/Spearman par modele/horizon, sur echantillon COMMUN ;
          test paire (block bootstrap) IC_FULL - IC_baseline ;
          regression incrementale real ~ pred_base + pred_full (t-stat block-boot).
"""
import os, sys, math, csv, urllib.request
import numpy as np
from bisect import insort, bisect_left

sys.path.insert(0, "/Users/paulgregoire/Desktop/theo/macro-quant-regime-engine/engine")
import yfetch

CACHE = "/tmp/fredcache"; os.makedirs(CACHE, exist_ok=True)
np.random.seed(42)

FRED = ["DGS10","DFII10","T10YIE","VIXCLS","T10Y2Y","DTWEXBGS","DCOILBRENTEU","DCOILWTICO","NASDAQCOM"]

def fetch(sid):
    fp = os.path.join(CACHE, sid + ".csv")
    if not os.path.exists(fp) or os.path.getsize(fp) < 50:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1900-01-01"
        with open(fp, "wb") as f: f.write(urllib.request.urlopen(url, timeout=30).read())
    d = {}
    with open(fp) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 2: continue
            dt, v = row[0], row[1]
            if v in (".", "", "NA"): continue
            try: d[dt] = float(v)
            except: pass
    return d

print("[1] data ...", file=sys.stderr)
raw = {s: fetch(s) for s in FRED}
raw["GOLD"] = yfetch.fetch("GC=F")
raw["COPPER"] = yfetch.fetch("HG=F")
raw["VIX3M"] = yfetch.fetch("^VIX3M")
raw["VIXY"] = yfetch.fetch("^VIX")   # spot VIX Yahoo, pour la pente (meme source que VIX3M)

cal = sorted(raw["DGS10"].keys()); N = len(cal)
year = np.array([int(d[:4]) for d in cal])
FFILL = 3
def align(sd):
    out = np.full(N, np.nan); last = np.nan; gap = 0
    for i, d in enumerate(cal):
        if d in sd: out[i] = sd[d]; last = sd[d]; gap = 0
        elif not math.isnan(last) and gap < FFILL: out[i] = last; gap += 1
        else: out[i] = np.nan
    return out
LV = {k: align(v) for k, v in raw.items()}

# ---------------- features (identiques au backtest) ----------------
K_IMP = 5; W0 = 252; WINSOR = 2.5
def dbps(x, k):
    r = np.full(N, np.nan); r[k:] = (x[k:] - x[:-k]) * 100.0; return r
def expanding_z(x):
    out = np.full(N, np.nan); cnt = 0; s = 0.0; s2 = 0.0
    for i in range(N):
        xi = x[i]
        if math.isfinite(xi):
            cnt += 1; s += xi; s2 += xi * xi
            if cnt >= W0:
                mu = s / cnt; var = max(s2 / cnt - mu * mu, 1e-12)
                out[i] = (xi - mu) / math.sqrt(var)
    return out
def expanding_pctl(x):
    """percentile causal de x[i] parmi x[:i+1] (warmup W0)."""
    out = np.full(N, np.nan); buf = []
    for i in range(N):
        xi = x[i]
        if math.isfinite(xi):
            insort(buf, xi)
            if len(buf) >= W0:
                out[i] = bisect_left(buf, xi) / max(len(buf) - 1, 1)
    return out

usd5 = np.full(N, np.nan); usd5[K_IMP:] = np.log(LV["DTWEXBGS"][K_IMP:] / LV["DTWEXBGS"][:-K_IMP]) * 100
bmom = np.full(N, np.nan); bmom[20:] = np.log(LV["DCOILBRENTEU"][20:] / LV["DCOILBRENTEU"][:-20]) * 100
_cg = np.where((LV["GOLD"] > 0) & np.isfinite(LV["COPPER"]), LV["COPPER"] / LV["GOLD"], np.nan)
gmom = np.full(N, np.nan)
for i in range(20, N):
    a, b = _cg[i], _cg[i - 20]
    if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0: gmom[i] = math.log(a / b) * 100
# pente term structure VIX : log(VIX3M/VIX)*100  (>0 = contango)
ts_raw = np.where((LV["VIX3M"] > 0) & (LV["VIXY"] > 0), np.log(LV["VIX3M"] / LV["VIXY"]) * 100, np.nan)
# vol realisee 22j du Nasdaq (HAR/VRP-lite)
lnq = np.where(LV["NASDAQCOM"] > 0, np.log(LV["NASDAQCOM"]), np.nan)
ret = np.full(N, np.nan); ret[1:] = lnq[1:] - lnq[:-1]
rv22 = np.full(N, np.nan)
for i in range(22, N):
    w = ret[i - 21:i + 1]
    if np.all(np.isfinite(w)): rv22[i] = np.std(w) * math.sqrt(252) * 100.0

raw_feat = {
    "d10_5": dbps(LV["DGS10"], K_IMP), "dreal_5": dbps(LV["DFII10"], K_IMP),
    "dbe_5": dbps(LV["T10YIE"], K_IMP), "vix_lvl": LV["VIXCLS"],
    "slope": LV["T10Y2Y"], "dusd_5": usd5,
    "brwti": LV["DCOILBRENTEU"] - LV["DCOILWTICO"], "brent_mom": bmom, "growth": gmom,
    "ts_slope": ts_raw,
}
FEAT9 = ["d10_5","dreal_5","dbe_5","vix_lvl","slope","dusd_5","brwti","brent_mom","growth"]
Zf = {fn: np.clip(expanding_z(raw_feat[fn]), -WINSOR, WINSOR) for fn in raw_feat}

# echantillon valide = celui du backtest (9 features finies) -> comparaison a periode identique
Z9 = np.column_stack([Zf[fn] for fn in FEAT9])
valid = np.all(np.isfinite(Z9), axis=1)
vidx = np.where(valid)[0]; gidx = vidx.copy(); nvalid = len(vidx); vyear = year[vidx]
print(f"    valid={nvalid} {cal[vidx[0]]}..{cal[vidx[-1]]}", file=sys.stderr)

MODELS_KNN = {
    "FULL":    FEAT9,
    "VIXONLY": ["vix_lvl"],
    "NOVIX":   [f for f in FEAT9 if f != "vix_lvl"],
    "VIXTS":   ["vix_lvl", "ts_slope"],
}
ZV = {m: np.column_stack([Zf[fn] for fn in cols])[vidx] for m, cols in MODELS_KNN.items()}
OKV = {m: np.all(np.isfinite(ZV[m]), axis=1) for m in ZV}   # VIXTS a des NaN (VIX3M debute 2006)

vix_lvl_v = LV["VIXCLS"][vidx]
vix_pctl_v = expanding_pctl(LV["VIXCLS"])[vidx]
rv22_v = rv22[vidx]

# ---------------- labels ----------------
HORIZONS = [5, 10, 20]
def fwd_vix(t, h):
    if t + h >= N: return np.nan
    a, b = LV["VIXCLS"][t], LV["VIXCLS"][t + h]
    return (b - a) if (math.isfinite(a) and math.isfinite(b)) else np.nan
FR = {h: np.array([fwd_vix(t, h) for t in range(N)]) for h in HORIZONS}
FRV = {h: FR[h][vidx] for h in HORIZONS}   # label indexe en espace "valide"

# ---------------- machinerie causale ----------------
REFRESH = 63; EMBARGO = 5; KFLOOR = 60

def build_metric(Zsub, ok, upto_pos):
    rows = np.where(ok[:upto_pos + 1])[0]
    sub = Zsub[rows]
    mu = sub.mean(axis=0); sd = sub.std(axis=0) + 1e-12
    Zs = (sub - mu) / sd
    C = np.corrcoef(Zs, rowvar=False) if Zsub.shape[1] > 1 else np.array([[1.0]])
    C = np.atleast_2d(C)
    ev, V = np.linalg.eigh(C); ev = np.clip(ev, 1e-6, None)
    Pall = ((Zsub - mu) / sd) @ V
    return Pall, ev

def knn_pred(it, h, Pall, ev, ok):
    gi = gidx[it]; gj = gidx[:it]
    cond = (gj + h <= gi) & ((gi - gj) > EMBARGO) & ok[:it]
    elig = np.where(cond)[0]
    if len(elig) < KFLOOR: return np.nan
    diff = Pall[elig] - Pall[it]
    D = np.sqrt(np.sum(diff * diff / ev, axis=1))
    k = max(KFLOOR, int(0.05 * len(elig)))
    sel = elig[np.argsort(D)[:k]]
    fa = FRV[h][sel]; fa = fa[np.isfinite(fa)]
    return float(np.mean(fa)) if len(fa) >= KFLOOR * 0.5 else np.nan

def ols_pred(it, h, X):
    """OLS causale expanding : R ~ [1, X] sur le passe purge (prefixe)."""
    gi = gidx[it]; gj = gidx[:it]
    cond = (gj + h <= gi) & ((gi - gj) > EMBARGO)
    elig = np.where(cond)[0]
    if len(elig) < 250: return np.nan
    y = FRV[h][elig]; A = X[elig]
    m = np.isfinite(y) & np.all(np.isfinite(A), axis=1) if A.ndim > 1 else np.isfinite(y) & np.isfinite(A)
    A = np.atleast_2d(A.T).T[m]; y = y[m]
    if len(y) < 250: return np.nan
    xt = np.atleast_2d(X[it].T).T.ravel() if X.ndim > 1 else np.array([X[it]])
    if not np.all(np.isfinite(xt)): return np.nan
    G = np.column_stack([np.ones(len(y)), A])
    try: beta, *_ = np.linalg.lstsq(G, y, rcond=None)
    except Exception: return np.nan
    return float(beta[0] + beta[1:] @ xt)

START = next(i for i in range(nvalid) if vyear[i] >= 2012 and i >= 500)
print(f"[2] walk-forward {cal[gidx[START]]} .. {cal[gidx[-1]]} ({nvalid-START} jours)", file=sys.stderr)

X_OLS    = vix_lvl_v.reshape(-1, 1)
X_OLSLOG = np.where(vix_lvl_v > 0, np.log(vix_lvl_v), np.nan).reshape(-1, 1)
X_VRP    = np.column_stack([vix_lvl_v, rv22_v])

NAMES = ["FULL","VIXONLY","NOVIX","VIXTS","PCTL","OLS","OLSLOG","VRP"]
res = {h: {m: [] for m in NAMES} for h in HORIZONS}
res_meta = {h: {"date": [], "year": [], "real": []} for h in HORIZONS}

metric = {}
last_ref = {}
for m in MODELS_KNN:
    metric[m] = build_metric(ZV[m], OKV[m], START); last_ref[m] = START

for it in range(START, nvalid):
    for m in MODELS_KNN:
        if it - last_ref[m] >= REFRESH:
            metric[m] = build_metric(ZV[m], OKV[m], it); last_ref[m] = it
    for h in HORIZONS:
        real = FRV[h][it]
        if not math.isfinite(real): continue
        res_meta[h]["date"].append(cal[gidx[it]]); res_meta[h]["year"].append(int(vyear[it]))
        res_meta[h]["real"].append(float(real))
        for m in MODELS_KNN:
            Pall, ev = metric[m]
            res[h][m].append(knn_pred(it, h, Pall, ev, OKV[m]) if OKV[m][it] else np.nan)
        p = vix_pctl_v[it]
        res[h]["PCTL"].append(-float(p) if math.isfinite(p) else np.nan)   # VIX haut -> baisse attendue
        res[h]["OLS"].append(ols_pred(it, h, X_OLS))
        res[h]["OLSLOG"].append(ols_pred(it, h, X_OLSLOG))
        res[h]["VRP"].append(ols_pred(it, h, X_VRP))
    if (it - START) % 500 == 0: print(f"    {cal[gidx[it]]}", file=sys.stderr)

# ---------------- metriques ----------------
def rankdata(a):
    a = np.asarray(a, float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    return ((start + csum + 1) / 2.0)[inv]
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 10: return np.nan
    xd = x - x.mean(); yd = y - y.mean(); d = math.sqrt((xd @ xd) * (yd @ yd))
    return float(xd @ yd / d) if d > 0 else np.nan
def spearman(x, y): return pearson(rankdata(x), rankdata(y))

def block_idx(n, L, B, rng):
    p = 1.0 / max(L, 1); out = np.empty((B, n), dtype=int)
    for b in range(B):
        s = np.empty(n, dtype=int); i = rng.integers(n)
        for j in range(n):
            s[j] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        out[b] = s
    return out

def ic_ci(pred, real, L, B=1000, seed=7):
    rng = np.random.default_rng(seed); idx = block_idx(len(real), L, B, rng)
    st = np.array([pearson(pred[i], real[i]) for i in idx])
    return float(np.percentile(st, 5)), float(np.percentile(st, 95))

def ic_diff_ci(p1, p2, real, L, B=1000, seed=11):
    rng = np.random.default_rng(seed); idx = block_idx(len(real), L, B, rng)
    st = np.array([pearson(p1[i], real[i]) - pearson(p2[i], real[i]) for i in idx])
    return float(np.percentile(st, 5)), float(np.percentile(st, 95)), float(np.mean(st > 0))

def incr_t(pred_base, pred_full, real, L, B=1000, seed=13):
    """real ~ a + b*base + c*full : distribution block-boot de c (et de b)."""
    G = np.column_stack([np.ones(len(real)), (pred_base - pred_base.mean()) / (pred_base.std() + 1e-12),
                         (pred_full - pred_full.mean()) / (pred_full.std() + 1e-12)])
    rng = np.random.default_rng(seed); idx = block_idx(len(real), L, B, rng)
    bs = np.empty((B, 2))
    for k, i in enumerate(idx):
        beta, *_ = np.linalg.lstsq(G[i], real[i], rcond=None)
        bs[k] = beta[1:]
    beta0, *_ = np.linalg.lstsq(G, real, rcond=None)
    return beta0[1:], np.percentile(bs, [5, 95], axis=0)

out_lines = []
def P(s=""):
    print(s); out_lines.append(s)

for h in HORIZONS:
    real_all = np.array(res_meta[h]["real"])
    preds = {m: np.array(res[h][m], float) for m in NAMES}
    ok_common = np.isfinite(real_all) & np.all(np.column_stack([np.isfinite(preds[m]) for m in NAMES]), axis=1)
    P("")
    P("=" * 96)
    P(f"HORIZON {h}j — cible = variation du VIX (points).  n commun = {int(ok_common.sum())} "
      f"({res_meta[h]['date'][int(np.argmax(ok_common))]} .. {res_meta[h]['date'][-1]})")
    P("=" * 96)
    r = real_all[ok_common]
    P(f"{'modele':10} {'n':>5} {'IC_pearson':>11} {'IC_5%':>8} {'IC_95%':>8} {'IC_spear':>9}   vs FULL: {'dIC':>7} {'[5%':>8} {'95%]':>8} {'P(>0)':>6}")
    ic_full = pearson(preds["FULL"][ok_common], r)
    for m in NAMES:
        pm = preds[m][ok_common]
        icp = pearson(pm, r); ics = spearman(pm, r)
        lo, hi = ic_ci(pm, r, h)
        if m == "FULL":
            P(f"{m:10} {len(r):>5} {icp:>11.4f} {lo:>8.3f} {hi:>8.3f} {ics:>9.4f}")
        else:
            d = ic_full - icp
            dlo, dhi, pg = ic_diff_ci(preds["FULL"][ok_common], pm, r, h)
            P(f"{m:10} {len(r):>5} {icp:>11.4f} {lo:>8.3f} {hi:>8.3f} {ics:>9.4f}   "
              f"{d:>16.4f} {dlo:>8.3f} {dhi:>8.3f} {pg:>6.2f}")
    P("")
    P("Regression incrementale  real ~ a + b*pred_baseline + c*pred_FULL  (predicteurs standardises,")
    P("coeff en points de VIX par 1 sd ; IC 5-95% block bootstrap L=h) :")
    for m in ["VIXONLY", "PCTL", "OLS", "VIXTS", "VRP"]:
        b, ci = incr_t(preds[m][ok_common], preds["FULL"][ok_common], r, h)
        P(f"  base={m:8}  b_base={b[0]:+7.3f} [{ci[0,0]:+.3f},{ci[1,0]:+.3f}]    "
          f"c_FULL={b[1]:+7.3f} [{ci[0,1]:+.3f},{ci[1,1]:+.3f}]"
          + ("   <- FULL n'ajoute rien" if ci[0,1] <= 0 <= ci[1,1] else "   <- FULL ajoute"))

    # correlation entre predictions
    P("")
    P("Correlation des predictions (a quel point FULL EST la baseline) :")
    for m in ["VIXONLY","PCTL","OLS","VIXTS","VRP","NOVIX"]:
        P(f"  corr(FULL, {m:8}) = {pearson(preds['FULL'][ok_common], preds[m][ok_common]):+.3f}")

    # IC par annee : FULL vs VIXONLY vs NOVIX
    yrs = np.array(res_meta[h]["year"])[ok_common]
    P("")
    P(f"{'annee':>6} {'n':>5} {'FULL':>7} {'VIXONLY':>8} {'NOVIX':>7} {'OLS':>7}")
    for yy in sorted(set(yrs.tolist())):
        mm = yrs == yy
        if mm.sum() < 20: continue
        P(f"{yy:>6} {int(mm.sum()):>5} "
          f"{pearson(preds['FULL'][ok_common][mm], r[mm]):>7.3f} "
          f"{pearson(preds['VIXONLY'][ok_common][mm], r[mm]):>8.3f} "
          f"{pearson(preds['NOVIX'][ok_common][mm], r[mm]):>7.3f} "
          f"{pearson(preds['OLS'][ok_common][mm], r[mm]):>7.3f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_vix_out.txt"), "w") as f:
    f.write("\n".join(out_lines))
print("\n[done]", file=sys.stderr)
