#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST 2 — les features macro predisent-elles le RESIDU causal du niveau du VIX ?

Le test 1 (ablation_vix.py) a montre : NOVIX ~ 0, et FULL bat par une OLS a 2 params.
Objection restante : le k-NN est un estimateur bruite (inefficacite, pas absence
d'information), et l'orthogonalisation implicite etait grossiere. On refait proprement,
100% causal, en enlevant le confondant "efficacite d'estimateur" :

  A chaque jour de decision t et horizon h :
    1. OLS causale expanding sur le passe PURGE (gj+h<=gi, embargo 5) :
           R_{j,h} = a + b * VIX_j + eps
       -> (a_t, b_t) n'utilisent que de l'information <= t.
    2. Residu causal d'un jour passe j :  e_j = R_{j,h} - (a_t + b_t * VIX_j)
       Residu realise du jour t        :  e_t = R_{t,h} - (a_t + b_t * VIX_t)
       (e_t = erreur OOS de la baseline de niveau -> c'est CA qu'il reste a expliquer)
    3. Predicteurs du residu :
         RESID_KNN8   k-NN sur les 8 features macro (sans vix_lvl) -> moyenne des e_j
         RESID_KNN9   k-NN sur les 9 features du moteur            -> moyenne des e_j
         RESID_KNNVIX k-NN sur vix_lvl seul  (CONTROLE : doit donner ~0)
    4. Version PARAMETRIQUE (pas de bruit k-NN) :
         OLS_BASE  R ~ 1 + VIX
         OLS_AUG   R ~ 1 + VIX + les 8 features macro (z causaux winsorises)
         OLS_TS    R ~ 1 + VIX + ts_slope        (controle : term structure)
       -> IC(OLS_AUG) vs IC(OLS_BASE), test apparie block bootstrap.
    5. Feature par feature : IC(z_f(t), e_t) pour chacune des 8 (Bonferroni 8 tests).

Verdict : si IC(residu) ~ 0 partout, il n'y a AUCUN signal macro sur le VIX,
estimateur-independant.
"""
import os, sys, math, csv, urllib.request
import numpy as np

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
raw["GOLD"] = yfetch.fetch("GC=F"); raw["COPPER"] = yfetch.fetch("HG=F")
raw["VIX3M"] = yfetch.fetch("^VIX3M"); raw["VIXY"] = yfetch.fetch("^VIX")

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

usd5 = np.full(N, np.nan); usd5[K_IMP:] = np.log(LV["DTWEXBGS"][K_IMP:] / LV["DTWEXBGS"][:-K_IMP]) * 100
bmom = np.full(N, np.nan); bmom[20:] = np.log(LV["DCOILBRENTEU"][20:] / LV["DCOILBRENTEU"][:-20]) * 100
_cg = np.where((LV["GOLD"] > 0) & np.isfinite(LV["COPPER"]), LV["COPPER"] / LV["GOLD"], np.nan)
gmom = np.full(N, np.nan)
for i in range(20, N):
    a, b = _cg[i], _cg[i - 20]
    if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0: gmom[i] = math.log(a / b) * 100
ts_raw = np.where((LV["VIX3M"] > 0) & (LV["VIXY"] > 0), np.log(LV["VIX3M"] / LV["VIXY"]) * 100, np.nan)

raw_feat = {
    "d10_5": dbps(LV["DGS10"], K_IMP), "dreal_5": dbps(LV["DFII10"], K_IMP),
    "dbe_5": dbps(LV["T10YIE"], K_IMP), "vix_lvl": LV["VIXCLS"],
    "slope": LV["T10Y2Y"], "dusd_5": usd5,
    "brwti": LV["DCOILBRENTEU"] - LV["DCOILWTICO"], "brent_mom": bmom, "growth": gmom,
    "ts_slope": ts_raw,
}
FEAT9  = ["d10_5","dreal_5","dbe_5","vix_lvl","slope","dusd_5","brwti","brent_mom","growth"]
MACRO8 = [f for f in FEAT9 if f != "vix_lvl"]
Zf = {fn: np.clip(expanding_z(raw_feat[fn]), -WINSOR, WINSOR) for fn in raw_feat}

Z9 = np.column_stack([Zf[fn] for fn in FEAT9])
valid = np.all(np.isfinite(Z9), axis=1)
vidx = np.where(valid)[0]; gidx = vidx.copy(); nvalid = len(vidx); vyear = year[vidx]
print(f"    valid={nvalid} {cal[vidx[0]]}..{cal[vidx[-1]]}", file=sys.stderr)

ZKNN = {"KNN8": np.column_stack([Zf[f] for f in MACRO8])[vidx],
        "KNN9": Z9[vidx],
        "KNNVIX": Zf["vix_lvl"][vidx].reshape(-1, 1)}
M8   = np.column_stack([Zf[f] for f in MACRO8])[vidx]        # regresseurs macro standardises
vix_v = LV["VIXCLS"][vidx]
ts_v  = Zf["ts_slope"][vidx]
ok_ts = np.isfinite(ts_v)

HORIZONS = [5, 10, 20]
def fwd_vix(t, h):
    if t + h >= N: return np.nan
    a, b = LV["VIXCLS"][t], LV["VIXCLS"][t + h]
    return (b - a) if (math.isfinite(a) and math.isfinite(b)) else np.nan
FRV = {h: np.array([fwd_vix(t, h) for t in range(N)])[vidx] for h in HORIZONS}

REFRESH = 63; EMBARGO = 5; KFLOOR = 60; MINFIT = 250

def build_metric(Zsub, upto_pos):
    sub = Zsub[:upto_pos + 1]
    mu = sub.mean(axis=0); sd = sub.std(axis=0) + 1e-12
    Zs = (sub - mu) / sd
    C = np.atleast_2d(np.corrcoef(Zs, rowvar=False)) if Zsub.shape[1] > 1 else np.array([[1.0]])
    ev, V = np.linalg.eigh(C); ev = np.clip(ev, 1e-6, None)
    return ((Zsub - mu) / sd) @ V, ev

def elig_prefix(it, h):
    gi = gidx[it]; gj = gidx[:it]
    return np.where((gj + h <= gi) & ((gi - gj) > EMBARGO))[0]

def fit_ols(Xcols, elig, h):
    """OLS causale sur le passe purge. Xcols = liste de colonnes (n_valid,)."""
    y = FRV[h][elig]
    A = np.column_stack([c[elig] for c in Xcols])
    m = np.isfinite(y) & np.all(np.isfinite(A), axis=1)
    if m.sum() < MINFIT: return None
    G = np.column_stack([np.ones(int(m.sum())), A[m]])
    try: beta, *_ = np.linalg.lstsq(G, y[m], rcond=None)
    except Exception: return None
    return beta

def apply_ols(beta, Xcols, i):
    x = np.array([c[i] for c in Xcols])
    if not np.all(np.isfinite(x)): return np.nan
    return float(beta[0] + beta[1:] @ x)

START = next(i for i in range(nvalid) if vyear[i] >= 2012 and i >= 500)
print(f"[2] walk-forward {cal[gidx[START]]} .. {cal[gidx[-1]]} ({nvalid-START} j)", file=sys.stderr)

NAMES = ["RESID_KNN8","RESID_KNN9","RESID_KNNVIX","OLS_BASE","OLS_AUG","OLS_TS"]
out = {h: {k: [] for k in NAMES + ["real","resid","date","year"]} for h in HORIZONS}
feat_ic = {h: {f: {"x": [], "e": []} for f in MACRO8 + ["ts_slope"]} for h in HORIZONS}

metric = {}; last_ref = {}
for m in ZKNN:
    metric[m] = build_metric(ZKNN[m], START); last_ref[m] = START

M8cols = [M8[:, j] for j in range(M8.shape[1])]

for it in range(START, nvalid):
    for m in ZKNN:
        if it - last_ref[m] >= REFRESH:
            metric[m] = build_metric(ZKNN[m], it); last_ref[m] = it
    for h in HORIZONS:
        real = FRV[h][it]
        if not math.isfinite(real): continue
        elig = elig_prefix(it, h)
        if len(elig) < MINFIT: continue
        beta = fit_ols([vix_v], elig, h)
        if beta is None: continue
        pred_base = apply_ols(beta, [vix_v], it)
        e_t = real - pred_base                        # residu OOS de la baseline de niveau
        e_past = FRV[h] - (beta[0] + beta[1] * vix_v)  # residus causaux du passe (coeffs <= t)

        out[h]["date"].append(cal[gidx[it]]); out[h]["year"].append(int(vyear[it]))
        out[h]["real"].append(float(real)); out[h]["resid"].append(float(e_t))
        out[h]["OLS_BASE"].append(pred_base)

        # --- k-NN sur le residu ---
        for tag, m in (("RESID_KNN8","KNN8"), ("RESID_KNN9","KNN9"), ("RESID_KNNVIX","KNNVIX")):
            Pall, ev = metric[m]
            if len(elig) < KFLOOR:
                out[h][tag].append(np.nan); continue
            diff = Pall[elig] - Pall[it]
            D = np.sqrt(np.sum(diff * diff / ev, axis=1))
            k = max(KFLOOR, int(0.05 * len(elig)))
            sel = elig[np.argsort(D)[:k]]
            ea = e_past[sel]; ea = ea[np.isfinite(ea)]
            out[h][tag].append(float(np.mean(ea)) if len(ea) >= KFLOOR * 0.5 else np.nan)

        # --- versions parametriques ---
        b_aug = fit_ols([vix_v] + M8cols, elig, h)
        out[h]["OLS_AUG"].append(apply_ols(b_aug, [vix_v] + M8cols, it) if b_aug is not None else np.nan)
        b_ts = fit_ols([vix_v, ts_v], elig, h) if ok_ts[it] else None
        out[h]["OLS_TS"].append(apply_ols(b_ts, [vix_v, ts_v], it) if b_ts is not None else np.nan)

        for f in MACRO8:
            feat_ic[h][f]["x"].append(float(Zf[f][gidx[it]])); feat_ic[h][f]["e"].append(float(e_t))
        feat_ic[h]["ts_slope"]["x"].append(float(ts_v[it]) if ok_ts[it] else np.nan)
        feat_ic[h]["ts_slope"]["e"].append(float(e_t))
    if (it - START) % 500 == 0: print(f"    {cal[gidx[it]]}", file=sys.stderr)

# ------------------ metriques ------------------
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
    p = 1.0 / max(L, 1); o = np.empty((B, n), dtype=int)
    for b in range(B):
        s = np.empty(n, dtype=int); i = rng.integers(n)
        for j in range(n):
            s[j] = i; i = rng.integers(n) if rng.random() < p else (i + 1) % n
        o[b] = s
    return o
def ic_ci(pred, real, L, B=1000, seed=7):
    rng = np.random.default_rng(seed); idx = block_idx(len(real), L, B, rng)
    st = np.array([pearson(pred[i], real[i]) for i in idx])
    return float(np.percentile(st, 5)), float(np.percentile(st, 95))
def ic_diff_ci(p1, p2, real, L, B=1000, seed=11):
    rng = np.random.default_rng(seed); idx = block_idx(len(real), L, B, rng)
    st = np.array([pearson(p1[i], real[i]) - pearson(p2[i], real[i]) for i in idx])
    return float(np.percentile(st, 5)), float(np.percentile(st, 95)), float(np.mean(st > 0))

lines = []
def P(s=""):
    print(s); lines.append(s)

for h in HORIZONS:
    d = {k: np.array(out[h][k], float) for k in NAMES + ["real","resid"]}
    yrs = np.array(out[h]["year"])
    ok = np.isfinite(d["real"]) & np.isfinite(d["resid"]) & np.all(
        np.column_stack([np.isfinite(d[k]) for k in NAMES]), axis=1)
    e = d["resid"][ok]; r = d["real"][ok]
    P(""); P("=" * 92)
    P(f"HORIZON {h}j — n={int(ok.sum())} ({out[h]['date'][int(np.argmax(ok))]} .. {out[h]['date'][-1]})")
    P(f"  ecart-type du residu causal e_t = {e.std():.2f} pts de VIX "
      f"(vs {r.std():.2f} pour DVIX brut -> la baisse de niveau explique "
      f"{100*(1-e.var()/r.var()):.1f}% de la variance)")
    P("=" * 92)
    P("A) PREDICTION DU RESIDU CAUSAL e_t (ce qui reste apres la reversion de niveau)")
    P(f"   {'predicteur':14} {'IC_pearson':>11} {'IC_5%':>8} {'IC_95%':>8} {'IC_spear':>9}")
    for k in ["RESID_KNN8","RESID_KNN9","RESID_KNNVIX"]:
        pk = d[k][ok]; lo, hi = ic_ci(pk, e, h)
        note = "  <- CONTROLE (doit etre ~0)" if k == "RESID_KNNVIX" else ""
        P(f"   {k:14} {pearson(pk,e):>11.4f} {lo:>8.3f} {hi:>8.3f} {spearman(pk,e):>9.4f}{note}")
    P("")
    P("B) VERSION PARAMETRIQUE (sans bruit k-NN) — IC sur DVIX brut")
    for k in ["OLS_BASE","OLS_AUG","OLS_TS"]:
        pk = d[k][ok]; lo, hi = ic_ci(pk, r, h)
        P(f"   {k:14} {pearson(pk,r):>11.4f} {lo:>8.3f} {hi:>8.3f} {spearman(pk,r):>9.4f}")
    dlo, dhi, pg = ic_diff_ci(d["OLS_AUG"][ok], d["OLS_BASE"][ok], r, h)
    P(f"   dIC(AUG-BASE) = {pearson(d['OLS_AUG'][ok],r)-pearson(d['OLS_BASE'][ok],r):+.4f}"
      f"  [{dlo:+.3f},{dhi:+.3f}]  P(AUG>BASE)={pg:.2f}")
    dlo, dhi, pg = ic_diff_ci(d["OLS_TS"][ok], d["OLS_BASE"][ok], r, h)
    P(f"   dIC(TS -BASE) = {pearson(d['OLS_TS'][ok],r)-pearson(d['OLS_BASE'][ok],r):+.4f}"
      f"  [{dlo:+.3f},{dhi:+.3f}]  P(TS >BASE)={pg:.2f}")
    P("")
    P("C) FEATURE PAR FEATURE : IC(z_f(t), e_t)   [8 tests -> seuil Bonferroni |IC| a t=2,7]")
    for f in MACRO8 + ["ts_slope"]:
        x = np.array(feat_ic[h][f]["x"], float); ee = np.array(feat_ic[h][f]["e"], float)
        mk = np.isfinite(x) & np.isfinite(ee)
        if mk.sum() < 100: continue
        lo, hi = ic_ci(x[mk], ee[mk], h)
        sig = "  *" if (lo > 0 or hi < 0) else ""
        P(f"   {f:12} n={int(mk.sum()):>5}  IC={pearson(x[mk],ee[mk]):>+7.4f}  [{lo:+.3f},{hi:+.3f}]{sig}")
    P("")
    P("D) IC du residu par annee (RESID_KNN8)")
    row = []
    for yy in sorted(set(yrs[ok].tolist())):
        mm = yrs[ok] == yy
        if mm.sum() < 20: continue
        row.append(f"{yy}:{pearson(d['RESID_KNN8'][ok][mm], e[mm]):+.2f}")
    P("   " + " · ".join(row))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resid_vix_out.txt"), "w") as f:
    f.write("\n".join(lines))
print("\n[done]", file=sys.stderr)
