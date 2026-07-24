#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qstat.py — boite a outils econometrique numpy-only (partagee).
Extraite et VALIDEE dans macro_quant_pairs.py (tests synthetiques :
bruit blanc -> ADF-46/KPSS0.22 ; random walk -> ADF-1.3/KPSS20 ;
AR(1)0.95 -> half-life 11j vs 13j theorique).

CAVEAT : p-values ADF/EG = APPROX (interpolation valeurs critiques MacKinnon),
suffisantes pour ranking + FDR. Go/no-go final : reconfirmer statsmodels.
"""
import urllib.request, csv, os, sys, math
import numpy as np

CACHE = "/tmp/fredcache"; os.makedirs(CACHE, exist_ok=True)

# --- valeurs critiques (asymptotiques) --- #
ADF_CRIT   = {0.01: -3.4336, 0.05: -2.8621, 0.10: -2.5671}   # ADF standard, constante
EG_CRIT_N2 = {0.01: -3.9001, 0.05: -3.3377, 0.10: -3.0462}   # Engle-Granger N=2, MacKinnon
KPSS_CRIT  = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}  # KPSS niveau

# ------------------------------------------------------------------ #
# Data FRED
# ------------------------------------------------------------------ #
def fetch(sid):
    fp = os.path.join(CACHE, sid + ".csv")
    if not os.path.exists(fp) or os.path.getsize(fp) < 50:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1900-01-01"
        try:
            with open(fp, "wb") as f:
                f.write(urllib.request.urlopen(url, timeout=30).read())
        except Exception as e:
            print(f"    ! fetch {sid} echoue: {e}", file=sys.stderr); return {}
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

def build_calendar(raw, keep):
    allkeys = set()
    for sid in keep: allkeys |= set(raw[sid].keys())
    return sorted(allkeys)

def align(sd, cal, ffill=3):
    N = len(cal); out = np.full(N, np.nan); last = np.nan; gap = 0
    for i, d in enumerate(cal):
        if d in sd: out[i] = sd[d]; last = sd[d]; gap = 0
        elif math.isfinite(last) and gap < ffill: out[i] = last; gap += 1
        else: out[i] = np.nan
    return out

# ------------------------------------------------------------------ #
# Econometrie
# ------------------------------------------------------------------ #
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    nobs, k = X.shape
    sigma2 = float(resid @ resid) / max(nobs - k, 1)
    return beta, resid, sigma2, np.linalg.pinv(X.T @ X)

def adf(y, regression="c", maxlag=None):
    """ADF, selection lag par BIC. Retourne (tstat, usedlag, nobs)."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    n = len(y)
    if n < 20: return (np.nan, 0, 0)
    dy = np.diff(y); m = len(dy)
    if maxlag is None:
        maxlag = int(np.floor(12 * ((n / 100.0) ** 0.25)))
        maxlag = max(0, min(maxlag, m // 2 - 2))
    best = None
    for p in range(0, maxlag + 1):
        rows = np.arange(p, m)
        if len(rows) - (p + (regression == "c") + 1) < 10: break
        cols = [y[rows]]
        for k in range(1, p + 1): cols.append(dy[rows - k])
        X = np.column_stack(cols)
        if regression == "c": X = np.column_stack([np.ones(len(rows)), X])
        target = dy[rows]
        beta, resid, sigma2, XtX_inv = _ols(X, target)
        nobs = len(rows); rss = float(resid @ resid)
        bic = nobs * math.log(rss / nobs + 1e-300) + X.shape[1] * math.log(nobs)
        gi = 1 if regression == "c" else 0
        se_g = math.sqrt(sigma2 * XtX_inv[gi, gi])
        tstat = beta[gi] / se_g if se_g > 0 else np.nan
        if best is None or bic < best[0]:
            best = (bic, tstat, p, nobs)
    if best is None: return (np.nan, 0, 0)
    return (best[1], best[2], best[3])

def approx_pvalue(stat, crit):
    if not math.isfinite(stat): return np.nan
    pts = sorted(crit.items(), key=lambda kv: kv[1])
    if stat <= pts[0][1]:
        (p1, c1), (p2, c2) = pts[0], pts[1]
        return max(1e-4, p1 + (p2 - p1) / (c2 - c1) * (stat - c1))
    if stat >= pts[-1][1]:
        (p1, c1) = pts[-1]
        return min(0.99, p1 + (0.90 - p1) / (0.0 - c1) * (stat - c1))
    for (p1, c1), (p2, c2) in zip(pts, pts[1:]):
        if c1 <= stat <= c2:
            return p1 + (p2 - p1) * (stat - c1) / (c2 - c1)
    return np.nan

def verdict(stat, crit):
    for lvl in sorted(crit):
        if math.isfinite(stat) and stat <= crit[lvl]: return lvl
    return None

def kpss(y, nlags=None):
    """KPSS niveau. H0=stationnaire. Bartlett kernel."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]; n = len(y)
    if n < 20: return np.nan
    e = y - y.mean(); S = np.cumsum(e)
    if nlags is None: nlags = int(np.floor(4 * ((n / 100.0) ** 0.25)))
    s2 = float(e @ e) / n
    for l in range(1, nlags + 1):
        w = 1.0 - l / (nlags + 1.0)
        s2 += 2.0 * w * float(e[l:] @ e[:-l]) / n
    return float(np.sum(S ** 2)) / (n ** 2 * s2) if s2 > 0 else np.nan

def kpss_rejects(stat, level=0.05):
    """True = rejette H0(stationnaire) au seuil `level` -> NON stationnaire."""
    return math.isfinite(stat) and stat > KPSS_CRIT[level]

def half_life(spread):
    s = np.asarray(spread, float); s = s[np.isfinite(s)]
    if len(s) < 20: return np.nan
    ds = np.diff(s); lag = s[:-1]
    X = np.column_stack([np.ones(len(lag)), lag])
    beta, *_ = np.linalg.lstsq(X, ds, rcond=None)
    b = beta[1]
    if b >= 0 or (1 + b) <= 0: return np.inf
    return -math.log(2) / math.log(1 + b)

def engle_granger(logy, logx):
    """OLS logy = a + b logx ; ADF(nc) sur le residu. Retourne dict ou None."""
    mask = np.isfinite(logy) & np.isfinite(logx)
    ly, lx = logy[mask], logx[mask]
    if len(ly) < 100: return None
    X = np.column_stack([np.ones(len(lx)), lx])
    beta, resid, *_ = _ols(X, ly)
    stat, lag, nobs = adf(resid, regression="nc")
    return dict(alpha=beta[0], beta=beta[1], spread=resid, adf=stat, adf_lag=lag,
                nobs=nobs, pval=approx_pvalue(stat, EG_CRIT_N2),
                lvl=verdict(stat, EG_CRIT_N2), kpss=kpss(resid),
                hl=half_life(resid), npts=int(mask.sum()))

def kalman_hedge(logy, logx, delta=1e-4, R=None, beta0=1.0):
    """
    Filtre de Kalman : regression LINEAIRE DYNAMIQUE  logy_t = a_t + b_t*logx_t + e_t.
    L'etat [a_t, b_t] suit une MARCHE ALEATOIRE -> le hedge ratio s'ADAPTE en continu.

    delta : vitesse d'adaptation de l'etat (Vw = delta/(1-delta)). Petit = lent/rigide,
            grand = rapide/souple. ATTENTION : delta trop grand -> b tracke tout ->
            e_t~0 trivialement "stationnaire" (degenere). On garde delta PETIT et fixe.
    R     : bruit d'observation (Ve). None -> variance du residu OLS full-sample (echelle).

    Retourne dict de tableaux alignes sur l'entree (NaN si obs manquante) :
      alpha[t], beta[t]   = etat filtre (hedge ratio dynamique),
      e[t]                = INNOVATION = spread Kalman (temps reel, sans look-ahead),
      S[t]                = variance de l'innovation,
      z[t]                = e/sqrt(S) = z-score tradeable en direct.
    """
    logy = np.asarray(logy, float); logx = np.asarray(logx, float)
    n = len(logy)
    if R is None:
        m = np.isfinite(logy) & np.isfinite(logx)
        if m.sum() >= 30:
            X = np.column_stack([np.ones(m.sum()), logx[m]])
            _, resid, sig2, _ = _ols(X, logy[m]); R = max(sig2, 1e-8)
        else:
            R = 1e-3
    Vw = delta / (1.0 - delta); Q = Vw * np.eye(2)
    beta = np.array([0.0, beta0]); P = np.eye(2)
    a_o = np.full(n, np.nan); b_o = np.full(n, np.nan)
    e_o = np.full(n, np.nan); s_o = np.full(n, np.nan); z_o = np.full(n, np.nan)
    inited = False
    for t in range(n):
        y, x = logy[t], logx[t]
        if not (math.isfinite(y) and math.isfinite(x)): continue
        if not inited:
            beta = np.array([0.0, beta0]); P = np.eye(2); inited = True
        H = np.array([1.0, x])
        P = P + Q                              # predict (etat = marche aleatoire)
        e = y - float(H @ beta)                # innovation (etat PREDIT -> pas de look-ahead)
        S = float(H @ P @ H) + R
        K = (P @ H) / S
        beta = beta + K * e                    # update
        P = P - np.outer(K, H) @ P
        a_o[t] = beta[0]; b_o[t] = beta[1]
        e_o[t] = e; s_o[t] = S; z_o[t] = e / math.sqrt(S) if S > 0 else np.nan
    return dict(alpha=a_o, beta=b_o, e=e_o, S=s_o, z=z_o, R=R, delta=delta)


def benjamini_hochberg(pvals, q=0.10):
    p = np.asarray(pvals, float); mfin = np.isfinite(p)
    idx = np.where(mfin)[0]; pv = p[idx]; m = len(pv)
    if m == 0: return np.zeros(len(p), dtype=bool), np.nan
    order = np.argsort(pv); thresh = q * np.arange(1, m + 1) / m
    passed = pv[order] <= thresh
    kmax = int(np.max(np.where(passed)[0])) if passed.any() else -1
    rej = np.zeros(len(p), dtype=bool)
    if kmax >= 0: rej[idx[order[:kmax + 1]]] = True
    return rej, (thresh[kmax] if kmax >= 0 else 0.0)
