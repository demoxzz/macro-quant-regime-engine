---
title: "Macro Quant Engine — Méthodologie & formules (v1)"
type: quant
statut: en-cours
tier: procedural
confidence: 60
created: 2026-07-14
updated: 2026-07-14
decay-date: 2027-01-14
hallucination-risk: low
validated-by: self
topic: quant-methodo
methode: "base rates conditionnels au régime (k-NN Mahalanobis) + block bootstrap"
instruments: ["taux", "credit", "vol", "fx", "commodities", "indices"]
tags: [type/quant, topic/quant, topic/macro, statut/en-cours]
source: "FRED (Federal Reserve Economic Data)"
sources: ["[[Macro/Quant/engine/macro_quant_engine.py]]"]
related: ["[[Wiki/macro/Cadres-de-Lecture]]", "[[Wiki/macro/Cadre-Oil-Benchmarks-Spreads]]", "[[Learning/2026-07-15 - Stationnarite, Lead-Lag & Regimes cross-asset (guidage PG)]]", "[[CLAUDE]]"]
---

# Macro Quant Engine — Méthodologie & formules (v1)

> **Couche 2** de la vision à deux étages. La Couche 1 (daily AMT/niveaux) dit **OÙ** est le prix (spatial, contemporain, mise en condition). La Couche 2 (ce moteur) dit **À QUELLE FRÉQUENCE** un régime comparable a été suivi de tel move (probabiliste, forward, dimensionnement de conviction). **Complémentaires, pas redondantes.**

> Principe directeur, en réponse à la critique quant : *un seuil (« 4,6 % = régime hawkish activé ») ou un « gardien » n'a de valeur que si l'histoire montre un **lift** vs le baseline, mesuré avec un **n** honnête et un **IC robuste**.* Ce moteur remplace les affirmations posées par des base rates datés + intervalles.

---

## 0. Ce que le moteur produit

Pour le **régime du jour** (vecteur de features macro standardisées), il trouve les **k jours historiques les plus analogues**, puis pour chaque asset et chaque horizon *h* ∈ {5, 10, 20 j} calcule :
- **P_cond** = fréquence conditionnelle (ex. P(asset baisse | régime))
- **LIFT** = P_cond − P_uncond (le seul chiffre qui compte : l'écart au hasard)
- **mean_cond / median_cond** du rendement forward
- **IC 90 %** par block bootstrap, **n_eff**, **tag de confiance** 🔴🟡🟢

Un résultat n'est un **signal** que si l'IC bootstrap **exclut** la valeur inconditionnelle.

---

## 1. Univers (FRED, quotidien)

| Rôle | Séries |
|---|---|
| **Conditionnement + réponse** (`both`) | DGS2, DGS10, DFII10 (10Y réel TIPS), T10YIE (breakeven), T10Y2Y (pente), VIXCLS, DTWEXBGS (USD broad), DEXJPUS, DCOILBRENTEU |
| **Réponse seule** (`resp`) | DFF, DGS5, DGS30, BAMLH0A0HYM2 (HY OAS), BAMLC0A0CM (IG OAS), DEXUSEU, DCOILWTICO, DHHNGSP (NatGas), NASDAQCOM, SP500, DJIA |

**Type de transform par nature d'actif :**
- `price` (indices, FX, commodities) → **log-rendements**
- `yield` (taux, spreads crédit, OAS) → **variations en points de base (bps)**
- `vol` (VIX) → **niveau** (features) et **variation en points** (forward)

---

## 2. Nettoyage des données (méticuleux — anti-look-ahead)

1. **Calendrier maître** = jours ouvrés du marché obligataire (DGS10). Toutes les séries y sont réindexées.
2. **Forward-fill LIMITÉ à 3 jours** ouvrés : comble un décalage de férié entre marchés (bonds fermés / actions ouvertes) **sans fabriquer** de données au-delà. > 3 j de trou → NaN.
3. **Outliers** : z robuste MAD, |z| > 8 **flaggé, jamais supprimé** (les jours 2008/2020 sont des faits, pas du bruit).
   - z robuste : `z = 0.6745·(x − médiane)/MAD`, `MAD = médiane(|x − médiane|)`.
4. **Jour valide** = jour où **toutes** les features de conditionnement existent (pas d'imputation silencieuse dans la matrice de régime).

> **Contrainte d'échantillon (v1)** : la fenêtre valide démarre **2007-01** (limitée par DFII10/T10YIE = TIPS/breakeven depuis 2003 + warmup 252 j). HY/IG OAS ne remontant qu'à 2023-07 **dans cet environnement FRED**, ils sont **exclus du conditionnement** (sinon la matrice s'écrase à 2 ans) et gardés en **asset-réponse seulement** — d'où leur n_eff faible et leur tag 🔴.

---

## 3. Formules

### (1) Prix → log-rendements
$$r_t = \ln(P_t / P_{t-1}), \qquad R_{t,h} = \ln(P_{t+h}/P_t)\times 100 \ \ [\%]$$

### (2) Taux / spreads → bps
$$\Delta_k(t) = (y_t - y_{t-k})\times 100 \ [\text{bps}], \qquad \Delta^{fwd}_h(t) = (y_{t+h}-y_t)\times 100$$

### (3) Vol → niveau (features) / points (forward)
$$\text{fwd}_h^{VIX}(t) = V_{t+h} - V_t \ [\text{points VIX}]$$

### (4) Standardisation **causale** (expanding z, warmup W₀ = 252)
$$z_t = \frac{x_t - \mu_{0:t}}{\sigma_{0:t}}, \quad \mu,\sigma \text{ calculés sur } [0..t] \text{ SEULEMENT}$$
→ aucun jour n'utilise d'information postérieure à *t*. C'est le cœur de l'anti-look-ahead.

### (5) Vecteur de régime
Features (k_impulse = 5 j) : `d10_5` (Δ10Y nominal), `dreal_5` (Δ10Y réel), `dbe_5` (Δbreakeven), `vix_lvl`, `slope` (2s10s niveau), `dusd_5` (var. 5 j USD broad %), `brwti` (spread Brent−WTI), `brent_mom` (momentum Brent 20 j).
Régime du jour = `v* = (z_f)` au dernier jour valide.

### (6) Distance de Mahalanobis via whitening PCA
Sur la matrice de corrélation **R** des z-features : `R = V Λ Vᵀ` (eigen-décomposition).
$$p_t = V^\top z_t \ \text{(projection PCA)}, \qquad D(t) = \sqrt{\sum_j \frac{(p_{t,j}-p^*_j)^2}{\lambda_j}}$$
Le whitening par λ_j **égalise les axes corrélés** : un régime n'est pas surpondéré parce que deux features bougent ensemble. Les **analogues A** = k plus proches voisins par D(t), avec **exclusion du voisinage direct** |i−j| > 20 (anti-recouvrement trivial). k = max(120, 5 % de l'échantillon).

> *Caveat causalité v1* : la métrique (matrice de covariance) est estimée en plein échantillon → léger look-ahead **sur la métrique de distance uniquement** ; features et labels restent strictement causaux. **v2** : covariance expanding.

### (7) Base rate conditionnel + lift
$$P_{cond} = \frac{1}{|A|}\sum_{t\in A}\mathbf{1}[R_{t,h}<0], \qquad \boxed{\text{LIFT} = P_{cond}-P_{uncond}}$$
P_uncond = même stat sur **tous** les jours valides (baseline). Sans lift, un base rate ne dit rien.

### (8) IC par **stationary block bootstrap** (Politis-Romano 1994)
Les fenêtres forward se **chevauchent** (R_{t,h} et R_{t+1,h} partagent h−1 jours) → t-stat naïf **faux** (autocorrélation induite). On rééchantillonne des **blocs** de longueur géométrique moyenne L = h (prob. de redémarrage p = 1/L), B = 2000 répétitions → percentiles 5/95.

### (9) n effectif honnête
$$n_{eff} = |A| / h$$
243 analogues à h = 10 → **24,3 obs indépendantes**, pas 243. C'est ce n_eff qui pilote le tag.

### (10) Tags de confiance
| Tag | n_eff | Signal ? |
|---|---|---|
| 🔴 | < 20 | fragile, indicatif |
| 🟡 | 20–60 | exploitable avec prudence |
| 🟢 | > 60 | robuste |

**Signal = OUI** uniquement si l'IC 90 % bootstrap **exclut** la valeur inconditionnelle.

---

## 4. Lecture — ce que le moteur PEUT et NE PEUT PAS dire

**PEUT** : « historiquement, dans un régime dont la signature de features ressemble à aujourd'hui (n_eff obs indép.), l'asset X a été négatif Y % du temps à 10 j, soit +Z pts vs baseline, IC [a,b] excluant le hasard. »

**NE PEUT PAS** : prédire *ce* move. Base rate ≠ prévision. Un régime « défavorable aux indices 60 % du temps » laisse 40 % de contre-exemples. Le moteur **dimensionne la conviction**, il ne remplace pas la Couche 1.

**Limites v1 explicites :**
- Fenêtre 2007-2026 → un seul grand cycle de resserrement + QE massif ; sur-représentation post-GFC.
- k-NN Mahalanobis suppose une pertinence **linéaire** des features ; pas d'interactions non linéaires.
- Métrique de distance plein-échantillon (cf. caveat §6).
- SP500/DJIA sur FRED = **2016+ seulement** → n_eff bas, tag 🔴 (préférer NASDAQCOM qui remonte à 1971).
- Or spot quotidien absent (séries FRED discontinuées) → **trou connu**, à combler v2.
- **Latence FRED** : Brent/WTI/NatGas s'arrêtent ~3-4 j avant la date courante → le régime « du jour » est daté à la **dernière donnée complète** (honnête, pas intraday).

---

## 4bis. Validation OOS (backtest walk-forward) — RÈGLE DURE

Un base rate n'a de valeur que s'il **prédit hors-échantillon**. Backtest causal + purgé (2012-2026, 3630 jours) — détail : [[2026-07-14 - Backtest Validation]].

**Verdict** : sur 8 assets testés, **seul le VIX** a un IC OOS significatif (**+0,170 @10j, t=+3,22, \*\*\***, stable 13/15 années). Nasdaq (IC −0,036), USD (≈0), FX, Brent, taux → **pas de skill directionnel OOS** (t<1). Le flag « signal » contemporain de l'engine v1 est **trompeur** (il a flaggé USD/NQ, nuls OOS).

> **RÈGLE** : la Couche 2 n'émet une **conclusion forward directionnelle** que pour un asset à **IC OOS significatif** (aujourd'hui : VIX seul). Les autres base rates sont montrés en **contexte de régime**, jamais comme un pari directionnel. Le vrai edge du modèle = **timing de la volatilité**, pas la direction actions/FX/taux.

## 4ter. v1.1 (2026-07-24) — améliorations post-review mentor
- **(A) Winsorizing ±2,5σ** sur toutes les features avant Mahalanobis → un outlier de crise (`brent_mom` −2,6σ) ne peut plus écraser la distance. **Mesuré : RENFORCE le VIX** (IC OOS 0,170→**0,202**, hold-out test 0,16→0,19, 14/15 ans). Moteur + backtest.
- **(A) Flag de dominance** : part de chaque feature (`z_i²/Σz_j²`) ; **flag si >40%** (matching quasi-univarié). Dans JSON + note + dashboard. Ex. 24/07 : `brent_mom` = 57%.
- **(B) Oil frais** : Brent/WTI FRED prolongés depuis le **dernier jour réel** avec les **futures Yahoo** (BZ=F/CL=F), return-chaining (écrase le ffill périmé). Corrige la latence **et** dé-périme `brent_mom` (stale le sous-estimait : +1,18→+1,64σ).
- **(C) Multi-horizon + t-stat VIX** surfacés : IC 5/10/20j = 0,19/0,20/0,27 ; t = 5,0/3,8/3,7.
- **(D) JAMBE CROISSANCE** = `growth` = momentum 20j **cuivre/or** (HG=F/GC=F). 9ᵉ feature de conditionnement. C'est l'axe qui manquait : reflation (croissance↑) et stagflation (croissance↓) **partagent la jambe inflation** → sans growth, le moteur ne peut pas les distinguer. Signe + = growth-on. **VIX survit** (IC OOS 0,192, hold-out test 0,188, 12/15 ans). Ex. 24/07 : growth +0,64 → **reflation confirmée par calcul** (pas stagflation). Schema DB → **v2**.
- **Non-FRED via Yahoo** : MOVE `^MOVE`, BTC `BTC-USD`, Or `GC=F`, cuivre `HG=F` — MOVE/BTC/Or resp-only **réfutés OOS** ; cuivre = input feature growth (VIX reste seul asset validé).

## 5. Roadmap v2 (reste à faire)
- **(E) Cible vol réalisée** (S&P/oil/BTC) : chercher une 2ᵉ étoile sur le **2nd moment** (pas la direction).
- Second proxy croissance (cycliques/défensives XLY/XLP) en complément du cuivre/or.
- Covariance **expanding** pour la métrique (causalité pleine).
- Test de robustesse : sensibilité du lift à k, L, fenêtre de départ.
- Covariance **expanding** pour la métrique (causalité pleine).
- Séparer un **sous-régime oil** du régime taux/vol.
- Test de robustesse : sensibilité du lift à k, L, fenêtre de départ.
