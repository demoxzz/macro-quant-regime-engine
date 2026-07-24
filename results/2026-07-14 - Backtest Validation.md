---
title: Backtest Validation — Macro Quant Engine v1 (walk-forward causal 2012-2026)
type: quant
statut: complet
tier: procedural
confidence: 75
created: 2026-07-14
updated: 2026-07-14
decay-date: 2027-07-14
hallucination-risk: low
validated-by: quant-backtest
topic: quant-backtest
methode: walk-forward causal + purge, IC / quintiles / calibration / strat, block bootstrap
instruments:
  - vix
  - taux
  - fx
  - commodities
  - indices
tags:
  - type/quant
  - topic/quant
  - topic/backtest
  - statut/complet
source: FRED via macro_quant_backtest.py (run 2026-07-14, 3630 jours de decision)
sources:
  - "[[Wiki/macro/Macro-Quant-Methodo]]"
  - "[[Macro/Quant/engine/macro_quant_backtest.py]]"
related:
  - "[[2026-07-14 - Macro Quant Daily]]"
---

# 🔬 Backtest Validation — Macro Quant Engine v1

> **Question** : le modèle de base rates conditionnels au régime a-t-il un **pouvoir prédictif hors-échantillon**, et peut-on faire confiance aux probas du jour ? **Réponse courte : oui pour le VIX uniquement ; non pour tout le reste.**

## Protocole (anti-triche)
Walk-forward **strictement causal + purgé**, 2012-01 → 2026-07, **3630 jours de décision** :
- Métrique de whitening PCA **ré-estimée sur données ≤ t** (rafraîchie tous les 63 j) — pas de leak de covariance (corrige le caveat full-sample de l'engine).
- Analogues pris **uniquement dans le passé** (j < t) **ET** fenêtre forward révolue (**PURGE** : global(j)+h ≤ global(t)) → label de l'analogue connu à t, zéro recouvrement avec le futur de t.
- Prédiction = base rate des analogues passés ; label = R_{t,h} réalisé après t.
- Métriques : **IC** (corr pred↔réalisé), monotonie **quintiles**, **calibration** P(neg)/Brier, **stratégie causale** (seuil = médiane glissante) sur jours **non chevauchants**, découpage **par année**. Signif. via block bootstrap + t = IC·√(n_eff−2), n_eff = n/h.

## Résultat principal — horizon 10 j

| Asset | IC (Pearson) | IC (Spearman) | t-stat | Signif. | Q5−Q1 | Strat Sharpe (vs BH) | Verdict |
|---|---:|---:|---:|:--:|---:|---:|---|
| **VIX** | **+0,170** | **+0,182** | **+3,22** | **\*\*\*** | **+2,03 pt** | **+0,23** (−0,02) | ✅ **signal réel** |
| Pente 2s10s | +0,047 | +0,052 | +0,90 | — | +1,37 bps | +0,26 (−0,19) | bruit |
| UST 10Y | +0,041 | +0,036 | +0,78 | — | +1,83 bps | +0,05 (+0,23) | bruit |
| Brent | +0,030 | +0,050 | +0,57 | — | +0,79% | −0,05 (−0,07) | bruit |
| UST 2Y | +0,010 | −0,004 | +0,19 | — | +0,24 bps | −0,05 (+0,39) | bruit |
| USD broad | −0,002 | +0,002 | −0,03 | — | −0,07% | +0,18 (+0,38) | ❌ zéro skill |
| EUR/USD | −0,009 | −0,023 | −0,17 | — | −0,20% | −0,34 (−0,12) | bruit |
| **Nasdaq** | **−0,036** | −0,018 | −0,68 | — | −0,12% | −0,19 (+0,85) | ❌ **pire que random** |

*(\*\*\* survit à une correction de Bonferroni pour 8 tests : seuil t≈2,7. Le VIX est le seul à passer.)*

## Le VIX en détail (le seul edge validé)

| Horizon | IC | t-stat | Q5−Q1 | Strat mean | Strat Sharpe | vs Buy&Hold |
|---|---:|---:|---:|---:|---:|---:|
| 5 j | +0,148 | +3,99 | +1,25 pt | +0,44 pt | **+0,90** | −0,02 |
| 10 j | +0,170 | +3,22 | +2,03 pt | +0,23 pt | +0,23 | −0,02 |
| 20 j | +0,227 | +3,04 | +3,63 pt | +0,81 pt | +0,55 | −0,02 |

**IC par année (13/15 positifs)** : 2012 +0,05 · 2013 −0,09 · 2014 +0,21 · 2015 +0,38 · 2016 +0,18 · 2017 +0,22 · 2018 +0,26 · 2019 +0,22 · 2020 +0,04 · 2021 +0,42 · 2022 +0,34 · 2023 +0,01 · 2024 +0,40 · 2025 +0,42 · 2026 +0,20.
→ **Stable et signé** dans presque tous les régimes. La vol cluster et mean-reverte de façon conditionnelle au régime — le modèle capte ça. Cohérent avec la littérature (la vol est la variable macro la plus prévisible ; la direction actions à 10 j ≈ marche aléatoire + drift, d'où l'IC nul sur NQ).

## Interprétation honnête
1. **Le modèle a UN edge : le timing de la volatilité.** Le régime dit « la vol va-t-elle monter/baisser » avec un IC modeste (~0,17) mais robuste et significatif. C'est exploitable (VIX, VXX, straddles, dimensionnement du risque).
2. **Les « signaux » directionnels sur actions / USD / FX / taux ne survivent PAS OOS.** Le flag « signal YES » de l'engine v1 (basé sur l'IC contemporain excluant le baseline) est **trompeur** : il a flaggé USD et NQ, tous deux à IC OOS ≈ 0. → **ne jamais trader la direction de ces assets sur la base de ce modèle.**
3. **Validation empirique de la critique quant** : la majorité des seuils/biais directionnels étaient des corrélations contemporaines sans persistance forward. Le backtest les élimine et garde ce qui tient.

## Impact sur le système
- **Couche 2 (`/macro-quant`)** : n'émet une conclusion **forward directionnelle** que pour les assets à **IC OOS significatif** (aujourd'hui : **VIX seul**). Les autres base rates sont affichés en **contexte** (position du régime), marqués « pas de skill OOS — direction non exploitable ».
- **Engine v1** : remplacer le flag « signal » contemporain par le **filtre IC OOS** de ce backtest. Porter la **métrique causale** (expanding, rafraîchie) en v2 — ce backtest en est le prototype validé.
- **Probas du jour (09/07)** : la seule lecture forward défendable = **VIX pred +1,6 pt @10j (IC 0,170)** → régime historiquement suivi d'une remontée modeste de vol. NQ −0,73% / USD +0,41% / Brent / taux → **ignorer la direction** (IC ≈ 0). ⚠️ le caveat vintage Brent (pré-spike) joue peu ici puisque ces assets ne sont de toute façon pas prévisibles ; côté VIX, un choc oil non capté ne ferait que **renforcer** le biais vol-up.

## Limites du backtest & v2
- IC 0,17 = **modeste** en absolu (mais réel et tradeable pour la vol). Pas un Graal.
- Un seul grand cycle (2012-2026, post-GFC/QE) → sur-représentation d'un régime de vol structurellement basse.
- Multiple testing géré (Bonferroni) mais 8 assets × 3 horizons ; VIX passe, les marginaux non.
- **v2** : winsoriser les features extrêmes (`brent_mom` à −2,6σ domine le matching), séparer un sous-régime oil, ajouter MOVE (vol taux — probablement aussi prévisible que le VIX), tester la sensibilité à k et à la fenêtre de départ, et vérifier si l'edge VIX tient net de coûts sur un instrument réel.
