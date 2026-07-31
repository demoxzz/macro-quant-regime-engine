---
title: "MOC Macro-Quant — état du projet & chantiers"
type: moc
statut: en-cours
tier: working
confidence: 80
created: 2026-07-24
updated: 2026-07-24
decay-date: 2027-07-24
hallucination-risk: low
validated-by: self
topic: macro-quant-moc
tags: [type/moc, topic/quant, topic/research, statut/en-cours]
source: ""
sources: []
related: ["[[Cockpit Quant]]", "[[Wiki/macro/Macro-Quant-Methodo]]"]
---

# 🗺️ MOC Macro-Quant — état du projet & chantiers

> **Doc vivante de recherche.** Point d'entrée unique du projet Couche 2 (base rates conditionnels au régime). Organisé **par chantier**, avec l'état validé (source de vérité), le journal des tests, le backlog d'idées à challenger, et les retours externes. À tenir à jour à chaque avancée.

## 🎯 Vision
Dire **à quelle fréquence** un régime historiquement comparable a été suivi de tel move (probabiliste, forward), pour dimensionner la conviction. **But final ≠ un signal joli statistiquement, mais un edge EXPLOITABLE net de coûts.** Complémentaire de la Couche 1 (AMT/niveaux live).

---

## 🚧 Chantiers

| # | Chantier | Statut | Détail |
|---|---|:--:|---|
| **C0** | Cadre & méthodo (features causales, Mahalanobis, anti-lookahead, winsor ±2,5σ, flag dominance) | ✅ | [[Wiki/macro/Macro-Quant-Methodo]] |
| **C1** | Validation OOS (walk-forward, DSR/PBO/hold-out, multi-horizon, t-stat, médiane) | ✅ | [[Macro/Quant/research/2026-07-14 - Backtest Validation]] · [[Macro/Quant/research/2026-07-23 - Backtest Robustesse (DSR PBO Holdout)]] |
| **C2** | Univers d'assets (tester candidats, garder les validés) | 🔄 | MOVE/BTC/Or/vol réalisée testés → réfutés (cf. journal ↓) |
| **C3** | Jambes du régime (inflation + **croissance** cuivre/or → reflation vs stagflation) | ✅ | daily [[Macro/Quant/analysis/macro-quant/2026-07-24 - Macro Quant Daily]] |
| **C4** | Base de données & vintage point-in-time + lecture cross-day | ✅ | `Macro/Quant/db/SCHEMA.md` · `analyze_db.py` |
| **C5** | **VERS LA RENTABILITÉ** (instrument tradable, coûts, strat dynamique) | 🔄 | **Phase 1 faite** : edge VIX **non tradable standalone** (IC 0,19→0,04 sur VIXY, contango) ; piste = **overlay de dé-risquage sur carry**. [[Macro/Quant/research/2026-07-24 - C5 Rentabilite — Signal VIX sur VIXY (contango)]] |
| **C6** | **AMT regime-detection** (projet sœur, pas v1) — quand le fade/retour-à-la-valeur marche | 🔄 **actif** | look-ahead levé ; 60m réfuté MAIS **à 5m edge causal ROBUSTE** (fade extension ≥2,5 ATR en balance, SIG multi-H, hit 61-70% ; trend=0) → **FirstRate multi-années justifié** (caveat : 60j only). [[Macro/Quant/amt-regime/2026-07-27 - C6 step1 — Thermometre causal (hypothese refutee 60m)]] |

---

## ✅ Source de vérité — état validé (OOS)

| Asset / cible | Statut OOS | Rôle |
|---|:--:|---|
| **VIX** (Δ implicite) | ✅ **VALIDÉ** — IC 0,19 @10j (t 3,6), 12-14/15 ans, hold-out 0,19 ; **⚠️ capture ~1% de la queue** | **seul edge — timing de vol EN MOYENNE. PAS une protection anti-krach** (aveugle aux spikes) |
| Actions / USD / FX / taux / oil (direction) | ❌ bruit OOS | contexte de régime |
| **MOVE** (vol taux) | ❌ réfuté (IC instable) | contexte |
| **BTC** (direction) | ❌ réfuté (overfit : IS 0,17 → OOS −0,06) | contexte |
| **Or** (direction) | ❌ réfuté (IC nég. train+test) | contexte |
| **Vol réalisée** (oil/S&P/or) | ❌ réfuté (piège persistance : niveau 0,5 mais Δvol ≈ 0) | — |

> **Règle** : conclusion directionnelle **uniquement** pour un asset à IC OOS significatif → **VIX seul**. Tout le reste = contexte, jamais un pari.

---

## 🧪 Journal des tests (validés / réfutés)
| Date | Test | Verdict | Note |
|---|---|---|---|
| 14/07 | Backtest v1 (8 assets, direction) | VIX seul validé | [[Macro/Quant/research/2026-07-14 - Backtest Validation]] |
| 23/07 | DSR + PBO + hold-out (façon paires) | VIX tient ; strat naïve échoue DSR ; PBO~100% si tri par Sharpe | [[Macro/Quant/research/2026-07-23 - Backtest Robustesse (DSR PBO Holdout)]] |
| 23/07 | MOVE (vol taux) | réfuté | Robustesse §4bis |
| 23/07 | BTC (direction) | réfuté (overfit) | daily 23/07 |
| 24/07 | Or (direction) | réfuté | daily 24/07 |
| 24/07 | Winsor ±2,5σ | ✅ renforce VIX (0,17→0,20) | daily 24/07 |
| 24/07 | Jambe croissance (cuivre/or) | ✅ gardée (VIX survit) — tranche reflation/stagflation | daily 24/07 |
| 24/07 | Vol réalisée (E) | réfuté (persistance) | [[Macro/Quant/research/2026-07-24 - Test E — Vol realisee (piege persistance)]] |
| 24/07 | **C5 — signal sur VIXY (contango)** | edge **non tradable standalone** (IC 0,04) ; overlay carry surestimé ; rien ne passe DSR | [[Macro/Quant/research/2026-07-24 - C5 Rentabilite — Signal VIX sur VIXY (contango)]] |
| 24/07 | **Test de queue (critique PG)** | signal **aveugle aux krachs** : capture ~1% de la queue (mars 2020 +49 prédit −2) ; **winsor innocenté** (1% avec/sans). IC moyen ≠ résistance de queue | méthodo §v1.1 + C5 §queue |
| 27/07 | **C6 step-1 — thermomètre AVWAP causal** | look-ahead levé ; **hypothèse RÉFUTÉE à 60m** (artefact) | [[Macro/Quant/amt-regime/2026-07-27 - C6 step1 — Thermometre causal (hypothese refutee 60m)]] |
| 27/07 | **C6 step-2 — test 5m (gratuit Yahoo) + sweep H** | à 5m **edge causal ROBUSTE** : fade extension ≥2,5 ATR en **balance** = +0,25→+0,71 ATR, SIG sur 4 H/5, hit 61-70% ; **trend = 0** (thèse vindiquée). Caveat : 60j only → **FirstRate multi-années justifié** | même note (step-2) |

---

## 📥 Backlog — idées à challenger (moi / IA / externes)
- ⬜ **[buy-side] C5 — instrument tradable** : tester le signal VIX sur **VIXY / VX futures** (subit le **contango** ~80% du temps). *L'edge survit-il au roll ?* = make-or-break.
- ⬜ **[buy-side] C5 — modèle de coûts** (spread + commissions + slippage) → **Sharpe net déflaté**. Réutiliser le modèle coûts des paires actions.
- ⬜ **[buy-side] C5 — stratégie dynamique** : sizing ∝ conviction + filtre term-structure (ne longer la vol que si contango faible/backwardation), au lieu du long/short au seuil.
- ⬜ **[buy-side] C6 — AMT regime-detection** : le moteur doit dire **quand** le régime mean-reversion-à-la-valeur est actif (vs trending) → edge institutionnel (prix d'équilibre + volume/temps).
- ⬜ **[moi] 2ᵉ proxy croissance** : cycliques/défensives (XLY/XLP) en complément du cuivre/or.
- ⬜ **[moi] covariance expanding** pour la métrique Mahalanobis (causalité pleine, retirer le léger look-ahead sur la métrique).
- ⬜ **[moi] robustesse** : sensibilité du lift à k, L, fenêtre de départ ; élargir les périodes de test (conseil buy-side #7).

---

## 🗣️ Retours externes (log)
| Source | Verdict | Ce que ça a déclenché |
|---|---|---|
| **PG** (guidage initial) | idée cross-asset → base rates | tout le projet ; cf. [[Learning/2026-07-15 - Stationnarite, Lead-Lag & Regimes cross-asset (guidage PG)]] |
| **MW** (auto-éval) | discipline OOS, tuer ses faux edges | pivot paires + rigueur ; [[Learning/2026-07-17 - Auto-evaluation recherche quant & positionnement (regard MW)]] |
| **Mentor (AMT)** | « signature d'un vrai » ; n faible + feature dominante ; multi-horizon/t-stat ; **jambe croissance manquante** | A (winsor+dominance) · B (oil frais) · C (multi-horizon/t) · **D (jambe croissance)** — tous appliqués 24/07 |
| **un intervenant buy-side** | « très solide V1 » ; il manque **la réalité** : oublier spot VIX, calc de coûts, strat dynamique ; bosser par chantier ; AMT-regime = jackpot | → **chantier C5** (rentabilité) + cette doc + C6 |

---

## 🔗 Ressources
- **Méthodo** : [[Wiki/macro/Macro-Quant-Methodo]] · **Cockpit** : [[Cockpit Quant]] · **Schéma DB** : `Macro/Quant/db/SCHEMA.md`
- **Fiches notions** : [[Learning/notions/00 - MOC Quant Research (notions)]] (IC, t-stat, DSR/PBO, vol clustering, persistance, look-ahead…)
- **Code** : `Macro/Quant/engine/` (`macro_quant_engine.py`, `macro_quant_backtest.py`, `macro_quant_daily.py`, `analyze_db.py`)
