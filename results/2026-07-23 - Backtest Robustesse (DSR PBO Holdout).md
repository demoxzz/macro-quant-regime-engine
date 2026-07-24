---
title: Backtest Robustesse — Regime Engine façon paires (DSR + PBO + Hold-out)
type: quant
statut: complet
tier: procedural
confidence: 78
created: 2026-07-23
updated: 2026-07-23
decay-date: 2027-07-23
hallucination-risk: low
validated-by: quant-backtest
topic: quant-backtest
methode: Deflated Sharpe Ratio (Bailey-LdP 2014) + PBO/CSCV (Bailey 2017) + vrai hold-out train/test
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
source: FRED via macro_quant_backtest.py (run 2026-07-23, robustness appendix)
sources:
  - "[[2026-07-14 - Backtest Validation]]"
  - "[[Wiki/macro/Macro-Quant-Methodo]]"
  - "[[Macro/Quant/engine/macro_quant_backtest.py]]"
related:
  - "[[2026-07-14 - Macro Quant Daily]]"
  - "[[2026-07-23 - Macro Quant Daily]]"
  - "[[2026-07-22 - Pairs Futures (edition tradable)]]"
---

# 🔬 Backtest Robustesse — Regime Engine (même rigueur que le dossier paires)

> **Question** : le backtest v1 disait « VIX seul a un IC OOS réel ». Mais on n'avait pas appliqué au régime les 3 juges de López de Prado utilisés sur les paires : **Deflated Sharpe** (le Sharpe survit-il à 24 essais ?), **PBO** (choisir le meilleur asset = surapprendre ?), **vrai hold-out** (tient-il sur un bloc jamais vu ?). Fait maintenant. Verdict : **le signal VIX passe le hold-out ; la stratégie de trading naïve échoue à la déflation ; trier par Sharpe est un piège quasi-certain.**

Run walk-forward causal + purgé 2012→2026 (3635 jours), appendix robustesse dans [macro_quant_backtest.py](Macro/Quant/engine/macro_quant_backtest.py). Figures : [fig18](fig18_regime_dsr.png), [fig19](fig19_regime_holdout.png).

---

## 1. Hold-out — le juge le plus dur (TRAIN <2019 / TEST ≥2019 jamais vu)

Sélection de l'asset **uniquement sur le TRAIN** (meilleur IC), jugement sur un TEST qui contient **COVID 2020 + 2022 + 2025-26** — jamais vu à la sélection.

| Asset | IC train (<2019) | IC test (≥2019) | SR test (ann.) | |
|---|---:|---:|---:|:--|
| **VIX** | **+0,189** | **+0,161** | **+0,37** | ⬅ **pick IS → tient OOS** |
| Pente 2s10s | +0,075 | +0,013 | −0,26 | s'effondre OOS |
| UST 10Y | +0,047 | +0,037 | +0,03 | ≈ 0 |
| Nasdaq | +0,013 | **−0,066** | −0,08 | pire que random OOS |
| Brent | +0,013 | +0,034 | +0,06 | bruit |
| USD broad | +0,006 | −0,025 | +0,08 | bruit |
| UST 2Y | −0,001 | +0,012 | +0,02 | bruit |
| EUR/USD | −0,017 | +0,005 | −0,19 | bruit |

➡️ **L'asset qu'on aurait choisi en aveugle sur le train (VIX) reste le meilleur hors-échantillon et garde ~85% de son IC** (0,189 → 0,161). Aucun autre n'a d'IC exploitable ni avant ni après. **C'est la validation forte : l'edge VIX-régime est réel et persistant dans le temps**, pas un artefact de période.

---

## 2. Deflated Sharpe — la *stratégie* naïve ne convertit PAS l'IC en Sharpe robuste

Univers d'essais = 8 assets × 3 horizons = **24 configs**, V(SR)=0,0024. DSR = proba que le vrai Sharpe batte le max attendu sous le null de N essais.

| Asset | SR/période | Sharpe ann. | DSR N=8 | DSR N=24 | DSR N=72 |
|---|---:|---:|---:|---:|---:|
| Pente 2s10s | 0,051 | 0,26 | 0,35 | 0,19 | 0,10 |
| **VIX** | 0,047 | **0,23** | 0,33 | **0,18** | 0,10 |
| USD broad | 0,036 | 0,18 | 0,25 | 0,12 | 0,06 |
| UST 10Y | 0,011 | 0,05 | 0,13 | 0,05 | 0,02 |
| … (autres) | <0 | <0 | <0,06 | <0,02 | <0,01 |

➡️ **Aucun asset n'atteint DSR 0,95.** Même le VIX plafonne à 0,18. **Nuance capitale** : l'edge existe au niveau du **signal (IC 0,16 hold-out)**, mais la **stratégie de trading naïve** (long/short vol au seuil = médiane glissante, jours non chevauchants) **ne le monétise pas** en un Sharpe déflaté. → Le VIX-régime doit servir d'**overlay de dimensionnement du risque / timing de vol**, pas de stratégie PnL standalone au seuil brut. Monétisation à retravailler (structure de terme VIX, straddles, sizing conditionnel).

---

## 3. PBO (CSCV, 16 blocs) — trier les assets par Sharpe = surapprentissage quasi-certain

**PBO = 99,98 %** · meilleur asset IS (par Sharpe full-sample) = **2s10s**, qui **ne tient pas OOS** (IC test +0,013, SR test −0,26). logit(λ) médian = −2,08.

➡️ Choisir l'asset « qui a le meilleur backtest Sharpe » tombe sous la médiane OOS ~100 % du temps. **Ne JAMAIS classer par Sharpe de backtest** — c'est du bruit sur-ajusté. Le seul tri qui survit est le **tri par IC OOS**, et il ne garde que le VIX (§1).

---

## 4. Synthèse — ce que les 3 juges ajoutent au backtest v1

| Dimension | v1 (14/07) | Robustesse (23/07) | Apport |
|---|---|---|---|
| Signal réel ? | VIX IC 0,17 | **VIX survit au hold-out** (0,19→0,16) | ✅ confirme + prouve la persistance temporelle |
| Stratégie tradable ? | Sharpe 0,23 « ok » | **DSR 0,18 — échoue à la déflation** | ⚠️ le Sharpe brut était trompeur : signal ≠ stratégie |
| Tri des assets ? | Bonferroni | **PBO ~100 % si tri par Sharpe** | ❌ interdit de sélectionner par Sharpe ; IC OOS uniquement |

**Conclusion honnête** (fidèle à la discipline anti-auto-illusion [[feedback_quant_no_self_deception]]) : le régime cross-asset **prédit la volatilité, et seulement elle**, de façon robuste et persistante. Mais **la valeur est dans le signal, pas dans un PnL au seuil naïf** — il faut un module d'exécution vol dédié pour l'exploiter, sinon c'est un thermomètre de risque, pas une machine à Sharpe. Pour toute **direction** (actions/USD/oil/taux) : muet, et honnête de l'être.

---

## 4bis. Test **MOVE (vol taux)** — hypothèse « prévisible comme le VIX » RÉFUTÉE

MOVE ajouté au panel de réponse (source Yahoo `^MOVE`, 2002→, via `yfetch`). Hypothèse : la vol taux mean-reverte comme la vol actions → devrait passer le hold-out. **Testé, rejeté.**

| Test | VIX | MOVE | Verdict MOVE |
|---|---:|---:|:--|
| IC OOS full (10j) | +0,169 | **+0,020** | ≈ 0 |
| IC hold-out train / test | +0,189 / +0,161 | **−0,031 / +0,059** | signe instable |
| DSR (N=24) | 0,18 | 0,012 | néant |
| IC par année | **+ dans 13/15 ans** | **erratique** | pas d'edge stable |

**IC MOVE année par année** : 2020 **+0,44** · 2021 +0,28 — mais 2012 −0,08 · 2013 −0,18 · 2016 −0,17 · 2019 −0,09 · **2022 −0,26** · 2023 −0,02 · 2024 −0,03 · **2026 −0,30**. → l'IC agrégé n'est porté que par **2020** ; MOVE est **négatif dans les grosses années de vol taux (2022, 2026)**, l'inverse de ce qu'exigerait un vrai edge. La vol taux ne mean-reverte pas de façon régime-conditionnelle exploitable dans ce cadre.

**Contre-test** (test équitable) : ajout d'une feature de conditionnement `move_lvl` symétrique à `vix_lvl` → IC MOVE monte à +0,055 mais reste porté par 2020 seul, et la feature perturbe la géométrie du régime validé sans gain robuste. **Rejeté** (degré de liberté que le §3 PBO condamne). Régime gardé à **8 features**. MOVE conservé comme **asset-réponse de contexte** (affiché dans le daily, marqué non-tradable en direction), pas comme edge validé.

> Enseignement : l'edge VIX n'est **pas** « la vol est prévisible » en général — c'est spécifiquement **la vol actions**, et en grande partie parce que le régime **conditionne sur `vix_lvl`**. La symétrie vol-actions↔vol-taux ne tient pas empiriquement.

## 5. À faire (v2)
- Module de monétisation vol propre (term structure VIX/VX, straddles delta-hedgés) → re-tester la DSR sur un instrument réel net de coûts.
- ~~Ajouter MOVE~~ → **fait, réfuté** (§4bis).
- Refaire le hold-out avec split 2016 puis 2021 (sensibilité à la date de coupure).
- **Gold** toujours manquant (trou data FRED).
