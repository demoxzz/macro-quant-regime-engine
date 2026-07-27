---
title: "Test E — Vol réalisée comme cible : le piège de la persistance"
type: quant
statut: complet
tier: procedural
confidence: 82
created: 2026-07-24
updated: 2026-07-24
decay-date: 2027-07-24
hallucination-risk: low
validated-by: quant-backtest
topic: quant-vol-realisee
methode: "cibles rvol (niveau) vs rvchg (Δ log-vol forward/trailing) — hold-out OOS"
instruments: ["oil", "gold", "spx", "btc", "vol"]
tags: [type/quant, topic/quant, topic/vol, topic/backtest, statut/complet]
source: "macro_quant_backtest.py (TEST_RVOL, run 2026-07-24)"
sources: ["[[Learning/notions/Volatility Clustering & GARCH]]", "[[Macro/Quant/research/2026-07-23 - Backtest Robustesse (DSR PBO Holdout)]]"]
related: ["[[Macro/Quant/analysis/macro-quant/2026-07-24 - Macro Quant Daily]]", "[[Learning/notions/Vol realisee — niveau vs changement (piege persistance)]]"]
---

# 🔬 Test E — la vol réalisée est-elle une 2ᵉ étoile ?

> **Hypothèse (roadmap E)** : le seul edge validé (VIX) est un signal de **volatilité** (2nd moment). Donc la **vol réalisée** d'autres assets (oil, or, S&P, BTC) devrait aussi être prévisible par le régime → candidate à une 2ᵉ étoile.
> **Verdict : RÉFUTÉ, proprement.** Ce qui semblait un edge massif était un **artefact de persistance**.

## Méthode
Deux cibles forward, testées au hold-out (train <2019 / test ≥2019) :
- **`rvol`** = **niveau** de vol réalisée annualisée (%) sur (t, t+h].
- **`rvchg`** = **changement** = `log(RV_forward / RV_trailing)` — le régime prédit-il l'**expansion/contraction** de vol, **au-delà** de sa persistance ? *C'est le vrai test.*

## Résultat

| Cible | NIVEAU (`rvol`) IC OOS | test hold-out | **Δ VOL (`rvchg`) IC OOS** | test |
|---|---:|---:|---:|---:|
| Vol S&P | **+0,535** (t 8,2) | +0,52 | **+0,013** | +0,09 |
| Vol Oil | **+0,434** (t 8,2) | +0,44 | **−0,006** | −0,04 |
| Vol Or | +0,225 (t 4,3) | +0,39 | **−0,021** | −0,03 |
| Vol BTC | +0,115 | +0,08 | **−0,071** | −0,05 |
| *VIX (réf, Δ implicite)* | — | — | **+0,192** | +0,19 |

## Lecture honnête
- Les IC de **niveau (0,22-0,53)** sont réels et tiennent OOS — **mais entièrement portés par la persistance** (clustering GARCH) : *« la vol reste haute quand elle l'est »*. Le régime capte l'état de vol courant (via `vix_lvl`) et la vol est collante. **Trivial, non incrémental.**
- Quand on retire le niveau et qu'on teste le **changement (`rvchg`)** : **IC ≈ 0 partout**. Le régime **ne prédit PAS l'expansion de vol**. Un modèle naïf « vol_future ≈ vol_passée » capte déjà tout le signal de niveau.
- **Asymétrie clé** : Δ **VIX** (vol *implicite*) survit (0,19) ; Δ vol *réalisée* ≈ 0. → l'edge du VIX ne vit **pas** dans la vol réalisée mais dans la **dynamique de la vol implicite** (mean-reversion du niveau d'implicite / prime de risque de vol). Ce sont deux objets différents.

## Conclusion
**Pas de 2ᵉ étoile.** VIX reste le seul edge validé. Le test illustre le **piège de la persistance** : un IC spectaculaire (0,5) sur un **niveau** hyper-autocorrélé n'est PAS un edge tant qu'on n'a pas prouvé qu'il bat la persistance triviale (tester le **changement**). Sans le contrôle `rvchg`, on « trouvait » 3 faux edges (vol oil/S&P/or).

Fiche méthodo : [[Learning/notions/Vol realisee — niveau vs changement (piege persistance)]].

## Code
`macro_quant_backtest.py` : `TEST_RVOL=True` pour réactiver (désactivé par défaut — pseudo-assets `RV_*`/`RVC_*`). Ne pollue pas le run standard.
