---
title: "C5 — Rentabilité : le signal VIX survit-il au contango sur VIXY ?"
type: quant
statut: complet
tier: procedural
confidence: 80
created: 2026-07-24
updated: 2026-07-24
decay-date: 2027-07-24
hallucination-risk: low
validated-by: quant-backtest
topic: c5-rentabilite-vixy
methode: "signal causal VIX -> position VIXY (ETF futures court-terme), net de coûts, hold-out + DSR"
instruments: ["vixy", "vix"]
tags: [type/quant, topic/quant, topic/vol, topic/rentabilite, statut/complet]
source: "macro_quant_backtest.py (appendix C5, run 2026-07-24)"
sources: ["[[Macro/Quant/research/00 - MOC Macro-Quant (etat & chantiers)]]", "[[Macro/Quant/research/2026-07-23 - Backtest Robustesse (DSR PBO Holdout)]]"]
related: []
---

# 🔬 C5 — L'edge VIX est-il TRADABLE ? (le test du contango)

> **Question (retour buy-side)** : le signal +0,19 IC est sur le **spot VIX**, non tradable. Sur l'instrument réel (**VIXY**, ETF futures VIX court-terme), le **contango** fait saigner le long-vol (~80% du temps). Le signal survit-il ?
> **Verdict : l'edge n'est PAS directement tradable en long-vol. Mais il a de la valeur comme OVERLAY de dé-risquage sur un carry short-vol.**

## La réalité du contango
**VIXY buy&hold 2012→2026 : −99,996%** (573 920 → 21,78, splits ajustés). Être long vol te **ruine** — le spot VIX oscille (12-80) mais l'instrument roule et bleed. C'est LA réalité que le spot cache.

## Le chiffre qui tranche
| IC du signal contre… | valeur |
|---|---:|
| ΔVIX **spot** (ce qu'on avait) | **+0,192** |
| **rendement VIXY** (l'instrument tradable) | **+0,043** |

→ le pouvoir prédictif **s'effondre de 0,19 à 0,04** sur l'instrument réel. Le contango décorrèle le signal du P&L tradable.

## Stratégies (pas non chevauchants 10j, net de coûts 5-15 bps)
| Stratégie | Sharpe | ret/an | maxDD | Shp <2019 | Shp ≥2019 |
|---|---:|---:|---:|---:|---:|
| B&H VIXY (long vol) | −1,06 | −71% | −1014 | −1,32 | −0,89 |
| Short-vol (carry nu) | **+1,06** | +71% | −151 | +1,31 | +0,89 |
| Signal long/flat | −0,93 | −37% | −539 | −1,41 | −0,55 |
| Signal long/short | −0,04 | ≈0 | −282 | −0,46 | +0,26 |
| **Carry + filtre vol-up** | +0,62 | +34% | **−101** | +0,53 | +0,69 |

## Lecture honnête
1. **Le signal seul ne fait pas un edge tradable long-vol** : *long/short* ≈ **breakeven** (Sharpe −0,04), *long/flat* saigne. Le spot-edge est **statistique, pas directement monétisable**. buy-side avait raison.
2. **Le seul truc qui gagne = SHORT vol (carry)** — mais c'est la **prime de risque de vol** (bien connue), **pas mon edge**, et son maxDD ici (−151) **sous-estime massivement** le vrai risque de queue (Volmageddon 2018, mars 2020 = −80/−96% en jours ; l'échantillonnage 10j les lisse). *« Ramasser des pièces devant un rouleau compresseur ».*
3. **La VRAIE valeur du signal = filtre de risque sur le carry** : *« Carry + filtre vol-up »* a un Sharpe plus bas que le carry nu (0,62 vs 1,06) **MAIS le plus petit drawdown (−101)** ET une **meilleure stabilité OOS** (train 0,53 / test 0,69, vs 1,31/0,89 pour le carry nu). → le régime dit **QUAND réduire l'exposition short-vol** avant un spike. Usage institutionnel légitime.

## Caveats (ne pas se raconter d'histoires)
- **DSR de la meilleure @15bps = 0,58 → PAS robuste** (5 stratégies testées). Rien ne passe encore le filtre strict de déflation.
- Le **risque de queue du short-vol est sous-estimé** par l'échantillonnage non chevauchant → ne PAS célébrer le Sharpe 1,06.
- Robuste aux coûts (5 vs 15 bps ≈ identique) → ce n'est **pas** les coûts le tueur, c'est le **contango**.

## Conclusion & suite
**C5 phase 1 = verdict clair** : edge VIX non tradable en standalone ; **piste réelle = overlay de dé-risquage sur carry short-vol** (à approfondir). Reste à faire :
- Equity curve **quotidienne** (overlapping) pour révéler le vrai risque de queue du short-vol.
- **Sizing dynamique** ∝ conviction + filtre term-structure (ratio VIX/VIX3M) au lieu du seuil binaire.
- Tester d'autres instruments (VX future front + roll explicite, ou options straddle delta-hedgées).
- **Ne rien trader tant que le DSR net ne passe pas.**

Code : `macro_quant_backtest.py` appendix C5. Cf. [[Macro/Quant/research/00 - MOC Macro-Quant (etat & chantiers)]].
