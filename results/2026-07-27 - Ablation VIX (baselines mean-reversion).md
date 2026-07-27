---
title: "Ablation VIX — l'edge est-il autre chose que la réversion du niveau ?"
type: quant
statut: complet
tier: procedural
confidence: 90
created: 2026-07-27
updated: 2026-07-27
decay-date: 2027-07-27
hallucination-risk: low
validated-by: quant-backtest
topic: quant-ablation-vix
methode: "ablation de features + baselines de niveau (k-NN / percentile / OLS causale / term structure / VRP) + résidu orthogonalisé causalement — même harnais walk-forward purgé"
instruments: ["vix", "vixy"]
tags: [type/quant, topic/quant, topic/vol, topic/backtest, statut/complet]
source: "engine/ablation_vix.py + engine/resid_vix.py (run 2026-07-27)"
sources: ["[[Macro/Quant/research/2026-07-14 - Backtest Validation]]", "[[Macro/Quant/research/2026-07-23 - Backtest Robustesse (DSR PBO Holdout)]]", "[[Macro/Quant/research/2026-07-24 - Test E — Vol realisee (piege persistance)]]", "[[Macro/Quant/research/2026-07-24 - C5 Rentabilite — Signal VIX sur VIXY (contango)]]"]
related: ["[[Wiki/macro/Macro-Quant-Methodo]]", "[[Macro/Quant/research/00 - MOC Macro-Quant (etat & chantiers)]]"]
---

# 🔬 Ablation VIX — le seul edge validé survit-il à ses propres baselines ?

> **Objection (retour externe, 27/07)** : la feature `vix_lvl` est dans le vecteur de régime, et la cible est la **variation future du VIX**. Or le VIX est fortement mean-reverting. Le moteur apprend peut-être seulement *« VIX haut → baisse probable »*. Avant de parler d'edge macro, il faut le comparer à : VIX seul, percentile du VIX, régression/AR sur le VIX, VIX + pente de la courbe des futures, baseline HAR/vol.
> **Verdict : l'objection est validée, et plus durement que formulée. Il n'y a pas d'edge macro. L'unique résultat OOS positif est la réversion du niveau du VIX spot, qu'une régression à deux paramètres capture MIEUX que le moteur à 9 features.**

C'est le pendant du [[Macro/Quant/research/2026-07-24 - Test E — Vol realisee (piege persistance)|piège de la persistance]] : là on avait tué un IC de niveau sur la vol réalisée ; ici c'est le **même piège appliqué au VIX lui-même**, et cette fois il emporte l'edge principal.

## Méthode

Harnais **strictement identique** à `macro_quant_backtest.py` — métrique PCA-whitening ré-estimée sur données ≤ t (refresh 63 j), purge `global(j)+h ≤ global(t)`, embargo 5 j, k = max(60 ; 5 % des éligibles), 2012-2026, n ≈ 3630 jours de décision. **Seul le jeu de features varie.** Cible unique : `ΔVIX = VIXCLS[t+h] − VIXCLS[t]` (points).

*Contrôle de fidélité* : le modèle complet reproduit **IC 0,1919 @10j** contre 0,192 publié en §4ter du methodo. Le harnais est le bon.

Toutes les baselines sont **causales et purgées comme le moteur** : l'OLS est refittée chaque jour sur le seul passé dont le label était révolu.

## Test 1 — le moteur vs les baselines de niveau

Horizon 10 j (headline), échantillon commun n = 3630. IC = Pearson, intervalles = stationary block bootstrap (L = h, B = 1000). Colonne de droite = test **apparié** ΔIC(FULL − baseline).

| Modèle | features | IC | IC 5-95 % | ΔIC vs FULL | P(FULL > base) |
|---|---|---:|---|---:|---:|
| **FULL** | les 9 du moteur | **0,192** | [0,079 ; 0,310] | — | — |
| VIXONLY | `vix_lvl` seul (k-NN) | 0,216 | [0,112 ; 0,318] | −0,024 | 0,20 |
| **NOVIX** | **les 8 autres, sans VIX** | **−0,057** | [−0,114 ; 0,006] | +0,249 | 1,00 |
| VIXTS | VIX + pente VIX3M/VIX | 0,258 | [0,159 ; 0,366] | −0,066 | 0,06 |
| PCTL | percentile expanding du VIX | 0,277 | [0,181 ; 0,378] | −0,085 | **0,00** |
| VRP | OLS `ΔVIX ~ VIX + RV22(NDX)` | 0,271 | [0,157 ; 0,392] | −0,079 | **0,00** |
| **OLS** | **OLS causale `ΔVIX ~ a + b·VIX`** | **0,322** | [0,194 ; 0,449] | **−0,130** | **0,00** |

Aux trois horizons (5/10/20 j) : FULL 0,173 / 0,192 / 0,249 · **OLS 0,249 / 0,322 / 0,410** · NOVIX −0,019 / −0,057 / −0,063.

**Trois lectures convergentes :**

1. **Les 8 features macro n'ont aucune information sur ΔVIX.** `NOVIX` = **−0,057**, négatif, et négatif dans 11 années sur 15. Pas « elles ajoutent peu » : elles n'ajoutent rien.
2. **Le moteur est battu significativement par deux paramètres.** ΔIC(FULL − OLS) = −0,130, IC 5-95 % [−0,181 ; −0,076], P(FULL>OLS) = 0,00. Et l'OLS est **positive 15 années sur 15** (0,42 → 0,74), là où FULL tombe à −0,03 (2013) et 0,00 (2023).
3. **Régression incrémentale** `real ~ a + b·pred_base + c·pred_FULL` (prédicteurs standardisés, coeff en points de VIX / 1 σ) :

```
base=OLS      b_base= +1,555 [+1,100,+1,931]    c_FULL= -0,124 [-0,666,+0,326]
base=PCTL     b_base= +1,156 [+0,848,+1,467]    c_FULL= +0,188 [-0,362,+0,641]
base=VIXONLY  b_base= +0,721 [+0,438,+1,004]    c_FULL= +0,471 [-0,010,+0,893]
```

Conditionnellement à **n'importe quelle** baseline de niveau, le coefficient du moteur n'est pas distinguable de zéro — point estimate **négatif** face à l'OLS.

Que l'IC monte avec l'horizon (0,17 → 0,19 → 0,25 pour FULL ; 0,25 → 0,32 → 0,41 pour l'OLS) est la signature exacte d'une réversion vers la moyenne, pas d'un signal macro.

## Test 2 — les features macro expliquent-elles le RÉSIDU du niveau ?

Objection restante au test 1 : le k-NN est un **estimateur bruité** (moyenne sur ~180 voisins) — une partie de l'écart pourrait être de l'inefficacité, pas une absence d'information. Test 2 supprime ce confondant.

À chaque jour t : OLS causale `R_{j,h} = a + b·VIX_j` sur le passé purgé → **résidu réalisé** `e_t = R_{t,h} − (a_t + b_t·VIX_t)` (= l'erreur OOS de la baseline de niveau). On cherche à prédire `e_t`.

**Ce qui reste à expliquer est l'essentiel** : la réversion de niveau n'explique que **5,0 % / 8,1 % / 14,3 %** de la variance de ΔVIX à 5/10/20 j. Si les features macro avaient une once de signal, il y a de la place.

**A. k-NN sur le résidu** (h = 10) :

| Prédicteur du résidu | IC | IC 5-95 % |
|---|---:|---|
| k-NN 8 features macro | −0,088 | [−0,138 ; −0,032] |
| k-NN 9 features (moteur) | +0,004 | [−0,073 ; +0,091] |
| *k-NN `vix_lvl` seul (CONTRÔLE)* | *−0,051* | *[−0,129 ; +0,032]* |

Le contrôle n'est pas exactement nul (l'OLS linéaire laisse un reste **non linéaire** de niveau) : **|IC| ≈ 0,05-0,08 est le plancher d'artefact de la procédure**. Le k-NN macro est exactement à ce plancher → aucun signal, et le signe négatif ne doit pas être sur-interprété.

**B. Version paramétrique — le test décisif** (pas de bruit k-NN, mêmes régresseurs pour tous) :

| Modèle | IC @5j | IC @10j | IC @20j | ΔIC vs BASE @10j | P(AUG>BASE) |
|---|---:|---:|---:|---:|---:|
| `ΔVIX ~ VIX` | 0,249 | **0,322** | 0,410 | — | — |
| `ΔVIX ~ VIX + 8 macro` | 0,172 | **0,233** | 0,336 | **−0,089** [−0,114 ; −0,054] | **0,00** |
| `ΔVIX ~ VIX + ts_slope` | 0,240 | 0,281 | 0,369 | −0,041 [−0,066 ; −0,011] | 0,01 |

**Avec un estimateur efficace, ajouter les 8 features macro DÉGRADE l'IC OOS de façon significative aux trois horizons.** Elles sont un coût de variance d'estimation à rendement informationnel nul. L'objection « c'est juste le k-NN qui est bruité » est éliminée.

**C. Feature par feature — IC(z_f(t), e_t)**, 8 tests (seuil Bonferroni) :

| Feature | IC @5j | IC @10j | IC @20j | Lecture |
|---|---:|---:|---:|---|
| `d10_5`, `dreal_5`, `dbe_5`, `brent_mom` | ≈ 0 | ≈ 0 | ≈ 0 | rien |
| `dusd_5`, `brwti` | −0,03 | −0,04 | −0,06 | sous le plancher d'artefact |
| `growth` | +0,061 | +0,074 | +0,023 | marginal, s'éteint à 20 j |
| **`slope` (2s10s)** | −0,053 | −0,065 | **−0,104** | faible mais **monotone en horizon** |
| **`ts_slope` (VIX3M/VIX)** | **+0,166** | **+0,201** | **+0,208** | **seul signal net — et ce n'est pas une feature macro** |

## Lecture honnête

- **Il n'y a pas d'edge macro.** Ni en k-NN, ni en paramétrique, ni sur le résidu. Les 9 features cross-asset se réduisent, pour tout ce qui est mesurable OOS, à **`vix_lvl`** — et le moteur en fait un usage moins efficace qu'une régression à deux paramètres.
- **Le seul signal résiduel net est la structure par terme du VIX** (`ts_slope`, IC 0,17-0,21 stable aux trois horizons, ~2,5× le plancher d'artefact). Ce n'est **pas** un signal macro : c'est une variable de la surface de vol. Et c'est presque tautologique — **le marché des futures price déjà la réversion attendue du VIX**. C'est exactement le contango qui a tué le trade dans [[Macro/Quant/research/2026-07-24 - C5 Rentabilite — Signal VIX sur VIXY (contango)|C5]]. Prédire le résidu avec la variable qui *facture* ce résidu n'est pas de l'argent gratuit.
- **Tension à noter** : `ts_slope` a un IC de +0,20 avec le résidu, mais l'ajouter linéairement à la régression de niveau **dégrade** l'IC OOS (0,281 vs 0,322). Cause probable : forte colinéarité avec le niveau (le contango est raide quand le VIX est bas) → coefficients instables en expanding. C'est le seul point ouvert du test.
- **`slope` (2s10s)** est le seul candidat macro non trivial : IC résiduel négatif et **croissant en magnitude avec l'horizon** (−0,05 → −0,10). Cohérent économiquement (courbe plate/inversée → vol future plus haute que ne le dit la réversion). Mais 8 tests ont été menés, l'effet est faible, et il n'a pas amélioré la régression augmentée. **À traiter comme une hypothèse à tester proprement, pas comme un résultat.**
- **Rappel de cadrage** : même la baseline gagnante n'explique que **8 % de la variance de ΔVIX @10j**, sur le **spot**, non tradable, et l'IC s'effondre à **0,04 sur VIXY** ([[Macro/Quant/research/2026-07-24 - C5 Rentabilite — Signal VIX sur VIXY (contango)|C5]]) avec un Sharpe long/short de −0,04 et un [[Macro/Quant/research/2026-07-23 - Backtest Robustesse (DSR PBO Holdout)|DSR de 0,18]].

## Conclusion — reformulation défendable

> Le moteur ne possède pas d'edge macro. Son unique résultat OOS positif est la **réversion vers la moyenne du niveau du VIX spot**, qu'une régression causale à deux paramètres capture **mieux** (IC 0,32 vs 0,19 @10j ; 15/15 années positives). Les 8 features macro contribuent **négativement**, y compris avec un estimateur paramétrique efficace et y compris sur le résidu orthogonalisé causalement. Et cette prévision ne survit pas au passage à l'instrument.

La formule « **le vrai edge du modèle = timing de la volatilité** » (§4bis du methodo, note du 14/07) est un **surclaim** : le résultat réel est « **prévision de la réversion du niveau du VIX spot, sous-performant une AR triviale, non monétisable** ».

Ça déplace le projet de *« quel est mon edge ? »* vers *« ai-je un edge du tout ? »*. La réponse actuelle est **non** — résultat négatif propre, publiable, et cohérent avec la discipline anti-auto-illusion du projet.

## Ce que ça ouvre

1. **Ne pas jeter le moteur — changer sa cible.** Il n'a jamais été testé sur ce pour quoi il est bâti : un **classifieur de régime** (reflation/stagflation/risk-off), pas un prédicteur de rendement. Une cible catégorielle honnête reste non testée.
2. **`ts_slope` proprement orthogonalisé** au niveau (au lieu d'ajouté linéairement) — le seul angle où un résidu prédictible existe encore.
3. **`slope` 2s10s** comme hypothèse macro isolée, testée seule, hold-out dédié, sans les 7 autres features autour.
4. **Abandonner ΔVIX spot comme cible de validation.** Toute métrique construite dessus est dominée par la réversion de niveau ; l'unité de compte doit être le P&L d'un instrument.

## Code

`engine/ablation_vix.py` (test 1) et `engine/resid_vix.py` (test 2). Autonomes, numpy seul, réutilisent `yfetch`. Ne touchent pas au run quotidien. Sorties texte complètes régénérables en ~2 et ~7 minutes.
