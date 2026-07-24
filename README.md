# Macro Quant — Regime Engine (base rates conditionnels au régime)

Petit moteur perso. L'idée : décrire le **régime macro cross-asset du jour** (taux, courbe,
vol, FX, commos, indices) comme un vecteur de features standardisées de façon **causale**,
retrouver ses **k plus proches voisins historiques** (Mahalanobis via whitening PCA), et en
tirer des **base rates conditionnels** sur les rendements forward — avec `n_eff`, lift vs
baseline, et IC par **stationary block bootstrap** (fenêtres forward chevauchantes → t-stat
naïf faux).

Data : **FRED** (séries quotidiennes, fetch en direct) + `^MOVE` via Yahoo. Deps : `numpy`
+ stdlib. Rien à installer d'autre, aucune clé API.

---

## ⚠️ Le verdict honnête d'abord (lis `results/` pour le détail)

Je me suis appliqué les 3 juges de López de Prado. Résultat, sans enjoliver :

- **Le signal VIX-régime est réel et persistant.** Hold-out TRAIN <2019 / TEST ≥2019
  (COVID+2022+2025-26 jamais vus) : l'IC passe **0,189 → 0,161**, garde ~85%. Aucun autre
  asset n'a d'IC exploitable.
- **La stratégie de trading naïve NE monétise PAS cet IC.** Deflated Sharpe sur 24 configs :
  le VIX plafonne à **DSR 0,18**, personne n'atteint 0,95. → signal ≠ stratégie. C'est un
  **overlay de risque / timing de vol**, pas une machine à Sharpe au seuil brut.
- **Trier les assets par Sharpe de backtest = surapprentissage quasi-certain.** PBO (CSCV)
  = **99,98 %**. Le seul tri qui survit OOS est le tri par **IC hors-échantillon**, et il ne
  garde que le VIX.
- Pour toute **direction** (actions/USD/oil/taux) : le moteur est **muet, et honnête de l'être**.
- Hypothèse « MOVE prévisible comme le VIX » → **testée, réfutée** (IC porté par 2020 seul,
  négatif en 2022 et 2026). Voir §4bis de la note robustesse.

Bref : je ne te vends pas un edge magique. Je te partage un cadre où **ce qui marche et ce
qui ne marche pas est mesuré proprement** — c'est là-dessus que ton regard risk/quant m'intéresse.

---

## Anti-look-ahead (les garde-fous)

- Standardisation **expanding** (mu/sigma sur `[0..t]` only), forward returns `t→t+h`.
- IC via **stationary block bootstrap** (Politis-Romano 1994), L = h, pour l'autocorr induite
  par le chevauchement. `n_eff = |A| / h` (obs. indépendantes effectives, pas `|A|` brut).
- Backtest **walk-forward causal + purgé**. Hold-out = sélection de l'asset **sur le train seul**.
- Tag confiance : 🔴 `n_eff<20` · 🟡 20–60 · 🟢 >60 ; « signal » seulement si l'IC bootstrap
  **exclut** la valeur inconditionnelle.

---

## Lancer

```bash
python3 engine/macro_quant_engine.py     # régime du jour + base rates forward (console)
python3 engine/macro_quant_backtest.py   # backtest + appendix robustesse (DSR/PBO/hold-out)
python3 engine/macro_quant_daily.py      # persiste le run du jour dans la db
```

Premier run = téléchargement FRED (caché dans `/tmp/fredcache`). `numpy` requis
(`pip install numpy`). Les scripts `make_*_charts.py` de mon repo (matplotlib) ne sont pas
inclus — pas nécessaires pour lire les résultats.

---

## Contenu

```
README.md                  <- ce fichier
engine/
  macro_quant_engine.py    <- moteur : régime → k-NN Mahalanobis → base rates + bootstrap
  macro_quant_backtest.py  <- backtest walk-forward + robustesse (DSR, PBO/CSCV, hold-out)
  macro_quant_daily.py     <- run quotidien persisté
  qstat.py                 <- stats (bootstrap, DSR, PBO, IC…)
  yfetch.py                <- fetch Yahoo (^MOVE, absent de FRED)
methodo/
  Macro-Quant-Methodo.md   <- méthodo détaillée + formules
results/
  2026-07-14 - Backtest Validation.md
  2026-07-23 - Backtest Robustesse (DSR PBO Holdout).md   <- ⭐ le plus intéressant
  2026-07-24 - Macro Quant Daily.md
  figures/                 <- fig18 (DSR), fig19 (hold-out), dashboard db
db/
  SCHEMA.md, base_rates.csv, regime_features.csv, runs/  <- snapshots persistés
```

## Ce sur quoi ton avis m'aiderait

1. La **monétisation vol** : passer d'un IC 0,16 sur le VIX à un instrument réel net de coûts
   (structure de terme VX, straddles delta-hedgés, sizing conditionnel). C'est le trou du v2.
2. La **sensibilité du hold-out** à la date de coupure (2016 / 2019 / 2021).
3. Tout **leak causal** que j'aurais raté — c'est ma hantise sur ce genre de setup.
