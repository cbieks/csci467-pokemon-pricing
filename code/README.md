# Pricing Pokémon

CSCI 467 (USC, Fall 2026) course project by Cameron Bieker. Predicts the TCGplayer market price of an English Pokémon TCG card from its printed attributes (and, for the final report, its artwork).

## Data

- Source: [TCGdex API](https://tcgdex.dev) (`https://api.tcgdex.net/v2/en`), snapshot of November 2, 2026.
- `data/cards.csv` is the processed dataset used for every result: 19,080 priced cards with features, target (`log_price`), and split assignments (`split_id`, `split_ood`).
- `data/raw/` (not committed) holds the cached JSON responses. Running `fetch_data.py` today gives *current* prices, not the November 2 snapshot, so use the committed `cards.csv` to reproduce the report.

## Reproduce

Run from this `code/` folder.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere

# 1. Download sets and cards (~10 min, ~21k requests, resumable)
.venv/Scripts/python src/fetch_data.py

# 2. Flatten raw JSON into data/cards.csv and assign splits
.venv/Scripts/python src/build_dataset.py

# 3. Train baselines, Ridge, XGBoost; tune on dev; evaluate on ID and OOD test sets
.venv/Scripts/python src/train.py
```

`train.py` writes to `results/`:

| File | Contents |
|---|---|
| `metrics.json` | Dev/test/OOD metrics per method, per-rarity RMSE, chosen hyperparameters |
| `ridge_grid.csv`, `xgb_grid.csv` | Dev-set hyperparameter sweeps |
| `xgb_importance.csv` | XGBoost gain importance |
| `dev_predictions.csv` | XGBoost dev predictions |
| `dev_error_sample.csv` | Random sample of 50 dev cards off by more than 2x (error analysis) |
| `figures/` | `price_hist.pdf`, `rarity_rmse.pdf` |

All randomness is seeded (`SEED = 0`).
