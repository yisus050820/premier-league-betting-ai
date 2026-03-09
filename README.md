# ⚽ Premier League Betting AI

Machine learning system for detecting value bets on Over 2.5 goals in the Premier League. Trained on 10 seasons of data (2015–2026) using an ensemble of XGBoost, LightGBM, and Random Forest with stacking.

## Performance (backtest, zero data leakage)

| Metric | Value |
|---|---|
| AUC | 0.9408 |
| Win Rate | 80.0% |
| ROI (flat betting) | +48.1% |
| Brier Skill Score | 0.586 |
| p-value | 0.0009 |
| Value bets found | 40 / 357 matches (11.2%) |

## Quick Start

```bash
pip install -r requirements.txt
```

Edit `src/11_weekly_picks.py` with this week's fixtures and odds, then run:

```bash
py src/11_weekly_picks.py
```

## How It Works

1. **Feature engineering** — 50 features including rolling goals, shots, possession, ELO ratings, Poisson expected goals, rest days, and live table position
2. **Ensemble model** — XGBoost + LightGBM + Random Forest combined via a logistic regression meta-model (stacking)
3. **Calibration** — Platt scaling on a held-out validation set to produce reliable probabilities
4. **Value detection** — compares model probability vs bookmaker implied probability; bets only when edge > 5%
5. **Kelly sizing** — fractional Kelly (15%) to size each bet relative to bankroll

## Project Structure

```
├── data/
│   ├── 15-16/          # overview.csv, shots_possession.csv, odds.csv
│   ├── ...
│   └── 25-26/          # current season data
├── models/
│   ├── over25_model.pkl        # production ensemble
│   └── over25_model_eval.pkl   # evaluation model (no test leakage)
├── predictions/
│   ├── value_bets.csv
│   ├── backtest_results.csv
│   └── equity_curve.png
├── processed/
│   └── premier_league_features_v3.csv
└── src/
    ├── config.py                   # features list, Kelly, edge threshold
    ├── predict_helper.py           # ensemble inference helper
    ├── 01_merge_datasets.py        # combine all season CSVs
    ├── 02_feature_engineering.py   # build 50 features (~15 min)
    ├── 03_elo_features.py          # ELO with season decay + margin K
    ├── 04_poisson_model.py         # expected goals via Poisson
    ├── 05_train_model.py           # train XGBoost + LightGBM + RF + stacking
    ├── 06_value_bets.py            # generate value bets on test set
    ├── 07_backtest.py              # simulate P&L
    ├── 08_equity_curve.py          # plot equity curve
    ├── 10_statistical_validation.py # binomial test, bootstrap, Monte Carlo
    └── 11_weekly_picks.py          # ⭐ weekly gameweek analysis
```

## Weekly Usage (5 min)

1. Open `src/11_weekly_picks.py`
2. Update `BANKROLL` with your current balance
3. Update `FIXTURES` with this week's matches and Over 2.5 odds from your bookmaker
4. Run `py src/11_weekly_picks.py`
5. Bet only the matches marked ✅ VALUE BET at the exact stake shown

```python
BANKROLL = 500

FIXTURES = [
    ("Arsenal",   "Chelsea",   1.75),
    ("Liverpool", "Man City",  1.90),
    # format: ("Home", "Away", over_2.5_odds)
]
```

## Monthly Update (20–30 min)

Update `data/25-26/` with new match data, then retrain:

```bash
py src/02_feature_engineering.py
py src/03_elo_features.py
py src/04_poisson_model.py
py src/05_train_model.py
py src/06_value_bets.py
py src/07_backtest.py
py src/10_statistical_validation.py
```

## New Season

1. Create `data/26-27/` with `overview.csv`, `shots_possession.csv`, `odds.csv`
2. Add `"26-27"` to the seasons list in `src/01_merge_datasets.py`
3. Run the full pipeline from `01_merge_datasets.py` onward

## Data Format

Each season folder requires 3 CSV files with matching columns:

**overview.csv** — `id, matchDate, Country, League, Season, homeTeam, awayTeam, referee, FTHG, FTAG, FTR`

**shots_possession.csv** — `id, matchDate, ..., HBPFT, ABPFT, HTSFT, ATSFT, HSONFT, ASONFT, ...`

**odds.csv** — `id, matchDate, ..., H, D, A, O25, U25, O35, U35, BTTSY, BTTSN, ...`

Dates must be in `DD-MM-YY HH:MM` format.

## Model Details

### Ensemble Architecture

```
XGBoost  ──┐
LightGBM ──┼──► Logistic Regression (meta-model) ──► Platt Calibration ──► P(Over 2.5)
Rand Forest┘
```

Meta-model weights learned on validation set (2023–2025):
- XGBoost: 2.71
- LightGBM: 2.20
- Random Forest: 1.70

### Top Features by Importance

1. `combined_attack` — composite of goals + shots on target + shots
2. `away_conceded_last3` / `home_conceded_last3`
3. `away_goals_last3` / `home_goals_last3`
4. `home_goals_at_home_last5` / `home_conceded_at_home_last5`
5. ELO ratings with season decay and goal-margin K factor

### Walk-Forward Validation Split

```
Train:       1665 matches  (Aug 2016 → Nov 2023)
Validation:   357 matches  (Nov 2023 → Jan 2025)  ← calibration + stacking
Test:         357 matches  (Jan 2025 → Mar 2026)  ← never seen during training
```

## Risk Management

- **Kelly fraction**: 15% (conservative) — change `KELLY_FRAC` in `11_weekly_picks.py`
- **Minimum edge**: 5% — change `MIN_EDGE` in `11_weekly_picks.py`
- **Stop loss**: if bankroll drops 20% in a month, stop and review
- **Validation threshold**: run `10_statistical_validation.py` monthly; stop if p-value > 0.10

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=1.7
lightgbm>=3.3
scipy>=1.10
matplotlib>=3.7
joblib>=1.3
```
