import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_helper import load_model, predict_proba as ensemble_predict
from config import FEATURES, VAL_RATIO, MIN_EDGE, MAX_MODEL_PROB, MIN_KELLY, KELLY_FRACTION

print("Loading evaluation model (trained on past only)...")

# Use the eval model — NOT the production model
# The eval model was never trained on the test set
model_bundle = load_model("models/over25_model_eval.pkl")

print("Loading dataset...")

df = pd.read_csv("processed/premier_league_features_v3.csv")

df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)

# ------------------------------------------------
# USE ONLY THE TEST PERIOD (last 15% of data)
# Same split as training script
# ------------------------------------------------

split_val = int(len(df) * VAL_RATIO)
df = df.iloc[split_val:].copy()

print(f"Test period: {df['matchDate'].min().date()} to {df['matchDate'].max().date()}")
print(f"Matches:     {len(df)}")

features = FEATURES

# ------------------------------------------------
# PREDICT
# ------------------------------------------------

print("Predicting probabilities...")

df["model_prob_over25"] = ensemble_predict(model_bundle, df[features])

# Book implied probability (remove vig)
df["book_prob_over25"] = 1 / df["O25"]

# Raw edge
df["value"] = df["model_prob_over25"] - df["book_prob_over25"]

# ------------------------------------------------
# KELLY CRITERION (fractional)
# Tells you what % of bankroll to bet
# f = (b*p - q) / b
# b = decimal odds - 1
# p = model probability
# q = 1 - p
# We use 1/4 Kelly for safety
# ------------------------------------------------

def kelly_fraction(prob, odds, fraction=0.25):
    b = odds - 1
    p = prob
    q = 1 - p
    k = (b * p - q) / b
    k = max(k, 0)               # never bet negative kelly
    return round(k * fraction, 4)

df["kelly"] = df.apply(
    lambda r: kelly_fraction(r["model_prob_over25"], r["O25"]),
    axis=1
)

# ------------------------------------------------
# FILTER VALUE BETS
# Edge > 5% AND model prob < 80% (avoid extreme predictions)
# AND kelly > 0 (model says bet)
# ------------------------------------------------

value_bets = df[
    (df["value"] > MIN_EDGE) &
    (df["model_prob_over25"] < MAX_MODEL_PROB) &
    (df["kelly"] > MIN_KELLY)
].copy()

print(f"\nValue bets found: {len(value_bets)} out of {len(df)} matches ({len(value_bets)/len(df):.1%})")

# ------------------------------------------------
# OUTPUT COLUMNS
# ------------------------------------------------

value_bets = value_bets[[
    "matchDate",
    "homeTeam",
    "awayTeam",
    "FTHG",
    "FTAG",
    "total_goals",
    "O25",
    "model_prob_over25",
    "book_prob_over25",
    "value",
    "kelly"
]].sort_values("value", ascending=False)

print("")
print("Top 15 value bets by edge:")
print(value_bets.head(15).to_string(index=False))

value_bets.to_csv("predictions/value_bets.csv", index=False)
print("\nSaved to predictions/value_bets.csv")