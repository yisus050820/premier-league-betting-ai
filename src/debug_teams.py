import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES

xgb_model, calibrator = joblib.load("models/over25_model.pkl")

df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed")
df = df.sort_values("matchDate").reset_index(drop=True)

fixtures = [
    ("Chelsea",        "Newcastle"),
    ("Arsenal",        "Everton"),
    ("West Ham",       "Manchester City"),
    ("Manchester Utd", "Aston Villa"),
    ("Liverpool",      "Tottenham"),
]

# Show training data distribution for key features
print("=== TRAINING DATA RANGES (for context) ===")
for col in ["combined_attack", "home_goals_last5", "away_goals_last5", "home_conceded_last5"]:
    print(f"  {col}: min={df[col].min():.2f} mean={df[col].mean():.2f} max={df[col].max():.2f} p25={df[col].quantile(0.25):.2f} p75={df[col].quantile(0.75):.2f}")

print()
print("=== FEATURE VALUES PER FIXTURE ===")

for home_team, away_team in fixtures:
    lh = df[df["homeTeam"] == home_team].iloc[-1]
    la = df[df["awayTeam"] == away_team].iloc[-1]

    combined = lh["home_attack_score"] + la["away_attack_score"]

    exp_home = (lh["home_goals_last5"] + la["away_conceded_last5"]) / 2
    exp_away = (la["away_goals_last5"] + lh["home_conceded_last5"]) / 2

    print(f"\n{home_team} vs {away_team}:")
    print(f"  home_attack_score:    {lh['home_attack_score']:.3f}  (from {lh['matchDate']})")
    print(f"  away_attack_score:    {la['away_attack_score']:.3f}  (from {la['matchDate']})")
    print(f"  combined_attack:      {combined:.3f}")
    print(f"  home_goals_last5:     {lh['home_goals_last5']:.2f}")
    print(f"  away_goals_last5:     {la['away_goals_last5']:.2f}")
    print(f"  home_conceded_last5:  {lh['home_conceded_last5']:.2f}")
    print(f"  away_conceded_last5:  {la['away_conceded_last5']:.2f}")
    print(f"  exp_home_goals:       {exp_home:.2f}")
    print(f"  exp_away_goals:       {exp_away:.2f}")
    print(f"  home_elo:             {lh['home_elo']:.0f}")
    print(f"  away_elo:             {la['away_elo']:.0f}")