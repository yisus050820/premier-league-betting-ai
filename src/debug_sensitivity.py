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

latest_season = df["Season"].iloc[-1]
season_df = df[df["Season"] == latest_season]

# Build Chelsea vs Newcastle manually with exact same formula as 02_feature_engineering
home_team = "Chelsea"
away_team = "Newcastle"

lh = df[df["homeTeam"] == home_team].iloc[-1]
la = df[df["awayTeam"] == away_team].iloc[-1]
sh = season_df[season_df["homeTeam"] == home_team]
sa = season_df[season_df["awayTeam"] == away_team]

# Season averages
s_goals_h = sh["FTHG"].mean()
s_conc_h  = sh["FTAG"].mean()
s_goals_a = sa["FTAG"].mean()
s_conc_a  = sa["FTHG"].mean()

print(f"Chelsea season avg home: goals={s_goals_h:.2f} conceded={s_conc_h:.2f}")
print(f"Newcastle season avg away: goals={s_goals_a:.2f} conceded={s_conc_a:.2f}")
print()

# Recent form
r_goals_h = lh["home_goals_last5"]
r_sot_h   = lh["home_sot_last5"]
r_shots_h = lh["home_shots_last5"]
r_goals_a = la["away_goals_last5"]
r_sot_a   = la["away_sot_last5"]
r_shots_a = la["away_shots_last5"]

print(f"Chelsea recent home: goals={r_goals_h:.2f} sot={r_sot_h:.2f} shots={r_shots_h:.2f}")
print(f"Newcastle recent away: goals={r_goals_a:.2f} sot={r_sot_a:.2f} shots={r_shots_a:.2f}")
print()

# Blended
W = 0.60
b_goals_h = W*s_goals_h + (1-W)*r_goals_h
b_goals_a = W*s_goals_a + (1-W)*r_goals_a
b_sot_h   = r_sot_h
b_sot_a   = r_sot_a
b_shots_h = r_shots_h
b_shots_a = r_shots_a

home_atk = (b_goals_h + b_sot_h/5 + b_shots_h/15) / 3
away_atk = (b_goals_a + b_sot_a/5 + b_shots_a/15) / 3
combined = home_atk + away_atk

print(f"Blended values: home_goals={b_goals_h:.2f} away_goals={b_goals_a:.2f}")
print(f"home_attack_score={home_atk:.3f}  away_attack_score={away_atk:.3f}  combined={combined:.3f}")
print()

# Compare with training distribution
print("Training distribution for combined_attack:")
print(f"  mean={df['combined_attack'].mean():.3f}  std={df['combined_attack'].std():.3f}")
print(f"  p10={df['combined_attack'].quantile(0.1):.3f}  p25={df['combined_attack'].quantile(0.25):.3f}")
print(f"  p50={df['combined_attack'].quantile(0.5):.3f}  p75={df['combined_attack'].quantile(0.75):.3f}")
print(f"  p90={df['combined_attack'].quantile(0.9):.3f}")
print()
print(f"  Our combined_attack={combined:.3f} is at percentile {(df['combined_attack'] < combined).mean()*100:.1f}%")
print()

# Now test: what combined_attack value gives ~55% Over probability?
print("=== SENSITIVITY TEST ===")
print("What prob does the model give for different combined_attack values?")

# Use Chelsea vs Newcastle base row but vary combined_attack
base_row = {
    "home_goals_last5":             b_goals_h,
    "away_goals_last5":             b_goals_a,
    "home_goals_last3":             b_goals_h,
    "away_goals_last3":             b_goals_a,
    "home_goals_ewm":               b_goals_h,
    "away_goals_ewm":               b_goals_a,
    "home_conceded_last5":          W*s_conc_h + (1-W)*lh["home_conceded_last5"],
    "away_conceded_last5":          W*s_conc_a + (1-W)*la["away_conceded_last5"],
    "home_conceded_last3":          lh["home_conceded_last3"],
    "away_conceded_last3":          la["away_conceded_last3"],
    "home_goals_at_home_last5":     b_goals_h,
    "home_conceded_at_home_last5":  W*s_conc_h + (1-W)*lh["home_conceded_last5"],
    "away_goals_away_last5":        b_goals_a,
    "away_conceded_away_last5":     W*s_conc_a + (1-W)*la["away_conceded_last5"],
    "home_shots_last5":             r_shots_h,
    "away_shots_last5":             r_shots_a,
    "home_sot_last5":               r_sot_h,
    "away_sot_last5":               r_sot_a,
    "home_pos_last5":               lh["home_pos_last5"],
    "away_pos_last5":               la["away_pos_last5"],
    "home_shot_accuracy":           lh["home_shot_accuracy"],
    "away_shot_accuracy":           la["away_shot_accuracy"],
    "home_conversion_last5":        lh["home_conversion_last5"],
    "away_conversion_last5":        la["away_conversion_last5"],
    "home_form_last5":              lh["home_form_last5"],
    "away_form_last5":              la["away_form_last5"],
    "form_diff":                    lh["home_form_last5"] - la["away_form_last5"],
    "h2h_avg_goals":                3.0,
    "h2h_over25_rate":              0.6,
    "home_attack_score":            home_atk,
    "away_attack_score":            away_atk,
    "combined_attack":              combined,
    "home_elo":                     lh["home_elo"],
    "away_elo":                     la["away_elo"],
    "elo_diff":                     lh["home_elo"] - la["away_elo"],
    "elo_home_win_prob":            1/(1+10**((la["away_elo"]-lh["home_elo"]-65)/400)),
    "elo_sum":                      lh["home_elo"] + la["away_elo"],
    "exp_home_goals":               (b_goals_h + W*s_conc_a+(1-W)*la["away_conceded_last5"]) / 2,
    "exp_away_goals":               (b_goals_a + W*s_conc_h+(1-W)*lh["home_conceded_last5"]) / 2,
}

test_values = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]
print(f"  {'combined_attack':>16}  {'Prob O2.5':>10}")
for val in test_values:
    r = base_row.copy()
    r["combined_attack"] = val
    row_df = pd.DataFrame([r])
    raw = xgb_model.predict_proba(row_df[FEATURES])[:,1].reshape(-1,1)
    prob = calibrator.predict_proba(raw)[0,1]
    marker = " <-- current" if abs(val - combined) < 0.15 else ""
    print(f"  {val:>16.1f}  {prob:>9.1%}{marker}")