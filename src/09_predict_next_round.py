import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES

print("Loading model...")
xgb_model, calibrator = joblib.load("models/over25_model.pkl")

print("Loading historical dataset...")
df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed", dayfirst=False)
df = df.sort_values("matchDate").reset_index(drop=True)

# ------------------------------------------------
# NEXT ROUND FIXTURES — update every gameweek
# ------------------------------------------------

fixtures = [
    ("Chelsea",        "Newcastle"),
    ("Arsenal",        "Everton"),
    ("West Ham",       "Manchester City"),
    ("Manchester Utd", "Aston Villa"),
    ("Liverpool",      "Tottenham"),
]

# ------------------------------------------------
# BLEND RATIO
# Recent form can be noisy at season end / start
# 60% season average + 40% last-5 = more stable predictions
# Change to 0.0/1.0 if you have fresh in-season data
# ------------------------------------------------

SEASON_WEIGHT = 0.60
RECENT_WEIGHT = 0.40

# ------------------------------------------------
# BUILD TEAM STATS
# ------------------------------------------------

print("Building team stats (season avg + recent form blend)...")

teams = pd.unique(df[["homeTeam", "awayTeam"]].values.ravel())
team_stats = {}

# Get most recent season in data
latest_season = df["Season"].iloc[-1]
season_df = df[df["Season"] == latest_season]

for team in teams:
    home_games    = df[df["homeTeam"] == team].sort_values("matchDate")
    away_games    = df[df["awayTeam"] == team].sort_values("matchDate")
    season_home   = season_df[season_df["homeTeam"] == team]
    season_away   = season_df[season_df["awayTeam"] == team]

    if len(home_games) == 0 and len(away_games) == 0:
        continue

    last_home = home_games.iloc[-1] if len(home_games) > 0 else None
    last_away = away_games.iloc[-1] if len(away_games) > 0 else None
    last_any  = pd.concat([home_games, away_games]).sort_values("matchDate").iloc[-1]

    # ELO from most recent match
    elo = last_any["home_elo"] if last_any["homeTeam"] == team else last_any["away_elo"]

    # Season averages (stable baseline)
    s_goals_h    = season_home["FTHG"].mean() if len(season_home) > 0 else df[df["homeTeam"]==team]["FTHG"].mean()
    s_goals_a    = season_away["FTAG"].mean()  if len(season_away) > 0 else df[df["awayTeam"]==team]["FTAG"].mean()
    s_conc_h     = season_home["FTAG"].mean()  if len(season_home) > 0 else df[df["homeTeam"]==team]["FTAG"].mean()
    s_conc_a     = season_away["FTHG"].mean()  if len(season_away) > 0 else df[df["awayTeam"]==team]["FTHG"].mean()

    # Recent form features (from last home/away match)
    def get_h(col, fallback):
        return float(last_home[col]) if last_home is not None else fallback
    def get_a(col, fallback):
        return float(last_away[col]) if last_away is not None else fallback

    # Blended goals and conceded
    def blend_h(recent_val, season_val):
        return RECENT_WEIGHT * recent_val + SEASON_WEIGHT * season_val
    def blend_a(recent_val, season_val):
        return RECENT_WEIGHT * recent_val + SEASON_WEIGHT * season_val

    blended_goals_h   = blend_h(get_h("home_goals_last5",    s_goals_h), s_goals_h)
    blended_goals_a   = blend_a(get_a("away_goals_last5",    s_goals_a), s_goals_a)
    blended_conc_h    = blend_h(get_h("home_conceded_last5", s_conc_h),  s_conc_h)
    blended_conc_a    = blend_a(get_a("away_conceded_last5", s_conc_a),  s_conc_a)

    # Shot / possession stats (recent only — less noisy)
    shots_h   = get_h("home_shots_last5",   12.0)
    shots_a   = get_a("away_shots_last5",   10.0)
    sot_h     = get_h("home_sot_last5",     4.0)
    sot_a     = get_a("away_sot_last5",     3.5)
    pos_h     = get_h("home_pos_last5",     52.0)
    pos_a     = get_a("away_pos_last5",     48.0)
    acc_h     = get_h("home_shot_accuracy", 0.33)
    acc_a     = get_a("away_shot_accuracy", 0.35)
    conv_h    = get_h("home_conversion_last5", 0.15)
    conv_a    = get_a("away_conversion_last5", 0.15)
    form_h    = get_h("home_form_last5",    1.2)
    form_a    = get_a("away_form_last5",    1.0)

    # Attack score using blended goals
    home_attack = (blended_goals_h + sot_h/5 + shots_h/15) / 3
    away_attack = (blended_goals_a + sot_a/5 + shots_a/15) / 3

    team_stats[team] = {
        # Blended home stats
        "home_goals_last5":             blended_goals_h,
        "home_goals_last3":             blend_h(get_h("home_goals_last3", s_goals_h), s_goals_h),
        "home_goals_ewm":               blend_h(get_h("home_goals_ewm",   s_goals_h), s_goals_h),
        "home_conceded_last5":          blended_conc_h,
        "home_conceded_last3":          blend_h(get_h("home_conceded_last3", s_conc_h), s_conc_h),
        "home_goals_at_home_last5":     blended_goals_h,
        "home_conceded_at_home_last5":  blended_conc_h,
        "home_shots_last5":             shots_h,
        "home_sot_last5":               sot_h,
        "home_pos_last5":               pos_h,
        "home_shot_accuracy":           acc_h,
        "home_conversion_last5":        conv_h,
        "home_form_last5":              form_h,
        "home_attack_score":            home_attack,
        # Blended away stats
        "away_goals_last5":             blended_goals_a,
        "away_goals_last3":             blend_a(get_a("away_goals_last3", s_goals_a), s_goals_a),
        "away_goals_ewm":               blend_a(get_a("away_goals_ewm",   s_goals_a), s_goals_a),
        "away_conceded_last5":          blended_conc_a,
        "away_conceded_last3":          blend_a(get_a("away_conceded_last3", s_conc_a), s_conc_a),
        "away_goals_away_last5":        blended_goals_a,
        "away_conceded_away_last5":     blended_conc_a,
        "away_shots_last5":             shots_a,
        "away_sot_last5":               sot_a,
        "away_pos_last5":               pos_a,
        "away_shot_accuracy":           acc_a,
        "away_conversion_last5":        conv_a,
        "away_form_last5":              form_a,
        "away_attack_score":            away_attack,
        # ELO
        "elo": elo,
    }

# ------------------------------------------------
# BUILD PREDICTION ROWS
# ------------------------------------------------

rows = []

for home_team, away_team in fixtures:
    missing = [t for t in [home_team, away_team] if t not in team_stats]
    if missing:
        print(f"  WARNING: {missing} not found in data — skipping")
        continue

    h = team_stats[home_team]
    a = team_stats[away_team]

    elo_diff          = h["elo"] - a["elo"]
    elo_home_win_prob = 1 / (1 + 10 ** ((a["elo"] - h["elo"] - 65) / 400))
    elo_sum           = h["elo"] + a["elo"]

    exp_home_goals  = (h["home_goals_last5"] + a["away_conceded_last5"]) / 2
    exp_away_goals  = (a["away_goals_last5"] + h["home_conceded_last5"]) / 2
    combined_attack = h["home_attack_score"] + a["away_attack_score"]

    past_h2h = df[
        ((df["homeTeam"] == home_team) & (df["awayTeam"] == away_team)) |
        ((df["homeTeam"] == away_team) & (df["awayTeam"] == home_team))
    ].tail(5)
    h2h_avg_goals   = past_h2h["total_goals"].mean() if len(past_h2h) >= 2 else df["total_goals"].mean()
    h2h_over25_rate = past_h2h["over25"].mean()      if len(past_h2h) >= 2 else df["over25"].mean()

    row = {
        "home_goals_last5":             h["home_goals_last5"],
        "away_goals_last5":             a["away_goals_last5"],
        "home_goals_last3":             h["home_goals_last3"],
        "away_goals_last3":             a["away_goals_last3"],
        "home_goals_ewm":               h["home_goals_ewm"],
        "away_goals_ewm":               a["away_goals_ewm"],
        "home_conceded_last5":          h["home_conceded_last5"],
        "away_conceded_last5":          a["away_conceded_last5"],
        "home_conceded_last3":          h["home_conceded_last3"],
        "away_conceded_last3":          a["away_conceded_last3"],
        "home_goals_at_home_last5":     h["home_goals_at_home_last5"],
        "home_conceded_at_home_last5":  h["home_conceded_at_home_last5"],
        "away_goals_away_last5":        a["away_goals_away_last5"],
        "away_conceded_away_last5":     a["away_conceded_away_last5"],
        "home_shots_last5":             h["home_shots_last5"],
        "away_shots_last5":             a["away_shots_last5"],
        "home_sot_last5":               h["home_sot_last5"],
        "away_sot_last5":               a["away_sot_last5"],
        "home_pos_last5":               h["home_pos_last5"],
        "away_pos_last5":               a["away_pos_last5"],
        "home_shot_accuracy":           h["home_shot_accuracy"],
        "away_shot_accuracy":           a["away_shot_accuracy"],
        "home_conversion_last5":        h["home_conversion_last5"],
        "away_conversion_last5":        a["away_conversion_last5"],
        "home_form_last5":              h["home_form_last5"],
        "away_form_last5":              a["away_form_last5"],
        "form_diff":                    h["home_form_last5"] - a["away_form_last5"],
        "h2h_avg_goals":                h2h_avg_goals,
        "h2h_over25_rate":              h2h_over25_rate,
        "home_attack_score":            h["home_attack_score"],
        "away_attack_score":            a["away_attack_score"],
        "combined_attack":              combined_attack,
        "home_elo":                     h["elo"],
        "away_elo":                     a["elo"],
        "elo_diff":                     elo_diff,
        "elo_home_win_prob":            elo_home_win_prob,
        "elo_sum":                      elo_sum,
        "exp_home_goals":               exp_home_goals,
        "exp_away_goals":               exp_away_goals,
    }
    rows.append((home_team, away_team, row))

# ------------------------------------------------
# CLIP TO TRAINING RANGE
# ------------------------------------------------

pred_df = pd.DataFrame([r for _, _, r in rows])
for col in pred_df.columns:
    if col in df.columns:
        pred_df[col] = pred_df[col].clip(
            lower=df[col].quantile(0.01),
            upper=df[col].quantile(0.99)
        )

# ------------------------------------------------
# PREDICT
# ------------------------------------------------

print("Predicting probabilities...")

raw_probs = xgb_model.predict_proba(pred_df[FEATURES])[:, 1].reshape(-1, 1)
probs     = calibrator.predict_proba(raw_probs)[:, 1]

# ------------------------------------------------
# DISPLAY
# ------------------------------------------------

print("")
print("=" * 70)
print(f"  NEXT ROUND PREDICTIONS — Over 2.5 Goals  (season blend {int(SEASON_WEIGHT*100)}/{int(RECENT_WEIGHT*100)})")
print("=" * 70)
print(f"  {'Home':<22} {'Away':<22} {'Prob O2.5':>10}  {'xG Home':>7}  {'xG Away':>7}  Signal")
print("-" * 70)

for i, (home_team, away_team, row) in enumerate(rows):
    prob = probs[i]
    xgh  = row["exp_home_goals"]
    xga  = row["exp_away_goals"]
    if prob >= 0.70:
        signal = "STRONG ✅"
    elif prob >= 0.60:
        signal = "LIKELY"
    elif prob >= 0.50:
        signal = "LEAN"
    else:
        signal = "UNDER ❌"
    print(f"  {home_team:<22} {away_team:<22} {prob:>9.1%}  {xgh:>7.2f}  {xga:>7.2f}  {signal}")

print("=" * 70)
print(f"\nNote: Using {int(SEASON_WEIGHT*100)}% season avg + {int(RECENT_WEIGHT*100)}% recent form.")
print("Add 25-26 data when available for real-time predictions.")
print("Compare Prob O2.5 vs bookmaker O25 odds to find value bets.")