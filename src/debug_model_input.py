import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES

xgb_model, calibrator = joblib.load("models/over25_model.pkl")

df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed", dayfirst=False)
df = df.sort_values("matchDate").reset_index(drop=True)

latest_season = df["Season"].iloc[-1]
season_df = df[df["Season"] == latest_season]

print(f"Latest season: {latest_season}")
print(f"Season matches: {len(season_df)}")
print()

# Replicate exactly what 11_weekly_picks does for Liverpool vs Tottenham
for home_team, away_team in [("Liverpool", "Tottenham"), ("Chelsea", "Newcastle"), ("Arsenal", "Everton")]:
    sh = season_df[season_df["homeTeam"] == home_team]
    sa = season_df[season_df["awayTeam"] == away_team]

    goals_h = sh["FTHG"].mean()
    goals_a = sa["FTAG"].mean()
    conc_h  = sh["FTAG"].mean()
    conc_a  = sa["FTHG"].mean()

    last_home = df[df["homeTeam"] == home_team].iloc[-1]
    last_away = df[df["awayTeam"] == away_team].iloc[-1]

    sot_h   = float(last_home["home_sot_last5"])
    sot_a   = float(last_away["away_sot_last5"])
    shots_h = float(last_home["home_shots_last5"])
    shots_a = float(last_away["away_shots_last5"])

    atk_h = (goals_h + sot_h/5 + shots_h/15) / 3
    atk_a = (goals_a + sot_a/5 + shots_a/15) / 3
    combined = atk_h + atk_a

    exp_home = (goals_h + conc_a) / 2
    exp_away = (goals_a + conc_h) / 2

    form_h = sh.apply(lambda r: 3 if r["FTR"]=="H" else (1 if r["FTR"]=="D" else 0), axis=1).mean()
    form_a = sa.apply(lambda r: 3 if r["FTR"]=="A" else (1 if r["FTR"]=="D" else 0), axis=1).mean()

    print(f"{home_team} vs {away_team}:")
    print(f"  goals_h={goals_h:.2f}  goals_a={goals_a:.2f}")
    print(f"  conc_h={conc_h:.2f}   conc_a={conc_a:.2f}")
    print(f"  sot_h={sot_h:.2f}     sot_a={sot_a:.2f}")
    print(f"  shots_h={shots_h:.2f}  shots_a={shots_a:.2f}")
    print(f"  atk_h={atk_h:.3f}    atk_a={atk_a:.3f}    combined={combined:.3f}")
    print(f"  exp_home={exp_home:.2f}  exp_away={exp_away:.2f}")
    print(f"  form_h={form_h:.2f}    form_a={form_a:.2f}")

    # Build row and predict
    row = {
        "home_goals_last5": goals_h, "away_goals_last5": goals_a,
        "home_goals_last3": goals_h, "away_goals_last3": goals_a,
        "home_goals_ewm":   goals_h, "away_goals_ewm":   goals_a,
        "home_conceded_last5": conc_h, "away_conceded_last5": conc_a,
        "home_conceded_last3": conc_h, "away_conceded_last3": conc_a,
        "home_goals_at_home_last5":    goals_h,
        "home_conceded_at_home_last5": conc_h,
        "away_goals_away_last5":       goals_a,
        "away_conceded_away_last5":    conc_a,
        "home_shots_last5": shots_h, "away_shots_last5": shots_a,
        "home_sot_last5":   sot_h,   "away_sot_last5":   sot_a,
        "home_pos_last5":   float(last_home["home_pos_last5"]),
        "away_pos_last5":   float(last_away["away_pos_last5"]),
        "home_shot_accuracy": float(last_home["home_shot_accuracy"]),
        "away_shot_accuracy": float(last_away["away_shot_accuracy"]),
        "home_conversion_last5": goals_h/sot_h if sot_h>0 else 0.15,
        "away_conversion_last5": goals_a/sot_a if sot_a>0 else 0.15,
        "home_form_last5":  form_h, "away_form_last5":  form_a,
        "form_diff":        form_h - form_a,
        "h2h_avg_goals":    3.0,    "h2h_over25_rate":  0.6,
        "home_attack_score": atk_h, "away_attack_score": atk_a,
        "combined_attack":  combined,
        "home_elo": float(last_home["home_elo"]),
        "away_elo": float(last_away["away_elo"]),
        "elo_diff": float(last_home["home_elo"]) - float(last_away["away_elo"]),
        "elo_home_win_prob": 1/(1+10**((float(last_away["away_elo"])-float(last_home["home_elo"])-65)/400)),
        "elo_sum":  float(last_home["home_elo"]) + float(last_away["away_elo"]),
        "exp_home_goals": exp_home,
        "exp_away_goals": exp_away,
    }

    pred_df = pd.DataFrame([row])
    for col in pred_df.columns:
        if col in df.columns:
            pred_df[col] = pred_df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))

    # Show what clipping does
    print(f"  combined_attack before clip: {combined:.3f}")
    print(f"  combined_attack after clip:  {pred_df['combined_attack'].iloc[0]:.3f}")
    print(f"  form_h before clip: {form_h:.3f}")
    print(f"  form_h after clip:  {pred_df['home_form_last5'].iloc[0]:.3f}")

    raw = xgb_model.predict_proba(pred_df[FEATURES])[:,1].reshape(-1,1)
    prob = calibrator.predict_proba(raw)[0,1]
    print(f"  >>> MODEL PROB: {prob:.1%}")
    print()