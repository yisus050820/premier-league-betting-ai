import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES

# ================================================================
# 11_weekly_picks.py
# ================================================================
# INSTRUCCIONES:
#   1. Actualiza FIXTURES con los partidos de la jornada
#   2. Para cada partido agrega la cuota O25 de tu casa de apuestas
#   3. Corre: py src/11_weekly_picks.py
#   4. El script te dice exactamente qué apostar y cuánto
# ================================================================

# ----------------------------------------------------------------
# TU BANKROLL ACTUAL (actualiza esto cada semana)
# ----------------------------------------------------------------

BANKROLL = 500  # en dolares/euros

# ----------------------------------------------------------------
# FIXTURES DE LA JORNADA
# Formato: ("HomeTeam", "AwayTeam", odds_O25)
# odds_O25 = cuota de Over 2.5 en tu casa de apuestas
# Ejemplo: si pone 1.75, escribe 1.75
# ----------------------------------------------------------------

FIXTURES = [
    ("Chelsea",        "Newcastle",        1.45),
    ("Arsenal",        "Everton",          1.80),
    ("West Ham",       "Manchester City",  1.47),
    ("Manchester Utd", "Aston Villa",      1.66),
    ("Liverpool",      "Tottenham",        1.47),
]

# ----------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------

MIN_EDGE      = 0.05   # edge minimo para considerar value bet
KELLY_FRAC    = 0.15   # fraccion Kelly (conservador)

# ================================================================
# NO TOCAR DE AQUI PARA ABAJO
# ================================================================

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_helper import load_model, predict_proba as ensemble_predict

print("Loading model...")
model_bundle = load_model("models/over25_model.pkl")

print("Loading historical dataset...")
df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed", dayfirst=False)
df = df.sort_values("matchDate").reset_index(drop=True)

latest_season = df["Season"].iloc[-1]
season_df     = df[df["Season"] == latest_season]
teams_in_data = pd.unique(df[["homeTeam", "awayTeam"]].values.ravel())

# ----------------------------------------------------------------
# BUILD TEAM STATS (same as 09)
# ----------------------------------------------------------------

team_stats = {}

for team in teams_in_data:
    sh = season_df[season_df["homeTeam"] == team]
    sa = season_df[season_df["awayTeam"] == team]
    if len(sh) == 0 and len(sa) == 0:
        continue

    goals_h = sh["FTHG"].mean() if len(sh) > 0 else df[df["homeTeam"]==team]["FTHG"].mean()
    goals_a = sa["FTAG"].mean()  if len(sa) > 0 else df[df["awayTeam"]==team]["FTAG"].mean()
    conc_h  = sh["FTAG"].mean()  if len(sh) > 0 else df[df["homeTeam"]==team]["FTAG"].mean()
    conc_a  = sa["FTHG"].mean()  if len(sa) > 0 else df[df["awayTeam"]==team]["FTHG"].mean()

    last_home = df[df["homeTeam"] == team].iloc[-1] if len(df[df["homeTeam"]==team]) > 0 else None
    last_away = df[df["awayTeam"] == team].iloc[-1] if len(df[df["awayTeam"]==team]) > 0 else None

    def geth(col, default): return float(last_home[col]) if last_home is not None else default
    def geta(col, default): return float(last_away[col]) if last_away is not None else default

    shots_h = geth("home_shots_last5", 12.0)
    shots_a = geta("away_shots_last5", 10.0)
    sot_h   = geth("home_sot_last5",   4.0)
    sot_a   = geta("away_sot_last5",   3.5)
    pos_h   = geth("home_pos_last5",  52.0)
    pos_a   = geta("away_pos_last5",  48.0)
    acc_h   = (sh["HSONFT"] / sh["HTSFT"].replace(0, np.nan)).mean() if len(sh) > 0 else 0.33
    acc_a   = (sa["ASONFT"] / sa["ATSFT"].replace(0, np.nan)).mean() if len(sa) > 0 else 0.33
    conv_h  = goals_h / sot_h if sot_h > 0 else 0.15
    conv_a  = goals_a / sot_a if sot_a > 0 else 0.15
    form_h  = sh.apply(lambda r: 3 if r["FTR"]=="H" else (1 if r["FTR"]=="D" else 0), axis=1).mean() if len(sh) > 0 else 1.2
    form_a  = sa.apply(lambda r: 3 if r["FTR"]=="A" else (1 if r["FTR"]=="D" else 0), axis=1).mean() if len(sa) > 0 else 0.9
    atk_h   = (goals_h + sot_h/5 + shots_h/15) / 3
    atk_a   = (goals_a + sot_a/5 + shots_a/15) / 3
    all_g   = df[(df["homeTeam"]==team)|(df["awayTeam"]==team)].iloc[-1]
    elo     = all_g["home_elo"] if all_g["homeTeam"]==team else all_g["away_elo"]

    # Rest days — days since last match
    all_matches = df[(df["homeTeam"]==team)|(df["awayTeam"]==team)].sort_values("matchDate")
    if len(all_matches) >= 2:
        last_match_dt = all_matches.iloc[-1]["matchDate"]
        rest_h = (pd.Timestamp.now() - last_match_dt).days
        rest_a = rest_h  # same team, same last match
    else:
        rest_h = rest_a = 7

    # Table position from current season
    s_all = pd.concat([
        sh[["matchDate","FTR","FTHG","FTAG"]].assign(venue="home"),
        sa[["matchDate","FTR","FTHG","FTAG"]].assign(venue="away"),
    ]) if len(sh) > 0 and len(sa) > 0 else (
        sh[["matchDate","FTR","FTHG","FTAG"]].assign(venue="home") if len(sh) > 0
        else sa[["matchDate","FTR","FTHG","FTAG"]].assign(venue="away")
    )

    pts = 0
    gf  = 0
    ga  = 0
    for _, r in s_all.iterrows():
        if r["venue"] == "home":
            gf += r["FTHG"]; ga += r["FTAG"]
            pts += 3 if r["FTR"]=="H" else (1 if r["FTR"]=="D" else 0)
        else:
            gf += r["FTAG"]; ga += r["FTHG"]
            pts += 3 if r["FTR"]=="A" else (1 if r["FTR"]=="D" else 0)

    # Estimate table position from points (rough: top=1, bottom=20)
    all_team_pts = {}
    for t in teams_in_data:
        tsh = season_df[season_df["homeTeam"]==t]
        tsa = season_df[season_df["awayTeam"]==t]
        tp  = 0
        if len(tsh) > 0:
            tp += (tsh["FTR"]=="H").sum()*3 + (tsh["FTR"]=="D").sum()
        if len(tsa) > 0:
            tp += (tsa["FTR"]=="A").sum()*3 + (tsa["FTR"]=="D").sum()
        all_team_pts[t] = tp

    sorted_by_pts = sorted(all_team_pts, key=all_team_pts.get, reverse=True)
    table_pos = sorted_by_pts.index(team) + 1 if team in sorted_by_pts else 10

    team_stats[team] = {
        "goals_h": goals_h, "goals_a": goals_a,
        "conc_h":  conc_h,  "conc_a":  conc_a,
        "shots_h": shots_h, "shots_a": shots_a,
        "sot_h":   sot_h,   "sot_a":   sot_a,
        "pos_h":   pos_h,   "pos_a":   pos_a,
        "acc_h":   acc_h,   "acc_a":   acc_a,
        "conv_h":  conv_h,  "conv_a":  conv_a,
        "form_h":  form_h,  "form_a":  form_a,
        "atk_h":   atk_h,   "atk_a":   atk_a,
        "elo":        elo,
        "rest_days_h": rest_h,
        "rest_days_a": rest_a,
        "table_pos":   table_pos,
        "table_pts":   pts,
        "table_gd":    gf - ga,
    }

# ----------------------------------------------------------------
# PREDICT + EVALUATE EACH FIXTURE
# ----------------------------------------------------------------

rows     = []
fixtures_clean = []

for home_team, away_team, odds_o25 in FIXTURES:
    missing = [t for t in [home_team, away_team] if t not in team_stats]
    if missing:
        print(f"  WARNING: {missing} not found in data — skipping")
        continue

    h = team_stats[home_team]
    a = team_stats[away_team]

    elo_diff          = h["elo"] - a["elo"]
    elo_home_win_prob = 1 / (1 + 10**((a["elo"] - h["elo"] - 65) / 400))
    elo_sum           = h["elo"] + a["elo"]
    exp_home          = (h["goals_h"] + a["conc_a"]) / 2
    exp_away          = (a["goals_a"] + h["conc_h"]) / 2
    combined          = h["atk_h"] + a["atk_a"]

    past_h2h = df[
        ((df["homeTeam"]==home_team)&(df["awayTeam"]==away_team)) |
        ((df["homeTeam"]==away_team)&(df["awayTeam"]==home_team))
    ].tail(5)
    h2h_goals  = past_h2h["total_goals"].mean() if len(past_h2h)>=2 else df["total_goals"].mean()
    h2h_over25 = past_h2h["over25"].mean()      if len(past_h2h)>=2 else df["over25"].mean()

    row = {
        "home_goals_last5": h["goals_h"], "away_goals_last5": a["goals_a"],
        "home_goals_last3": h["goals_h"], "away_goals_last3": a["goals_a"],
        "home_goals_ewm":   h["goals_h"], "away_goals_ewm":   a["goals_a"],
        "home_conceded_last5": h["conc_h"], "away_conceded_last5": a["conc_a"],
        "home_conceded_last3": h["conc_h"], "away_conceded_last3": a["conc_a"],
        "home_goals_at_home_last5":    h["goals_h"],
        "home_conceded_at_home_last5": h["conc_h"],
        "away_goals_away_last5":       a["goals_a"],
        "away_conceded_away_last5":    a["conc_a"],
        "home_shots_last5": h["shots_h"], "away_shots_last5": a["shots_a"],
        "home_sot_last5":   h["sot_h"],   "away_sot_last5":   a["sot_a"],
        "home_pos_last5":   h["pos_h"],   "away_pos_last5":   a["pos_a"],
        "home_shot_accuracy":    h["acc_h"],  "away_shot_accuracy":    a["acc_a"],
        "home_conversion_last5": h["conv_h"], "away_conversion_last5": a["conv_a"],
        "home_form_last5":  h["form_h"],  "away_form_last5":  a["form_a"],
        "form_diff":        h["form_h"] - a["form_a"],
        "h2h_avg_goals":    h2h_goals,    "h2h_over25_rate":  h2h_over25,
        "home_attack_score": h["atk_h"],  "away_attack_score": a["atk_a"],
        "combined_attack":  combined,
        "home_elo": h["elo"], "away_elo": a["elo"],
        "elo_diff": elo_diff, "elo_home_win_prob": elo_home_win_prob,
        "elo_sum":  elo_sum,
        "exp_home_goals": exp_home, "exp_away_goals": exp_away,
        # --- Rest days ---
        "home_rest_days":  h["rest_days_h"],
        "away_rest_days":  a["rest_days_a"],
        "rest_diff":       h["rest_days_h"] - a["rest_days_a"],
        "home_fatigued":   int(h["rest_days_h"] < 4),
        "away_fatigued":   int(a["rest_days_a"] < 4),
        # --- Table position ---
        "home_table_pos":  h["table_pos"],
        "away_table_pos":  a["table_pos"],
        "table_pos_diff":  a["table_pos"] - h["table_pos"],
        "home_table_pts":  h["table_pts"],
        "away_table_pts":  a["table_pts"],
        "home_table_gd":   h["table_gd"],
        "away_table_gd":   a["table_gd"],
    }
    rows.append(row)
    fixtures_clean.append((home_team, away_team, odds_o25, exp_home, exp_away))

# clip to training range
pred_df = pd.DataFrame(rows)
for col in pred_df.columns:
    if col in df.columns:
        pred_df[col] = pred_df[col].clip(
            lower=df[col].quantile(0.01),
            upper=df[col].quantile(0.99)
        )

raw   = ensemble_predict(model_bundle, pred_df[FEATURES])
probs = raw

# ----------------------------------------------------------------
# DISPLAY RESULTS
# ----------------------------------------------------------------

print()
print("=" * 70)
print(f"  JORNADA — ANÁLISIS DE VALUE BETS")
print(f"  Bankroll: ${BANKROLL:.0f}  |  Kelly: {int(KELLY_FRAC*100)}%  |  Edge mínimo: {int(MIN_EDGE*100)}%")
print("=" * 70)

total_exposure = 0
bets_to_place  = []

for i, (home_team, away_team, odds_o25, xgh, xga) in enumerate(fixtures_clean):
    prob      = probs[i]
    book_prob = 1 / odds_o25
    edge      = prob - book_prob

    # Kelly stake
    b = odds_o25 - 1
    kelly_raw  = (b * prob - (1 - prob)) / b
    kelly_frac = max(kelly_raw * KELLY_FRAC, 0)
    stake      = round(BANKROLL * kelly_frac, 2)
    potential  = round(stake * (odds_o25 - 1), 2)

    is_value = edge >= MIN_EDGE and prob < 0.85

    print()
    print(f"  {home_team} vs {away_team}")
    print(f"  {'─'*40}")
    print(f"  xG:          {xgh:.2f} - {xga:.2f}  (total {xgh+xga:.2f})")
    print(f"  Modelo O2.5: {prob:.1%}")
    print(f"  Book O2.5:   {book_prob:.1%}  (@ {odds_o25})")
    print(f"  Edge:        {edge:+.1%}")

    if is_value:
        print(f"  ✅ VALUE BET")
        print(f"  Stake:       ${stake:.2f}  ({kelly_frac:.1%} del bankroll)")
        print(f"  Potencial:   +${potential:.2f}  si Over")
        total_exposure += stake
        bets_to_place.append((home_team, away_team, odds_o25, stake, potential, prob, edge))
    elif edge > 0:
        print(f"  ⚡ Edge positivo pero pequeño ({edge:+.1%}) — no apostar")
    else:
        print(f"  ❌ Sin value  (book mejor que modelo)")

print()
print("=" * 70)
print(f"  RESUMEN DE LA JORNADA")
print("=" * 70)

if bets_to_place:
    print(f"  Apuestas recomendadas: {len(bets_to_place)}")
    print(f"  Exposición total:      ${total_exposure:.2f}  ({total_exposure/BANKROLL:.1%} del bankroll)")
    print()
    print(f"  {'Partido':<35} {'Odds':>5}  {'Stake':>7}  {'Edge':>7}  {'Potencial':>10}")
    print(f"  {'─'*70}")
    for ht, at, odds, stake, pot, prob, edge in bets_to_place:
        match = f"{ht} vs {at}"
        print(f"  {match:<35} {odds:>5.2f}  ${stake:>6.2f}  {edge:>+6.1%}  +${pot:>8.2f}")
    print()
    print(f"  Si todas ganan: +${sum(p for _,_,_,_,p,_,_ in bets_to_place):.2f}")
    print(f"  Si todas pierden: -${total_exposure:.2f}")
else:
    print("  Sin value bets esta jornada. No apostar.")
    print("  Espera la siguiente jornada.")

print("=" * 70)
print()
print("RECUERDA:")
print("  - Solo apuesta lo que indica el stake")
print("  - Actualiza BANKROLL cada semana con tu saldo real")
print("  - Registra cada apuesta para seguimiento")