import pandas as pd
import numpy as np

print("Loading feature dataset...")

df = pd.read_csv("processed/premier_league_features.csv")

df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed", dayfirst=False)
df = df.sort_values("matchDate").reset_index(drop=True)


# ---------------------------------
# ELO RATINGS — IMPROVED
# Changes vs original:
# 1. K factor scales with goal margin (bigger wins = bigger ELO swing)
# 2. Separate home ELO advantage tracked
# 3. Season decay — ELO regresses toward 1500 each new season
# ---------------------------------

teams = pd.concat([df["homeTeam"], df["awayTeam"]]).unique()

elo        = {team: 1500.0 for team in teams}
K_BASE     = 20
HOME_ADV   = 65     # home team gets +65 ELO points advantage in expectation calc

home_elo_list = []
away_elo_list = []

current_season = None

print("Calculating improved ELO ratings...")

for idx, row in df.iterrows():

    season = row.get("Season", None)

    # Season reset: regress ELO 30% toward 1500 each new season
    # This prevents one great season from dominating forever
    if season and season != current_season:
        current_season = season
        for team in elo:
            elo[team] = elo[team] * 0.70 + 1500 * 0.30

    home = row["homeTeam"]
    away = row["awayTeam"]

    Ra = elo[home]
    Rb = elo[away]

    home_elo_list.append(Ra)
    away_elo_list.append(Rb)

    # Expected score WITH home advantage factored in
    Ea = 1 / (1 + 10 ** ((Rb - Ra - HOME_ADV) / 400))
    Eb = 1 - Ea

    # Actual result
    if row["FTR"] == "H":
        Sa, Sb = 1.0, 0.0
    elif row["FTR"] == "A":
        Sa, Sb = 0.0, 1.0
    else:
        Sa, Sb = 0.5, 0.5

    # K factor scales with goal margin
    # 1 goal diff = K*1.0, 2 goals = K*1.5, 3+ goals = K*1.75
    goal_diff = abs(int(row["FTHG"]) - int(row["FTAG"]))
    if goal_diff == 0 or goal_diff == 1:
        K = K_BASE * 1.0
    elif goal_diff == 2:
        K = K_BASE * 1.5
    else:
        K = K_BASE * 1.75

    elo[home] = Ra + K * (Sa - Ea)
    elo[away] = Rb + K * (Sb - Eb)


df["home_elo"] = home_elo_list
df["away_elo"] = away_elo_list
df["elo_diff"] = df["home_elo"] - df["away_elo"]

# ELO win probability for home team (useful feature)
df["elo_home_win_prob"] = 1 / (1 + 10 ** ((df["away_elo"] - df["home_elo"] - HOME_ADV) / 400))

# ELO sum — higher total ELO = both teams are strong = more goals expected
df["elo_sum"] = df["home_elo"] + df["away_elo"]


# ---------------------------------
# SAVE
# ---------------------------------

df.to_csv("processed/premier_league_features_v2.csv", index=False)

print("ELO features created")
print("Rows:   ", len(df))
print("Columns:", len(df.columns))
print("")
print("New ELO features added:")
print("  home_elo, away_elo    — with season decay and goal-margin K factor")
print("  elo_diff              — home minus away")
print("  elo_home_win_prob     — probability home wins based on ELO")
print("  elo_sum               — total quality of the match")