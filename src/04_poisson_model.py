import pandas as pd
import numpy as np
from scipy.stats import poisson

print("Loading dataset...")

df = pd.read_csv("processed/premier_league_features_v2.csv")

df["matchDate"] = pd.to_datetime(df["matchDate"])

df = df.sort_values("matchDate")


# ------------------------------------------------
# LEAGUE AVERAGE GOALS
# ------------------------------------------------

league_home_avg = df["FTHG"].mean()
league_away_avg = df["FTAG"].mean()

print("League avg home goals:", league_home_avg)
print("League avg away goals:", league_away_avg)


# ------------------------------------------------
# TEAM ATTACK / DEFENSE STRENGTH
# ------------------------------------------------

print("Calculating team strengths...")

home_attack = df.groupby("homeTeam")["FTHG"].mean() / league_home_avg
home_defense = df.groupby("homeTeam")["FTAG"].mean() / league_away_avg

away_attack = df.groupby("awayTeam")["FTAG"].mean() / league_away_avg
away_defense = df.groupby("awayTeam")["FTHG"].mean() / league_home_avg


# ------------------------------------------------
# EXPECTED GOALS FOR EACH MATCH
# ------------------------------------------------

exp_home_goals = []
exp_away_goals = []

print("Calculating expected goals...")

for _, row in df.iterrows():

    home = row["homeTeam"]
    away = row["awayTeam"]

    home_attack_strength = home_attack[home]
    away_defense_strength = away_defense[away]

    away_attack_strength = away_attack[away]
    home_defense_strength = home_defense[home]

    home_xg = league_home_avg * home_attack_strength * away_defense_strength
    away_xg = league_away_avg * away_attack_strength * home_defense_strength

    exp_home_goals.append(home_xg)
    exp_away_goals.append(away_xg)


df["exp_home_goals"] = exp_home_goals
df["exp_away_goals"] = exp_away_goals


# ------------------------------------------------
# POISSON OVER 2.5 PROBABILITY
# ------------------------------------------------

print("Calculating Over 2.5 probability...")

over25_prob = []

for _, row in df.iterrows():

    home_lambda = row["exp_home_goals"]
    away_lambda = row["exp_away_goals"]

    prob = 0

    for i in range(6):
        for j in range(6):

            goals_prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

            if i + j > 2:
                prob += goals_prob

    over25_prob.append(prob)


df["poisson_over25_prob"] = over25_prob


# ------------------------------------------------
# SAVE DATASET
# ------------------------------------------------

df.to_csv("processed/premier_league_features_v3.csv", index=False)

print("Poisson features added")

print("Rows:", len(df))
print("Columns:", len(df.columns))