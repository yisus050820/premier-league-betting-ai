import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_csv("processed/premier_league_master.csv")

print("Rows:", len(df))
print("Columns:", len(df.columns))

# ---------------------------------------------------
# SORT BY DATE
# ---------------------------------------------------

df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed", dayfirst=False)
df = df.sort_values("matchDate").reset_index(drop=True)

# ---------------------------------------------------
# BASIC TARGETS
# ---------------------------------------------------

print("Creating basic features...")

df["total_goals"]   = df["FTHG"] + df["FTAG"]
df["over25"]        = (df["total_goals"] > 2.5).astype(int)
df["home_conceded"] = df["FTAG"]
df["away_conceded"] = df["FTHG"]


# ---------------------------------------------------
# HELPER: weighted rolling mean
# Gives more weight to recent matches
# weights = [1, 2, 3, 4, 5] for window=5 (5 = most recent)
# ---------------------------------------------------

def weighted_rolling(series, window=5):
    weights = np.arange(1, window + 1, dtype=float)

    def wmean(x):
        if len(x) < window:
            return np.nan
        return np.dot(x, weights) / weights.sum()

    return (
        series
        .rolling(window)
        .apply(wmean, raw=True)
    )


# ---------------------------------------------------
# HELPER: exponential weighted mean (alternative)
# span=5 means ~5 match half-life
# ---------------------------------------------------

def ewm_rolling(series, span=5):
    return series.ewm(span=span, min_periods=5).mean()


# ---------------------------------------------------
# ROLLING GOALS — weighted (last 5 and last 3)
# ---------------------------------------------------

print("Creating weighted rolling goal features...")

# Last 5 weighted
df["home_goals_last5"] = (
    df.groupby("homeTeam")["FTHG"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_goals_last5"] = (
    df.groupby("awayTeam")["FTAG"]
    .transform(lambda x: weighted_rolling(x, 5))
)

# Last 3 weighted (captures hot/cold streaks better)
df["home_goals_last3"] = (
    df.groupby("homeTeam")["FTHG"]
    .transform(lambda x: weighted_rolling(x, 3))
)

df["away_goals_last3"] = (
    df.groupby("awayTeam")["FTAG"]
    .transform(lambda x: weighted_rolling(x, 3))
)

# Exponential weighted (long memory, recent bias)
df["home_goals_ewm"] = (
    df.groupby("homeTeam")["FTHG"]
    .transform(lambda x: ewm_rolling(x, span=5))
)

df["away_goals_ewm"] = (
    df.groupby("awayTeam")["FTAG"]
    .transform(lambda x: ewm_rolling(x, span=5))
)


# ---------------------------------------------------
# ROLLING GOALS CONCEDED — weighted
# ---------------------------------------------------

print("Creating weighted defense features...")

df["home_conceded_last5"] = (
    df.groupby("homeTeam")["home_conceded"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_conceded_last5"] = (
    df.groupby("awayTeam")["away_conceded"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["home_conceded_last3"] = (
    df.groupby("homeTeam")["home_conceded"]
    .transform(lambda x: weighted_rolling(x, 3))
)

df["away_conceded_last3"] = (
    df.groupby("awayTeam")["away_conceded"]
    .transform(lambda x: weighted_rolling(x, 3))
)


# ---------------------------------------------------
# HOME / AWAY SPLITS
# A team can be very different at home vs away
# ---------------------------------------------------

print("Creating home/away split features...")

# Home team: how do they score/concede specifically AT HOME?
df["home_goals_at_home_last5"] = (
    df.groupby("homeTeam")["FTHG"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["home_conceded_at_home_last5"] = (
    df.groupby("homeTeam")["FTAG"]
    .transform(lambda x: weighted_rolling(x, 5))
)

# Away team: how do they score/concede specifically AWAY?
df["away_goals_away_last5"] = (
    df.groupby("awayTeam")["FTAG"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_conceded_away_last5"] = (
    df.groupby("awayTeam")["FTHG"]
    .transform(lambda x: weighted_rolling(x, 5))
)


# ---------------------------------------------------
# ROLLING SHOTS — weighted
# ---------------------------------------------------

print("Creating shots features...")

df["home_shots_last5"] = (
    df.groupby("homeTeam")["HTSFT"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_shots_last5"] = (
    df.groupby("awayTeam")["ATSFT"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["home_sot_last5"] = (
    df.groupby("homeTeam")["HSONFT"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_sot_last5"] = (
    df.groupby("awayTeam")["ASONFT"]
    .transform(lambda x: weighted_rolling(x, 5))
)


# ---------------------------------------------------
# ROLLING POSSESSION — weighted
# ---------------------------------------------------

print("Creating possession features...")

df["home_pos_last5"] = (
    df.groupby("homeTeam")["HBPFT"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_pos_last5"] = (
    df.groupby("awayTeam")["ABPFT"]
    .transform(lambda x: weighted_rolling(x, 5))
)


# ---------------------------------------------------
# SHOT EFFICIENCY
# ---------------------------------------------------

print("Creating efficiency metrics...")

df["home_shot_accuracy"] = df["HSONFT"] / df["HTSFT"]
df["away_shot_accuracy"] = df["ASONFT"] / df["ATSFT"]

# Rolling shot conversion (goals per shot on target)
df["home_conversion_last5"] = (
    df.groupby("homeTeam").apply(
        lambda g: weighted_rolling(g["FTHG"] / g["HSONFT"].replace(0, np.nan), 5),
        include_groups=False
    ).reset_index(level=0, drop=True)
)

df["away_conversion_last5"] = (
    df.groupby("awayTeam").apply(
        lambda g: weighted_rolling(g["FTAG"] / g["ASONFT"].replace(0, np.nan), 5),
        include_groups=False
    ).reset_index(level=0, drop=True)
)


# ---------------------------------------------------
# MOMENTUM — last 5 points earned
# Win=3, Draw=1, Loss=0
# Tells you if a team is on a good or bad run
# ---------------------------------------------------

print("Creating momentum features...")

def match_points_home(row):
    if row["FTR"] == "H": return 3
    if row["FTR"] == "D": return 1
    return 0

def match_points_away(row):
    if row["FTR"] == "A": return 3
    if row["FTR"] == "D": return 1
    return 0

df["home_points"] = df.apply(match_points_home, axis=1)
df["away_points"] = df.apply(match_points_away, axis=1)

df["home_form_last5"] = (
    df.groupby("homeTeam")["home_points"]
    .transform(lambda x: weighted_rolling(x, 5))
)

df["away_form_last5"] = (
    df.groupby("awayTeam")["away_points"]
    .transform(lambda x: weighted_rolling(x, 5))
)

# Form differential — stronger predictor than raw form
df["form_diff"] = df["home_form_last5"] - df["away_form_last5"]


# ---------------------------------------------------
# HEAD-TO-HEAD (H2H)
# Average total goals in last 5 meetings between these teams
# ---------------------------------------------------

print("Creating head-to-head features...")

h2h_goals = []
h2h_over25_rate = []

for idx, row in df.iterrows():
    home = row["homeTeam"]
    away = row["awayTeam"]
    date = row["matchDate"]

    # All past meetings between these two teams (either side home/away)
    past = df[
        (df["matchDate"] < date) &
        (
            ((df["homeTeam"] == home) & (df["awayTeam"] == away)) |
            ((df["homeTeam"] == away) & (df["awayTeam"] == home))
        )
    ].tail(5)  # last 5 meetings

    if len(past) >= 2:
        h2h_goals.append(past["total_goals"].mean())
        h2h_over25_rate.append(past["over25"].mean())
    else:
        h2h_goals.append(np.nan)
        h2h_over25_rate.append(np.nan)

df["h2h_avg_goals"]     = h2h_goals
df["h2h_over25_rate"]   = h2h_over25_rate

print("H2H done.")


# ---------------------------------------------------
# COMBINED ATTACK SCORE
# Simple composite: goals + sot + shots / 3 (normalized)
# ---------------------------------------------------

df["home_attack_score"] = (
    df["home_goals_last5"].fillna(0) +
    df["home_sot_last5"].fillna(0) / 5 +
    df["home_shots_last5"].fillna(0) / 15
) / 3

df["away_attack_score"] = (
    df["away_goals_last5"].fillna(0) +
    df["away_sot_last5"].fillna(0) / 5 +
    df["away_shots_last5"].fillna(0) / 15
) / 3

# Combined expected goals proxy
df["combined_attack"] = df["home_attack_score"] + df["away_attack_score"]


# ---------------------------------------------------
# DAYS OF REST BETWEEN MATCHES
# Tired teams score less and concede more
# ---------------------------------------------------

print("Creating rest days features...")

def compute_rest_days(df, team_col, date_col):
    rest = {}
    last_match = {}
    result = []
    for _, row in df.iterrows():
        team = row[team_col]
        date = row[date_col]
        if team in last_match:
            days = (date - last_match[team]).days
        else:
            days = 7  # default — assume normal rest for first match
        result.append(days)
        last_match[team] = date
    return result

home_rest = []
away_rest = []

# Build per-team last match date as we iterate chronologically
last_match_date = {}
for _, row in df.iterrows():
    ht = row["homeTeam"]
    at = row["awayTeam"]
    dt = row["matchDate"]

    home_rest.append((dt - last_match_date[ht]).days if ht in last_match_date else 7)
    away_rest.append((dt - last_match_date[at]).days if at in last_match_date else 7)

    last_match_date[ht] = dt
    last_match_date[at] = dt

df["home_rest_days"] = home_rest
df["away_rest_days"] = away_rest
df["rest_diff"]      = df["home_rest_days"] - df["away_rest_days"]

# Fatigue flag: < 4 days rest = playing on short turnaround
df["home_fatigued"]  = (df["home_rest_days"] < 4).astype(int)
df["away_fatigued"]  = (df["away_rest_days"] < 4).astype(int)


# ---------------------------------------------------
# TABLE POSITION AT TIME OF MATCH
# Calculated from actual results up to that point
# Uses only past data — no leakage
# ---------------------------------------------------

print("Creating table position features...")

home_pos_list = []
away_pos_list = []
home_pts_list = []
away_pts_list = []
home_gd_list  = []
away_gd_list  = []

for idx, row in df.iterrows():
    season  = row["Season"]
    match_date = row["matchDate"]
    ht      = row["homeTeam"]
    at      = row["awayTeam"]

    # Past matches in same season before this match
    past = df[(df["Season"] == season) & (df["matchDate"] < match_date)]

    if len(past) == 0:
        home_pos_list.append(10)
        away_pos_list.append(10)
        home_pts_list.append(0)
        away_pts_list.append(0)
        home_gd_list.append(0)
        away_gd_list.append(0)
        continue

    # Compute standings from past matches
    teams = pd.unique(past[["homeTeam", "awayTeam"]].values.ravel())
    standings = {}
    for t in teams:
        h_games = past[past["homeTeam"] == t]
        a_games = past[past["awayTeam"] == t]
        pts  = (h_games["FTR"] == "H").sum() * 3 + (h_games["FTR"] == "D").sum()
        pts += (a_games["FTR"] == "A").sum() * 3 + (a_games["FTR"] == "D").sum()
        gf   = h_games["FTHG"].sum() + a_games["FTAG"].sum()
        ga   = h_games["FTAG"].sum() + a_games["FTHG"].sum()
        standings[t] = {"pts": pts, "gd": gf - ga}

    # Sort by points then GD
    sorted_teams = sorted(standings.keys(),
                          key=lambda t: (standings[t]["pts"], standings[t]["gd"]),
                          reverse=True)

    pos_map = {t: i+1 for i, t in enumerate(sorted_teams)}
    n_teams = len(sorted_teams)

    h_pos = pos_map.get(ht, n_teams // 2)
    a_pos = pos_map.get(at, n_teams // 2)
    h_pts = standings.get(ht, {"pts": 0})["pts"]
    a_pts = standings.get(at, {"pts": 0})["pts"]
    h_gd  = standings.get(ht, {"gd": 0})["gd"]
    a_gd  = standings.get(at, {"gd": 0})["gd"]

    home_pos_list.append(h_pos)
    away_pos_list.append(a_pos)
    home_pts_list.append(h_pts)
    away_pts_list.append(a_pts)
    home_gd_list.append(h_gd)
    away_gd_list.append(a_gd)

df["home_table_pos"]  = home_pos_list
df["away_table_pos"]  = away_pos_list
df["table_pos_diff"]  = df["away_table_pos"] - df["home_table_pos"]  # positive = home team higher
df["home_table_pts"]  = home_pts_list
df["away_table_pts"]  = away_pts_list
df["home_table_gd"]   = home_gd_list
df["away_table_gd"]   = away_gd_list

# ---------------------------------------------------
# CLEAN UP
# ---------------------------------------------------

df.replace([float("inf"), -float("inf")], np.nan, inplace=True)

print("Dropping rows without sufficient history...")
df = df.dropna()

print(f"\nFeature engineering completed")
print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")

df.to_csv("processed/premier_league_features.csv", index=False)
print("Saved to: processed/premier_league_features.csv")