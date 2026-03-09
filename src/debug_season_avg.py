import pandas as pd

df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed")
df = df.sort_values("matchDate").reset_index(drop=True)

# Check season 24-25 averages vs last-5 for key teams
season = df[df["Season"] == "2024/2025"]

teams = ["Chelsea", "Arsenal", "Liverpool", "West Ham", "Manchester Utd",
         "Newcastle", "Everton", "Manchester City", "Aston Villa", "Tottenham"]

print("=== SEASON 24-25 AVERAGES ===")
print(f"{'Team':<20} {'Avg Goals H':>11} {'Avg Goals A':>11} {'Avg Conc H':>10} {'Avg Conc A':>10}")
print("-" * 65)

for team in teams:
    h = season[season["homeTeam"] == team]
    a = season[season["awayTeam"] == team]
    avg_gh = h["FTHG"].mean() if len(h) > 0 else 0
    avg_ga = a["FTAG"].mean() if len(a) > 0 else 0
    avg_ch = h["FTAG"].mean() if len(h) > 0 else 0
    avg_ca = a["FTHG"].mean() if len(a) > 0 else 0
    print(f"{team:<20} {avg_gh:>11.2f} {avg_ga:>11.2f} {avg_ch:>10.2f} {avg_ca:>10.2f}")