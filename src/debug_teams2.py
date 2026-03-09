import pandas as pd
import numpy as np

df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"], format="mixed")
df = df.sort_values("matchDate").reset_index(drop=True)
s = df[df["Season"] == "2025/2026"]

print(f"Partidos 25-26 en dataset: {len(s)}")
print(f"Fecha mas reciente: {s['matchDate'].max()}")
print()

for team in ["Liverpool", "Chelsea", "Arsenal", "Tottenham", "Newcastle", "Everton"]:
    sh = s[s["homeTeam"] == team]
    sa = s[s["awayTeam"] == team]
    lh = df[df["homeTeam"] == team].iloc[-1]
    la = df[df["awayTeam"] == team].iloc[-1]

    print(f"{team}:")
    print(f"  Partidos 25-26: {len(sh)} casa / {len(sa)} fuera")
    print(f"  Avg goles casa={sh['FTHG'].mean():.2f}  fuera={sa['FTAG'].mean():.2f}")
    print(f"  Avg conced casa={sh['FTAG'].mean():.2f}  fuera={sa['FTHG'].mean():.2f}")
    print(f"  Ultimo home: {lh['matchDate']}  goals_last5={lh['home_goals_last5']:.2f}  conc_last5={lh['home_conceded_last5']:.2f}  combined_attack={lh['combined_attack']:.3f}")
    print(f"  Ultimo away: {la['matchDate']}  goals_last5={la['away_goals_last5']:.2f}  conc_last5={la['away_conceded_last5']:.2f}  combined_attack={la['combined_attack']:.3f}")
    print()