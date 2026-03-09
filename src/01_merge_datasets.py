import pandas as pd
import os

BASE_DIR = "data"

seasons = [
    "15-16","16-17","17-18","18-19",
    "19-20","20-21","21-22","22-23",
    "23-24","24-25","25-26"
]

all_matches = []

for season in seasons:

    print(f"Processing season {season}")

    overview_path = f"{BASE_DIR}/{season}/overview.csv"
    shots_path    = f"{BASE_DIR}/{season}/shots_possession.csv"
    odds_path     = f"{BASE_DIR}/{season}/odds.csv"

    overview = pd.read_csv(overview_path)
    shots    = pd.read_csv(shots_path)
    odds     = pd.read_csv(odds_path)

    # -------------------------------------------------------
    # FIX DATES — format is YY-MM-DD HH:MM (e.g. 25-05-25 17:00)
    # Must force dayfirst=False and format explicitly
    # Without this, pandas guesses wrong and creates time-travel leakage
    # -------------------------------------------------------

    for source in [overview, shots, odds]:
        if "matchDate" in source.columns:
            source["matchDate"] = pd.to_datetime(
                source["matchDate"],
                format="%d-%m-%y %H:%M",
                dayfirst=True,
                errors="coerce"
            )

    # merge overview + shots
    df = overview.merge(
        shots,
        on="id",
        how="left",
        suffixes=("", "_shots")
    )

    # merge odds
    df = df.merge(
        odds,
        on="id",
        how="left",
        suffixes=("", "_odds")
    )

    # verify dates parsed correctly
    if df["matchDate"].isna().any():
        n_bad = df["matchDate"].isna().sum()
        print(f"  WARNING: {n_bad} dates failed to parse in season {season}")

    date_min = df["matchDate"].min()
    date_max = df["matchDate"].max()
    print(f"  Dates: {date_min.date()} to {date_max.date()} — {len(df)} matches")

    all_matches.append(df)

master_df = pd.concat(all_matches, ignore_index=True)

# drop duplicate columns from merges
columns_to_drop = [
    col for col in master_df.columns
    if col.endswith("_shots") or col.endswith("_odds")
]
master_df = master_df.drop(columns=columns_to_drop)

# sort by date — critical for all downstream rolling features
master_df = master_df.sort_values("matchDate").reset_index(drop=True)

# final date sanity check
print("")
print("Date range check:")
print(f"  Earliest: {master_df['matchDate'].min()}")
print(f"  Latest:   {master_df['matchDate'].max()}")
print(f"  Any nulls: {master_df['matchDate'].isna().sum()}")

# verify no future dates (sanity check)
import datetime
today = pd.Timestamp.now()
future = master_df[master_df["matchDate"] > today]
if len(future) > 0:
    print(f"  WARNING: {len(future)} matches have future dates — check your data")
else:
    print("  All dates look valid.")

output_path = "processed/premier_league_master.csv"
master_df.to_csv(output_path, index=False)

print("")
print("Dataset maestro creado")
print(f"Matches totales: {len(master_df)}")
print(f"Guardado en: {output_path}")