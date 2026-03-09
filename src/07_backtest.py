import pandas as pd
import numpy as np

print("Loading value bets...")

df = pd.read_csv("predictions/value_bets.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)

print(f"Bets to backtest: {len(df)}")

# ------------------------------------------------
# RESULT
# ------------------------------------------------

df["over25_result"] = (df["total_goals"] > 2.5).astype(int)

# ------------------------------------------------
# FLAT BETTING (1 unit per bet)
# ------------------------------------------------

df["profit_flat"] = np.where(
    df["over25_result"] == 1,
    df["O25"] - 1,
    -1.0
)

# ------------------------------------------------
# KELLY SIZING
# Bet kelly% of current bankroll
# Start with 100 units
# ------------------------------------------------

bankroll = 100.0
kelly_profits = []
bankroll_history = [bankroll]

for _, row in df.iterrows():
    stake = bankroll * row["kelly"]
    if row["over25_result"] == 1:
        profit = stake * (row["O25"] - 1)
    else:
        profit = -stake
    kelly_profits.append(profit)
    bankroll += profit
    bankroll_history.append(bankroll)

df["profit_kelly"] = kelly_profits

# ------------------------------------------------
# STATS
# ------------------------------------------------

bets   = len(df)
wins   = (df["over25_result"] == 1).sum()
losses = bets - wins
win_rate = wins / bets

# Flat
flat_profit = df["profit_flat"].sum()
flat_roi    = flat_profit / bets

# Kelly
kelly_profit = df["profit_kelly"].sum()
kelly_roi    = kelly_profit / 100  # relative to starting bankroll

# Average odds
avg_odds = df["O25"].mean()

# Average edge
avg_edge = df["value"].mean()

# Max drawdown (flat)
df["cumprofit_flat"] = df["profit_flat"].cumsum()
roll_max = df["cumprofit_flat"].cummax()
drawdown = df["cumprofit_flat"] - roll_max
max_drawdown = drawdown.min()

print("")
print("=" * 50)
print("  BACKTEST RESULTS")
print("=" * 50)
print(f"  Period:          {df['matchDate'].min().date()} to {df['matchDate'].max().date()}")
print(f"  Total bets:      {bets}")
print(f"  Wins:            {wins}  ({win_rate:.1%})")
print(f"  Losses:          {losses}")
print(f"  Avg odds:        {avg_odds:.2f}")
print(f"  Avg edge:        {avg_edge:.1%}")
print("")
print("  --- FLAT BETTING (1 unit) ---")
print(f"  Total profit:    {flat_profit:+.2f} units")
print(f"  ROI:             {flat_roi:+.1%}")
print(f"  Max drawdown:    {max_drawdown:.2f} units")
print("")
print("  --- KELLY BETTING (1/4 Kelly, 100u bankroll) ---")
print(f"  Final bankroll:  {bankroll:.2f} units")
print(f"  Total profit:    {kelly_profit:+.2f} units")
print(f"  ROI:             {kelly_roi:+.1%}")
print("=" * 50)
print("")

if flat_roi > 0.05:
    print("Strong positive ROI. Model has real edge.")
elif flat_roi > 0:
    print("Positive ROI but small. Monitor with paper betting first.")
else:
    print("Negative ROI. Model needs more work before betting real money.")

# ------------------------------------------------
# SAVE
# ------------------------------------------------

df.to_csv("predictions/backtest_results.csv", index=False)
print("\nBacktest saved to predictions/backtest_results.csv")