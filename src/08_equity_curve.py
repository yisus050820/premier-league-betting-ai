import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

print("Loading backtest results...")

df = pd.read_csv("predictions/backtest_results.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)

df["bet_number"] = range(1, len(df) + 1)

# Cumulative curves
df["equity_flat"]  = df["profit_flat"].cumsum()

# Kelly cumulative from 100 starting bankroll
bankroll = 100.0
kelly_curve = [bankroll]
for p in df["profit_kelly"]:
    bankroll += p
    kelly_curve.append(bankroll)
kelly_curve = kelly_curve[1:]
df["equity_kelly"] = kelly_curve

# ------------------------------------------------
# PLOT
# ------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Premier League Over 2.5 — Betting Model Backtest", fontsize=14, fontweight="bold")

# --- Flat betting ---
ax1 = axes[0]
color = "green" if df["equity_flat"].iloc[-1] > 0 else "red"
ax1.plot(df["bet_number"], df["equity_flat"], color=color, linewidth=1.5)
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.fill_between(df["bet_number"], df["equity_flat"], 0,
                 where=df["equity_flat"] >= 0, alpha=0.15, color="green")
ax1.fill_between(df["bet_number"], df["equity_flat"], 0,
                 where=df["equity_flat"] < 0, alpha=0.15, color="red")
ax1.set_title("Flat Betting (1 unit per bet)")
ax1.set_ylabel("Profit (units)")
ax1.set_xlabel("Bet Number")
ax1.grid(True, alpha=0.3)

final_flat = df["equity_flat"].iloc[-1]
ax1.annotate(f"Final: {final_flat:+.1f}u",
             xy=(df["bet_number"].iloc[-1], final_flat),
             xytext=(-60, 15), textcoords="offset points",
             fontsize=10, fontweight="bold",
             color="green" if final_flat > 0 else "red")

# --- Kelly betting ---
ax2 = axes[1]
color2 = "green" if df["equity_kelly"].iloc[-1] > 100 else "red"
ax2.plot(df["bet_number"], df["equity_kelly"], color=color2, linewidth=1.5)
ax2.axhline(100, color="gray", linestyle="--", linewidth=0.8)
ax2.fill_between(df["bet_number"], df["equity_kelly"], 100,
                 where=df["equity_kelly"] >= 100, alpha=0.15, color="green")
ax2.fill_between(df["bet_number"], df["equity_kelly"], 100,
                 where=df["equity_kelly"] < 100, alpha=0.15, color="red")
ax2.set_title("1/4 Kelly Betting (starting bankroll: 100 units)")
ax2.set_ylabel("Bankroll (units)")
ax2.set_xlabel("Bet Number")
ax2.grid(True, alpha=0.3)

final_kelly = df["equity_kelly"].iloc[-1]
ax2.annotate(f"Final: {final_kelly:.1f}u",
             xy=(df["bet_number"].iloc[-1], final_kelly),
             xytext=(-60, 15), textcoords="offset points",
             fontsize=10, fontweight="bold",
             color="green" if final_kelly > 100 else "red")

plt.tight_layout()
plt.savefig("predictions/equity_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("Chart saved to predictions/equity_curve.png")