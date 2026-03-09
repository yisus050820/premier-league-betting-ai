import pandas as pd
import numpy as np
from scipy import stats

print("Loading backtest results...")
df = pd.read_csv("predictions/backtest_results.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)
df["over25_result"] = (df["total_goals"] > 2.5).astype(int)

n_bets   = len(df)
n_wins   = (df["over25_result"] == 1).sum()
win_rate = n_wins / n_bets
roi      = df["profit_flat"].sum() / n_bets
avg_odds = df["O25"].mean()

print(f"\nBets: {n_bets} | Wins: {n_wins} | Win rate: {win_rate:.1%} | ROI: {roi:+.1%}")

# ------------------------------------------------
# TEST 1: IS THE WIN RATE STATISTICALLY SIGNIFICANT?
# Null hypothesis: model has no edge, true win rate = book's implied prob
# ------------------------------------------------

book_win_rate = (1 / df["O25"]).mean()
print(f"\nBook implied win rate: {book_win_rate:.1%}")
print(f"Model actual win rate: {win_rate:.1%}")

# Binomial test: is our win rate significantly above book's expectation?
from scipy.stats import binomtest
result = binomtest(n_wins, n_bets, book_win_rate, alternative="greater")
p_value_winrate = result.pvalue

print(f"\n--- TEST 1: Win Rate vs Book Expectation ---")
print(f"  p-value: {p_value_winrate:.4f}")
if p_value_winrate < 0.05:
    print(f"  ✅ SIGNIFICANT (p<0.05) — win rate is above book expectation")
elif p_value_winrate < 0.10:
    print(f"  ⚡ MARGINAL (p<0.10) — some evidence of edge, needs more bets")
else:
    print(f"  ❌ NOT SIGNIFICANT (p>{p_value_winrate:.2f}) — could be luck")

# ------------------------------------------------
# TEST 2: IS THE ROI SIGNIFICANT?
# Bootstrap confidence interval for ROI
# ------------------------------------------------

N_BOOTSTRAP = 10000
bootstrap_rois = []

for _ in range(N_BOOTSTRAP):
    sample = df["profit_flat"].sample(n=n_bets, replace=True)
    bootstrap_rois.append(sample.mean())

bootstrap_rois = np.array(bootstrap_rois)
ci_low  = np.percentile(bootstrap_rois, 2.5)
ci_high = np.percentile(bootstrap_rois, 97.5)
p_positive = (bootstrap_rois > 0).mean()

print(f"\n--- TEST 2: Bootstrap ROI Confidence Interval ({N_BOOTSTRAP:,} simulations) ---")
print(f"  Observed ROI:    {roi:+.1%}")
print(f"  95% CI:          [{ci_low:+.1%}, {ci_high:+.1%}]")
print(f"  P(ROI > 0):      {p_positive:.1%}")

if ci_low > 0:
    print(f"  ✅ ENTIRE confidence interval is positive — strong evidence of edge")
elif ci_low > -0.05:
    print(f"  ⚡ CI includes slightly negative — likely edge but needs more data")
else:
    print(f"  ❌ Wide CI including negative values — too few bets to be certain")

# ------------------------------------------------
# TEST 3: MONTE CARLO — could this be luck?
# Simulate 10,000 bettors with NO edge (random bets at same odds)
# See what % achieve our ROI by chance
# ------------------------------------------------

print(f"\n--- TEST 3: Monte Carlo — Probability this is luck ---")
print(f"  Simulating 10,000 random bettors at same odds...")

N_SIMULATIONS = 10000
simulated_rois = []

for _ in range(N_SIMULATIONS):
    # Random bettor: wins with probability = 1/odds (no edge)
    rand_wins  = np.random.binomial(1, 1 / df["O25"].values)
    rand_profit = np.where(rand_wins == 1, df["O25"].values - 1, -1.0)
    simulated_rois.append(rand_profit.mean())

simulated_rois = np.array(simulated_rois)
p_luck = (simulated_rois >= roi).mean()

print(f"  Our ROI:                {roi:+.1%}")
print(f"  Random bettor avg ROI:  {simulated_rois.mean():+.1%}")
print(f"  Random bettor best 5%:  {np.percentile(simulated_rois, 95):+.1%}")
print(f"  P(luck >= our ROI):     {p_luck:.1%}")

if p_luck < 0.05:
    print(f"  ✅ Only {p_luck:.1%} of random bettors beat us — very unlikely to be luck")
elif p_luck < 0.15:
    print(f"  ⚡ {p_luck:.1%} of random bettors beat us — probably real edge, needs more bets")
else:
    print(f"  ❌ {p_luck:.1%} of random bettors could achieve this ROI — likely luck")

# ------------------------------------------------
# TEST 4: MINIMUM BETS NEEDED FOR SIGNIFICANCE
# ------------------------------------------------

print(f"\n--- TEST 4: Sample Size Analysis ---")

# How many bets do we need to confirm this edge with 95% confidence?
# Based on observed win rate and book win rate
from math import ceil, sqrt

p_model = win_rate
p_book  = book_win_rate
# z-score for 95% confidence
z = 1.645
# minimum n from proportion test
n_needed = ceil(
    (z * sqrt(p_book * (1 - p_book)) + z * sqrt(p_model * (1 - p_model))) ** 2
    / (p_model - p_book) ** 2
)

print(f"  Current bets:     {n_bets}")
print(f"  Bets needed (95% confidence): {n_needed}")

if n_bets >= n_needed:
    print(f"  ✅ You have enough bets to confirm edge statistically")
else:
    pct = n_bets / n_needed * 100
    print(f"  ⚠️  You have {pct:.0f}% of the bets needed — paper bet until you reach {n_needed}")

# ------------------------------------------------
# FINAL VERDICT
# ------------------------------------------------

print(f"\n{'='*50}")
print(f"  STATISTICAL VERDICT")
print(f"{'='*50}")

signals = [
    p_value_winrate < 0.10,
    ci_low > -0.05,
    p_luck < 0.15,
]
positive = sum(signals)

if positive == 3:
    print(f"  🟢 STRONG: 3/3 tests positive")
    print(f"     Edge appears real. Start with minimum stakes ($5-10/bet)")
    print(f"     Track every bet. Review after 50 more real bets.")
elif positive == 2:
    print(f"  🟡 MODERATE: 2/3 tests positive")
    print(f"     Promising but not conclusive. Paper bet for 1 more month.")
    print(f"     Re-run this script when you have {n_needed} total backtest bets.")
else:
    print(f"  🔴 WEAK: {positive}/3 tests positive")
    print(f"     Do NOT bet real money yet. Likely overfitting or luck.")
print(f"{'='*50}")

df.to_csv("predictions/statistical_validation.csv", index=False)