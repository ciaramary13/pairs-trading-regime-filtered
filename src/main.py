from src.data import download_data, split_data
from src.cointegration import engle_granger
from src.signal import (
    compute_zscore,
    generate_signals,
    compute_vol_regime,
    compute_deviation_regime,
    compute_halflife,
    compute_halflife_regime,
    compute_coint_regime,
    compute_hurst,
    compute_hurst_regime,
    compute_joint_regime,
    compute_combined_regime,
)
from src.backtest import backtest
from src.metrics import summary_stats

import matplotlib.pyplot as plt
import json
import numpy as np
import yfinance as yf

# ======================================================================
# 1. Load and split data
# ======================================================================
data = download_data(["KO", "PEP"])
train, test = split_data(data)

y_train, x_train = train["KO"], train["PEP"]
y_test,  x_test  = test["KO"],  test["PEP"]

# ======================================================================
# 2. Fit cointegration model on TRAIN
# ======================================================================
res_train = engle_granger(y_train, x_train)

print(f"Train ADF p-value : {res_train['p_value']:.4f}")
print(f"Train beta        : {res_train['beta']:.4f}")
print(f"Train alpha       : {res_train['alpha']:.4f}")

if res_train["p_value"] > 0.05:
    print("WARNING: ADF p-value > 0.05 — cointegration not confirmed at 5%.")

alpha      = res_train["alpha"]
beta       = res_train["beta"]
train_mean = res_train["spread"].mean()
train_std  = res_train["spread"].std()

# ======================================================================
# 3. Build test spread and z-score (no look-ahead)
# ======================================================================
spread_test = y_test - alpha - beta * x_test
z_test      = compute_zscore(spread_test, train_mean=train_mean, train_std=train_std)

# ======================================================================
# 4. Base signals
# ======================================================================
signals_test = generate_signals(z_test, entry=2.0, exit=0.5)

# ======================================================================
# 5. Compute all regimes
# ======================================================================
print("\nComputing regimes (rolling windows — may take a moment)...")

regime_vol, vol = compute_vol_regime(spread_test, window=30)
regime_dev      = compute_deviation_regime(z_test, threshold=1.0)
half_lives      = compute_halflife(spread_test, window=60)
regime_hl       = compute_halflife_regime(half_lives, min_hl=5, max_hl=60)
regime_coint    = compute_coint_regime(spread_test, window=60, pvalue_threshold=0.10)
hurst           = compute_hurst(spread_test, window=60)
regime_hurst    = compute_hurst_regime(hurst, threshold=0.45)
regime_joint    = compute_joint_regime(regime_hl, regime_hurst)

print("Regimes computed.")

# ======================================================================
# 6. Helper: apply a regime filter to signals
# ======================================================================
def apply_regime_filter(signals, regime):
    """Zero out signals where the lagged regime == 0 (no look-ahead)."""
    filtered = signals.copy()
    filtered[regime.shift(1) == 0] = 0
    return filtered

# ======================================================================
# 7. Filtered backtests
# ======================================================================
results_base  = backtest(y_test, x_test, beta, alpha, signals_test)

results_vol   = backtest(y_test, x_test, beta, alpha,
                         apply_regime_filter(signals_test, regime_vol))
results_dev   = backtest(y_test, x_test, beta, alpha,
                         apply_regime_filter(signals_test, regime_dev))
results_hl    = backtest(y_test, x_test, beta, alpha,
                         apply_regime_filter(signals_test, regime_hl))
results_coint = backtest(y_test, x_test, beta, alpha,
                         apply_regime_filter(signals_test, regime_coint))
results_hurst = backtest(y_test, x_test, beta, alpha,
                         apply_regime_filter(signals_test, regime_hurst))
results_joint = backtest(y_test, x_test, beta, alpha,
                         apply_regime_filter(signals_test, regime_joint))

# Attach regime labels to base results for conditional analysis
results_base["regime_vol"]   = regime_vol.shift(1)
results_base["regime_dev"]   = regime_dev.shift(1)
results_base["regime_hl"]    = regime_hl.shift(1)
results_base["regime_coint"] = regime_coint.shift(1)
results_base["regime_hurst"] = regime_hurst.shift(1)
results_base["regime_joint"] = regime_joint.shift(1)
results_base["half_life"]    = half_lives.shift(1)
results_base["hurst"]        = hurst.shift(1)

# ======================================================================
# 8. Print all strategy stats
# ======================================================================
def print_stats(label, results):
    stats = summary_stats(results)
    print(f"\n--- {label} ---")
    for k, v in stats.items():
        print(f"  {k:>15}: {v:.4f}")
    return stats

print_stats("Base strategy (no filter)", results_base)
print_stats("Vol-filtered   (Regime 1)", results_vol)
print_stats("Dev-filtered   (Regime 2)", results_dev)
print_stats("Half-life      (Regime 3)", results_hl)
print_stats("Coint-rolling  (Regime 4)", results_coint)
print_stats("Hurst          (Regime 5)", results_hurst)
print_stats("HL + Hurst     (Joint)",    results_joint)

# ======================================================================
# 9. Regime-conditional performance (within each regime state)
# ======================================================================
print("\n====== Regime-conditional performance ======")
for label, col in [
    ("Vol",   "regime_vol"),
    ("Dev",   "regime_dev"),
    ("HL",    "regime_hl"),
    ("Coint", "regime_coint"),
    ("Hurst", "regime_hurst"),
    ("Joint", "regime_joint"),
]:
    active   = results_base[results_base[col] == 1]
    inactive = results_base[results_base[col] == 0]
    sh_a = summary_stats(active)["Sharpe"]
    sh_i = summary_stats(inactive)["Sharpe"]
    print(f"  {label:6}  active={len(active):4d}d  Sharpe={sh_a:+.3f}  |  "
          f"inactive={len(inactive):4d}d  Sharpe={sh_i:+.3f}")

# ======================================================================
# 10. Half-life sensitivity analysis
# ======================================================================
print("\n====== Half-life filter sensitivity ======")
print(f"  {'HL range':>12}  {'active':>6}  {'Sharpe':>7}  {'MDD':>7}  {'Ann.Ret':>8}")
for min_hl, max_hl in [(3, 30), (5, 45), (5, 60), (5, 90), (10, 60), (10, 45)]:
    r_hl  = compute_halflife_regime(half_lives, min_hl=min_hl, max_hl=max_hl)
    sig   = apply_regime_filter(signals_test, r_hl)
    res   = backtest(y_test, x_test, beta, alpha, sig)
    s     = summary_stats(res)
    days  = int((r_hl.shift(1) == 1).sum())
    print(f"  [{min_hl:2d} – {max_hl:2d} days]  {days:6d}  "
          f"{s['Sharpe']:7.3f}  {s['Max Drawdown']:7.3f}  {s['Ann. Return']:8.3f}")

# ======================================================================
# 11. Descriptive stats for OU / Hurst
# ======================================================================
hl_valid    = half_lives.dropna()
hurst_valid = hurst.dropna()

print(f"\nHalf-life:  mean={hl_valid.mean():.1f}d  median={hl_valid.median():.1f}d  "
      f"min={hl_valid.min():.1f}d  max={hl_valid.max():.1f}d")
print(f"Hurst:      mean={hurst_valid.mean():.3f}  median={hurst_valid.median():.3f}  "
      f"% < 0.5={100*(hurst_valid < 0.5).mean():.1f}%  "
      f"% < 0.45={100*(hurst_valid < 0.45).mean():.1f}%")

# ======================================================================
# 12. Save outputs
# ======================================================================
results_base.to_csv("results/tables/strategy_results.csv")

with open("results/stats.json", "w") as f:
    json.dump({k: float(v) for k, v in summary_stats(results_base).items()},
              f, indent=4)

# ======================================================================
# 13. Plots
# ======================================================================

# --- Plot 1: Cumulative returns — all strategies ---
fig, ax = plt.subplots(figsize=(13, 5))
results_base["cumulative"].plot( ax=ax, label="Base",          linewidth=2,  color="black")
results_hl["cumulative"].plot(   ax=ax, label="Half-life",     linestyle="-.",color="steelblue", linewidth=2)
results_joint["cumulative"].plot(ax=ax, label="HL + Hurst",    linestyle="-",color="green",     linewidth=2)
results_hurst["cumulative"].plot(ax=ax, label="Hurst",         linestyle=":", color="purple")
results_vol["cumulative"].plot(  ax=ax, label="Vol filter",    linestyle="--",color="orange")
results_dev["cumulative"].plot(  ax=ax, label="Dev filter",    linestyle="--",color="red")
results_coint["cumulative"].plot(ax=ax, label="Coint rolling", linestyle="--",color="grey")
ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Out-of-Sample Cumulative Returns — All Regime Filters")
ax.set_ylabel("Cumulative return")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/cumulative.png", dpi=150)
plt.show()

# --- Plot 2: Spread and z-score ---
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

spread_test.plot(ax=axes[0], color="steelblue")
axes[0].axhline(train_mean, color="red", linestyle="--", linewidth=0.8, label="Train mean")
axes[0].set_title("Test Spread  (KO − α − β·PEP)")
axes[0].legend(fontsize=9)

z_test.plot(ax=axes[1], color="darkorange")
for level, style in [(2, "--"), (-2, "--"), (0.5, ":"), (-0.5, ":")]:
    axes[1].axhline(level, linestyle=style, color="grey", linewidth=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Z-Score — entry ±2, exit ±0.5")

plt.tight_layout()
plt.savefig("results/figures/spread.png", dpi=150)
plt.show()

# --- Plot 3: Rolling half-life and Hurst ---
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

half_lives.plot(ax=axes[0], color="purple")
axes[0].axhline(5,  color="red",   linestyle="--", linewidth=0.8, label="Min HL (5d)")
axes[0].axhline(60, color="green", linestyle="--", linewidth=0.8, label="Max HL (60d)")
axes[0].set_title("Rolling OU Half-Life (days)")
axes[0].set_ylim(0, 120)
axes[0].legend(fontsize=9)

hurst.plot(ax=axes[1], color="teal")
axes[1].axhline(0.5,  color="black", linestyle="-",  linewidth=0.8, label="Random walk (0.5)")
axes[1].axhline(0.45, color="red",   linestyle="--", linewidth=0.8, label="Regime threshold (0.45)")
axes[1].set_title("Rolling Hurst Exponent")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/figures/halflife_hurst.png", dpi=150)
plt.show()

# --- Plot 4: Rolling cointegration regime ---
fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)

regime_coint.plot(ax=axes[0], color="navy", drawstyle="steps-post")
axes[0].set_title("Rolling Cointegration Regime  (1 = confirmed, 0 = avoid)")
axes[0].set_yticks([0, 1])
axes[0].set_ylabel("Regime")

results_coint["cumulative"].plot(ax=axes[1], color="darkgreen")
axes[1].axhline(1, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Cumulative Return — Cointegration-Filtered Strategy")
axes[1].set_ylabel("Cumulative return")

plt.tight_layout()
plt.savefig("results/figures/coint_regime.png", dpi=150)
plt.show()

# --- Plot 5: Z-score with signal overlay ---
fig, ax = plt.subplots(figsize=(13, 4))
z_test.plot(ax=ax, color="darkorange", alpha=0.7, label="Z-score")
for level, style in [(2, "--"), (-2, "--"), (0.5, ":"), (-0.5, ":")]:
    ax.axhline(level, linestyle=style, color="grey", linewidth=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Z-Score with Entry/Exit Bands")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/zscore.png", dpi=150)
plt.show()

# --- Plot 6: Rolling Sharpe and rolling beta vs S&P 500 ---
# Download SPX returns aligned to test period
print("\nDownloading S&P 500 for rolling beta calculation...")
spx_raw = yf.download("^GSPC",
                       start=test.index[0].strftime("%Y-%m-%d"),
                       end=test.index[-1].strftime("%Y-%m-%d"),
                       auto_adjust=True, progress=False)["Close"]
spx_ret = spx_raw.squeeze().pct_change()

# Align with strategy returns (axis=0 required when aligning two Series)
strat_ret = results_base["strategy_returns"]
spx_ret, strat_aligned = spx_ret.align(strat_ret, join="inner", axis=0)

# Rolling 30-day Sharpe
rolling_sr = strat_ret.rolling(30).apply(
    lambda r: np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan,
    raw=True
)

# Rolling 60-day beta vs SPX
rolling_cov  = strat_aligned.rolling(60).cov(spx_ret)
rolling_var  = spx_ret.rolling(60).var()
rolling_beta = rolling_cov / rolling_var

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

rolling_sr.plot(ax=axes[0], color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axhline(1, color="green", linestyle="--", linewidth=0.8, label="Sharpe = 1")
axes[0].set_title("Rolling 30-Day Sharpe Ratio (annualised)")
axes[0].legend(fontsize=9)

rolling_beta.plot(ax=axes[1], color="firebrick")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Rolling 60-Day Beta vs S&P 500")
axes[1].set_ylabel("Beta")

plt.tight_layout()
plt.savefig("results/figures/rolling_sharpe_beta.png", dpi=150)
plt.show()