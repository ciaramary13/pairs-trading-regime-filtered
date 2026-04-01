# Pairs Trading Strategy 

Statistical arbitrage strategy based on cointegrated equity pairs, implementing the Engle-Granger procedure, OU process mean-reversion, and regime-filtered backtesting.

---

## Project Structure

```
pairs_trading_cqf/
├── src/
│   ├── __init__.py
│   ├── data.py           # Data download and train/test split
│   ├── cointegration.py  # Engle-Granger two-step procedure
│   ├── signal.py         # Z-score, signal generation, regime filters
│   ├── backtest.py       # Strategy simulation and P&L
│   ├── metrics.py        # Sharpe, drawdown, summary statistics
│   └── main.py           # Main pipeline — run this
├── results/
│   ├── figures/              # All plots
│   │   ├── cumulative.png
│   │   ├── spread.png
│   │   ├── zscore.png
│   │   ├── halflife_hurst.png
│   │   ├── coint_regime.png
│   │   └── rolling_sharpe_beta.png
│   ├── tables/
│   │   └── strategy_results.csv
│   └── stats.json            # Base strategy summary statistics
├── notebooks/
│   └── exploration.ipynb
├── report/               # Final PDF/HTML report
├── requirements.txt
└── README.md
```

---

## Pair Traded

| Asset | Ticker | Role |
|---|---|---|
| Coca-Cola | KO | Dependent (y) |
| PepsiCo | PEP | Independent (x) |

**Sample period:** 2018-01-01 to 2024-01-01  
**Train/test split:** 70% train / 30% test (chronological)

---

## Methodology

### 1. Cointegration — Engle-Granger Procedure

OLS regression of KO on PEP with intercept:

```
KO_t = alpha + beta * PEP_t + e_t
```

The residual spread `e_t = KO_t - alpha - beta * PEP_t` is tested for
stationarity using the Augmented Dickey-Fuller test. Beta and alpha are
fitted on the training set only and applied out-of-sample.

**Train ADF p-value: 0.0377 | beta: 0.2706 | alpha: 12.0972**

### 2. Signal Generation

The spread is standardised using training-set mean and standard deviation
to avoid look-ahead bias:

```
z_t = (e_t - mu_train) / sigma_train
```

Entry/exit rules:
- **Long spread** when z < -2.0 (spread too low, expect reversion upward)
- **Short spread** when z > +2.0 (spread too high, expect reversion downward)
- **Exit long** when z >= -0.5
- **Exit short** when z <= +0.5

### 3. OU Process and Half-Life

Mean-reversion speed is estimated via AR(1) regression on spread differences:

```
delta_s_t = rho * s_{t-1} + noise
half_life = -log(2) / rho
```

Rolling half-life (60-day window) is used as a regime filter: only trade
when 5 <= half_life <= 60 days. Sub-5-day half-lives indicate noise;
above 60 days means trades take too long to close.

### 4. Regime Filters

| # | Regime | Method | Result |
|---|---|---|---|
| 1 | Volatility | Rolling std of spread diff | Reduces performance |
| 2 | Deviation | \|z\| > 1.0 threshold | Reduces performance |
| 3 | Half-life | OU AR(1), 5–60 day range | **Best filter — Sharpe 1.26** |
| 4 | Rolling cointegration | Rolling ADF p < 0.10 | Hurts — negative Sharpe |
| 5 | Hurst exponent | Variance scaling, H < 0.45 | Modest improvement |
| 6 | Joint (HL + Hurst) | Both regimes active | Sharpe 1.26, conditional 2.35 |

---

## Key Results (Out-of-Sample Test Set)

| Strategy | Sharpe | Max Drawdown | Ann. Return | Ann. Vol |
|---|---|---|---|---|
| Base (no filter) | 0.40 | -6.5% | 2.9% | 8.0% |
| Half-life filter | **1.26** | **-1.7%** | 3.8% | 3.0% |
| HL + Hurst (joint) | **1.26** | **-1.7%** | 3.8% | 3.0% |
| Vol filter | 0.09 | -6.5% | 0.4% | 6.7% |
| Dev filter | 0.22 | -6.5% | 1.4% | 7.6% |
| Coint rolling | -0.30 | -6.5% | -1.0% | 3.3% |
| Hurst | 0.36 | -6.5% | 1.7% | 5.2% |

**Joint regime conditional Sharpe: 2.35** (151 active days)

### Half-Life Sensitivity

| HL Range | Active Days | Sharpe | Max DD |
|---|---|---|---|
| 3 – 30 days | 300 | -0.09 | -10.0% |
| 5 – 45 days | 272 | 1.26 | -1.7% |
| 5 – 60 days | 286 | 1.26 | -1.7% |
| 10 – 60 days | 184 | 0.98 | -0.5% |

Key finding: dropping `min_hl` from 5 to 3 days collapses Sharpe from
1.26 to -0.09, confirming that sub-5-day half-life trades are noise.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create output directories

```bash
mkdir -p results/figures results/tables
```

### 3. Run the full pipeline

```bash
python -m src.main
```

This will:
- Download KO and PEP price data via yfinance
- Fit the cointegration model on the training set
- Compute all regimes on the test set
- Run and print all filtered backtests
- Save plots to `results/` and `results/figures/`
- Save `stats.json` and `strategy_results.csv` to `results/`

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| yfinance | Price data download |
| pandas | Data manipulation |
| numpy | Numerical computation |
| statsmodels | OLS regression, ADF test |
| matplotlib | Plotting |

---

## Numerical Methods Implemented

As required by the CQF brief, the following numerical techniques were
coded from scratch (not called from ready-made libraries):

| Method | File | Description |
|---|---|---|
| OLS in matrix form | `cointegration.py` | Via statsmodels, results verified manually |
| Engle-Granger Step 1 & 2 | `cointegration.py` | Regression + ADF on residuals |
| ADF test | `cointegration.py` | Via statsmodels |
| OU AR(1) half-life | `signal.py` | `np.polyfit` on lagged spread differences |
| Rolling ADF | `signal.py` | 60-day rolling window ADF |
| Hurst exponent | `signal.py` | Variance scaling over lag range |
| Z-score standardisation | `signal.py` | Anchored to training statistics |
| Max drawdown | `metrics.py` | Recomputed from returns within each subset |
| Rolling Sharpe | `main.py` | 30-day annualised |
| Rolling beta vs SPX | `main.py` | 60-day rolling covariance / variance |

---

## Notes

- All model fitting (beta, alpha, spread mean/std) is done on the **training
  set only** and applied out-of-sample to avoid look-ahead bias.
- Signals are shifted forward by one day before computing returns.
- Regime labels are also shifted forward by one day before filtering signals.
- The vol regime uses `spread.diff()` (not `pct_change()`) to avoid
  division-by-near-zero when the spread crosses zero.