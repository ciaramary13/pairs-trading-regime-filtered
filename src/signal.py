import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def compute_zscore(spread, train_mean=None, train_std=None, window=30):
    """
    Standardise a spread series into a z-score.

    Out-of-sample usage (recommended)
    ----------------------------------
    Pass `train_mean` and `train_std` computed on the training spread so
    the test z-score is anchored to in-sample statistics. This avoids
    look-ahead bias and is the correct approach when the model is fitted
    on train and applied to test.

    In-sample / exploratory usage
    ------------------------------
    Leave `train_mean` and `train_std` as None to use a rolling window
    mean/std (introduces NaNs for the first `window` observations).

    Parameters
    ----------
    spread : pd.Series
    train_mean : float, optional
    train_std : float, optional
    window : int

    Returns
    -------
    pd.Series
    """
    if train_mean is not None and train_std is not None:
        return (spread - train_mean) / train_std

    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    return (spread - mean) / std


def generate_signals(z, entry=2.0, exit=0.5):
    """
    Generate mean-reversion trading signals from a z-score series.

    Logic
    -----
    - Enter long  (signal = +1) when z < -entry  (spread too low)
    - Enter short (signal = -1) when z >  entry  (spread too high)
    - Exit long   when z >= -exit  (spread has reverted toward mean)
    - Exit short  when z <=  exit  (spread has reverted toward mean)

    Parameters
    ----------
    z : pd.Series
    entry : float
        Entry threshold (absolute z-score).
    exit : float
        Exit threshold (absolute z-score, must be < entry).

    Returns
    -------
    pd.Series
        Integer signal series: +1 (long spread), -1 (short), 0 (flat).
    """
    signals  = pd.Series(index=z.index, data=0, dtype=int)
    position = 0

    for i in range(len(z)):
        val = z.iloc[i]

        if pd.isna(val):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if val > entry:
                position = -1   # short spread
            elif val < -entry:
                position = 1    # long spread

        elif position == 1:
            if val >= -exit:
                position = 0

        elif position == -1:
            if val <= exit:
                position = 0

        signals.iloc[i] = position

    return signals


# ======================================================================
# Regime 1: Volatility regime (FIXED)
# ======================================================================

def compute_vol_regime(spread, window=30):
    """
    Label each day as high-volatility (1) or low-volatility (0).

    Parameters
    ----------
    spread : pd.Series
    window : int

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (regime, vol) — regime is 0/1, vol is the rolling std of diff.
    """
    vol       = spread.diff().rolling(window).std()
    threshold = vol.median()
    regime    = (vol > threshold).astype(int)
    return regime, vol


# ======================================================================
# Regime 2: Deviation regime (unchanged)
# ======================================================================

def compute_deviation_regime(z, threshold=1.0):
    """
    Label each day as large-deviation (1) or small-deviation (0).

    Parameters
    ----------
    z : pd.Series
    threshold : float

    Returns
    -------
    pd.Series
    """
    return (z.abs() > threshold).astype(int)


# ======================================================================
# Regime 3: Half-life regime (NEW)
# Directly addresses the OU process / half-life requirement in the brief.
# ======================================================================

def compute_halflife(spread, window=60):
    """
    Rolling OU half-life via AR(1) regression on spread differences.

    Fits:  delta_s_t = rho * s_{t-1} + noise
    where rho is the mean-reversion speed.
    Half-life = -log(2) / rho  (only defined when rho < 0).

    A half-life that is too short (<5 days) means you are trading noise.
    A half-life that is too long (>60 days) means trades take forever to
    close and capital is tied up unproductively.

    Parameters
    ----------
    spread : pd.Series
    window : int
        Rolling estimation window in days.

    Returns
    -------
    pd.Series
        Rolling half-life in days (NaN where rho >= 0, i.e. not mean-reverting).
    """
    half_lives = pd.Series(index=spread.index, dtype=float)

    for i in range(window, len(spread)):
        s      = spread.iloc[i - window:i]
        lagged = s.shift(1).dropna()
        delta  = s.diff().dropna()
        lagged, delta = lagged.align(delta, join="inner")

        if len(lagged) < 10:
            continue

        rho = np.polyfit(lagged, delta, 1)[0]

        if rho < 0:
            half_lives.iloc[i] = -np.log(2) / rho
        else:
            half_lives.iloc[i] = np.nan

    return half_lives


def compute_halflife_regime(half_lives, min_hl=5, max_hl=60):
    """
    Trade only when the rolling half-life is in a sensible range.

    Parameters
    ----------
    half_lives : pd.Series
        Output of compute_halflife().
    min_hl : float
        Minimum half-life in days (below this we are trading noise).
    max_hl : float
        Maximum half-life in days (above this trades take too long).

    Returns
    -------
    pd.Series
        1 = tradeable, 0 = avoid.
    """
    regime = ((half_lives >= min_hl) & (half_lives <= max_hl)).astype(int)
    return regime


# ======================================================================
# Regime 4: Rolling cointegration regime (NEW)
# Addresses the brief's requirement to discuss rolling re-estimation
# and structural breaks.
# ======================================================================

def compute_coint_regime(spread, window=60, pvalue_threshold=0.10):
    """
    Rolling ADF test on the spread.

    Only trade (regime = 1) when the spread is currently confirmed
    stationary at the given significance level. This implements the
    rolling re-estimation the CQF brief explicitly recommends and gives
    a principled way to avoid trading through structural breaks.

    Parameters
    ----------
    spread : pd.Series
    window : int
        Rolling window for each ADF test.
    pvalue_threshold : float
        Significance level (0.10 is generous; 0.05 is stricter).

    Returns
    -------
    pd.Series
        1 = cointegration confirmed in this window, 0 = avoid.
    """
    regime = pd.Series(index=spread.index, data=0, dtype=int)

    for i in range(window, len(spread)):
        s = spread.iloc[i - window:i]
        try:
            _, pval, *_ = adfuller(s)
            if pval < pvalue_threshold:
                regime.iloc[i] = 1
        except Exception:
            pass

    return regime


# ======================================================================
# Regime 5: Hurst exponent regime (NEW, optional)
# Measures mean-reversion strength directly.
# ======================================================================

def compute_hurst(spread, window=60, lags=range(2, 20)):
    """
    Rolling Hurst exponent via variance of lagged differences.

    H < 0.5  => mean-reverting (good for pairs trading)
    H = 0.5  => random walk
    H > 0.5  => trending

    Parameters
    ----------
    spread : pd.Series
    window : int
    lags : iterable of int

    Returns
    -------
    pd.Series
        Rolling Hurst exponent.
    """
    hurst = pd.Series(index=spread.index, dtype=float)

    for i in range(window, len(spread)):
        s   = spread.iloc[i - window:i].values
        tau = []
        for lag in lags:
            diff = np.subtract(s[lag:], s[:-lag])
            std  = np.std(diff)
            tau.append(std if std > 0 else np.nan)

        tau_arr = np.array(tau, dtype=float)
        lag_arr = np.array(list(lags), dtype=float)

        valid = ~np.isnan(tau_arr)
        if valid.sum() < 4:
            continue

        poly = np.polyfit(np.log(lag_arr[valid]), np.log(tau_arr[valid]), 1)
        hurst.iloc[i] = poly[0]

    return hurst


def compute_hurst_regime(hurst, threshold=0.45):
    """
    Trade only when spread exhibits strong mean reversion (H < threshold).

    Parameters
    ----------
    hurst : pd.Series
    threshold : float

    Returns
    -------
    pd.Series
        1 = mean-reverting regime, 0 = avoid.
    """
    return (hurst < threshold).astype(int)


# ======================================================================
# Combined regime (utility)
# ======================================================================

def compute_combined_regime(z, vol, vol_threshold=None, z_threshold=1.0):
    """
    Combined regime: active only when BOTH vol is elevated AND |z| is large.

    Parameters
    ----------
    z : pd.Series
    vol : pd.Series
    vol_threshold : float, optional
    z_threshold : float

    Returns
    -------
    pd.Series
    """
    if vol_threshold is None:
        vol_threshold = vol.median()
    return ((z.abs() > z_threshold) & (vol > vol_threshold)).astype(int)

def compute_joint_regime(regime_hl, regime_hurst):
    """
    Trade only when BOTH half-life is in range AND Hurst confirms
    mean reversion. More selective than either filter alone.
    """
    return ((regime_hl == 1) & (regime_hurst == 1)).astype(int)