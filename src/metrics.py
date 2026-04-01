import numpy as np
import pandas as pd


def sharpe_ratio(returns, freq=252):
    """
    Annualised Sharpe ratio (assumes zero risk-free rate).

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    freq : int
        Trading days per year (252 for equities).

    Returns
    -------
    float
    """
    if returns.std() == 0:
        return np.nan
    return np.sqrt(freq) * returns.mean() / returns.std()


def max_drawdown(returns):
    """
    Maximum drawdown computed from a returns series.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns (not cumulative).

    Returns
    -------
    float
        Maximum drawdown (negative number, e.g. -0.15 means -15%).
    """
    cumulative = (1 + returns).cumprod()
    peak       = cumulative.cummax()
    drawdown   = (cumulative - peak) / peak
    return drawdown.min()


def summary_stats(results):
    """
    Compute summary performance statistics from a backtest results DataFrame.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain a 'strategy_returns' column.  Typically the output
        of backtest(), but safe to call on any row-subset of it.

    Returns
    -------
    dict
        Keys: Sharpe, Max Drawdown, Total Return, Ann. Return, Ann. Vol,
              Num Trades (approximate, counts signal changes).
    """
    returns = results["strategy_returns"].dropna()

    if len(returns) == 0:
        return {k: np.nan for k in
                ["Sharpe", "Max Drawdown", "Total Return",
                 "Ann. Return", "Ann. Vol"]}

    cumulative   = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # Annualised return and volatility
    n_days    = len(returns)
    ann_ret   = (1 + total_return) ** (252 / n_days) - 1
    ann_vol   = returns.std() * np.sqrt(252)

    return {
        "Sharpe":       sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Total Return": total_return,
        "Ann. Return":  ann_ret,
        "Ann. Vol":     ann_vol,
    }