import pandas as pd


def backtest(y, x, beta, alpha, signals):
    """
    Simulate a pairs-trading strategy and return a results DataFrame.

    The spread P&L is computed from the percentage returns of each leg.
    A long-spread position (signal = +1) profits when y rises relative
    to x; a short-spread position (signal = -1) profits when y falls
    relative to x.

    Signals are shifted forward by one day so that today's signal is
    acted on tomorrow's open (no look-ahead).

    Parameters
    ----------
    y : pd.Series
        Price series for the dependent asset.
    x : pd.Series
        Price series for the independent asset.
    beta : float
        Cointegrating weight from Engle-Granger Step 1.
    alpha : float
        Intercept from Engle-Granger Step 1 (kept for reference;
        does not affect return calculation directly).
    signals : pd.Series
        Integer signal series (+1 long, -1 short, 0 flat).

    Returns
    -------
    pd.DataFrame
        Columns: strategy_returns, cumulative.

    Notes on beta interpretation
    ----------------------------
    beta here is a price-level OLS coefficient.  For a truly
    dollar-neutral trade you would need beta * (x_price / y_price)
    shares of x per share of y.  The percentage-return approximation
    used below is standard for academic backtests; flag this in the
    project report.
    """
    returns_y = y.pct_change()
    returns_x = x.pct_change()

    # Spread return: long 1 unit of y, short beta units of x
    spread_returns = returns_y - beta * returns_x

    # Use yesterday's signal to trade today (shift(1) prevents look-ahead)
    strategy_returns = signals.shift(1) * spread_returns

    results = pd.DataFrame({
        "strategy_returns": strategy_returns,
        "cumulative":       (1 + strategy_returns).cumprod(),
    })

    return results