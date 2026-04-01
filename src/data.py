import yfinance as yf
import pandas as pd


def download_data(tickers, start="2018-01-01", end="2024-01-01"):
    """
    Download adjusted closing prices for the given tickers.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols, e.g. ["KO", "PEP"].
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame of closing prices, NaN rows dropped.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna()
    return data


def split_data(data, split_ratio=0.7):
    """
    Split a time-series DataFrame into train and test sets chronologically.

    Parameters
    ----------
    data : pd.DataFrame
        Full price data.
    split_ratio : float
        Fraction of data used for training (default 0.7).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test) DataFrames.
    """
    split = int(len(data) * split_ratio)
    train = data.iloc[:split]
    test = data.iloc[split:]
    return train, test