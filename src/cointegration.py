import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def engle_granger(y, x):
    """
    Engle-Granger two-step cointegration procedure.

    Step 1: OLS regression of y on x (with intercept).
    Step 2: ADF test on the residual spread.

    Parameters
    ----------
    y : pd.Series
        Dependent price series.
    x : pd.Series
        Independent price series.

    Returns
    -------
    dict with keys:
        alpha   – OLS intercept
        beta    – OLS slope (cointegrating weight)
        spread  – stationary residual: y - alpha - beta * x
        adf_stat – ADF test statistic on the spread
        p_value  – MacKinnon p-value for ADF
        model   – fitted OLS model (for diagnostics)
    """
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()

    alpha = model.params.iloc[0]   # intercept
    beta  = model.params.iloc[1]   # slope

    # Correct spread: residual is y minus the full fitted line
    spread = y - alpha - beta * x

    adf_stat, p_value, _, _, _, _ = adfuller(spread)

    return {
        "alpha":    alpha,
        "beta":     beta,
        "spread":   spread,
        "adf_stat": adf_stat,
        "p_value":  p_value,
        "model":    model,
    }