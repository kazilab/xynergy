"""Quality scores, AUC/AAC, XEPTO, and Drug Sensitivity Score (DSS).

Adapted from xynergy008's scores.py and tools.py to work with xynergy's
Polars-based pipeline and 4PL function conventions.
"""

import numpy as np
import scipy.integrate as spi
from scipy import stats
from scipy.integrate import quad
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from xynergy.fit import _ll4, fit_curve


def quality_scores(original_values, fitted_values):
    """Calculate quality-of-fit metrics between observed and predicted values.

    Parameters
    ----------
    original_values : array-like
        Observed response values.
    fitted_values : array-like
        Model-predicted response values.

    Returns
    -------
    dict
        Dictionary with keys: r2, adjusted_r2, syx, rmse, shapiro_p,
        explained_variance, max_error, rmae, mape.
    """
    original_values = np.asarray(original_values, dtype=float)
    fitted_values = np.asarray(fitted_values, dtype=float)

    n = len(original_values)
    k = 1  # number of predictors

    r2 = r2_score(original_values, fitted_values)
    ar2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan

    syx = np.sqrt(np.sum((original_values - fitted_values) ** 2) / max(n - 2, 1))
    rmse = np.sqrt(mean_squared_error(original_values, fitted_values))

    residuals = original_values - fitted_values
    if n >= 3:
        _, shapiro_p = stats.shapiro(residuals)
    else:
        shapiro_p = np.nan

    evar = explained_variance_score(original_values, fitted_values)
    merr = max_error(original_values, fitted_values)
    rmae = np.sqrt(mean_absolute_error(original_values, fitted_values))
    mape = mean_absolute_percentage_error(original_values, fitted_values)

    return {
        "r2": round(r2, 4),
        "adjusted_r2": round(ar2, 4),
        "syx": round(syx, 4),
        "rmse": round(rmse, 4),
        "shapiro_p": round(shapiro_p, 4),
        "explained_variance": round(evar, 4),
        "max_error": round(merr, 4),
        "rmae": round(rmae, 4),
        "mape": round(mape, 4),
    }


def cal_auc_aac(
    slope,
    min_val,
    max_val,
    log10_ic50,
    log10_doses,
    baseline=0,
):
    """Calculate AUC and AAC from 4PL fit parameters.

    Parameters
    ----------
    slope : float
        Hill slope.
    min_val : float
        Minimum response (lower asymptote).
    max_val : float
        Maximum response (upper asymptote).
    log10_ic50 : float
        Log10-transformed IC50.
    log10_doses : array-like
        Log10-transformed drug concentrations.
    baseline : float, default 0
        Baseline response for AUC calculation.

    Returns
    -------
    dict
        Dictionary with keys 'auc' and 'aac'.
    """
    log10_doses = np.asarray(log10_doses, dtype=float)
    dose_min = np.min(log10_doses)
    dose_max = np.max(log10_doses)

    def curve_minus_baseline(x):
        return _ll4(x, slope, min_val, max_val, log10_ic50) - baseline

    def max_minus_curve(x):
        return max_val - _ll4(x, slope, min_val, max_val, log10_ic50)

    auc, _ = quad(curve_minus_baseline, dose_min, dose_max)
    aac, _ = quad(max_minus_curve, dose_min, dose_max)

    return {"auc": round(auc, 2), "aac": round(aac, 2)}


def xepto_score(
    slope,
    min_val,
    max_val,
    log10_ic50,
    baseline_for_auc,
    integration_limit=1,
):
    """Calculate the XEPTO potency score.

    The XEPTO score quantifies the proportion of the dose-response curve above
    a baseline within a specific integration window around the IC50.

    Parameters
    ----------
    slope : float
        Hill slope.
    min_val : float
        Minimum response (lower asymptote).
    max_val : float
        Maximum response (upper asymptote).
    log10_ic50 : float
        Log10-transformed IC50 (integration starts here).
    baseline_for_auc : float
        Baseline response subtracted from the curve.
    integration_limit : float, default 1
        Width of integration window in log10 dose units from log10_ic50.

    Returns
    -------
    float
        XEPTO50 score (percentage).
    """

    def curve_minus_baseline(x):
        return _ll4(x, slope, min_val, max_val, log10_ic50) - 50

    integration_result, _ = quad(
        curve_minus_baseline,
        log10_ic50,
        log10_ic50 + integration_limit,
    )

    area_of_interest = (max_val - baseline_for_auc) * integration_limit
    if area_of_interest == 0:
        return 0.0

    xepto50 = (integration_result / area_of_interest) * 100
    return round(xepto50, 2)


def auc_from_params(
    log10_dose_min,
    log10_dose_max,
    slope,
    min_val,
    max_val,
    log10_ic50,
):
    """Calculate AUC from 4PL parameters over a dose range.

    Parameters
    ----------
    log10_dose_min : float
        Minimum log10 drug concentration.
    log10_dose_max : float
        Maximum log10 drug concentration.
    slope : float
        Hill slope.
    min_val : float
        Minimum response (lower asymptote, baseline).
    max_val : float
        Maximum response (upper asymptote).
    log10_ic50 : float
        Log10-transformed IC50.

    Returns
    -------
    float
        Area under the 4PL curve.
    """
    result, _ = spi.quad(
        _ll4, log10_dose_min, log10_dose_max, args=(slope, min_val, max_val, log10_ic50)
    )
    return result


def dss(
    ic50,
    slope,
    max_val,
    min_con_tested,
    max_con_tested,
    y=10,
    dss_type=2,
    con_scale=1e-9,
):
    """Calculate Drug Sensitivity Score (DSS).

    Computes a normalized area-based drug sensitivity metric from 4PL curve
    parameters. Three DSS variants are available.

    Parameters
    ----------
    ic50 : float
        IC50 in the same units as min/max_con_tested.
    slope : float
        Hill slope.
    max_val : float
        Maximum response (upper asymptote, percent).
    min_con_tested : float
        Minimum concentration tested.
    max_con_tested : float
        Maximum concentration tested.
    y : float, default 10
        Activity threshold (minimum response to count as active).
    dss_type : int, default 2
        DSS variant (1, 2, or 3).
    con_scale : float, default 1e-9
        Concentration scaling factor (e.g. 1e-9 for nanomolar inputs).

    Returns
    -------
    float or None
        DSS score, or None if inputs are invalid.
    """
    a = float(max_val)
    b = float(slope)
    d = 0  # min response fixed at 0
    ic50 = float(ic50)
    min_con_tested = float(min_con_tested)
    max_con_tested = float(max_con_tested)
    min_con = np.log10(min_con_tested * con_scale)
    x2 = np.log10(max_con_tested * con_scale)

    if any(np.isnan(v) for v in [ic50, b, a, min_con, x2]):
        return None

    if ic50 >= max_con_tested:
        return 0

    if b == 0:
        return 0

    if a > 100:
        a = 100

    if b < 0:
        b = -b

    c = np.log10(ic50 * con_scale)

    if a > y:
        if y != 0:
            x1 = c - ((np.log(a - y) - np.log(y - d)) / (b * np.log(10)))
            if x1 < min_con:
                x1 = min_con
            elif x1 > x2:
                x1 = x2
        else:
            x1 = min_con

        area, _ = spi.quad(_ll4, x1, x2, args=(b, d, a, c))
        int_y = area - y * (x2 - x1)

        total_area = (x2 - min_con) * (100 - y)
        norm_area = 0
        if dss_type == 1:
            norm_area = (int_y / total_area) * 100
        elif dss_type == 2:
            norm_area = (int_y / total_area) * 100 / np.log10(a) if a > 1 else 0
            if norm_area > 50:
                norm_area = 0
        elif dss_type == 3:
            norm_area = (
                (int_y / total_area)
                * 100
                * (np.log10(100) / np.log10(a) if a > 1 else 0)
                * ((x2 - x1) / (x2 - min_con) if x2 != min_con else 0)
            )

        if norm_area < 0 or norm_area > 100:
            return 0
        else:
            return round(norm_area, 2)
    else:
        return 0


def xeptosync(
    matrix,
    doses_a,
    doses_b,
    fit_a,
    fit_b,
    con_scale_a=1e-9,
    con_scale_b=1e-9,
    baseline=10,
    integration_limit=1,
):
    """Calculate synchronized XEPTO score across a drug combination matrix.

    Computes XEPTO scores for each drug alone, for each row/column of the
    combination matrix, and for the diagonal, then returns their sum as
    a measure of combination potency.

    Parameters
    ----------
    matrix : numpy.ndarray
        2D response matrix where rows correspond to doses_a and columns to
        doses_b. First row/column should be single-agent responses (dose=0
        of the other drug).
    doses_a : array-like
        Doses for drug A (row labels, including 0).
    doses_b : array-like
        Doses for drug B (column labels, including 0).
    fit_a : dict
        Fit parameters for drug A with keys: slope, min, max, ic50.
    fit_b : dict
        Fit parameters for drug B with keys: slope, min, max, ic50.
    con_scale_a : float, default 1e-9
        Concentration scaling factor for drug A.
    con_scale_b : float, default 1e-9
        Concentration scaling factor for drug B.
    baseline : float, default 10
        Baseline for XEPTO AUC calculation.
    integration_limit : float, default 1
        Integration window in log10 dose units.

    Returns
    -------
    float
        The xeptosync score (sum of single-agent XEPTO scores).
    """
    doses_a = np.asarray(doses_a, dtype=float)
    doses_b = np.asarray(doses_b, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    # Single-agent responses (first column = drug A alone, first row = drug B alone)
    r1 = matrix[1:, 0]  # drug A responses
    r2 = matrix[0, 1:]  # drug B responses

    log10_ic50_a = np.log10(fit_a["ic50"] * con_scale_a)
    log10_ic50_b = np.log10(fit_b["ic50"] * con_scale_b)

    xepto_d1 = xepto_score(
        slope=fit_a["slope"],
        min_val=fit_a["min"],
        max_val=fit_a["max"],
        log10_ic50=log10_ic50_a,
        baseline_for_auc=baseline,
        integration_limit=integration_limit,
    )

    xepto_d2 = xepto_score(
        slope=fit_b["slope"],
        min_val=fit_b["min"],
        max_val=fit_b["max"],
        log10_ic50=log10_ic50_b,
        baseline_for_auc=baseline,
        integration_limit=integration_limit,
    )

    xsync = xepto_d1 + xepto_d2

    # Score each column (drug A varying, drug B fixed at each dose)
    combo_doses_a = doses_a[1:]
    for j in range(1, matrix.shape[1]):
        col_responses = matrix[1:, j]
        col_fit = fit_curve(combo_doses_a, col_responses)
        log10_ic50_col = np.log10(col_fit["ic50"] * con_scale_a) if col_fit["ic50"] > 0 else log10_ic50_a
        xepto_col = xepto_score(
            slope=col_fit["slope"],
            min_val=col_fit["min"],
            max_val=col_fit["max"],
            log10_ic50=log10_ic50_col,
            baseline_for_auc=baseline,
            integration_limit=integration_limit,
        )
        xsync += xepto_col

    # Score each row (drug B varying, drug A fixed at each dose)
    combo_doses_b = doses_b[1:]
    for i in range(1, matrix.shape[0]):
        row_responses = matrix[i, 1:]
        row_fit = fit_curve(combo_doses_b, row_responses)
        log10_ic50_row = np.log10(row_fit["ic50"] * con_scale_b) if row_fit["ic50"] > 0 else log10_ic50_b
        xepto_row = xepto_score(
            slope=row_fit["slope"],
            min_val=row_fit["min"],
            max_val=row_fit["max"],
            log10_ic50=log10_ic50_row,
            baseline_for_auc=baseline,
            integration_limit=integration_limit,
        )
        xsync += xepto_row

    return round(xsync, 2)
