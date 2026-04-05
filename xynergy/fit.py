import warnings

import lmfit as lm
import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.optimize as opt

from .util import _add_id_if_no_experiment_cols, make_list_if_str_or_none
from .validate import ensure_all_cols_in_df


def add_uncombined_drug_responses(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    experiment_cols: list[str] | str | None = "experiment_id",
    suffix="_resp",
    fit: bool = True,
    log: str = "all",
):
    """Add columns that contain responses if just drug A (or drug B) alone was
    added at their respective `dose_cols` concentration.

    Parameters
    ----------
    df: polars.DataFrame
        Usually the output from `tidy`

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string, default "response"
        The name of the column containing responses

    experiment_cols: list[str], string, or None, default "experiment_id"
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates.

    suffix: string, default "_resp"
        The suffix to give the uncombined drug response columns. By default,
        resultant columns will be of the form `dose_cols` + `suffix`

    fit: bool, default True
        Should the returned values should be fitted values predicted at that
        concentration, or just the raw, observed values at that point?

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Similar to input, with two additional columns `dose_a_resp` and
        `dose_b_resp` (name depends on the name of `dose_cols` as well as
        `suffix`), that correspond with expected response of drug A (or B) at
        that given concentration.

    """

    experiment_cols = make_list_if_str_or_none(experiment_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    ensure_all_cols_in_df(df, [response_col] + dose_cols + experiment_cols)

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )

    if not fit:
        out = _add_uncombined_drug_responses(
            df, dose_cols, response_col, experiment_cols, suffix, log
        )
    else:
        out = _add_uncombined_drug_fitted_responses(
            df, dose_cols, response_col, experiment_cols, suffix, log
        )

    if added_dummy:
        out = out.drop("experiment_id")

    return out


def _add_uncombined_drug_responses(
    df: pl.DataFrame,
    dose_cols: list[str],
    response_col: str,
    experiment_cols: list[str],
    suffix: str,
    log: str,
):
    """Add two additional columns to the input `df` that contain the single
    agent responses at that concentration, without the addition of the other
    drug. If this dataset were a matrix of dose combinations, you could think of
    it like the first row and column of of this matrix.

    df: polars.DataFrame
        Usually the output from `tidy`

    dose_cols: list
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string
        The name of the column containing responses

    experiment_cols: list[str]
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates.

    suffix: string
        The suffix to give the single response columns. By default, resultant
        columns will be of the form `dose_cols` + `suffix`

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    """

    min_doses = _get_min_concentrations(df, dose_cols, experiment_cols, log)

    with_a = _get_single_uncombined_drug_responses(
        df, min_doses, dose_cols[1], dose_cols[0], response_col, experiment_cols, suffix
    )
    with_b = _get_single_uncombined_drug_responses(
        df, min_doses, dose_cols[0], dose_cols[1], response_col, experiment_cols, suffix
    )
    df = df.join(with_a, on=experiment_cols + [dose_cols[0]], how="left")
    df = df.join(with_b, on=experiment_cols + [dose_cols[1]], how="left")
    return df


def _add_uncombined_drug_fitted_responses(
    df: pl.DataFrame,
    dose_cols: list[str],
    response_col: str,
    experiment_cols: list[str],
    suffix: str,
    log: str,
):
    fits = fit_individual_drugs(df, dose_cols, response_col, experiment_cols, log)

    tidy_df = df.unpivot(
        on=dose_cols, index=experiment_cols, variable_name="drug", value_name="dose"
    ).unique()

    with_params = tidy_df.join(fits, on=["drug"] + experiment_cols)
    with_response = with_params.with_columns(
        _l4("dose", "slope", "min", "max", "ic50").alias("tmp_resp")
    )

    # You'd think a pivot + join would be the move here. Unfortunately,
    # particularly in a synthetic situation where 'dose_a' and 'dose_b' are the
    # same, Weird Stuff happens, which could cause unexpected behavior down the
    # line. So we'll do it this way:
    a_resp = (
        with_response.filter(pl.col("drug") == dose_cols[0])
        .select("dose", "tmp_resp", pl.col(experiment_cols))
        .rename({"tmp_resp": dose_cols[0] + suffix})
    )
    b_resp = (
        with_response.filter(pl.col("drug") == dose_cols[1])
        .select("dose", "tmp_resp", pl.col(experiment_cols))
        .rename({"tmp_resp": dose_cols[1] + suffix})
    )
    return df.join(
        a_resp,
        left_on=[dose_cols[0]] + experiment_cols,
        right_on=["dose"] + experiment_cols,
    ).join(
        b_resp,
        left_on=[dose_cols[1]] + experiment_cols,
        right_on=["dose"] + experiment_cols,
    )


def fit_individual_drugs(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    experiment_cols: str | list[str] | None = ["experiment_id"],
    log: str = "all",
):
    """Fit individual drugs per experimental group from a tidy `polars.DataFrame`.
    Preferably will fit Drug A where `dose_b = 0` (and vice-versa), but if no
    conditions exist where `dose_b = 0`, chooses the lowest concentration and
    warns.

    Parameters
    ----------

    df: polars.DataFrame
        Usually the output from `tidy`

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string, default "response"
        The name of the column containing responses

    experiment_cols: list[str], string, or None, default "experiment_id"
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates.

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Number of rows equal to the number of experimental groups in `df`, with
        the following columns:

        - Columns included in `experiment_cols`
        - Parameters of each fit: `slope`, `max`, `min`, `ic50`
        - `drug`
            Which drug the fit refers to. Will use them names provided in `dose_cols`.

    """
    dose_cols = make_list_if_str_or_none(dose_cols)
    experiment_cols = make_list_if_str_or_none(experiment_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    ensure_all_cols_in_df(df, [response_col] + dose_cols + experiment_cols)

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )

    min_drug_concs = _get_min_concentrations(df, dose_cols, experiment_cols, log)
    out = pl.concat(
        [
            _fit_single_drug(
                df,
                min_drug_concs,
                dose_cols[0],
                dose_cols[1],
                response_col,
                experiment_cols,
            ),
            _fit_single_drug(
                df,
                min_drug_concs,
                dose_cols[1],
                dose_cols[0],
                response_col,
                experiment_cols,
            ),
        ]
    )

    if added_dummy:
        out = out.drop("experiment_id")

    return out


def _get_min_concentrations(
    df: pl.DataFrame, dose_cols: list[str], experiment_cols: list[str], log: str
):
    """Get the minimum concentration of each drug in each experiment

    :param dose_cols: A list of exactly two columns names that contain numeric
    values of agent dose

    :param experiment_cols: List of names of columns that uniquely
    describe/differentiate experiments

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    :return: A `polars.DataFrame` where each row is an experiment, containing
    the grouping experiment_cols and the dose_cols, with the values of the
    lowest concentration of each drug in each experiment (hopefully zero but not
    always!)

    """
    min_drug_concs = df.group_by(experiment_cols).agg(pl.col(dose_cols).min())

    if log in ["all", "warn"]:
        non_zero_mins = min_drug_concs.filter(
            (pl.col(dose_cols[0]) != 0) | (pl.col(dose_cols[1]) != 0)
        )
        if len(non_zero_mins) > 0:
            non_zero_mins_experiments = non_zero_mins[experiment_cols].unique()
            warnings.warn(
                f"The following experiments had non-zero minimum concentrations:\n{non_zero_mins_experiments}"
            )
    return min_drug_concs


def _get_single_uncombined_drug_responses(
    df: pl.DataFrame,
    min_doses: pl.DataFrame,
    min_drug_col: str,
    other_drug_col: str,
    response_col: str,
    experiment_cols: list[str],
    suffix: str,
):
    """Get responses of one drug at the other drug's minimum (preferably 0). If
    more than one value exist for a given dose in a given experimental group,
    the mean is taken.

    :param df: A `polars.DataFrame` containing concentrations and responses

    :param min_doses: The output of _get_min_concentrations

    :param min_drug_col: The column containing the doses of the drug whose
    response we DON'T care about

    :param other_drug_col: The column containing doses of the drug whose
    response we are interested in

    :param response_col: The name of the column containing responses

    :param experiment_cols: The names of columns that should be used to
    distinguish one dose pair's response from another.

    :param suffix: The suffix to give the single response columns.

    TODO Return value
    """
    just_min_doses = df.join(min_doses, on=experiment_cols + [min_drug_col], how="semi")

    return just_min_doses.group_by(experiment_cols + [other_drug_col]).agg(
        pl.col(response_col).mean().alias(other_drug_col + suffix)
    )


def _fit_single_drug(
    df: pl.DataFrame,
    min_drug_concs: pl.DataFrame,
    fitting_drug: str,
    other_drug: str,
    response_col: str,
    experiment_cols: list[str],
):
    subset = df.join(min_drug_concs, on=experiment_cols + [other_drug], how="semi")
    fits = []
    for key, group in subset.group_by(experiment_cols):
        params = fit_curve(group[fitting_drug], group[response_col])
        row = {"drug": fitting_drug, **params}
        if isinstance(key, tuple):
            row.update(dict(zip(experiment_cols, key)))
        else:
            row[experiment_cols[0]] = key
        fits.append(row)
    return pl.DataFrame(fits)


def fit_curve(
    doses: npt.NDArray[np.float64],
    inhibition: npt.NDArray[np.float64],
    n_param: int = 4,
    fit_method: str = "lm",
    min_inhibition: float | None = None,
    fit=None,
):
    if n_param not in {3, 4}:
        raise ValueError("n_param must be 3 or 4")

    if (n_param == 3) and (min_inhibition is None):
        raise ValueError("min_inhibition must be supplied for a 3 parameter fit")

    if fit_method not in {"curve_fit", "lm"}:
        raise ValueError("fit_method must be either 'curve_fit' or 'lm'")

    # deal with 0 doses getting their log taken
    doses = make_list_if_str_or_none(doses)
    log10_doses = [np.float64(-np.inf) if x == 0 else np.log10(x) for x in doses]
    log10_doses = np.nan_to_num(log10_doses, neginf=-100)
    if fit_method == "curve_fit":
        fit = _fit_curve_curve_fit(log10_doses, inhibition, n_param, min_inhibition)
        fit["ic50"] = 10 ** fit.pop("log10_ic50")
        return fit
    else:
        init_fit = fit
        if init_fit is None:
            init_fit = _fit_curve_curve_fit(
                log10_doses, inhibition, n_param, min_inhibition
            )
        fit = _fit_curve_lm(log10_doses, inhibition, n_param, min_inhibition, init_fit)
        fit["ic50"] = 10 ** fit.pop("log10_ic50")
        return fit


def _fit_curve_curve_fit(
    log10_doses,
    inhibition,
    n_param,
    min_inhibition,
):
    kwargs = {
        "log10_doses": log10_doses,
        "inhibition": inhibition,
        "n_param": n_param,
        "fit_method": "curve_fit",
        "fit": None,
        "min_inhibition": min_inhibition,
    }
    init_guesses = _make_init_guesses(**kwargs)
    lower, upper = _make_bounds(**kwargs)

    keys = ["slope", "min", "max", "log10_ic50"]
    if n_param == 3:
        curve_function = _make_ll3(min_inhibition)
        keys.remove("min")
    else:
        curve_function = _ll4

    init_guesses = [init_guesses[x] for x in keys]
    bounds = ([lower[x] for x in keys], [upper[x] for x in keys])

    vals = opt.curve_fit(
        f=curve_function,
        xdata=log10_doses,
        ydata=inhibition,
        p0=init_guesses,
        bounds=bounds,
        maxfev=100000,
    )[0]

    fit_values = dict(zip(keys, vals))

    return _sanitize_values(fit_values, log10_doses, inhibition)


def _fit_curve_lm(log10_doses, inhibition, n_param, min_inhibition, fit):
    if fit is None:
        fit = _fit_curve_curve_fit(log10_doses, inhibition, n_param, min_inhibition)
    kwargs = {
        "log10_doses": log10_doses,
        "inhibition": inhibition,
        "n_param": n_param,
        "fit_method": "lm",
        "fit": fit,
        "min_inhibition": min_inhibition,
    }
    init_guesses = _make_init_guesses(**kwargs)
    min_bounds, max_bounds = _make_bounds(**kwargs)
    if n_param == 3:
        curve_function = _make_ll3(min_inhibition)
    if n_param == 4:
        curve_function = _ll4
    model = lm.Model(curve_function)
    params = lm.Parameters()

    params.add(**_make_param("slope", init_guesses, min_bounds, max_bounds))
    params.add(**_make_param("max", init_guesses, min_bounds, max_bounds))
    params.add(**_make_param("log10_ic50", init_guesses, min_bounds, max_bounds))

    if n_param == 4:
        params.add(**_make_param("min", init_guesses, min_bounds, max_bounds))
    fit = model.fit(data=inhibition, params=params, log10_doses=log10_doses)

    # See if fitting with log10_ic50 as a median dose produces better fit. If it
    # does, use that fit.
    params["log10_ic50"].value = np.median(log10_doses)
    fit_median_ic50 = model.fit(data=inhibition, params=params, log10_doses=log10_doses)
    fit_std_residual = np.std(fit.residual)
    fit_median_ic50_std_residual = np.std(fit_median_ic50.residual)
    fit = fit if fit_std_residual < fit_median_ic50_std_residual else fit_median_ic50

    if fit.params["slope"].value <= 0.2:
        params["log10_ic50"].value = init_guesses["log10_ic50"]
        params["slope"].min = 0.1
        params["slope"].max = 2.5
        fit = model.fit(data=inhibition, params=params, log10_doses=log10_doses)
    # fit = _sanitize_ic50_after_lm(fit)
    # Must return fit parameters rather than fit itself due to
    # https://github.com/pola-rs/polars/issues/10189
    return fit.params.valuesdict()


def _make_init_guesses(
    log10_doses, inhibition, n_param, fit_method, fit, min_inhibition
):
    if fit_method == "curve_fit":
        guesses = dict({"slope": 1, "max": 100, "log10_ic50": np.median(log10_doses)})

        if n_param == 4:
            guesses["min"] = 0

        return guesses
    else:
        if fit is None:
            fit = _fit_curve_curve_fit(log10_doses, inhibition, n_param, min_inhibition)
        return fit


def _make_bounds(
    log10_doses,
    inhibition,
    n_param,
    fit_method,
    fit,
    min_inhibition,
):
    # Some bounds are the same regardless of fit type
    bounds_lower = {"slope": 0, "log10_ic50": np.min(log10_doses)}
    bounds_upper = {"slope": 4, "log10_ic50": np.max(log10_doses), "max": 100}

    # Min won't vary in 3 parameter fit - it's fixed. Hence, no bounds
    if n_param == 4:
        bounds_lower["min"] = 0

    if fit_method == "curve_fit":
        bounds_lower["max"] = min(np.nanmax(inhibition), 95)
        if n_param == 4:
            bounds_upper["min"] = max(np.nanmin(inhibition), 5)

    if fit_method == "lm":
        if fit is None:
            fit = _fit_curve_curve_fit(log10_doses, inhibition, n_param, min_inhibition)
        bounds_lower["max"] = min(90, fit["max"] * 0.9)
        if n_param == 4:
            bounds_upper["min"] = max(10, fit["min"] * 1.1)

    return (bounds_lower, bounds_upper)


# Value sanitization -----------------------------
def _sanitize_values(fit_values, log10_doses, inhibition):
    # Temp (?) fix - inhibition comes in as polars.Series
    inhibition = np.array(inhibition)

    # Sanitize minimum ---------------------------
    min_inhibition = np.nanmin(inhibition)
    min_inhibition = _coerce_between_bounds(0, min_inhibition, 99)
    if "min" in fit_values:
        min_fit = _coerce_between_bounds(0, fit_values["min"], 99)
        fit_values["min"] = min(min_inhibition, min_fit)
    if "min" not in fit_values:
        fit_values["min"] = min_inhibition

    # Sanitize maximum ---------------------------
    max_inhibition = np.nanmax(inhibition)
    max_inhibition = _coerce_between_bounds(1, max_inhibition, 100)
    max_fit = _coerce_between_bounds(1, fit_values["max"], 100)
    max_temp = max(max_inhibition, max_fit)

    running_average = pl.Series(inhibition).rolling_mean(3).to_numpy()
    max_run = max_temp
    # Detect if the last rolling average is lower than the prior peak, indicating
    # the curve has started to decline — cap max_run at that prior peak
    if np.nanmax(running_average[:-1]) > running_average[-1]:
        valid_mask = ~np.isnan(running_average)
        boolean_array = (running_average > running_average[-1]) & valid_mask
        max_run = np.nanmax(inhibition[boolean_array])
    if np.nanmax(inhibition) > max_run:
        valid_mask = ~np.isnan(inhibition)
        boolean_array = (inhibition > max_run) & valid_mask
        max_run = np.nanmean(inhibition[boolean_array]) + 1

    max_run = _coerce_between_bounds(1, max_run, 100)

    fit_values["max"] = np.nanmax([max_temp, max_run])

    if fit_values["min"] == fit_values["max"]:
        fit_values["max"] = fit_values["max"] + 0.001

    # Sanitize log10ic50_curve_fit ---------------
    log10_ic50_fit = fit_values["log10_ic50"]
    if fit_values["min"] >= fit_values["max"]:
        log10_ic50_fit = np.nanmax(log10_doses)
    elif log10_ic50_fit > np.nanmax(log10_doses):
        log10_ic50_fit = np.nanmax(log10_doses)
    elif log10_ic50_fit < np.nanmin(log10_doses):
        log10_ic50_fit = np.nanmin(log10_doses)

    if np.nanmax(inhibition) < 0:
        log10_ic50_fit = np.nanmax(log10_doses)
    elif np.nanmin(inhibition) > 100:
        log10_ic50_fit = np.nanmin(log10_doses)

    if np.nanmean(inhibition[:-2]) < 5:
        log10_ic50_fit = np.nanmax(log10_doses)

    fit_values["log10_ic50"] = log10_ic50_fit

    return fit_values


def _sanitize_ic50_after_lm(fit):
    negative_slope: bool = fit.params["slope"].value < 0
    low_max_observed_effect = np.nanmax(fit.data) < 10

    if negative_slope or low_max_observed_effect:
        fit.params["log10_ic50"].value = fit.userkws["log10_doses"].max()

    return fit


# Curve equations --------------------------------
def ll4(
    doses: npt.NDArray[np.float64] | float,
    slope: float,
    min: float,
    max: float,
    ic50: float,
):
    doses = make_list_if_str_or_none(doses)
    log10_doses = [np.float64(-np.inf) if x == 0 else np.log10(x) for x in doses]
    log10_ic50 = np.log10(ic50)
    return np.array([_single_ll4(x, slope, min, max, log10_ic50) for x in log10_doses])


# Single dose calculation
def _single_ll4(
    log10_dose: float, slope: float, min: float, max: float, log10_ic50: float
) -> float:
    if log10_dose == -np.inf:
        return min
    # 10.0 must be float or negative integer powers will cause python to freak
    # out
    return min + (max - min) / (1 + 10.0 ** (slope * (log10_ic50 - log10_dose)))


# Internal version of the function used largely for scipy optimization
def _ll4(log10_doses: float, slope: float, min: float, max: float, log10_ic50: float):
    return min + (max - min) / (1 + 10.0 ** (slope * (log10_ic50 - log10_doses)))


def _l4(dose: str, slope: str, min: str, max: str, ic50: str) -> pl.Expr:
    # A polars-compatible version of _ll4

    # Note that this is the same as _ll4, just using non-logged values.
    # https://web.archive.org/web/20241207014823/https://www.quantics.co.uk/blog/what-is-the-4pl-formula/

    # Note that ic50 and dose are the reciprocal of what is shown in the link
    # above because we are measuring % inhibition, not % survival
    return pl.col(min) + (pl.col(max) - pl.col(min)) / (
        1 + (pl.col(ic50) / pl.col(dose)) ** pl.col(slope)
    )


def inverse_ll4(response: float, slope: float, min: float, max: float, ic50: float) -> float:
    """Inverse of the 4-parameter logistic function.

    Given a response value, returns the dose that would produce it.

    Parameters
    ----------
    response : float
        The response value to invert.
    slope : float
        Hill slope.
    min : float
        Minimum response (lower asymptote).
    max : float
        Maximum response (upper asymptote).
    ic50 : float
        IC50 (untransformed, not log10).

    Returns
    -------
    float
        The dose producing the given response, or np.inf if outside bounds.
    """
    if response >= max or response <= min:
        return np.inf
    ratio = (response - min) / (max - response)
    if ratio <= 0:
        return np.inf
    return ic50 * ratio ** (1 / slope)


def _make_ll3(min: float):
    def _ll3(log10_doses, slope, max, log10_ic50):
        y = min + (max - min) / (1 + 10 ** (slope * (log10_ic50 - log10_doses)))
        return y

    return _ll3


# Utilities --------------------------------------
def _coerce_between_bounds(min_, x, max_):
    x = np.nanmax([min_, x])
    x = np.nanmin([max_, x])
    return x


def _make_param(name, init_guesses, min_bounds, max_bounds):
    return {
        "name": name,
        "value": init_guesses[name],
        "min": min_bounds[name],
        "max": max_bounds[name],
    }
