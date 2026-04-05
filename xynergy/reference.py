import numpy as np
import polars as pl

import xynergy.fit as fit
from xynergy.fit import (
    _add_uncombined_drug_fitted_responses,
    _add_uncombined_drug_responses,
)
from xynergy.util import _add_id_if_no_experiment_cols, make_list_if_str_or_none
from xynergy.validate import ensure_all_cols_in_df

# References for different synergy models

# Note that it is not always enough to take the difference between the observed
# response and given reference to get the 'correct' delta. For instance, ZIP
# does a bit of 'smoothing' on the responses before taking the difference from its reference.


def add_reference(
    df,
    dose_cols,
    response_col,
    experiment_cols=None,
    method: list[str] | str = ["bliss", "hsa", "loewe", "zip"],
    log: str = "all",
):
    """Add columns containing the reference for a given method.

    Parameters
    ----------
    df: polars.DataFrame
        Usually the output from `tidy` or one of its downstream functions

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string, default "response"
        The name of the column containing responses

    experiment_cols: list[str], string, or None, default "experiment_id"
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates.

    method: list[str] or str, default ["bliss", "hsa", "loewe", "zip"]
        The method used for calculating reference.

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Input with `[method]_ref` columns appended.


    Warnings
    --------
    Subtracting the reference from the observed value is sometimes, *but not
    always* the same as calculating the synergy score. If what you want is a
    measure of deviation of the observed response from the expected response,
    prefer `add_synergy`

    Notes
    -----
    Refer to `add_synergy` for details on individual synergy/reference models,
    such as their advantages, limitations, and how they are calculated

    """

    experiment_cols = make_list_if_str_or_none(experiment_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    ensure_all_cols_in_df(df, [response_col] + dose_cols + experiment_cols)

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )

    if "bliss" in method:
        df = _add_uncombined_drug_responses(
            df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
        )
        uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
        df = _bliss(df, uncombined_resp_cols)
        df = df.drop(uncombined_resp_cols)

    if "hsa" in method:
        df = _add_uncombined_drug_responses(
            df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
        )
        uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
        df = _hsa(df, uncombined_resp_cols)
        df = df.drop(uncombined_resp_cols)

    if "loewe" in method:
        fits = fit.fit_individual_drugs(
            df, dose_cols, response_col, experiment_cols, log=log
        )
        df = _add_uncombined_drug_fitted_responses(
            df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
        )
        uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
        df = _loewe(df, dose_cols, experiment_cols, uncombined_resp_cols, fits)
        df = df.drop(uncombined_resp_cols)

    if "zip" in method:
        df = _add_uncombined_drug_fitted_responses(
            df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
        )
        uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
        df = _zip(df, uncombined_resp_cols)
        df = df.drop(uncombined_resp_cols)

    if added_dummy:
        df = df.drop("experiment_id")

    return df


def _bliss(df: pl.DataFrame, uncombined_resp_cols):
    """Calculate Bliss response.

    The Bliss synergy reference model assumes that responses, when reported as
    fraction survived, can be used like probabilities, and that the expected
    probability (and thus response) of any given combination is simply `resp_a`
    x `resp_b` at their single-drug (that is, alone, uncombined) response
    values.

    Advantages of this model include its interpretability and the fact that it
    can be calculated from any two single-drug response values without needing
    any additional information (note that these advantages are not necessarily
    exclusive to the Bliss model).

    https://doi.org/10.1111/j.1744-7348.1939.tb06990.x

    :param df: A `polars.DataFrame`

    :param uncombined_resp_cols: A list of exactly two column names. These columns
    should contain the single-drug responses in percent inhibition (see
    `add_uncombined_drug_responses` for more information)

    :return: A polars.DataFrame with a `bliss_ref` column containing the expected
    reponse under Bliss assumptions.
    """
    a = uncombined_resp_cols[0]
    b = uncombined_resp_cols[1]
    df = df.with_columns(
        bliss_ref=(pl.col(a) + pl.col(b) - pl.col(a) * pl.col(b) / 100)
    )
    return df


def _hsa(df: pl.DataFrame, uncombined_resp_cols):
    """Calculate Highest Single Agent response

    The Highest Single Agent (HSA) reference model predicts that the combined
    response is just the higher of the two's single agent response.

    Advantages of this model is that it is dead simple to calculate and reason
    with. Like Bliss, it only requires two single-agent doses to calculate the
    expected combined dose (that is, it does not require you to know the whole
    fit of the single agent). A glaring disadvantage is that it does not
    distinguish 'additivity' from 'synergy', so even the sham case (a drug mixed
    with itself) will be considered to be synergy in this model. Thus, this
    model tends to only be useful in instances where one or both agent have no
    effect by themselves.

    :param df: A `polars.DataFrame`

    :param uncombined_resp_cols: A list of exactly two column names. These columns
    should contain the single-drug responses in percent inhibition (see
    `add_uncombined_drug_responses` for more information)

    :return: A polars.DataFrame with a `hsa_ref` column containing the expected
    reponse under Highest Single Agent assumptions.
    """
    df = df.with_columns(hsa_ref=pl.max_horizontal(uncombined_resp_cols))
    return df


def _zip(df: pl.DataFrame, uncombined_resp_cols):
    """Calculate ZIP reference.

    Identical to Bliss, except it expects fitted single response cols, and
    returns a column named `zip_ref` instead of `bliss_ref`.

    :param df: A `polars.DataFrame`

    :param uncombined_resp_cols: A list of exactly two column names. These columns
    should contain the single-drug responses in percent inhibition (see
    `add_uncombined_drug_responses` for more information)

    :return: A polars.DataFrame with a `zip_ref` column containing the expected
    reponse under ZIP assumptions.
    """
    a = uncombined_resp_cols[0]
    b = uncombined_resp_cols[1]
    df = df.with_columns(zip_ref=(pl.col(a) + pl.col(b) - pl.col(a) * pl.col(b) / 100))
    return df


def _loewe(
    df: pl.DataFrame,
    dose_cols,
    experiment_cols,
    uncombined_resp_cols,
    fits: pl.DataFrame,
):
    """Calculate Loewe additivity reference

    This is a clumsy implementation of 10.3389/fphar.2018.00031

    :param df: A `polars.DataFrame`

    :param dose_cols: A list of exactly two columns names that contain numeric
    values of agent dose

    :param experiment_cols: The names of columns that should be used to
    distinguish one dose pair's response from another.

    :param uncombined_resp_cols: A list of exactly two column names. These columns
    should contain the single-drug responses in percent inhibition (see
    `add_uncombined_drug_responses` for more information)

    :param fits: A `polars.DataFrame` with one uncombined drug fit per row, per
    experimental condition. The output will be similar to that from
    `fit_individual_drugs`

    :return: A polars.DataFrame with a `loewe_ref` column containing the expected
    reponse under Loewe additivity assumptions.
    """

    df = _add_a_and_b_in_a_terms(
        df,
        fits,
        dose_cols[0],
        experiment_cols,
        uncombined_resp_cols[0],
        uncombined_resp_cols[1],
        "_tmp_1",
    )

    df = _add_a_and_b_in_a_terms(
        df,
        fits,
        dose_cols[1],
        experiment_cols,
        uncombined_resp_cols[1],
        uncombined_resp_cols[0],
        "_tmp_2",
    )

    out = df.with_columns(
        pl.mean_horizontal("_tmp_1", "_tmp_2").alias("loewe_ref")
    ).drop("_tmp_1", "_tmp_2")
    return out


def _combination_index(dose_a, dose_b, response, slope_a, min_a, max_a, ic50_a, slope_b, min_b, max_b, ic50_b):
    """Calculate the Loewe Combination Index (CI) for a single dose pair.

    CI < 1 indicates synergy, CI = 1 indicates additivity, CI > 1 indicates
    antagonism.

    Parameters
    ----------
    dose_a, dose_b : float
        Doses of drug A and drug B.
    response : float
        Observed response at (dose_a, dose_b).
    slope_a, min_a, max_a, ic50_a : float
        4PL fit parameters for drug A.
    slope_b, min_b, max_b, ic50_b : float
        4PL fit parameters for drug B.

    Returns
    -------
    float or None
        Combination index, or None if it cannot be computed.
    """
    # For single-drug points, CI is not meaningful
    if dose_a == 0 or dose_b == 0:
        return None

    # Calculate the equivalent dose of each drug that alone would give
    # the observed response, using the inverse 4PL
    def _inverse_4pl(resp, slope, min_val, max_val, ic50):
        if resp >= max_val or resp <= min_val:
            return np.inf
        ratio = (resp - min_val) / (max_val - resp)
        if ratio <= 0:
            return np.inf
        return ic50 * ratio ** (1 / slope)

    equiv_a = _inverse_4pl(response, slope_a, min_a, max_a, ic50_a)
    equiv_b = _inverse_4pl(response, slope_b, min_b, max_b, ic50_b)

    if np.isinf(equiv_a) and np.isinf(equiv_b):
        return None

    ci = 0.0
    if not np.isinf(equiv_a) and equiv_a > 0:
        ci += dose_a / equiv_a
    if not np.isinf(equiv_b) and equiv_b > 0:
        ci += dose_b / equiv_b

    return ci


def _loewe_ci(
    df: pl.DataFrame,
    dose_cols,
    experiment_cols,
    fits: pl.DataFrame,
    response_col: str,
):
    """Add Loewe Combination Index column to the DataFrame.

    Parameters
    ----------
    df : polars.DataFrame
        Input data.
    dose_cols : list[str]
        Two dose column names.
    experiment_cols : list[str]
        Experiment grouping columns.
    fits : polars.DataFrame
        Individual drug fits (from fit_individual_drugs).
    response_col : str
        Name of the response column.

    Returns
    -------
    polars.DataFrame
        Input with 'loewe_ci' column appended.
    """
    fits_a = fits.filter(pl.col("drug") == dose_cols[0])
    fits_b = fits.filter(pl.col("drug") == dose_cols[1])

    w_fits = df.join(
        fits_a.select(experiment_cols + ["slope", "min", "max", "ic50"]),
        on=experiment_cols,
        suffix="_fit_a",
    ).join(
        fits_b.select(experiment_cols + ["slope", "min", "max", "ic50"]),
        on=experiment_cols,
        suffix="_fit_b",
    )

    s = [
        dose_cols[0],
        dose_cols[1],
        response_col,
        "slope",
        "min",
        "max",
        "ic50",
        "slope_fit_b",
        "min_fit_b",
        "max_fit_b",
        "ic50_fit_b",
    ]

    out = w_fits.with_columns(
        pl.struct(s)
        .map_elements(
            lambda x: _combination_index(
                x[s[0]], x[s[1]], x[s[2]],
                x[s[3]], x[s[4]], x[s[5]], x[s[6]],
                x[s[7]], x[s[8]], x[s[9]], x[s[10]],
            ),
            return_dtype=float,
        )
        .alias("loewe_ci"),
    ).select(pl.col(df.columns), pl.col("loewe_ci"))

    return out


def _a_and_b_in_a_terms(dose_a, resp_a, resp_b, slope, min, max, ic50):
    # The provided fit parameters must be those of the fit of 'a'

    # If the response of 'b' exceeds that possible to reach by 'a', just return
    # the response of 'b'
    if resp_b > max:
        return resp_b

    # If the response of 'b' is lower than can be reached by 'a', just return
    # the response of 'a'
    if resp_b < min:
        return resp_a

    # Otherwise, translate the response of 'b' into a concentration of 'a' that
    # would produce the same response:
    b_in_a_terms = ic50 * ((resp_b - min) / (max - resp_b)) ** (1 / slope)

    # Then add that dose with the original dose of a and see what the response
    # would be at that point:
    combined_dose = dose_a + b_in_a_terms

    # Catch case where combined_dose == 0
    # TODO add test
    if combined_dose == 0.0:
        return min

    response = min + (max - min) / (1 + (ic50 / combined_dose) ** slope)
    return response


def _add_a_and_b_in_a_terms(df, fits, dose_a, experiment_cols, resp_a, resp_b, colname):
    fits = fits.filter(pl.col("drug") == dose_a)
    w_fits = df.join(fits, on=experiment_cols)
    s = [dose_a, resp_a, resp_b, "slope", "min", "max", "ic50"]
    out = w_fits.with_columns(
        pl.struct(s)
        .map_elements(
            lambda x: _a_and_b_in_a_terms(
                x[s[0]], x[s[1]], x[s[2]], x[s[3]], x[s[4]], x[s[5]], x[s[6]]
            ),
            return_dtype=float,
        )
        .alias(colname),
    ).select(pl.col(df.columns), pl.col(colname))

    return out
