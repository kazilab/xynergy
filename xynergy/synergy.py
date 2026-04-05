import polars as pl

import .fit as xfit
from .fit import (
    _add_uncombined_drug_fitted_responses,
    _add_uncombined_drug_responses,
)
from .reference import _bliss, _hsa, _loewe, _loewe_ci, _zip
from .util import _add_id_if_no_experiment_cols, make_list_if_str_or_none
from .validate import ensure_all_cols_in_df


def add_synergy(
    df,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    experiment_cols: list[str] | str | None = "experiment_id",
    method: list[str] | str = ["bliss", "hsa", "loewe", "zip"],
    include_ci: bool = False,
    log: str = "all",
):
    """Add columns containing the synergy score for a given method.

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
        The method used for calculating synergy.

    include_ci: bool, default False
        If True and "loewe" is in method, also compute the Loewe Combination
        Index (CI) and add it as a ``loewe_ci`` column. CI < 1 indicates
        synergy, CI = 1 additivity, CI > 1 antagonism.

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Input with `[method]_syn` columns appended. These columns contain the
        synergy score, where positive numbers indicate synergy and negative
        numbers indicate antagonism


    Notes
    -----
    This function currently implements four models of synergy: Bliss
    independence, Highest Single Agent (HSA), Loewe additivity, and Zero
    Interaction Potency (ZIP). A very short description of each is provided
    below.

    **Bliss:** [1]_

    The Bliss independence reference model assumes that responses, when reported
    as fraction survived, can be used like probabilities, and that the expected
    probability (and thus response) of any given combination is simply `resp_a`
    x `resp_b` at their single-drug (that is, alone, uncombined) response
    values.

    Advantages of this model include its interpretability and the fact that it
    can be calculated from any two single-drug response values without needing
    any additional information (note that these advantages are not necessarily
    exclusive to the Bliss model).

    **HSA:**

    Highest Single Agent (HSA) predicts that the combined response is the higher
    of the two's single agent response.

    Advantages of this model is that it is dead simple to calculate and reason
    with. Like Bliss, it only requires two single-agent doses to calculate the
    expected combined dose (that is, it does not require you to know the whole
    fit of the single agent). A glaring disadvantage is that it does not
    distinguish 'additivity' from 'synergy', so even the sham case (a drug mixed
    with itself) will be considered to be synergy in this model. Thus, this
    model tends to only be useful in instances where one or both agent have no
    effect by themselves.


    **Loewe additity:** [2]_ [3]_

    Loewe additivity assumes that the combined effect of two drugs will be equal
    to the response acheived by drug A at (dose A) + (the dose of drug A
    required to achieve the response of drug B). See below for a more concrete
    example.

    It's a rather popular measure of synergy, in part because it returns a
    synergy of 0 in the sham case (drug A combined with drug A).

    Loewe additivity has a disadvantage in that it is poorly defined at the edge
    cases. If drug B achieves a response beyond the maximum effect of drug A, it
    is nonsensical to talk about the dose of A required to achieve the same
    effect.

    Under this model, the predicted response for 4nM drug B and 10nM drug A is
    (for example) calculated roughly as follows:

    1. Fit both drugs with a four parameter logistic fit

    2. Determine the response of drug B at 4nM (let's say it's 20% inhibition)

    3. Using the fit of drug A, determine what dose gives 20% inhibition (to do
       this, we use the inverse four-parameter logistic function). Let's say
       drug A achieved a 20% inhibition at 50nM.

    4. Add the original dose of drug A (10nM) to the drug B response in terms of
       drug A dose (50nM) to get the final dose (60nM)

    5. Find the response for drug A at 60nM

    6. Repeat this again, but convert drug A to drug B terms instead

    7. Take the average of the two responses



    **ZIP:** [4]_

    Zero Interaction Potency is somewhat different from the traditional synergy
    methods, in that it does not simply subtract the reference from the observed
    values. Rather, it calculates a synergy score by subtracting a reference
    model from a ZIP score, both of which will be described below. Due to the
    fact that it fits observed values before comparing them to a reference, it
    has a 'smoothing' effect.

    The reference model for ZIP is very similar to Bliss, except it uses fitted
    (to a four parameter logisitic fit) values for the single agent dose-responses
    rather than the raw observed values. This has a tendency to smooth
    irregularities.

    The ZIP score is calculated by fitting each row and column with a
    three-parameter logistic fit, with the fourth parameter - the minimum effect
    - fixed to be that row or columns '0 drug' condition (Put another way, if we
    took a column that had increasing 'Drug A', the minimum effect would be
    where 'Drug A' was 0, meaning the minimum effect would be whatever response
    'Drug B' had at that particular column's concentration). Each value of the
    matrix is now described by two equations - the row fit and the column fit.
    We can get a ZIP score for every point on the matrix by taking the average
    of the row fit and column fit at each point.


    References
    ----------
    .. [1] https://doi.org/10.1111/j.1744-7348.1939.tb06990.x

    .. [2] Loewe, S. The problem of synergism and antagonism of combined drugs. Arzneimittelforschung. 1953; 3:285-290

    .. [3] 10.3389/fphar.2018.00031

    .. [4] http://dx.doi.org/10.1016/j.csbj.2015.09.001

    """
    experiment_cols = make_list_if_str_or_none(experiment_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)
    method = make_list_if_str_or_none(method)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    ensure_all_cols_in_df(df, [response_col] + dose_cols + experiment_cols)

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )

    # These functions will utilize [method]_ref columns if they see them.
    # Otherwise they will generate them and destroy them at the end

    if "bliss" in method:
        has_ref = "bliss_ref" in df.columns

        if has_ref:
            df = df.with_columns(bliss_syn=pl.col(response_col) - pl.col("bliss_ref"))
        else:
            df = _add_uncombined_drug_responses(
                df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
            )
            uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
            df = _bliss(df, uncombined_resp_cols)
            df = df.drop(uncombined_resp_cols)
            df = df.with_columns(bliss_syn=pl.col(response_col) - pl.col("bliss_ref"))
            df = df.drop("bliss_ref")

    if "hsa" in method:
        has_ref = "hsa_ref" in df.columns

        if has_ref:
            df = df.with_columns(hsa_syn=pl.col(response_col) - pl.col("hsa_ref"))

        else:
            df = _add_uncombined_drug_responses(
                df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
            )
            uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
            df = _hsa(df, uncombined_resp_cols)
            df = df.drop(uncombined_resp_cols)
            df = df.with_columns(hsa_syn=pl.col(response_col) - pl.col("hsa_ref"))
            df = df.drop("hsa_ref")

    if "loewe" in method:
        has_ref = "loewe_ref" in df.columns

        if has_ref:
            df = df.with_columns(loewe_syn=pl.col(response_col) - pl.col("loewe_ref"))
            if include_ci:
                fits = xfit.fit_individual_drugs(
                    df, dose_cols, response_col, experiment_cols, log=log
                )
                df = _loewe_ci(df, dose_cols, experiment_cols, fits, response_col)

        else:
            fits = xfit.fit_individual_drugs(
                df, dose_cols, response_col, experiment_cols, log=log
            )
            df = _add_uncombined_drug_fitted_responses(
                df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
            )
            uncombined_resp_cols = [x + "_tmp" for x in dose_cols]
            df = _loewe(df, dose_cols, experiment_cols, uncombined_resp_cols, fits)
            df = df.drop(uncombined_resp_cols)
            df = df.with_columns(loewe_syn=pl.col(response_col) - pl.col("loewe_ref"))
            df = df.drop("loewe_ref")
            if include_ci:
                df = _loewe_ci(df, dose_cols, experiment_cols, fits, response_col)

    if "zip" in method:
        has_ref = "zip_ref" in df.columns

        df = _add_uncombined_drug_fitted_responses(
            df, dose_cols, response_col, experiment_cols, suffix="_tmp", log=log
        )
        uncombined_resp_cols = [x + "_tmp" for x in dose_cols]

        # TODO probably worth making _zip_score handle grouping
        grouped = df.group_by(experiment_cols)
        scores = []
        for _, group in grouped:
            score = _zip_score(group, dose_cols, uncombined_resp_cols, response_col)
            scores.append(score)

        scores = pl.concat(scores)

        # Sorting is preferred to a join since in the (usually synthetic) case
        # of a replicate (that is, same experiment, same dose combo) with the
        # exact same response the join will be ambiguous and thus you'll end up
        # with more rows than expected.
        sort_on = dose_cols + experiment_cols + [response_col]
        df = df.sort(sort_on).with_columns(scores.sort(sort_on)["zip_score"])

        if has_ref:
            df = df.with_columns(zip_syn=pl.col("zip_score") - pl.col("zip_ref"))
        else:
            df = _zip(df, uncombined_resp_cols)
            df = df.with_columns(zip_syn=pl.col("zip_score") - pl.col("zip_ref"))
            df = df.drop("zip_ref")

        df = df.drop(uncombined_resp_cols + ["zip_score"])

    if added_dummy:
        df = df.drop("experiment_id")

    return df


def _zip_score(
    df: pl.DataFrame,
    dose_cols: list[str],
    uncombined_resp_cols: list[str],
    resp_col: str,
):
    # The Zero Interaction Potency (ZIP) model (doi: 10.1016/j.csbj.2015.09.001)
    # NOTE: This is NOT the zip reference, but rather the ZIP *score*
    #
    # This function will either need to be passed only the group of interest or
    # be made aware of the grouping cols.

    # Additionally, it also needs to have some information about the combination
    # response data (as opposed to Bliss or HSA, which only needs to know
    # single-drug information)
    df = _calc_y_add_a_to_b(df, dose_cols, uncombined_resp_cols, resp_col, 1, 0)
    df = _calc_y_add_a_to_b(df, dose_cols, uncombined_resp_cols, resp_col, 0, 1)
    df = df.with_columns(zip_score=pl.mean_horizontal("y12", "y21"))

    return df


def _get_min(df: pl.DataFrame, cols: list[str]) -> float:
    return df.select(cols).min().min_horizontal()[0]


def _calc_y_add_a_to_b(
    df: pl.DataFrame,
    drug_cols,
    uncombined_resp_cols,
    resp_col,
    a_index: int,
    b_index: int,
):
    grouped_by_b = df.group_by(drug_cols[b_index])
    tmp = []
    for _, b in grouped_by_b:
        min_resp = _get_min(b, uncombined_resp_cols[b_index])

        fit = xfit.fit_curve(
            b[drug_cols[a_index]],
            b[resp_col],
            n_param=3,
            min_inhibition=min_resp,
        )

        b = b.with_columns(
            _calc_y_add_a_to_b_polars(
                fit, min_resp, drug_cols, uncombined_resp_cols, a_index, b_index
            )
        )
        tmp.append(b)

    df = pl.concat(tmp)
    return df


def _calc_y_add_a_to_b_polars(fit, min, drugs, singles, a, b):
    # Sometimes the fit returned has a VERY (<1e-50) small IC50, which
    # causes....issues, downstream.

    # Very small IC50s will converge to fit["max"], so we can get around that
    # issue early by just setting the result to fit["max"] if the IC50 is small

    # An important caveat is when the drug_cols[a_index] == 0, in which case the
    # result will be min_resp, regardless of IC50 value

    return (
        pl.when(pl.col(drugs[a]) == 0.0)
        .then(min)
        .when(fit["ic50"] < 1e-32)
        .then(fit["max"])
        .otherwise(
            (
                pl.col(singles[b])
                + fit["max"] * ((pl.col(drugs[a]) / fit["ic50"]) ** fit["slope"])
            )
            / (1 + (pl.col(drugs[a]) / fit["ic50"]) ** fit["slope"])
        )
        .alias("y" + str(a + 1) + str(b + 1))
    )
