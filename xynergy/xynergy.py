from xynergy.synergy import add_synergy
from xynergy.tidy import tidy
from xynergy.impute import pre_impute, post_impute
from xynergy.factor import matrix_factorize
from xynergy.util import make_list_if_str_or_none
import polars as pl


def xynergy(
    df: pl.DataFrame,
    dose_cols: list[str],
    response_col: str,
    experiment_cols: list[str] | str | None,
    response_is_percent: bool,
    complete_response_is_0: bool,
    pre_impute_method: str = "RBFSurface",
    pre_impute_target: str = "response",
    pre_impute_reference_for_target: str = "bliss",
    pre_impute_clip_response_bounds: tuple[float, float] | None = (0.0, 100.0),
    factorization_method: list[str] | str = ["NMF", "SVD", "RPCA"],
    synergy_method: list[str] | str = ["bliss", "hsa", "loewe", "zip"],
    use_single_drug_response_data: bool = True,
    post_impute_tuning: str = "Predefined",
    log: str = "all",
) -> pl.DataFrame:
    """Impute missing values and calculate synergy from a (or several) dose-response matrices.

    Parameters
    ----------
    df: polars.DataFrame
        Contains, minimally, one response and two agent doses per row.

    dose_cols: list
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string
        The name of the column that contains response data. Can be multiple
        columns, though they will be unpivoted to a single column.

    experiment_cols: list[str] or string, optional
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates. One common application of this
        might be to provide columns containing the names of drugs used.

    response_is_percent: bool
        Is the response a percentage (ranges from 0-100) or is it a
        probability/ratio (ranges from 0-1)?

    complete_response_is_0: bool
        Is the response reported as (for instance) survival, where a complete
        response would be 0? Or is it something like (again, for instance)
        killing, where a complete response would be 1 (in the case of
        `response_is_percent = False`) or 100 (in the case of `reponse_is_percent =
        True`)

    pre_impute_method: string, default "RBFSurface"
        - "RBFSurface" (recommended): RBF interpolation of Bliss residuals in
          log-dose space.  Exploits pharmacological smoothness of dose-response
          surfaces.  Very fast and generally the most accurate method.
        - "GaussianProcessSurface": Gaussian-process regression in log-dose
          space. Slower than RBFSurface, but a strong non-parametric surface
          benchmark.
        - "MatrixCompletion": Iterative rank-truncated SVD that exploits the
          low-rank structure of dose-response matrices.
        - "XGBR" (slowest, most accurate of the tabular methods),
        - "RandomForest" (roughly medium speed and accuracy),
        - "LassoCV" (fast, poor accuracy. Not recommended.),
        - Otherwise, default sklearn IterativeImputer (fastest, sometimes better
          accuracy than LassoCV)

    pre_impute_target: string, default "response"
        Passed through to `pre_impute`. Options are "response",
        "combo_effect", and "ensemble".

    pre_impute_reference_for_target: string, default "bliss"
        Passed through to `pre_impute` when
        `pre_impute_target in ["combo_effect", "ensemble"]`. Options currently
        include "bliss" and "hsa".

    pre_impute_clip_response_bounds: tuple[float, float] | None, default (0.0, 100.0)
        Bounds applied by `pre_impute` to reconstructed responses. Set to `None`
        to disable clipping.

    factorization_method: list[str] or str, default ["NMF", "SVD", "RPCA"]
        The method(s) used for matrix factorization. Options include NMF, SVD,
        PMF, and RPCA

    synergy_method: list[str] or str, default ["bliss", "hsa", "loewe", "zip"]
        The method used for calculating synergy.

    use_single_drug_response_data: bool, default True
        Some methods - like RandomForest - perform better when the dataset
        contains columns with the responses of, say, 'drug A' at 'dose_a' (no
        combination). If this parameter is `True`, automatically calculate this
        value and include it as data to be used for imputation. You might set
        this as `False` if you want to include your own data, for instance - in
        which case you would add the name of those columns to
        `additional_imputation_cols`. In general, this step can only help and is
        relatively quick.

    post_impute_tuning: string, default "Predefined"
        Strategy for tuning XGBoost hyperparameters during post-imputation.

        - "Predefined": Use fixed hyperparameters (very fast).
        - "RandomizedSearchCV": Random subset search (moderate speed).
        - "GridSearchCV": Exhaustive grid search (slowest).

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Data will be 'tidy', with each row containing a single response.
        Additionally, the following columns will be appended:

        - `resp_imputed` column (plus `[dose_cols]_resp` if
          `use_single_drug_response_data = True`). Contains response imputed by
          `pre_impute_method`
        - `response_[factorization_method]`. Contain the supplied response
          values approximated by indicated method
        - `[synergy_method]_syn`. Contain the synergy score, where positive
          numbers indicate synergy and negative numbers indicate antagonism

        Additionally, the missing `response_col` values are imputed. Will be
        modified (if necessary) to be '% inhibition style' (0 = no inhibition,
        100 = complete inhibition).

    Notes
    -----
    This function is essentially several functions in a trenchcoat: This
    function runs `tidy`, `pre_impute`, `matrix_factorize`, `post_impute`, and
    `add_synergy` in sequence. These functions can be called individually if you
    want a bit more control or to do something between each step

    For additional information, see the documentation of the individual
    functions.

    """
    experiment_cols = make_list_if_str_or_none(experiment_cols)

    tidied = tidy(
        df=df,
        dose_cols=dose_cols,
        response_col=response_col,
        experiment_cols=experiment_cols,
        response_is_percent=response_is_percent,
        complete_response_is_0=complete_response_is_0,
        log=log,
    )

    # `tidy` always returns an `experiment_id` column. Avoid duplicate names if
    # callers already supplied "experiment_id" in experiment_cols.
    experiment_cols = [x for x in experiment_cols if x != "experiment_id"] + [
        "experiment_id"
    ]

    pre_imputed = pre_impute(
        df=tidied,
        experiment_cols=experiment_cols,
        method=pre_impute_method,
        target=pre_impute_target,
        reference_for_target=pre_impute_reference_for_target,
        clip_response_bounds=pre_impute_clip_response_bounds,
        use_single_drug_response_data=use_single_drug_response_data,
        log=log,
    )

    factored = matrix_factorize(
        df=pre_imputed,
        experiment_cols=experiment_cols,
        method=factorization_method,
        log=log,
    )

    imputed = post_impute(df=factored, experiment_cols=experiment_cols, post_impute_tuning=post_impute_tuning, log=log)

    with_synergy = add_synergy(
        imputed, experiment_cols=experiment_cols, method=synergy_method, log=log
    )
    # Many args not exposed since there's no way for them to be preserved
    # through the analytical process, nor for them to be 'injected' within this
    # function
    return with_synergy
