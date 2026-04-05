import numpy as np
import numpy.linalg as la
import polars as pl
import scipy.optimize as opt
import sklearn.decomposition as decomp
from xynergy.lnmf import _lnmf

from xynergy.util import (
    _add_id_if_no_experiment_cols,
    make_list_if_str_or_none,
    venter,
)
from xynergy.validate import *

try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None

try:
    import optuna
except ModuleNotFoundError:
    optuna = None


def matrix_factorize(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "resp_imputed",
    experiment_cols: str | list[str] | None = "experiment_id",
    method: list[str] | str = ["NMF", "SVD", "PMF", "RPCA"],
    og_response_col: str | None = "response",
    log: str = "all",
):
    """Estimate dose-response data via matrix factorization.

    Parameters
    ----------
    df: polars.DataFrame
        Usually the output from `tidy` or one of its downstream functions

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string, default "resp_imputed"
        The name of the column containing responses. Should not contain missing
        values. In a typical workflow, this will be the pre-imputed responses

    experiment_cols: list[str], string, or None, default "experiment_id"
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates.

    method: list[str] or str, default ["NMF", "SVD", "PMF", "RPCA"]
        The method(s) used for matrix factorization

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Input with `[response_col]_[method]` columns appended. These columns
        contain the supplied response values approximated by the respective
        method(s)

    Notes
    -----
    If there are multiple responses per dose-pair per experiment, (that is,
    replicates), this function will silently take the mean.

    """
    experiment_cols = make_list_if_str_or_none(experiment_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)
    response_col = make_list_if_str_or_none(response_col)
    method = make_list_if_str_or_none(method)
    og_response_col = make_list_if_str_or_none(og_response_col)

    ensure_all_cols_in_df(
        df, cols=experiment_cols + dose_cols + response_col + og_response_col
    )

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    if len(response_col) != 1:
        raise ValueError("Length of response_col must be exactly 1")

    response_col = response_col[0]
    og_response_col = og_response_col[0]

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )

    # Ensure each dosepair for each experiment has only one response by taking
    # the mean

    df_summary = df.group_by(experiment_cols + dose_cols).agg(
        pl.col(response_col, og_response_col).mean()
    )  # NOTE: Will need to add "response" here for indicator

    if "NMF" in method:
        df_summary = _factor_by_group(
            df_summary, dose_cols, response_col, experiment_cols, _nmf, "NMF"
        )
    if "SVD" in method:
        df_summary = _factor_by_group(
            df_summary, dose_cols, response_col, experiment_cols, _svd, "SVD"
        )
    if "PMF" in method:
        df_summary = _factor_by_group(
            df_summary, dose_cols, response_col, experiment_cols, _pmf, "PMF"
        )
    if "RPCA" in method:
        df_summary = _factor_by_group(
            df_summary, dose_cols, response_col, experiment_cols, _rpca, "RPCA"
        )
    if "LNMF" in method:
        if optuna is None:
            raise ModuleNotFoundError(
                "optuna is required for LNMF factorization. Install optuna or "
                "remove 'LNMF' from the selected factorization methods."
            )

        if log == "info":
            optuna.logging.set_verbosity(optuna.logging.INFO)

        # optuna outputs wayyyyy more logs than everything else in this library,
        # so its 'all' feels more like a debug

        # To make things hopefully a little more homogenous, we're going to make
        # log == "all" relate to logging.WARNING in optuna
        if log == "all":
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if log == "warn":
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if log == "none":
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        df_summary = _factor_by_group(
            df_summary,
            dose_cols,
            response_col,
            experiment_cols,
            _lnmf,
            "LNMF",
            og_response_col,
        )
    # Unrecognized methods are silently ignored (no factorization column added)
    out = df.join(
        df_summary.drop(response_col),
        on=experiment_cols + dose_cols,
        how="right",
    )

    if added_dummy:
        out = out.drop("experiment_id")
    # TODO maintain experiment order (maybe using hole'd argument in loop)
    return out


def mf_combination(
    df,
    dose_cols=["dose_a", "dose_b"],
    response_col="resp_imputed",
    experiment_cols="experiment_id",
    og_response_col="response",
    log="all",
):
    """Run all four matrix factorization methods and combine results.

    This is a convenience wrapper that runs NMF, SVD, PMF, and RPCA together,
    equivalent to xynergy008's ``mf_combination`` function.

    Parameters
    ----------
    df : polars.DataFrame
        Input data (typically output from pre_impute).
    dose_cols : list[str]
        Two dose column names.
    response_col : str
        Name of the imputed response column.
    experiment_cols : list[str] or str
        Experiment grouping columns.
    og_response_col : str
        Original (non-imputed) response column name.
    log : str
        Verbosity level.

    Returns
    -------
    polars.DataFrame
        Input with columns for each factorization method appended.
    """
    return matrix_factorize(
        df,
        dose_cols=dose_cols,
        response_col=response_col,
        experiment_cols=experiment_cols,
        method=["NMF", "SVD", "PMF", "RPCA"],
        og_response_col=og_response_col,
        log=log,
    )


def _to_mat(experiment: pl.DataFrame, dose_cols: list[str], response_col: str):
    """Turns an 'experiment' into a matrix ammenable to matrix factorization.

    :param experiment: A polars.DataFrame describing a condition that only
    varies by drug concentrations - that is, not varying by drug type, cell line
    type, etc. This experiment should not have replicates.

    :param dose_cols: The names of the columns containing the concentrations of
    each drug

    :param response_col: The name of the column containing imputed responses

    :return: A matrix with dose_cols[0] increasing on from top to bottom on the
    rows and dose_cols[1] increasing from left to right on the columns
    """
    experiment = experiment.sort(dose_cols)
    mat = experiment.pivot(index=dose_cols[0], on=dose_cols[1], values=response_col)

    # Remove dose_a 'index' column
    mat = mat.drop(pl.col(dose_cols[0]))
    return mat.to_numpy()


def _from_mat(mat, name: str) -> pl.DataFrame:
    """Return an array to the order which it initially came. Kind of like the
    reverse of _prep_mat_from_exp.

    :param mat: Matrix of values

    :param name: Column name of resultant unraveled matrix

    :return: A single column `polars.DataFrame` named `colname`.
    """

    # Transformations necessary to the rows ends up the same order as the
    # input dataset
    long = pl.DataFrame(mat.transpose()).unpivot()
    values = long.select(pl.col("value").alias(name))
    return values


def _factor_by_group(
    x,
    dose_cols,
    response_col,
    experiment_cols,
    fn,
    method_name,
    og_response_col=None,
):
    factored = []
    for _, experiment in x.group_by(experiment_cols):
        experiment = experiment.sort(dose_cols)
        mat = _to_mat(experiment, dose_cols, response_col)

        if method_name == "LNMF":
            indicator_mat = _to_mat(experiment, dose_cols, og_response_col)
            indicator_mat = ~np.isnan(indicator_mat) * 1
            approx = fn(mat, indicator_mat)
        else:
            approx = fn(mat)

        approx_col = _from_mat(approx, f"{response_col}_{method_name}")
        experiment = pl.concat([experiment, approx_col], how="horizontal")
        experiment = experiment.drop(response_col)
        factored.append(experiment)
    all = pl.concat(factored)

    # Upstream summarization should ensure this is safe...
    all = all.sort(experiment_cols + dose_cols)
    x = x.sort(experiment_cols + dose_cols)
    x = x.with_columns(all[f"{response_col}_{method_name}"])
    return x


def _nmf(x):
    """Calculate something *like* 'composite of weighted penalized NMF' (cNMF)

    This is significantly different from the cNMF described in Ianevski et al.
    (DOI: 10.1038/s42256-019-0122-4) in that penalization is not required (as
    the input data already has values imputed).

    Like cNMF, rank is randomly selected to be 2 or 3 for each run of NMF.
    Similarly, the mode of the values is found TODO (determine optimal mode
    finding method and doc here).

    This is a lower-level function that consumes numeric matrices rather than
    tidy data. For typical xynergy workflows, call via `matrix_factorize`.

    :param x: A `np.array` to be approximated with cNMF.

    :return: An approximated version of the `np.array`
    """
    # Convert negatives to 0

    # Negative inhibitions DO happen, like when a drug makes cells grow faster
    # rather than kills them
    x = np.where(x < 0, 0.0, x)

    rng = np.random.default_rng(1337)

    final = pl.DataFrame()

    for i in range(120):
        rank = rng.choice([2, 3], 1)[0]
        model = decomp.NMF(
            rank,
            beta_loss="kullback-leibler",
            solver="mu",
            random_state=i,
            max_iter=500,
        )
        w = model.fit_transform(x)
        h = model.components_
        out = pl.DataFrame(w @ h).unpivot().drop("variable")
        out.columns = [str(i)]
        final = pl.concat([final, out], how="horizontal")
    modes = final.transpose().select(pl.all().map_batches(venter, return_dtype=pl.Float64, returns_scalar=True)).transpose()
    return modes.to_numpy().reshape(x.shape).transpose()


def _svd(x):
    """Matrix factorization via singular value decomposition (SVD)"""

    U, S, VT = la.svd(x)

    rank = S.shape[0]

    final = pl.DataFrame()
    for i in range(2, rank + 1):
        s = np.diag(S[:i])
        u = U[:, :i]
        vt = VT[:i, :]
        approx = u @ s @ vt
        out = pl.DataFrame(approx).unpivot().drop("variable")
        out.columns = [str(i)]
        final = pl.concat([final, out], how="horizontal")

    modes = final.transpose().select(pl.all().map_batches(venter, return_dtype=pl.Float64, returns_scalar=True)).transpose()
    return modes.to_numpy().reshape(x.shape).transpose()


def _rpca(x, l=None):
    """Factorize a DataFrame using robust principal component analysis (RPCA)"""
    if cp is None:
        raise ModuleNotFoundError(
            "cvxpy is required for RPCA factorization. Install cvxpy or "
            "remove 'RPCA' from the selected factorization methods."
        )

    if l is None:
        l = 1 / np.sqrt(max(x.shape))  # A common choice for lambda

    m, n = x.shape
    L = cp.Variable((m, n))
    S = cp.Variable((m, n))

    objective = cp.Minimize(cp.norm(L, "nuc") + l * cp.norm(S, "fro"))
    constraints = [L + S == x]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return S.value


def cost_function(params, num_rows, num_cols, num_features, x):
    """Cost function that calculates the MSE between known matrix values and predictions."""
    row_features = params[: num_rows * num_features].reshape((num_rows, num_features))
    col_features = params[num_rows * num_features :].reshape((num_cols, num_features))
    prediction = np.dot(row_features, col_features.T)
    mask = ~np.isnan(x)
    return np.sum((x[mask] - prediction[mask]) ** 2)


def _pmf(x):
    """Factorize a DataFrame using probabilistic matrix factorization."""

    num_features = 3
    num_iterations = 120
    num_rows, num_cols = x.shape
    final = pl.DataFrame()
    for i in range(num_iterations + 1):
        np.random.seed(i)
        row_features = np.random.rand(num_rows, num_features)
        col_features = np.random.rand(num_cols, num_features)

        # Flatten the initial parameters
        initial_params = np.hstack((row_features.ravel(), col_features.ravel()))

        # Optimize
        result = opt.minimize(
            cost_function,
            initial_params,
            args=(num_rows, num_cols, num_features, x),
            method="BFGS",
            options={"maxiter": 1000},
        )

        # Extract the optimized features
        optimized_params = result.x
        row_features = optimized_params[: num_rows * num_features].reshape(
            (num_rows, num_features)
        )
        col_features = optimized_params[num_rows * num_features :].reshape(
            (num_cols, num_features)
        )

        # Reconstruct the matrix
        approximation = pl.DataFrame(np.dot(row_features, col_features.T))
        out = approximation.unpivot().drop("variable")
        out.columns = [str(i)]
        final = pl.concat([final, out], how="horizontal")

    modes = final.transpose().select(pl.all().map_batches(venter, return_dtype=pl.Float64, returns_scalar=True)).transpose()
    return modes.to_numpy().reshape(x.shape).transpose()
