import numpy as np
import numpy.linalg as la
import polars as pl
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

# Needed to enable iterative imputation! Do not remove.
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

from .fit import add_uncombined_drug_responses
from .util import _add_id_if_no_experiment_cols, make_list_if_str_or_none
from .validate import ensure_all_cols_in_df


def _iterative_svd_complete(
    observed: np.ndarray,
    mask: np.ndarray,
    rank: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """Complete a partially observed matrix via iterative rank-truncated SVD.

    At each iteration the observed cells are pinned to their known values while
    the missing cells are filled with the current rank-*r* approximation.  The
    process converges to a low-rank matrix that agrees with the observations.

    Parameters
    ----------
    observed : np.ndarray
        Matrix with known values in observed positions (other positions are
        ignored; set them to 0).
    mask : np.ndarray
        Binary array of the same shape: 1.0 for observed, 0.0 for missing.
    rank : int
        Rank used for truncated SVD at each step.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on relative change in Frobenius norm.
    warm_start : np.ndarray or None
        Optional initial estimate for the completed matrix.  A good warm
        start (e.g. Bliss reference) accelerates convergence.

    Returns
    -------
    np.ndarray
        Completed matrix with the same shape as *observed*.
    """
    X = np.where(mask > 0.5, observed, 0.0)
    M = warm_start.copy() if warm_start is not None else X.copy()
    r = min(rank, *observed.shape)

    for _ in range(max_iter):
        M_prev = M
        Z = np.where(mask > 0.5, X, M)
        U, S, Vt = la.svd(Z, full_matrices=False)
        M = (U[:, :r] * S[:r]) @ Vt[:r, :]

        denom = la.norm(M_prev)
        if denom > 1e-10 and la.norm(M - M_prev) / denom < tol:
            break

    # Pin observed cells back to their exact values so downstream steps see
    # the original measurements for known points.
    M = np.where(mask > 0.5, X, M)
    return M


def _rbf_surface_complete(
    data: pl.DataFrame,
    dose_cols: list[str],
    response_col: str,
    target_col: str,
    target: str,
    single_resp_cols: list[str],
    experiment_cols: list[str],
    reference_for_target: str,
    rbf_kernel: str = "thin_plate_spline",
    rbf_smoothing: float = 0.05,
) -> pl.DataFrame:
    """Complete missing values via RBF interpolation on Bliss residuals in log-dose space.

    The key insight is that dose-response surfaces are smooth in log-dose
    coordinates, and the *deviation* from a Bliss independence baseline is
    particularly well-behaved for interpolation.  This method:

    1. Computes a no-interaction reference (Bliss or HSA) from fitted
       single-drug curves.
    2. Calculates the residual (observed - reference) at every observed point.
    3. Fits a thin-plate-spline RBF in (log10(dose_a), log10(dose_b)) space.
    4. Predicts the residual at missing points and reconstructs the response.

    This is fast, parameter-free, and exploits the pharmacological smoothness
    prior that generic tabular methods like XGBR cannot.
    """
    imputed_list = []
    for _, group in data.group_by(experiment_cols):
        group = group.sort(dose_cols)

        # Compute reference for this group
        if len(single_resp_cols) >= 2:
            if reference_for_target == "bliss":
                group = group.with_columns(
                    (
                        pl.col(single_resp_cols[0])
                        + pl.col(single_resp_cols[1])
                        - pl.col(single_resp_cols[0]) * pl.col(single_resp_cols[1]) / 100
                    ).alias("_rbf_reference")
                )
            else:
                group = group.with_columns(
                    pl.max_horizontal(single_resp_cols).alias("_rbf_reference")
                )
        else:
            group = group.with_columns(pl.lit(0.0).alias("_rbf_reference"))

        da_list = group[dose_cols[0]].to_list()
        db_list = group[dose_cols[1]].to_list()
        resp_list = group[response_col].to_list()
        ref_list = group["_rbf_reference"].to_list()
        n_rows = len(da_list)

        # Collect observed points in log-dose space
        obs_coords = []
        obs_resids = []
        for k in range(n_rows):
            v = resp_list[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                log_da = np.log10(da_list[k] + 1e-10)
                log_db = np.log10(db_list[k] + 1e-10)
                obs_coords.append([log_da, log_db])
                obs_resids.append(v - ref_list[k])

        obs_coords = np.array(obs_coords)
        obs_resids = np.array(obs_resids)

        if len(obs_coords) >= 3:
            rbf = RBFInterpolator(
                obs_coords, obs_resids,
                kernel=rbf_kernel, smoothing=rbf_smoothing,
            )
            # Predict for all rows
            all_coords = np.array([
                [np.log10(da_list[k] + 1e-10), np.log10(db_list[k] + 1e-10)]
                for k in range(n_rows)
            ])
            pred_resids = rbf(all_coords)
            imputed_vals = np.array(ref_list) + pred_resids
        else:
            # Too few observed points — fall back to reference
            imputed_vals = np.array(ref_list)

        if target == "combo_effect":
            group = group.with_columns(
                pl.Series("combo_effect_imputed", imputed_vals - np.array(ref_list))
            )
            group = group.with_columns(
                pl.Series("response_imputed_from_effect", imputed_vals)
            )
            group = group.with_columns(
                pl.col("response_imputed_from_effect").alias("resp_imputed")
            )
        else:
            group = group.with_columns(
                pl.Series("resp_imputed", imputed_vals)
            )

        group = group.drop("_rbf_reference")
        imputed_list.append(group)

    return pl.concat(imputed_list)


def _gaussian_process_surface_complete(
    data: pl.DataFrame,
    dose_cols: list[str],
    target_col: str,
    target: str,
    experiment_cols: list[str],
) -> pl.DataFrame:
    """Complete missing values with a Gaussian process in log-dose space.

    The model is fit separately to each experiment using the observed values of
    `target_col`, which is either the response itself or the residual
    interaction effect created upstream. Observed values are pinned back onto
    the output so the method behaves as an imputer rather than a denoiser.
    """
    imputed_list = []
    for _, group in data.group_by(experiment_cols):
        group = group.sort(dose_cols)

        dose_a = np.array(group[dose_cols[0]].to_list(), dtype=float)
        dose_b = np.array(group[dose_cols[1]].to_list(), dtype=float)
        target_vals = np.array(
            [np.nan if x is None else float(x) for x in group[target_col].to_list()],
            dtype=float,
        )
        all_coords = np.column_stack(
            (np.log10(dose_a + 1e-10), np.log10(dose_b + 1e-10))
        )
        observed_mask = ~np.isnan(target_vals)

        if observed_mask.sum() == 0:
            pred_target = np.full(group.height, np.nan)
        else:
            obs_coords = all_coords[observed_mask]
            obs_target = target_vals[observed_mask]
            unique_coords, inverse = np.unique(obs_coords, axis=0, return_inverse=True)
            unique_target = np.zeros(unique_coords.shape[0], dtype=float)
            unique_counts = np.zeros(unique_coords.shape[0], dtype=float)
            np.add.at(unique_target, inverse, obs_target)
            np.add.at(unique_counts, inverse, 1.0)
            unique_target = unique_target / unique_counts

            if unique_coords.shape[0] == 1:
                pred_target = np.full(group.height, unique_target[0], dtype=float)
            else:
                coord_scale = np.maximum(np.std(unique_coords, axis=0), 1e-2)
                kernel = (
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * RBF(length_scale=coord_scale, length_scale_bounds=(1e-2, 1e2))
                    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
                )
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=0,
                )
                try:
                    gp.fit(unique_coords, unique_target)
                    pred_target = gp.predict(all_coords)
                except Exception:
                    pred_target = np.full(group.height, unique_target.mean(), dtype=float)

            pred_target[observed_mask] = target_vals[observed_mask]

        if target == "combo_effect":
            reference = np.array(group["_target_reference"].to_list(), dtype=float)
            group = group.with_columns(
                pl.Series("combo_effect_imputed", pred_target),
                pl.Series("response_imputed_from_effect", reference + pred_target),
            ).with_columns(
                pl.col("response_imputed_from_effect").alias("resp_imputed")
            )
        else:
            group = group.with_columns(pl.Series("resp_imputed", pred_target))

        imputed_list.append(group)

    return pl.concat(imputed_list)


def pre_impute(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    experiment_cols: str | list[str] = "experiment_id",
    method: str = "RBFSurface",
    target: str = "response",
    reference_for_target: str = "bliss",
    ensemble_response_weight: float = 0.6,
    clip_response_bounds: tuple[float, float] | None = (0.0, 100.0),
    use_single_drug_response_data: bool = True,
    additional_imputation_cols: str | list[str] | None = None,
    log: str = "all",
):
    """Impute missing response data.

    Parameters
    ----------
    df: polars.DataFrame
        Usually the output from `tidy` or one of its downstream functions

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string, default "response"
        The name of the column containing responses and missing responses to be
        imputed

    experiment_cols: list[str], string, or None, default "experiment_id"
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates. Experiments are imputed separately,
        so as to prevent information leakage. These columns are used strictly
        for grouping and are not used for imputation.

    method: string, default "XGBR"
        - "RBFSurface" (recommended): RBF interpolation of Bliss residuals in
          log-dose space.  Exploits the pharmacological smoothness of
          dose-response surfaces.  Very fast and generally the most accurate
          method, especially when the observed cells are the single-drug edges
          and a positional diagonal.
        - "GaussianProcessSurface": Gaussian-process regression in log-dose
          space. Slower than RBFSurface, but a strong non-parametric surface
          baseline for benchmarking.
        - "MatrixCompletion": Iterative rank-truncated SVD that exploits the
          low-rank structure of dose-response matrices.
        - "XGBR" (slowest, most accurate of the tabular methods),
        - "RandomForest" (roughly medium speed and accuracy),
        - "LassoCV" (fast, poor accuracy. Not recommended.),
        - Otherwise, default sklearn IterativeImputer (fastest, sometimes better
          accuracy than LassoCV)

    target: string, default "response"
        What to impute. Options:

        - "response": Impute `response_col` directly (existing behavior).
        - "combo_effect": Impute residual interaction effect relative to
          `reference_for_target`, then reconstruct response.
        - "ensemble": Blend the `"response"` and `"combo_effect"` predictions
          using `ensemble_response_weight`.

    reference_for_target: string, default "bliss"
        Only used when `target = "combo_effect"`. Defines the no-interaction
        baseline used to create the target residual. Options:

        - "bliss"
        - "hsa"

    ensemble_response_weight: float, default 0.6
        Only used when `target = "ensemble"`. Weight assigned to the direct
        `"response"` prediction; the `"combo_effect"` prediction receives
        `1 - ensemble_response_weight`.

    clip_response_bounds: tuple[float, float] | None, default (0.0, 100.0)
        Bounds applied to reconstructed response columns (`resp_imputed`,
        `response_imputed_from_effect`) when available. Use `None` to disable.

    use_single_drug_response_data: bool, default True
        Some methods - like RandomForest - perform better when the dataset
        contains columns with the responses of, say, 'drug A' at 'dose_a' (no
        combination). If this parameter is `True`, automatically calculate this
        value and include it as data to be used for imputation. You might set
        this as `False` if you want to include your own data, for instance - in
        which case you would add the name of those columns to
        `additional_imputation_cols`. In general, this step can only help and is
        relatively quick.

    additional_imputation_cols: string, list[str], optional
        Additional column name(s) that should also be used for imputation.
        Columns not listed here will be dropped prior to imputation and rejoined
        afterwards.

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    Input df with a `resp_imputed` column (plus `[dose_cols]_resp` if
    `use_single_drug_response_data = True`).

    If `target = "combo_effect"`, additional columns are returned:

    - `combo_effect_imputed`
    - `response_imputed_from_effect`

    If `target = "ensemble"`, these additional columns are returned:

    - `resp_imputed_response`
    - `resp_imputed_combo_effect`
    - `resp_imputed_ensemble`

    """
    df_og = df.clone()

    experiment_cols = make_list_if_str_or_none(experiment_cols)
    additional_imputation_cols = make_list_if_str_or_none(additional_imputation_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    if len([response_col]) != 1:
        raise ValueError("Length of response_col must be exactly 1")

    if target not in ["response", "combo_effect", "ensemble"]:
        raise ValueError("`target` must be one of ['response', 'combo_effect', 'ensemble']")

    if reference_for_target not in ["bliss", "hsa"]:
        raise ValueError("`reference_for_target` must be one of ['bliss', 'hsa']")

    if not 0.0 <= ensemble_response_weight <= 1.0:
        raise ValueError("`ensemble_response_weight` must be between 0 and 1")

    if target == "ensemble":
        row_id_col = "__xynergy_ensemble_row_id"
        while row_id_col in df_og.columns:
            row_id_col = "_" + row_id_col

        df_with_row_id = df_og.with_row_index(row_id_col)
        common_args = dict(
            dose_cols=dose_cols,
            response_col=response_col,
            experiment_cols=experiment_cols,
            method=method,
            reference_for_target=reference_for_target,
            ensemble_response_weight=ensemble_response_weight,
            clip_response_bounds=clip_response_bounds,
            use_single_drug_response_data=use_single_drug_response_data,
            additional_imputation_cols=additional_imputation_cols,
            log=log,
        )

        response_out = pre_impute(
            df_with_row_id,
            target="response",
            **common_args,
        ).rename({"resp_imputed": "resp_imputed_response"})

        combo_out = pre_impute(
            df_with_row_id,
            target="combo_effect",
            **common_args,
        )

        combo_select = [row_id_col, "resp_imputed", "combo_effect_imputed", "response_imputed_from_effect"]
        for col in [dose_cols[0] + "_resp", dose_cols[1] + "_resp"]:
            if col in combo_out.columns and col not in response_out.columns:
                combo_select.append(col)

        out = response_out.join(
            combo_out.select(combo_select).rename({"resp_imputed": "resp_imputed_combo_effect"}),
            on=row_id_col,
            how="left",
        ).with_columns(
            pl.lit(ensemble_response_weight).alias("ensemble_response_weight"),
            pl.lit(1.0 - ensemble_response_weight).alias("ensemble_combo_effect_weight"),
        ).with_columns(
            (
                pl.col("ensemble_response_weight") * pl.col("resp_imputed_response")
                + pl.col("ensemble_combo_effect_weight") * pl.col("resp_imputed_combo_effect")
            ).alias("resp_imputed_ensemble")
        ).with_columns(
            pl.col("resp_imputed_ensemble").alias("resp_imputed")
        ).drop(row_id_col)

        if clip_response_bounds is not None:
            low, high = clip_response_bounds
            out = out.with_columns(
                pl.col("resp_imputed_ensemble").clip(low, high),
                pl.col("resp_imputed").clip(low, high),
            )

        return out

    if experiment_cols == ["experiment_id"] and "experiment_id" not in df_og.columns:
        experiment_cols = []

    df_og, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df_og, experiment_cols
    )

    required_cols = experiment_cols + additional_imputation_cols + dose_cols + [response_col]
    ensure_all_cols_in_df(df_og, cols=required_cols)

    row_id_col = "__xynergy_pre_impute_row_id"
    while row_id_col in df_og.columns:
        row_id_col = "_" + row_id_col
    df_og = df_og.with_row_index(row_id_col)

    saved_cols_list = [row_id_col] + required_cols
    df = df_og[saved_cols_list]
    if use_single_drug_response_data:
        df = add_uncombined_drug_responses(
            df, dose_cols, response_col, experiment_cols, log=log
        )
        single_resp_cols = [dose_cols[0] + "_resp", dose_cols[1] + "_resp"]
    else:
        single_resp_cols = []

    if target == "combo_effect":
        if len(single_resp_cols) == 0:
            df = add_uncombined_drug_responses(
                df, dose_cols, response_col, experiment_cols, log=log
            )
            single_resp_cols = [dose_cols[0] + "_resp", dose_cols[1] + "_resp"]

        if reference_for_target == "bliss":
            df = df.with_columns(
                _target_reference=(
                    pl.col(single_resp_cols[0])
                    + pl.col(single_resp_cols[1])
                    - pl.col(single_resp_cols[0]) * pl.col(single_resp_cols[1]) / 100
                )
            )
        else:
            df = df.with_columns(
                _target_reference=pl.max_horizontal(single_resp_cols)
            )
        target_col = "_combo_effect_target"
        df = df.with_columns(
            (pl.col(response_col) - pl.col("_target_reference")).alias(target_col)
        )
    else:
        target_col = response_col

    if method == "RBFSurface":
        imputed = _rbf_surface_complete(
            data=df,
            dose_cols=dose_cols,
            response_col=response_col,
            target_col=target_col,
            target=target,
            single_resp_cols=single_resp_cols,
            experiment_cols=experiment_cols,
            reference_for_target=reference_for_target,
        )

    elif method in ["GaussianProcessSurface", "GPSurface"]:
        imputed = _gaussian_process_surface_complete(
            data=df,
            dose_cols=dose_cols,
            target_col=target_col,
            target=target,
            experiment_cols=experiment_cols,
        )

    elif method == "MatrixCompletion":
        imputed_list = []
        for _, data in df.group_by(experiment_cols):
            data = data.sort(dose_cols)

            doses_a = sorted(data[dose_cols[0]].unique().to_list())
            doses_b = sorted(data[dose_cols[1]].unique().to_list())
            idx_a = {d: i for i, d in enumerate(doses_a)}
            idx_b = {d: i for i, d in enumerate(doses_b)}
            n_a, n_b = len(doses_a), len(doses_b)

            da_list = data[dose_cols[0]].to_list()
            db_list = data[dose_cols[1]].to_list()
            target_vals = data[target_col].to_list()
            n_rows = len(da_list)

            # Build target matrix (average replicates when present)
            val_sum = np.zeros((n_a, n_b))
            val_count = np.zeros((n_a, n_b))
            for k in range(n_rows):
                i, j = idx_a[da_list[k]], idx_b[db_list[k]]
                v = target_vals[k]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    val_sum[i, j] += v
                    val_count[i, j] += 1

            target_mat = np.zeros((n_a, n_b))
            observed = val_count > 0
            target_mat[observed] = val_sum[observed] / val_count[observed]
            mask = observed.astype(float)

            # Warm start: Bliss reference for direct response, zeros for residual
            if target == "combo_effect":
                warm = np.zeros((n_a, n_b))
            elif use_single_drug_response_data and len(single_resp_cols) >= 2:
                warm = np.zeros((n_a, n_b))
                sa_list = data[single_resp_cols[0]].to_list()
                sb_list = data[single_resp_cols[1]].to_list()
                for k in range(n_rows):
                    i, j = idx_a[da_list[k]], idx_b[db_list[k]]
                    sa, sb = sa_list[k], sb_list[k]
                    if (sa is not None and sb is not None
                            and not (isinstance(sa, float) and np.isnan(sa))
                            and not (isinstance(sb, float) and np.isnan(sb))):
                        warm[i, j] = sa + sb - sa * sb / 100
            else:
                warm = None

            # Rank 1 for combo_effect (interaction residual is near rank-1);
            # rank 3 for direct response (full surface needs more flexibility).
            default_rank = 1 if target == "combo_effect" else 3
            completed = _iterative_svd_complete(
                target_mat, mask,
                rank=min(default_rank, min(n_a, n_b) - 1),
                warm_start=warm,
            )

            # Map completed matrix back to DataFrame rows
            imputed_vals = np.empty(n_rows)
            for k in range(n_rows):
                i, j = idx_a[da_list[k]], idx_b[db_list[k]]
                imputed_vals[k] = completed[i, j]

            if target == "combo_effect":
                data = data.with_columns(
                    pl.Series("combo_effect_imputed", imputed_vals)
                )
                data = data.with_columns(
                    (pl.col("_target_reference") + pl.col("combo_effect_imputed")).alias(
                        "response_imputed_from_effect"
                    )
                )
                data = data.with_columns(
                    pl.col("response_imputed_from_effect").alias("resp_imputed")
                )
            else:
                data = data.with_columns(
                    pl.Series("resp_imputed", imputed_vals)
                )

            imputed_list.append(data)
        imputed = pl.concat(imputed_list)

    else:
        const_args = {"random_state": 0}
        if target == "response":
            const_args["min_value"] = 0
            const_args["max_value"] = 100
        if method == "XGBR":
            imputer = IterativeImputer(
                XGBRegressor(learning_rate=0.05, n_jobs=None), max_iter=10, **const_args
            )
        elif method == "LassoCV":
            imputer = IterativeImputer(
                LassoCV(max_iter=100000, n_jobs=None), max_iter=1000, **const_args
            )
        elif method == "RandomForest":
            imputer = IterativeImputer(
                RandomForestRegressor(n_jobs=None), max_iter=100, **const_args
            )
        else:
            imputer = IterativeImputer(
                max_iter=10, random_state=const_args["random_state"]
            )

        imputer.set_output(transform="polars")

        imputation_cols = additional_imputation_cols + dose_cols
        if use_single_drug_response_data:
            imputation_cols = imputation_cols + single_resp_cols
        imputation_cols.append(target_col)

        imputed_list = []
        for _, data in df.group_by(experiment_cols):
            imputed_df: pl.DataFrame = imputer.fit_transform(data[imputation_cols])
            if target == "combo_effect":
                data = data.with_columns(
                    pl.Series(imputed_df[target_col]).alias("combo_effect_imputed")
                )
                data = data.with_columns(
                    (pl.col("_target_reference") + pl.col("combo_effect_imputed")).alias(
                        "response_imputed_from_effect"
                    )
                )
                data = data.with_columns(
                    pl.col("response_imputed_from_effect").alias("resp_imputed")
                )
            else:
                data = data.with_columns(
                    pl.Series(imputed_df[target_col]).alias("resp_imputed")
                )
            imputed_list.append(data)
        imputed = pl.concat(imputed_list)

    if clip_response_bounds is not None:
        low, high = clip_response_bounds
        if "response_imputed_from_effect" in imputed.columns:
            imputed = imputed.with_columns(
                pl.col("response_imputed_from_effect").clip(low, high),
                pl.col("resp_imputed").clip(low, high),
            )
        else:
            imputed = imputed.with_columns(pl.col("resp_imputed").clip(low, high))

    if "_target_reference" in imputed.columns:
        imputed = imputed.drop("_target_reference", "_combo_effect_target")

    imputed_only_cols = [col for col in imputed.columns if col not in df_og.columns]
    rejoined = df_og.join(
        imputed.select([row_id_col] + imputed_only_cols),
        on=row_id_col,
        how="left",
    ).drop(row_id_col)

    if added_dummy:
        rejoined = rejoined.drop("experiment_id")

    return rejoined


def post_impute(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    experiment_cols: str | list[str] | None = "experiment_id",
    imputed_response_cols: list[str] | None = None,
    imputed_resp_prefix: str = "resp_imputed_",
    post_impute_tuning: str = "Predefined",
    log: str = "all",
):
    """Predict missing data using matrix factorization.

    Parameters
    ----------
    df: polars.DataFrame
        Usually the output from `tidy` or one of its downstream functions

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain *untransformed* numeric
        values of agent dose

    response_col: string, default "response"
        The name of the column containing responses and missing responses to be
        imputed

    experiment_cols: list[str], string, or None, default "experiment_id"
        The names of columns that should be used to distinguish one dose pair's
        response from another. If none are supplied, two rows with the same
        doses will be considered replicates. Experiments are imputed separately,
        so as to prevent information leakage. These columns are used strictly
        for grouping and are not used for imputation.

    imputed_response_cols: list[str], optional
        Columns to use for imputation. If unspecified, will use
        ``imputed_resp_prefix`` and use all columns that match

    imputed_resp_prefix: str, default ``"resp_imputed_"``
        Only used if ``imputed_response_cols`` is ``None``. When looking for columns
        to use for imputation, will use columns that have this prefix in their
        name.

    post_impute_tuning: string, default "Predefined"
        Strategy for tuning XGBoost hyperparameters in post-imputation.

        - "Predefined": Use fixed hyperparameters (learning_rate=0.1,
          max_depth=3, subsample=0.9, gamma=0.5, n_estimators=50). Very fast.
        - "RandomizedSearchCV": Sample a subset of the hyperparameter space
          (20 random combinations, up to 3-fold CV). Moderate speed.
        - "GridSearchCV": Exhaustive search over the full hyperparameter grid
          (324 combinations, up to 3-fold CV). Slowest but most thorough.

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    The same input with values in ``response_col`` imputed

    """
    experiment_cols = make_list_if_str_or_none(experiment_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    if len([response_col]) != 1:
        raise ValueError("Length of response_col must be exactly 1")

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )

    if imputed_response_cols is None:
        colnames = df.columns
        imputed_response_cols = [x for x in colnames if imputed_resp_prefix in x]

        if len(imputed_response_cols) == 0:
            raise ValueError(
                "`imputed_response_cols` is `None` and no columns with `resp_imputed_` found"
            )

    keep = dose_cols + imputed_response_cols

    if post_impute_tuning not in ["Predefined", "RandomizedSearchCV", "GridSearchCV"]:
        raise ValueError(
            "`post_impute_tuning` must be one of "
            "['Predefined', 'RandomizedSearchCV', 'GridSearchCV']"
        )

    space = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
        "gamma": [0, 0.5, 1],
        "n_estimators": [25, 50, 75, 100],
    }

    out = []
    grouped = df.group_by(experiment_cols)
    for _, group in grouped:
        train = group.filter(
            pl.col(response_col).is_not_null(), pl.col(response_col).is_not_nan()
        )
        test = group.filter(
            pl.col(response_col).is_null() | pl.col(response_col).is_nan()
        )

        effective_tuning = post_impute_tuning
        cv_splits = min(3, train.height)
        if effective_tuning != "Predefined" and cv_splits < 2:
            if log in {"all", "warn"}:
                print(
                    "Too few observed rows for cross-validation; "
                    "falling back to Predefined post-imputation XGBoost parameters."
                )
            effective_tuning = "Predefined"

        if effective_tuning == "Predefined":
            # Use fixed hyperparameters — no search, very fast
            predefined_params = {
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.9,
                "gamma": 0.5,
                "n_estimators": 50,
            }
            model = XGBRegressor(
                **predefined_params,
                verbosity=0,
                objective="reg:squarederror",
                random_state=0,
                n_jobs=None,
            )
            model.fit(train.select(keep), train[response_col])
            prediction = model.predict(test.select(keep))
            test = test.with_columns(
                pl.Series(response_col, prediction).cast(pl.Float32)
            )

        else:
            # RandomizedSearchCV or GridSearchCV
            if effective_tuning == "RandomizedSearchCV":
                opt = RandomizedSearchCV(
                    XGBRegressor(
                        verbosity=0,
                        objective="reg:squarederror",
                        random_state=0,
                        n_jobs=None,
                    ),
                    param_distributions=space,
                    n_iter=20,
                    scoring="neg_mean_squared_error",
                    cv=cv_splits,
                    verbose=0,
                    error_score="raise",
                    random_state=0,
                )
            else:  # GridSearchCV
                opt = GridSearchCV(
                    XGBRegressor(
                        verbosity=0,
                        objective="reg:squarederror",
                        random_state=0,
                        n_jobs=None,
                    ),
                    param_grid=space,
                    scoring="neg_mean_squared_error",
                    cv=cv_splits,
                    verbose=0,
                    error_score="raise",
                )

            opt.fit(train.select(keep), train[response_col])
            results = pl.DataFrame(opt.cv_results_, strict=False)
            top_10_params = results.sort("rank_test_score").head(10)["params"].to_list()

            predictions = []
            for params in top_10_params:
                model = XGBRegressor(
                    **params,
                    verbosity=0,
                    objective="reg:squarederror",
                    random_state=12,
                    n_jobs=None,
                )
                model.fit(train.select(keep), train[response_col])
                prediction = model.predict(test.select(keep))
                predictions.append(prediction)
            predictions_array = np.array(predictions)
            predictions_median = np.median(predictions_array, axis=0)
            test = test.with_columns(
                pl.Series(response_col, predictions_median).cast(pl.Float32)
            )

        # Prediction output is Float32, so to append the two we need to downcast
        # the original responses:
        train = train.with_columns(pl.col(response_col).cast(pl.Float32))
        out.append(pl.concat([train, test]))
    out = pl.concat(out)

    if added_dummy:
        out = out.drop("experiment_id")

    return out
