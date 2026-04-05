import numpy as np
import polars as pl
import scipy.stats as stats


def make_list_if_str_or_none(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        if not np.iterable(x):
            return np.array([x.item()])
        return x
    if isinstance(x, pl.Series):
        return x
    if not isinstance(x, list):
        return [x]
    return x


def _add_id_if_no_experiment_cols(
    df: pl.DataFrame, experiment_cols: list
) -> (pl.DataFrame, list[str], bool):
    """Add a 'dummy' experiment_id if no experiment_cols are provided

    Many functions operate on each experimental group, but it's entirely
    possible that there's only one experiment. Rather than make the user add an
    experimental column, which goes against the principle of least astonishment
    imo, just add a dummy column.

    :return: Three things:

    - The original `df`. If it had experiment_cols, nothing is changed. If it
      didn't, a new column named `experiment_id` is added.

    - A list of `experiment_cols`. If `experiment_cols` was not an empty list,
      returns input. Otherwise, returns `["experiment_id"]`

    - A boolean denoting if the dummy column was added. This is useful as it is
      completely possible for `df` to be supplied with `experiment_id` as a
      column - which isn't a dummy! But if we *do* have a dummy column, we want
      to remove it at the end of the function. By looking at the value of this
      boolean, we can determine if this function added a dummy column.

    """
    if experiment_cols == []:
        # We can name it with experiment_id since it's a reserved column name
        return df.with_columns(experiment_id=pl.lit(1)), ["experiment_id"], True

    return df, experiment_cols, False


def venter(x, k=None):
    """Find the Venter mode of a list of numbers

    Source is a stripped down and adapted version from the {modeest} R package.

    :param x: A list of numbers - continuous or integer (or a Polars Series)

    :return: A numeric - the Venter mode
    """
    y = sorted(x.to_list() if isinstance(x, pl.Series) else x)
    ny = len(y)
    if k is None:
        k = int(np.ceil(ny / 2) - 1)

    if k < 0 or k >= ny:
        raise ValueError(f"k must be in the range [0, {ny})")
    inf = y[0 : (ny - k)]
    sup = y[k:ny]
    diffs = np.array(sup) - np.array(inf)
    min = np.nanmin(diffs)
    i = int(np.floor(np.nanmean(np.where(diffs == min))))
    return (y[i] + y[i + k]) / 2


def binned_mode(x, n_bins):
    bins = stats.binned_statistic(x, x, "count", n_bins)
    most_freq_bin = stats.mode(bins.binnumber).mode
    bin_avg = (bins.bin_edges[most_freq_bin - 1] + bins.bin_edges[most_freq_bin]) / 2
    # binnumber starts from 1, so most_freq_bin - 1 is the index of left bin
    # edge. Average with its neighbor to find the bin center.
    return bin_avg


def unit_conversion(concentration_unit: str) -> float:
    """Convert a concentration unit name to a numerical scaling factor.

    Parameters
    ----------
    concentration_unit : str
        One of 'Molar', 'Millimolar', 'Micromolar', 'Nanomolar', 'Picomolar',
        'Microgram', 'Nanogram', 'Picogram'.

    Returns
    -------
    float
        The numerical scaling factor.

    Raises
    ------
    ValueError
        If the unit is not recognized.
    """
    units = {
        "Molar": 1,
        "Millimolar": 1e-3,
        "Micromolar": 1e-6,
        "Nanomolar": 1e-9,
        "Picomolar": 1e-12,
        "Microgram": 1e-6,
        "Nanogram": 1e-9,
        "Picogram": 1e-12,
    }
    if concentration_unit not in units:
        raise ValueError(
            f"Unknown concentration unit: {concentration_unit}. "
            f"Valid units: {list(units.keys())}"
        )
    return units[concentration_unit]


def outlier_remove(values, iqr_=1.5):
    """Detect outliers in a flat array using IQR method.

    Parameters
    ----------
    values : array-like
        Numeric values (NaN-safe).
    iqr_ : float, default 1.5
        IQR multiplier for outlier threshold.

    Returns
    -------
    numpy.ndarray
        Boolean mask where True = non-outlier.
    """
    arr = np.asarray(values, dtype=float)
    non_na = arr[~np.isnan(arr)]
    if len(non_na) == 0:
        return np.zeros_like(arr, dtype=bool)
    q1 = np.percentile(non_na, 25)
    q3 = np.percentile(non_na, 75)
    iqr = q3 - q1
    mask = ~((arr < (q1 - iqr_ * iqr)) | (arr > (q3 + iqr_ * iqr)))
    # NaN positions are not outliers
    mask[np.isnan(arr)] = False
    return mask


def remove_row_outliers(row, threshold=1):
    """Remove outliers from a row using IQR method, replacing with NaN.

    Parameters
    ----------
    row : array-like
        A row of numeric data.
    threshold : float, default 1
        IQR multiplier for outlier threshold.

    Returns
    -------
    numpy.ndarray
        Row with outliers replaced by NaN.
    """
    row = np.asarray(row, dtype=float).copy()
    non_na = row[~np.isnan(row)]
    if len(non_na) == 0:
        return row
    q1 = np.percentile(non_na, 25)
    q3 = np.percentile(non_na, 75)
    iqr = q3 - q1
    outliers = (row < (q1 - threshold * iqr)) | (row > (q3 + threshold * iqr))
    row[outliers] = np.nan
    return row


def to_mat(
    x,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
):
    """Turn an 'experiment' into a matrix ammenable to factorization.

    :param experiment: A polars.DataFrame describing a condition that only
    varies by drug concentrations - that is, not varying by drug type, cell line
    type, etc.

    :param dose_cols: The names of the columns containing the concentrations of
    each drug

    :param response_col: The name of the column containing imputed responses

    :return: A matrix with dose_cols[0] increasing on from top to bottom on the
    rows and dose_cols[1] increasing from left to right on the columns

    """
    x = x.sort(dose_cols)
    mat = x.pivot(
        index=dose_cols[0],
        on=dose_cols[1],
        values=response_col,
        aggregate_function="mean",
    )

    # Remove dose_a 'index' column
    mat = mat.drop(pl.col(dose_cols[0]))
    return mat.to_numpy()
