import warnings

import polars as pl

from .util import make_list_if_str_or_none
from .validate import ensure_all_cols_in_df


def tidy(
    df: pl.DataFrame,
    dose_cols: list[str],
    response_col: str | list[str],
    experiment_cols: str | list[str] | None = None,
    response_is_percent: bool = True,
    complete_response_is_0: bool = False,
    log: str = "all",
):
    """Prepare data for analysis with Xynergy.

    Normalizes column names and response values. Should do most of the data
    checking for you to determine if something's off *before* you start the
    analyses

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

    response_is_percent: bool, default True
        Is the response a percentage (ranges from 0-100) or is it a
        probability/ratio (ranges from 0-1)?

    complete_response_is_0: bool, default False
        Is the response reported as (for instance) survival, where a complete
        response would be 0? Or is it something like (again, for instance)
        killing, where a complete response would be 1 (in the case of
        `response_is_percent = False`) or 100 (in the case of `reponse_is_percent =
        True`)

    log: string, default "all"
        Verbosity of function. Options include "all", "warn", and "none".

        - If "all", will emit notes and warnings.
        - If "warn", will emit only warnings.
        - If "none", will not emit anything (except errors)

    Returns
    -------
    polars.DataFrame
        Each row will contain a single response, with the following columns

        * `dose_a`, `dose_b`
            Numeric concentrations of the two agents

        * `response`
            Will be modified (if necessary) to be '% inhibition style' (0 = no
            inhibition, 100 = complete inhibition).

        * `experiment_id`
            Contains integer IDs for each experiment. If `experiment_cols =
            None`, all ids will be 1.

        * Any columns supplied to `experiment_cols`

        All other columns will be dropped.

    """
    # Make str args of column names become lists
    response_col = make_list_if_str_or_none(response_col)
    experiment_cols = list(make_list_if_str_or_none(experiment_cols))
    dose_cols = make_list_if_str_or_none(dose_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    all_needed_cols = experiment_cols + dose_cols + response_col

    ensure_all_cols_in_df(df, all_needed_cols)
    if len(set(all_needed_cols)) != len(all_needed_cols):
        raise ValueError("Columns cannot be used in multiple arguments")

    df, experiment_cols = _enforce_no_reserved_names_used(
        df, response_col, experiment_cols, log
    )
    # Has updated experiment_cols, if necessary
    all_needed_cols = experiment_cols + dose_cols + response_col

    df = df[all_needed_cols]  # Drop any extraneous cols

    df = _enforce_one_response_col(df, response_col, log)

    df = df.rename({dose_cols[0]: "dose_a", dose_cols[1]: "dose_b"})

    df = _add_experiment_id_col(df, experiment_cols)

    _ensure_more_than_one_dose(df)

    if not response_is_percent:
        df = df.with_columns(pl.col("response") * 100)

    if complete_response_is_0:
        df = df.with_columns((100 - pl.col("response")).alias("response"))

    experiment_cols = experiment_cols + ["experiment_id"]

    # In case someone provides only the wells that have data in them, the
    # missing wells will need to be added
    all_dose_combos = (
        df.group_by(experiment_cols)
        .agg(pl.col("dose_a").unique(), pl.col("dose_b").unique())
        .explode("dose_a")
        .explode("dose_b")
    )
    df = all_dose_combos.join(df, on=["dose_a", "dose_b"] + experiment_cols, how="left")

    # For some reason, the order of the rows is non-deterministic with polars.
    # We can force a deterministic order by sorting at the end:
    df = df.sort("experiment_id", "dose_a", "dose_b", "response")

    return df


def _enforce_one_response_col(df: pl.DataFrame, response_col: list[str], log: str):
    if len(response_col) == 1:
        df = df.rename({response_col[0]: "response"})
        return df
    else:
        if log == "all":
            print("More than one response_col. Assuming replicates, unpivoting.")
        not_response_col = set(df.columns) - set(response_col)
        return df.unpivot(
            index=list(not_response_col),
            variable_name="replicate",
            value_name="response",
        )


def _ensure_more_than_one_dose(df):
    out = df.group_by("experiment_id").n_unique() == 1
    if any(out["dose_a"]) or any(out["dose_b"]):
        raise ValueError("Experimental group contains drug with only 1 dose")


def _add_experiment_id_col(df: pl.DataFrame, experiment_cols: list[str]):
    if len(experiment_cols) == 0:
        df = df.with_columns(experiment_id=pl.lit(1))
    else:
        # Solution from https://stackoverflow.com/a/74603070
        df = df.with_row_index("experiment_id").with_columns(
            pl.col("experiment_id").first().over(experiment_cols).rank("dense")
        )
    return df


def _enforce_no_reserved_names_used(
    df,
    response_col,
    experiment_cols,
    log: str,
):
    reserved_names = {"dose_a", "dose_b", "response", "experiment_id"}
    # 'replicate' only needed to unpivot data, and may be used in
    # experiment_cols otherwise. So it is 'conditionally reserved'
    if len(response_col) > 1:
        reserved_names.add("replicate")

    # dose_cols and response_cols can be named whatever they want because they
    # will be clobbered into a unified naming scheme pretty early on
    reserved_names_used = reserved_names.intersection(experiment_cols)
    if len(reserved_names_used) != 0:
        if log in ["all", "warn"]:
            warnings.warn(
                f"experiment_cols contain reserved names: {reserved_names_used}. Prepending with periods."
            )
        new_names = ["." + x for x in reserved_names_used]
        rename_dict = dict(zip(reserved_names_used, new_names))
        df = df.rename(rename_dict)
        experiment_cols = [rename_dict.get(x, x) for x in experiment_cols]
    return df, experiment_cols
