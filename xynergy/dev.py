import polars as pl
from xynergy.util import _add_id_if_no_experiment_cols, make_list_if_str_or_none
from xynergy.validate import ensure_all_cols_in_df


def rm_off_axis(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    experiment_cols: list[str] | str = "experiment_id",
):
    experiment_cols = make_list_if_str_or_none(experiment_cols)
    dose_cols = make_list_if_str_or_none(dose_cols)

    if len(dose_cols) != 2:
        raise ValueError("Length of dose_cols must be exactly 2")

    ensure_all_cols_in_df(df, dose_cols + experiment_cols)

    df, experiment_cols, added_dummy = _add_id_if_no_experiment_cols(
        df, experiment_cols
    )
    grouped = df.group_by(experiment_cols)
    doses = grouped.agg(pl.col(dose_cols).unique().sort())
    mismatched_doses = doses.with_columns(pl.col(dose_cols).list.len()).filter(
        pl.col(dose_cols[0]) != pl.col(dose_cols[1])
    )
    if len(mismatched_doses) > 0:
        raise ValueError("Number of unique doses in both drugs must be the same")
    diags = doses.explode(dose_cols)
    min_doses = (
        doses.with_columns(
            pl.col(dose_cols).list.first().name.suffix("_min"),
        )
        .explode(dose_cols)
        .with_columns(
            pl.struct(pl.col(dose_cols[0], dose_cols[1] + "_min")).alias("dosepair1"),
            pl.struct(pl.col(dose_cols[0] + "_min", dose_cols[1])).alias("dosepair2"),
        )
        .with_columns(pl.col("dosepair1", "dosepair2").struct.rename_fields(dose_cols))
        .unpivot(on=["dosepair1", "dosepair2"], index=experiment_cols)
        .unnest("value")
    )
    to_keep = diags.join(
        min_doses, how="full", on=dose_cols + list(experiment_cols), coalesce=True
    )
    out = df.join(to_keep, how="semi", on=dose_cols + list(experiment_cols))

    if added_dummy:
        out = out.drop("experiment_id")

    return out
