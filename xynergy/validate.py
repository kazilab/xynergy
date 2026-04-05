import polars as pl


def ensure_all_cols_in_df(df: pl.DataFrame, cols: list[str]):
    df_cols = set(df.columns)
    requested_cols = set(cols)
    missing_cols = requested_cols - df_cols
    if len(missing_cols) > 0:
        raise ValueError(f"Column(s) do not exist in dataset: {missing_cols}")
