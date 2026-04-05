import altair as alt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.interpolate import griddata


def plot_synergy_3d(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    response_label: str = "Response",
    colorscale: str = "Viridis",
    interpolate: bool = True,
) -> go.Figure:
    """Create an interactive 3D surface plot of synergy or response data.

    Uses evenly-spaced index axes with dose tick labels, matching the
    traditional synergy surface layout where doses (which often span
    orders of magnitude) are spaced uniformly.

    Parameters
    ----------
    df : polars.DataFrame
        Must contain at least the two dose columns and one response column.
    dose_cols : list[str]
        Two column names for drug doses.
    response_col : str
        Column with response/synergy values.
    response_label : str
        Label for the Z-axis and color bar.
    colorscale : str
        Plotly colorscale name.
    interpolate : bool
        If True, interpolate onto a fine grid for a smooth surface.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    a, b = dose_cols

    # Aggregate duplicates
    pivot = (
        df.group_by(dose_cols)
        .agg(pl.col(response_col).mean())
        .sort(dose_cols)
    )

    dose_a_vals = np.sort(pivot[a].unique().to_numpy())
    dose_b_vals = np.sort(pivot[b].unique().to_numpy())

    # Build the Z matrix on index-based grid (like the matplotlib version)
    z_matrix = np.full((len(dose_a_vals), len(dose_b_vals)), np.nan)
    a_idx = {v: i for i, v in enumerate(dose_a_vals)}
    b_idx = {v: i for i, v in enumerate(dose_b_vals)}
    for row in pivot.iter_rows(named=True):
        z_matrix[a_idx[row[a]], b_idx[row[b]]] = row[response_col]

    # Use integer indices for evenly-spaced axes
    x_indices = np.arange(len(dose_b_vals))
    y_indices = np.arange(len(dose_a_vals))

    if interpolate and min(len(dose_a_vals), len(dose_b_vals)) >= 3:
        x_coarse, y_coarse = np.meshgrid(x_indices, y_indices)
        valid = ~np.isnan(z_matrix)
        x_fine = np.linspace(0, len(dose_b_vals) - 1, 50)
        y_fine = np.linspace(0, len(dose_a_vals) - 1, 50)
        x_fine_grid, y_fine_grid = np.meshgrid(x_fine, y_fine)
        z_fine = griddata(
            (x_coarse[valid], y_coarse[valid]),
            z_matrix[valid],
            (x_fine_grid, y_fine_grid),
            method="cubic",
        )
        surf_x = x_fine
        surf_y = y_fine
    else:
        surf_x = x_indices
        surf_y = y_indices
        z_fine = z_matrix

    # Format dose tick labels
    def _fmt(v):
        return f"{v:g}" if v != 0 else "0"

    a_ticktext = [_fmt(v) for v in dose_a_vals]
    b_ticktext = [_fmt(v) for v in dose_b_vals]

    fig = go.Figure(
        data=[
            go.Surface(
                x=surf_x,
                y=surf_y,
                z=z_fine,
                colorscale=colorscale,
                colorbar=dict(title=response_label),
                opacity=0.85,
            )
        ]
    )
    # Compute summary scores from the raw (non-interpolated) matrix
    valid_vals = z_matrix[~np.isnan(z_matrix)]
    score_min = float(np.min(valid_vals)) if valid_vals.size else 0.0
    score_max = float(np.max(valid_vals)) if valid_vals.size else 0.0
    score_avg = float(np.mean(valid_vals)) if valid_vals.size else 0.0

    title = (
        f"<b>{response_label}</b>"
        f"  —  Avg: {score_avg:.2f}  |  Min: {score_min:.2f}  |  Max: {score_max:.2f}"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(
                title=b,
                tickvals=list(range(len(dose_b_vals))),
                ticktext=b_ticktext,
            ),
            yaxis=dict(
                title=a,
                tickvals=list(range(len(dose_a_vals))),
                ticktext=a_ticktext,
            ),
            zaxis=dict(title=response_label),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=550,
    )
    return fig


# TODO allow for axis labels
def plot_response_landscape(
    df: pl.DataFrame,
    dose_cols: list[str] = ["dose_a", "dose_b"],
    response_col: str = "response",
    reference_col=None,
    color_min=None,
    color_mid=None,
    color_max=None,
    scheme="viridis",
    response_label="Response",
):
    """Plot a single experiment.

    Parameters
    ----------
    df: polars.DataFrame
        Contains, minimally, one response and two agent dose per row.

    dose_cols: list, default ["dose_a", "dose_b"]
        A list of exactly two columns names that contain numeric values of agent
        dose

    response_col: string, default "response"
        The name of the column containing responses

    reference_col: string, optional
        Column to subtract from `response_col`

    color_min: float, optional
        Value to set as the lower range for the color scale. If unspecified,
        defaults to 0 if `reference_col` is `None`, otherwise will default to
        minimum value plotted

    color_mid: float, optional
        Value to set as the middle of the range for the color scale.

        If unspecified:

        - If `reference_col` is `None`, will be the average of `color_min` and
          `color_max`
        - If `reference_col` is not `None`:
            - Will be 0 if that is within the bounds of the plotted values
            - Otherwise, will be average of `color_min` and `color_max`

    color_max: float, optional
        Value to set as the upper range for the color scale. If unspecified,
        defaults to 100 if `reference_col` is `None`, otherwise will default to
        maximum value plotted

    scheme: string, default "viridis"
        Color scheme for response colors. For a list of available schemes, see:
        https://vega.github.io/vega/docs/schemes/

    response_label: string, default "Response"
        What to label the color scale

    Returns
    -------
    An altair.Chart with the first dose column on the X-axis, the second dose
    column on the Y-axis, and a colored grid of squares corresponding to the
    response (optionally minus the reference)

    Notes
    -----
    Replicates with same dosepair concentrations will have their response's mean
    taken before ploting

    """
    # Silently take mean of duplicates
    if reference_col is None:
        df = df.group_by(dose_cols).agg(pl.col(response_col).mean())
    else:
        df = df.group_by(dose_cols).agg(
            pl.col(response_col).mean(),
            pl.col(reference_col).mean(),
        )

    a, b = dose_cols
    a_order = df[a].unique().sort().cast(str)
    b_order = df[b].unique().sort(descending=True).cast(str)

    df = df.with_columns(pl.col(dose_cols).cast(str))

    if reference_col is None:
        df = df.with_columns(pl.col(response_col).alias(response_label))
        if color_min is None:
            color_min = 0
        if color_max is None:
            color_max = 100
        if color_mid is None:
            color_mid = (color_max + color_min) / 2.0
    else:
        df = df.with_columns(
            (pl.col(response_col) - pl.col(reference_col)).alias(response_label)
        )
        if color_min is None:
            color_min = df[response_label].min()
        if color_max is None:
            color_max = df[response_label].max()
        if color_mid is None:
            if (0 > color_min) and (0 < color_max):
                color_mid = 0
            else:
                color_mid = (color_min + color_max) / 2.0

    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{a}:O").sort(a_order),
            y=alt.Y(f"{b}:O").sort(b_order),
            color=alt.Color(response_label).scale(
                domainMin=color_min,
                domainMid=color_mid,
                domainMax=color_max,
                scheme=scheme,
            ),
        )
    )
