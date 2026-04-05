"""Matplotlib-based plotting functions for dose-response and synergy visualization.

These complement the Altair-based
``plot_response_landscape`` in ``plot.py``.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata

matplotlib.use("Agg")


def xplot(
    ic50,
    dose_interpolated,
    plot_min,
    plot_max,
    log10con,
    inhibition,
    inhibition_sem,
    log10dose_fine,
    fitted_curve_fine,
    conversion_unit,
    dose_interpolated_at,
    experiment,
    cell_line,
    drug_name,
    set_baseline_for_auc=0,
    set_integration_limit=1,
):
    """Plot a single-drug dose-response curve with IC50 annotation.

    Parameters
    ----------
    ic50 : float
        IC50 value (untransformed).
    dose_interpolated : float
        Dose at which interpolation is performed.
    plot_min, plot_max : float
        Y-axis limits for the response.
    log10con : array-like
        Log10-transformed drug concentrations.
    inhibition : array-like
        Observed response values.
    inhibition_sem : array-like or None
        SEM for response values (None to omit error bars).
    log10dose_fine : array-like
        Fine-grained log10 doses for smooth curve plotting.
    fitted_curve_fine : array-like
        Fitted curve values at log10dose_fine.
    conversion_unit : float
        Concentration scaling factor.
    dose_interpolated_at : float
        Percentage at which interpolation was performed.
    experiment : str
        Experiment identifier.
    cell_line : str
        Cell line name.
    drug_name : str
        Drug name.
    set_baseline_for_auc : float, default 0
        Baseline for AUC visualization.
    set_integration_limit : float, default 1
        Integration window in log10 units.

    Returns
    -------
    matplotlib.figure.Figure
        The dose-response plot.
    """
    min_scale = min(0, plot_min - 2)
    max_scale = max(100, plot_max + 2)
    log10dose_at_interpolated = np.log10(dose_interpolated * conversion_unit)
    log10ic50 = np.log10(ic50 * conversion_unit)

    fig, ax = plt.subplots()
    ax.set_ylim([min_scale, max_scale])

    if inhibition_sem is not None:
        ax.errorbar(
            log10con, inhibition, yerr=inhibition_sem,
            fmt="o", color="red", label="Actual Data",
        )
    else:
        ax.scatter(log10con, inhibition, color="red", label="Actual Data")

    ax.plot(
        log10dose_fine, fitted_curve_fine,
        color="lightblue", label="Fitted Logistic Curve",
    )

    ax.axvline(
        log10dose_at_interpolated, color="green", linestyle="--",
        label=f"Interpolated at {dose_interpolated_at}%",
    )
    ax.axvline(log10ic50, color="blue", linestyle="--", label="IC50")

    min_point = min_scale + 1
    mid_point = min_scale + 45
    ax.annotate(
        f"Interp@ {dose_interpolated_at}% = {dose_interpolated:.3f}",
        (log10dose_at_interpolated, mid_point), color="green", rotation=90,
    )
    ax.annotate(
        f"IC50 = {ic50:.3f}", (log10ic50, min_point), color="purple", rotation=90
    )

    ax.set_xlabel("Log10 Concentration (Molar)")
    ax.set_ylabel("Inhibition %")
    ax.set_title(f"Curve for Exp: {experiment} Cell line: {cell_line} Drug: {drug_name}")
    ax.legend()
    plt.tight_layout()
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Synergy surface / contour helpers
# ---------------------------------------------------------------------------

def _prepare_data_surface(data):
    """Sort and normalize data for surface plotting."""
    index_sorted = data.sort_index(ascending=True)
    sorted_data = index_sorted[sorted(index_sorted.columns, reverse=True)]
    vmin = np.nanmin(sorted_data.values)
    vmax = np.nanmax(sorted_data.values)
    avg = np.nanmean(sorted_data.values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return sorted_data, vmin, vmax, avg, norm


def _prepare_data_contour(data):
    """Sort and normalize data for contour plotting."""
    index_sorted = data.sort_index(ascending=True)
    sorted_data = index_sorted[sorted(index_sorted.columns, reverse=False)]
    vmin = np.nanmin(sorted_data.values)
    vmax = np.nanmax(sorted_data.values)
    avg = np.nanmean(sorted_data.values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return sorted_data, vmin, vmax, avg, norm


def _interpolate_data(dataframe):
    """Interpolate a dataframe onto a fine grid for smooth plotting."""
    x = np.arange(dataframe.shape[1])
    y = np.arange(dataframe.shape[0])
    x, y = np.meshgrid(x, y)
    z = dataframe.values

    x_fine = np.linspace(0, dataframe.shape[1] - 1, 300)
    y_fine = np.linspace(0, dataframe.shape[0] - 1, 300)
    x_fine, y_fine = np.meshgrid(x_fine, y_fine)
    z_fine = griddata(
        (x.flatten(), y.flatten()), z.flatten(), (x_fine, y_fine), method="cubic"
    )
    return x_fine, y_fine, z_fine


def _format_func(value, tick_number):
    return f"{value:.2f}"


def _plot_surface(
    dataframe, ax, x_fine, y_fine, z_fine, norm, vmin, vmax, avg,
    scoring_method, factor_1, factor_2, unit_1, unit_2,
):
    """Render a 3D surface on the given axes."""
    surf = ax.plot_surface(
        x_fine, y_fine, z_fine, cmap="viridis", edgecolor="none", norm=norm, alpha=0.75
    )
    ax.set_xlabel(f"{factor_2} ({unit_2})")
    ax.set_ylabel(f"{factor_1} ({unit_1})")
    ax.set_zlabel("score")
    ax.tick_params(axis="z", which="major", pad=0, labelsize=12, labelcolor="red")
    ax.set_xticks(range(len(dataframe.columns)))
    ax.set_xticklabels(dataframe.columns, rotation=0)
    ax.tick_params(axis="x", which="major", pad=0, labelsize=8, labelcolor="red")
    ax.set_yticks(range(len(dataframe.index)))
    ax.set_yticklabels(dataframe.index)
    ax.tick_params(axis="y", which="major", pad=0, labelsize=8, labelcolor="red")
    ax.set_title(
        f"Average {scoring_method} score: {avg:.2f} (min:{vmin:.2f} max:{vmax:.2f})",
        fontsize=10,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_func))
    plt.colorbar(
        surf, ax=ax, shrink=0.2, aspect=5,
        orientation="horizontal", location="bottom", pad=0.2,
    ).set_label("score", loc="center")
    return surf


def _plot_contour(
    dataframe, ax, x_fine, y_fine, z_fine, norm, vmin, vmax, avg,
    scoring_method, factor_1, factor_2, unit_1, unit_2,
):
    """Render a contour plot on the given axes."""
    contour = ax.contourf(
        x_fine, y_fine, z_fine, cmap="viridis", levels=10, norm=norm, alpha=0.5
    )
    ax.set_xlabel(f"{factor_2} ({unit_2})")
    ax.set_ylabel(f"{factor_1} ({unit_1})")
    ax.set_xticks(range(len(dataframe.columns)))
    ax.set_xticklabels(dataframe.columns, rotation=0)
    ax.tick_params(axis="x", which="major", pad=0, labelsize=8, labelcolor="red")
    ax.set_yticks(range(len(dataframe.index)))
    ax.set_yticklabels(dataframe.index)
    ax.set_title(
        f"Average {scoring_method} score: {avg:.2f} (min:{vmin:.2f} max:{vmax:.2f})",
        fontsize=10,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_func))
    ax.tick_params(axis="y", which="major", pad=0, labelsize=8, labelcolor="red")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=0.2, aspect=5).set_label("score", loc="bottom")
    return contour


# ---------------------------------------------------------------------------
# Public multi-panel synergy plots
# ---------------------------------------------------------------------------

def synergy_plots(
    bliss_over,
    loewe_over,
    hsa_over,
    zip_over,
    bioactive_factor_1="Drug A",
    bioactive_factor_2="Drug B",
    unit_bioactive_factor_1="nM",
    unit_bioactive_factor_2="nM",
):
    """Create a 4x2 grid of synergy plots (surface + contour) for all four methods.

    Parameters
    ----------
    bliss_over, loewe_over, hsa_over, zip_over : pandas.DataFrame
        Synergy score matrices (rows = drug A doses, columns = drug B doses).
    bioactive_factor_1, bioactive_factor_2 : str
        Drug names for axis labels.
    unit_bioactive_factor_1, unit_bioactive_factor_2 : str
        Concentration units for axis labels.

    Returns
    -------
    matplotlib.figure.Figure
        The multi-panel figure.
    """
    fig = plt.figure(figsize=(12, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig)

    dataframes = {
        "BLISS": bliss_over,
        "LOEWE": loewe_over,
        "HSA": hsa_over,
        "ZIP": zip_over,
    }
    plot_index = 0

    for name, df in dataframes.items():
        sorted_s, vmin_s, vmax_s, avg_s, norm_s = _prepare_data_surface(df)
        sorted_c, vmin_c, vmax_c, avg_c, norm_c = _prepare_data_contour(df)
        x_s, y_s, z_s = _interpolate_data(sorted_s)
        x_c, y_c, z_c = _interpolate_data(sorted_c)

        # Surface
        ax = fig.add_subplot(gs[plot_index], projection="3d")
        _plot_surface(
            sorted_s, ax, x_s, y_s, z_s, norm_s, vmin_s, vmax_s, avg_s, name,
            bioactive_factor_1, bioactive_factor_2,
            unit_bioactive_factor_1, unit_bioactive_factor_2,
        )
        plot_index += 1

        # Contour
        ax = fig.add_subplot(gs[plot_index])
        _plot_contour(
            sorted_c, ax, x_c, y_c, z_c, norm_c, vmin_c, vmax_c, avg_c, name,
            bioactive_factor_1, bioactive_factor_2,
            unit_bioactive_factor_1, unit_bioactive_factor_2,
        )
        plot_index += 1

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout(pad=1.5, w_pad=1)
    return fig


def synergy2plots(
    dataframe,
    bioactive_factor_1="Drug A",
    bioactive_factor_2="Drug B",
    unit_bioactive_factor_1="nM",
    unit_bioactive_factor_2="nM",
    scoring_method="Synergy",
):
    """Create a comprehensive 6-panel synergy plot for a single method.

    Panels: 3D surface, contour, heatmap, 3D bar, bubble (synergy), bubble
    (absolute).

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Synergy score matrix (rows = drug A doses, columns = drug B doses).
    bioactive_factor_1, bioactive_factor_2 : str
        Drug names for axis labels.
    unit_bioactive_factor_1, unit_bioactive_factor_2 : str
        Concentration units for axis labels.
    scoring_method : str
        Name of the synergy method (used in titles).

    Returns
    -------
    matplotlib.figure.Figure
        The 6-panel figure.
    """
    dataframe = dataframe.sort_index(ascending=True)
    dataframe = dataframe[sorted(dataframe.columns, reverse=True)]
    vmin = np.nanmin(dataframe.values)
    vmax = np.nanmax(dataframe.values)
    avg = np.nanmean(dataframe.values)
    norm = Normalize(vmin=vmin, vmax=vmax)

    x = np.arange(dataframe.shape[1])
    y = np.arange(dataframe.shape[0])
    x, y = np.meshgrid(x, y)
    z = dataframe.values

    x_fine = np.linspace(0, dataframe.shape[1] - 1, 300)
    y_fine = np.linspace(0, dataframe.shape[0] - 1, 300)
    x_fine, y_fine = np.meshgrid(x_fine, y_fine)
    z_fine = griddata(
        (x.flatten(), y.flatten()), z.flatten(), (x_fine, y_fine), method="cubic"
    )

    title = f"Average {scoring_method} score: {avg:.2f} (min:{vmin:.2f} max:{vmax:.2f})"
    f1, f2, u1, u2 = (
        bioactive_factor_1, bioactive_factor_2,
        unit_bioactive_factor_1, unit_bioactive_factor_2,
    )

    fig = plt.figure(figsize=(18, 12))

    # 1) 3D Surface
    ax1 = fig.add_subplot(231, projection="3d")
    surf = ax1.plot_surface(
        x_fine, y_fine, z_fine, cmap="viridis", edgecolor="none", norm=norm, alpha=0.75
    )
    ax1.set_xlabel(f"{f2} ({u2})")
    ax1.set_ylabel(f"{f1} ({u1})")
    ax1.set_zlabel("score")
    ax1.set_xticks(range(len(dataframe.columns)))
    ax1.set_xticklabels(dataframe.columns)
    ax1.set_yticks(range(len(dataframe.index)))
    ax1.set_yticklabels(dataframe.index)
    for axis in ["x", "y", "z"]:
        ax1.tick_params(axis=axis, labelsize=8, labelcolor="red")
    ax1.set_title(title, fontsize=10)
    plt.colorbar(
        surf, ax=ax1, shrink=0.2, aspect=5,
        orientation="horizontal", location="bottom", pad=0.2,
    ).set_label("score")

    # 2) Contour
    df_c = dataframe.sort_index(ascending=True)
    df_c = df_c[sorted(df_c.columns, reverse=False)]
    ax2 = fig.add_subplot(232)
    contour = ax2.contourf(
        x_fine, y_fine, z_fine, cmap="viridis", levels=10, norm=norm, alpha=0.5
    )
    ax2.set_xlabel(f"{f2} ({u2})")
    ax2.set_ylabel(f"{f1} ({u1})")
    ax2.set_xticks(range(len(df_c.columns)))
    ax2.set_xticklabels(df_c.columns)
    ax2.set_yticks(range(len(df_c.index)))
    ax2.set_yticklabels(df_c.index)
    for axis in ["x", "y"]:
        ax2.tick_params(axis=axis, labelsize=8, labelcolor="red")
    ax2.set_title(title, fontsize=10)
    ax2.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax2, shrink=0.2, aspect=5).set_label("score")

    # 3) Heatmap
    df_h = dataframe.sort_index(ascending=False)
    df_h = df_h[sorted(df_h.columns, reverse=False)]
    ax3 = fig.add_subplot(233)
    heatmap = ax3.imshow(
        df_h, cmap="viridis", aspect="equal", interpolation="nearest",
        norm=norm, alpha=0.5,
    )
    ax3.set_title(title, fontsize=10)
    ax3.set_xlabel(f"{f2} ({u2})")
    ax3.set_ylabel(f"{f1} ({u1})")
    ax3.set_xticks(range(len(df_h.columns)))
    ax3.set_xticklabels(df_h.columns)
    ax3.set_yticks(range(len(df_h.index)))
    ax3.set_yticklabels(df_h.index)
    for axis in ["x", "y"]:
        ax3.tick_params(axis=axis, labelsize=8, labelcolor="red")
    plt.colorbar(heatmap, ax=ax3, shrink=0.2, aspect=5).set_label("score")

    # 4) 3D Bar
    ax4 = fig.add_subplot(234, projection="3d")
    bar_norm = Normalize(vmin=z.min(), vmax=z.max())
    colors = cm.viridis(bar_norm(z.flatten()))
    ax4.bar3d(
        x.flatten(), y.flatten(), np.zeros_like(z.flatten()),
        0.5, 0.5, z.flatten(), color=colors, alpha=0.5,
    )
    ax4.set_title(title, fontsize=10)
    ax4.set_xlabel(f"{f2} ({u2})")
    ax4.set_ylabel(f"{f1} ({u1})")
    ax4.set_xticks(range(len(dataframe.columns)))
    ax4.set_xticklabels(dataframe.columns)
    ax4.set_yticks(range(len(dataframe.index)))
    ax4.set_yticklabels(dataframe.index)
    for axis in ["x", "y", "z"]:
        ax4.tick_params(axis=axis, labelsize=8, labelcolor="red")
    plt.colorbar(
        cm.ScalarMappable(norm=bar_norm, cmap="viridis"),
        ax=ax4, shrink=0.2, aspect=5,
    ).set_label("score")

    # 5) Bubble chart
    df_b = dataframe.sort_index(ascending=True)
    df_b = df_b[sorted(df_b.columns, reverse=False)]
    x_pos = np.repeat(range(len(df_b.columns)), len(df_b.index))
    y_pos = np.tile(range(len(df_b.index)), len(df_b.columns))
    sizes = df_b.values.flatten() * 100
    colors_b = df_b.values.flatten()

    ax5 = fig.add_subplot(235)
    bubble = ax5.scatter(x_pos, y_pos, s=sizes, c=colors_b, cmap="viridis", alpha=0.5)
    ax5.set_title(title, fontsize=10)
    ax5.set_xlabel(f"{f2} ({u2})")
    ax5.set_ylabel(f"{f1} ({u1})")
    ax5.set_xticks(range(len(df_b.columns)))
    ax5.set_xticklabels(df_b.columns)
    ax5.set_yticks(range(len(df_b.index)))
    ax5.set_yticklabels(df_b.index)
    for axis in ["x", "y"]:
        ax5.tick_params(axis=axis, labelsize=8, labelcolor="red")
    ax5.set_aspect("equal", adjustable="box")
    plt.colorbar(bubble, ax=ax5, shrink=0.2, aspect=5).set_label("score")

    # 6) Bubble chart (absolute sizes)
    sizes_abs = np.abs(df_b.values.flatten()) * 100
    ax6 = fig.add_subplot(236)
    ax6.scatter(x_pos, y_pos, s=sizes_abs, c=colors_b, cmap="viridis", alpha=0.5)
    ax6.set_title(title, fontsize=10)
    ax6.set_xlabel(f"{f2} ({u2})")
    ax6.set_ylabel(f"{f1} ({u1})")
    ax6.set_xticks(range(len(df_b.columns)))
    ax6.set_xticklabels(df_b.columns)
    ax6.set_yticks(range(len(df_b.index)))
    ax6.set_yticklabels(df_b.index)
    for axis in ["x", "y"]:
        ax6.tick_params(axis=axis, labelsize=8, labelcolor="red")
    ax6.set_aspect("equal", adjustable="box")
    abs_norm = Normalize(vmin=np.min(colors_b), vmax=np.max(colors_b))
    sm = cm.ScalarMappable(cmap="viridis", norm=abs_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax6, shrink=0.2, aspect=5).set_label("score")

    plt.tight_layout()
    return fig
