"""Streamlit web app for xynergy drug synergy analysis."""

import io
import os
import tempfile

# Matplotlib needs a writable config directory in this environment.
MPLCONFIGDIR = os.path.join(tempfile.gettempdir(), "xynergy-mpl")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)

import matplotlib
import polars as pl
import streamlit as st
from xynergy._meta import (
    APP_NAME,
    APP_TAGLINE,
    CONTACT_EMAIL,
    COPYRIGHT_HOLDER,
    DEVELOPED_BY,
    __version__,
)

matplotlib.use("Agg")


def _module_is_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


def _factorization_result_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("resp_imputed_")]


HAS_CVXPY = _module_is_importable("cvxpy")

st.set_page_config(page_title=APP_NAME, page_icon="🧬", layout="wide")

st.title(APP_NAME)
st.markdown(APP_TAGLINE)
st.caption(
    f"Version {__version__} | {DEVELOPED_BY} | {CONTACT_EMAIL} | © {COPYRIGHT_HOLDER}"
)

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")

response_is_percent = st.sidebar.toggle(
    "Response is percentage (0–100)",
    value=True,
    help="Turn off if your response ranges from 0–1.",
)
complete_response_is_0 = st.sidebar.toggle(
    "Complete response is 0",
    value=False,
    help=(
        "Enable if a complete response (e.g. full killing) is represented as 0 "
        "in your data."
    ),
)

pre_impute_method = st.sidebar.selectbox(
    "Pre-imputation method",
    ["RBFSurface", "GaussianProcessSurface", "MatrixCompletion",
     "XGBR", "RandomForest", "LassoCV"],
    index=0,
)

pre_impute_target = st.sidebar.selectbox(
    "Pre-imputation target",
    ["response", "combo_effect", "ensemble"],
    index=0,
)

pre_impute_reference = st.sidebar.selectbox(
    "Pre-imputation reference (for combo_effect/ensemble)",
    ["bliss", "hsa"],
    index=0,
)

clip_response = st.sidebar.toggle("Clip response to 0–100", value=True)
clip_bounds = (0.0, 100.0) if clip_response else None

factorization_options = ["NMF", "SVD", "PMF"]
if HAS_CVXPY:
    factorization_options.insert(2, "RPCA")

factorization_method = st.sidebar.selectbox(
    "Factorization method",
    factorization_options,
    index=0,
)

if not HAS_CVXPY:
    st.sidebar.info("`cvxpy` is unavailable, so `RPCA` is excluded.")

synergy_methods = st.sidebar.multiselect(
    "Synergy methods",
    ["bliss", "hsa", "loewe", "zip"],
    default=["bliss", "hsa", "loewe", "zip"],
)

use_single_drug = st.sidebar.toggle(
    "Use single-drug response data", value=True
)

post_impute_tuning = st.sidebar.selectbox(
    "XGBoost Parameters",
    ["Predefined", "RandomizedSearchCV", "GridSearchCV"],
    index=0,
    help=(
        "Predefined: fixed params (fast). "
        "RandomizedSearchCV: sampled search (moderate). "
        "GridSearchCV: exhaustive search (slow)."
    ),
)

log_level = st.sidebar.selectbox("Log verbosity", ["all", "warn", "none"], index=0)

# ---------------------------------------------------------------------------
# Data input
# ---------------------------------------------------------------------------
st.header("1. Load data")

use_example = st.toggle("Use example data instead of uploading a file")

df: pl.DataFrame | None = None
dose_cols: list[str] = []
response_col: str = ""
experiment_cols: list[str] | None = None

if use_example:
    from xynergy.example import get_example_xynergy_kwargs, load_example_data

    example_defaults = get_example_xynergy_kwargs()
    df = load_example_data()
    dose_cols = example_defaults["dose_cols"]
    response_col = example_defaults["response_col"]
    experiment_cols = example_defaults["experiment_cols"]
    response_is_percent = example_defaults["response_is_percent"]
    complete_response_is_0 = example_defaults["complete_response_is_0"]
    st.sidebar.info(
        "Bundled workbook example loaded with canonical columns and "
        "inhibition-style responses."
    )
    st.success(f"Example data loaded — {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(50).to_pandas(), width="stretch")
else:
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls", "tsv"],
    )
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            df = pl.read_csv(uploaded)
        elif name.endswith(".tsv"):
            df = pl.read_csv(uploaded, separator="\t")
        else:
            import pandas as pd
            df = pl.from_pandas(pd.read_excel(uploaded))

        st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(50).to_pandas(), width="stretch")

        all_cols = df.columns

        col1, col2 = st.columns(2)
        with col1:
            dose_a = st.selectbox("Dose column A", all_cols, index=0)
        with col2:
            dose_b = st.selectbox(
                "Dose column B", all_cols,
                index=min(1, len(all_cols) - 1),
            )
        dose_cols = [dose_a, dose_b]

        response_col = st.selectbox(
            "Response column",
            all_cols,
            index=min(2, len(all_cols) - 1),
        )

        remaining = [c for c in all_cols if c not in dose_cols + [response_col]]
        experiment_cols = st.multiselect(
            "Experiment columns (optional — used to distinguish experiments)",
            remaining,
        ) or None

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
st.header("2. Run analysis")

if df is not None and dose_cols and response_col:
    if not synergy_methods:
        st.warning("Select at least one synergy method.")
    elif st.button("Run xynergy", type="primary", use_container_width=True):
        from xynergy.xynergy import xynergy

        with st.spinner("Running xynergy pipeline…"):
            result = xynergy(
                df=df,
                dose_cols=dose_cols,
                response_col=response_col,
                experiment_cols=experiment_cols,
                response_is_percent=response_is_percent,
                complete_response_is_0=complete_response_is_0,
                pre_impute_method=pre_impute_method,
                pre_impute_target=pre_impute_target,
                pre_impute_reference_for_target=pre_impute_reference,
                pre_impute_clip_response_bounds=clip_bounds,
                factorization_method=factorization_method,
                synergy_method=synergy_methods,
                use_single_drug_response_data=use_single_drug,
                post_impute_tuning=post_impute_tuning,
                log=log_level,
            )
        st.session_state["result"] = result

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result: pl.DataFrame = st.session_state["result"]

    st.header("3. Results")
    st.dataframe(result.head(200).to_pandas(), width="stretch")

    # --- Download button ---
    csv_buf = io.BytesIO()
    result.write_csv(csv_buf)
    st.download_button(
        "Download full results as CSV",
        data=csv_buf.getvalue(),
        file_name="xynergy_results.csv",
        mime="text/csv",
    )

    # --- Plots ---
    st.header("4. Visualizations")

    from xynergy.plot import plot_response_landscape, plot_synergy_3d

    experiments = result["experiment_id"].unique().sort().to_list()
    selected_exp = st.selectbox("Select experiment", experiments)
    exp_df = result.filter(pl.col("experiment_id") == selected_exp)

    # Response landscape
    st.subheader("Response landscape")
    tab_2d_resp, tab_3d_resp = st.tabs(["2D Heatmap", "3D Surface"])
    with tab_2d_resp:
        chart = plot_response_landscape(exp_df)
        st.altair_chart(chart, width="stretch")
    with tab_3d_resp:
        fig_3d_resp = plot_synergy_3d(
            exp_df, response_col="response", response_label="Response"
        )
        st.plotly_chart(fig_3d_resp, width="stretch")

    # Synergy landscapes
    syn_cols = [c for c in result.columns if c.endswith("_syn")]
    if syn_cols:
        st.subheader("Synergy landscapes")
        selected_syn = st.selectbox("Synergy method to plot", syn_cols)
        # Derive display name: "bliss_syn" -> "BLISS Synergy"
        syn_display = selected_syn.replace("_syn", "").upper() + " Synergy"
        tab_2d_syn, tab_3d_syn = st.tabs(["2D Heatmap", "3D Surface"])
        with tab_2d_syn:
            syn_chart = plot_response_landscape(
                exp_df,
                response_col=selected_syn,
                scheme="redblue",
                response_label=syn_display,
            )
            st.altair_chart(syn_chart, width="stretch")
        with tab_3d_syn:
            fig_3d_syn = plot_synergy_3d(
                exp_df,
                response_col=selected_syn,
                response_label=syn_display,
                colorscale="RdBu",
            )
            st.plotly_chart(fig_3d_syn, width="stretch")

    # Factorization landscapes
    factor_cols = _factorization_result_columns(result.columns)
    if factor_cols:
        st.subheader("Factorization approximations")
        selected_factor = st.selectbox("Factorization method to plot", factor_cols)
        tab_2d_fac, tab_3d_fac = st.tabs(["2D Heatmap", "3D Surface"])
        with tab_2d_fac:
            factor_chart = plot_response_landscape(
                exp_df,
                response_col=selected_factor,
                response_label=selected_factor,
            )
            st.altair_chart(factor_chart, width="stretch")
        with tab_3d_fac:
            fig_3d_fac = plot_synergy_3d(
                exp_df,
                response_col=selected_factor,
                response_label=selected_factor,
            )
            st.plotly_chart(fig_3d_fac, width="stretch")
else:
    st.info("Configure parameters in the sidebar, load data above, then click **Run xynergy**.")
