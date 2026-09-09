"""Streamlit UI for time-series prefix backfill."""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from backfill import backfill_series, load_series_from_frame


st.set_page_config(
    page_title="Time Series Backfill",
    page_icon="📈",
    layout="wide",
)


def _demo_series() -> pd.DataFrame:
    """Synthetic weekday series (~9y), missing the first year of a 10y window."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2017-01-01", "2025-12-31")
    t = np.arange(len(idx))
    trend = 100 + 0.012 * t
    # Annual + Mon–Fri patterns on a business-day index
    seasonal = (
        8 * np.sin(2 * np.pi * t / 252)
        + 3 * np.sin(2 * np.pi * t / 5)
    )
    noise = rng.normal(0, 2.5, size=len(idx))
    for i in range(1, len(noise)):
        noise[i] = 0.4 * noise[i - 1] + noise[i]
    values = trend + seasonal + noise
    return pd.DataFrame({"date": idx, "value": values})


def _result_frame(result) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": result.series.index,
            "value": result.series.to_numpy(),
            "source": np.where(
                result.series.index.isin(result.generated.index),
                "generated",
                "observed",
            ),
        }
    )
    return df


def _plot_series(result) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.generated.index,
            y=result.generated.values,
            mode="lines",
            name="Generated",
            line=dict(color="#c45c26", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.observed.index,
            y=result.observed.values,
            mode="lines",
            name="Observed",
            line=dict(color="#1f4e79", width=1.6),
        )
    )
    if len(result.generated):
        junction = result.observed.index[0]
        fig.add_vline(
            x=junction,
            line_dash="dot",
            line_color="#666",
            annotation_text="start of observed",
            annotation_position="top left",
        )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
    )
    return fig


def _plot_distribution(result) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Value distribution", "Residuals pool"))

    fig.add_trace(
        go.Histogram(
            x=result.generated.values,
            name="Generated",
            opacity=0.65,
            marker_color="#c45c26",
            nbinsx=40,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=result.observed.values,
            name="Observed",
            opacity=0.55,
            marker_color="#1f4e79",
            nbinsx=40,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=result.residuals_pool.dropna().values,
            name="Residuals",
            marker_color="#5a7d5a",
            nbinsx=40,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        barmode="overlay",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0),
    )
    return fig


def main() -> None:
    st.title("Time Series Backfill")
    st.caption(
        "Generate a missing prefix while preserving seasonal structure and the "
        "historical residual distribution (STL + seasonal block bootstrap). "
        "By default only weekdays (Mon–Fri) are kept and generated."
    )

    with st.sidebar:
        st.header("Data")
        use_demo = st.toggle("Use demo series", value=True)
        uploaded = None
        if not use_demo:
            uploaded = st.file_uploader("CSV file", type=["csv"])

        st.header("Backfill settings")
        weekdays_only = st.toggle(
            "Weekdays only (Mon–Fri)",
            value=True,
            help="Drop weekends from the input and generate values only for business days.",
        )
        model = st.selectbox("Model", ["additive", "multiplicative"])
        auto_period = st.toggle("Auto season period", value=True)
        season_period = None
        if not auto_period:
            default_period = 5 if weekdays_only else 7
            season_period = st.number_input(
                "Season period", min_value=2, value=default_period, step=1
            )

        auto_block = st.toggle("Auto block size", value=True)
        block_size = None
        if not auto_block:
            default_block = 5 if weekdays_only else 7
            block_size = st.number_input(
                "Bootstrap block size", min_value=1, value=default_block, step=1
            )

        blend_len = st.number_input(
            "Seam blend length (0 = off)",
            min_value=0,
            value=0,
            step=1,
            help="Leave 0 to use the automatic default (≈ period/4).",
        )
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    # Load data
    if use_demo:
        raw = _demo_series()
        st.info(
            "Demo: **weekday** series from **2017-01-03** (first business day). "
            "Try desired start **2016-01-01** to synthesize the missing first year "
            "(weekends are skipped)."
        )
    elif uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        st.warning("Upload a CSV or enable the demo series.")
        st.stop()

    cols = list(raw.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox(
            "Date column",
            cols,
            index=0 if "date" not in [c.lower() for c in cols] else
            next(i for i, c in enumerate(cols) if c.lower() == "date"),
        )
    with c2:
        numeric_candidates = [
            c for c in cols if c != date_col and pd.api.types.is_numeric_dtype(raw[c])
        ]
        if not numeric_candidates:
            numeric_candidates = [c for c in cols if c != date_col]
        default_val = (
            next((i for i, c in enumerate(numeric_candidates) if c.lower() == "value"), 0)
            if numeric_candidates
            else 0
        )
        value_col = st.selectbox("Value column", numeric_candidates, index=default_val)
    with c3:
        st.write("")  # spacer

    try:
        series = load_series_from_frame(raw, date_col, value_col)
    except Exception as exc:
        st.error(f"Failed to parse series: {exc}")
        st.stop()

    obs_start = series.index.min().date()
    obs_end = series.index.max().date()
    default_desired = obs_start - timedelta(days=365)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Observed start", str(obs_start))
    with d2:
        st.metric("Observed end", str(obs_end))
    with d3:
        st.metric("Points", f"{len(series):,}")

    desired_start = st.date_input(
        "Desired series start date",
        value=default_desired,
        max_value=obs_start - timedelta(days=1),
        help="The completed series will begin on this date (or the next weekday "
        "if weekdays-only mode is on). The gap until the first observed date "
        "will be generated.",
    )

    run = st.button("Generate missing prefix", type="primary", use_container_width=False)

    if not run:
        st.subheader("Observed series preview")
        preview = go.Figure(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color="#1f4e79", width=1.4),
                name="Observed",
            )
        )
        preview.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Value",
        )
        st.plotly_chart(preview, use_container_width=True)
        with st.expander("Raw data head"):
            st.dataframe(raw.head(20), use_container_width=True)
        st.stop()

    blend_arg = None if blend_len == 0 else int(blend_len)

    with st.spinner("Backfilling…"):
        try:
            result = backfill_series(
                series,
                desired_start=pd.Timestamp(desired_start),
                model=model,
                season_period=int(season_period) if season_period else None,
                block_size=int(block_size) if block_size else None,
                blend_len=blend_arg,
                weekdays_only=bool(weekdays_only),
                random_state=int(random_state),
            )
        except Exception as exc:
            st.error(f"Backfill failed: {exc}")
            st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Generated points", f"{result.n_generated:,}")
    m2.metric("Frequency", result.freq)
    m3.metric("Season period", result.season_period)
    m4.metric("Model", result.model)

    if weekdays_only:
        weekend_count = int((result.series.index.dayofweek >= 5).sum())
        st.caption(
            f"Weekdays-only mode: output has **{weekend_count}** weekend rows "
            f"(expected 0)."
        )

    st.subheader("Completed series")
    st.plotly_chart(_plot_series(result), use_container_width=True)

    st.subheader("Distribution check")
    st.plotly_chart(_plot_distribution(result), use_container_width=True)

    out = _result_frame(result)
    st.subheader("Download")
    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button(
        "Download completed CSV",
        data=csv_buf.getvalue(),
        file_name="series_backfilled.csv",
        mime="text/csv",
    )

    with st.expander("Result table"):
        st.dataframe(out, use_container_width=True, height=320)

    with st.expander("Method"):
        st.markdown(
            """
1. Build a regular calendar from the desired start (**business days** when
   weekdays-only mode is on — weekends are dropped and never generated).
2. Decompose the observed part with **STL** (trend + seasonal + residual).
3. Extrapolate the **trend** backward with a local linear fit.
4. Replay the average **seasonal** profile.
5. Fill residuals via a **seasonal block bootstrap** from historical residuals
   (preserves marginal distribution and short-range dependence).
6. Optionally **blend** the seam with the first observed value.
            """
        )


if __name__ == "__main__":
    main()
