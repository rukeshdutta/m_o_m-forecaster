import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Scenario Planner (m/m + y/y Weighted)", layout="wide")

st.title("Scenario Planner — Weighted m/m & y/y (12‑month Forecast)")
st.write(
    """
Upload a long-format dataset with a date column, one value column, and one or more dimension columns (plus an optional metric column). This app forecasts the next 12 months for each (dimension[, metric]) group using a weighted blend of:

m/m seasonal factor: average of the last X years' month-to-month ratio for the same calendar transition.

y/y momentum: average of the last Y months' year-over-year ratios (using actual + forecasted values), applied to the value from the same month one year prior.


The baseline forecast is weight_mm * m/m_projection + weight_yoy * y/y_projection. Then we produce -5% and +5% scenarios around baseline.
"""
)

# =====================
# Helper Functions
# =====================

def normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        st.warning("Some dates could not be parsed and were dropped.")
        df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp("M")
    return df

def last_yoy_avg(values_period: pd.Series, last_period: pd.Period, debug_log: list, forecast_month: pd.Period, periods: int) -> float:
    months = [last_period - i for i in range(0, periods)]
    ratios = []
    for m in months:
        prev_year = m - 12
        if m in values_period.index and prev_year in values_period.index:
            num = values_period.loc[m]
            den = values_period.loc[prev_year]
            if pd.notna(num) and pd.notna(den) and den != 0:
                ratios.append(num / den)
                debug_log.append(f"{forecast_month}: YoY ratio {m} vs {prev_year} = {num}/{den} = {num/den:.3f}")
    if not ratios:
        return np.nan
    return float(np.mean(ratios))

def seasonal_mm_avg(values_period: pd.Series, target_period: pd.Period, debug_log: list, periods: int) -> float:
    pairs = []
    for k in range(1, periods + 1):
        this_year = target_period - 12 * k
        prev_month = this_year - 1
        if this_year in values_period.index and prev_month in values_period.index:
            num = values_period.loc[this_year]
            den = values_period.loc[prev_month]
            if pd.notna(num) and pd.notna(den) and den != 0:
                pairs.append(num / den)
                debug_log.append(f"{target_period}: m/m ratio {this_year} vs {prev_month} = {num}/{den} = {num/den:.3f}")
    if len(pairs) == 0:
        return np.nan
    return float(np.mean(pairs))

def compute_group_forecast(group_df: pd.DataFrame, date_col: str, value_col: str, horizon: int, w_mm: float, w_yoy: float, yoy_periods: int, mm_periods: int, debug: bool=False):
    g = group_df[[date_col, value_col]].dropna().copy()
    g = g.sort_values(date_col)

    s = g.set_index(date_col)[value_col].copy()
    s.index = s.index.to_period('M')
    idx = pd.period_range(s.index.min(), s.index.max(), freq='M')
    s = s.reindex(idx)

    last_actual = s.last_valid_index()
    if last_actual is None:
        return pd.DataFrame(), []

    values = s.copy()
    forecasts = []
    debug_log = []

    for h in range(1, horizon + 1):
        t = last_actual + h

        # Pass the new 'periods' arguments to the helper functions
        yoy_avg = last_yoy_avg(values, t - 1, debug_log, t, periods=yoy_periods)
        mm_ratio = seasonal_mm_avg(values, t, debug_log, periods=mm_periods)

        mm_proj = np.nan
        prev_period = t - 1
        if pd.notna(mm_ratio) and prev_period in values.index:
            prev_val = values.loc[prev_period]
            if pd.notna(prev_val):
                mm_proj = prev_val * mm_ratio

        yoy_proj = np.nan
        if pd.notna(yoy_avg):
            same_month_prev_year = t - 12
            if same_month_prev_year in values.index:
                base = values.loc[same_month_prev_year]
                if pd.notna(base):
                    yoy_proj = base * yoy_avg

        if pd.isna(mm_proj) and pd.isna(yoy_proj):
            baseline = np.nan
        elif pd.isna(mm_proj):
            baseline = yoy_proj
        elif pd.isna(yoy_proj):
            baseline = mm_proj
        else:
            baseline = w_mm * mm_proj + w_yoy * yoy_proj

        values.loc[t] = baseline

        downside = baseline * 0.95 if pd.notna(baseline) else np.nan
        upside = baseline * 1.05 if pd.notna(baseline) else np.nan

        forecasts.append({
            "date": t.to_timestamp('M'),
            "mm_component": mm_proj,
            "yoy_component": yoy_proj,
            "baseline": baseline,
            "downside_-5pct": downside,
            "upside_+5pct": upside,
        })

    out = pd.DataFrame(forecasts)
    return out, debug_log if debug else []

# =====================
# Sidebar — Inputs
# =====================

with st.sidebar:
    st.header("1) Upload your data")
    file = st.file_uploader("CSV or Excel (long format)", type=["csv", "xlsx", "xls"])

    if file is None:
        st.info("Upload a file to begin. A minimal example is shown below.")
        example = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=48, freq="MS"),
            "channel": np.random.choice(["corp", "retail", "dealer"], size=48),
            "metric": "AGA",
            "value": np.random.randint(100, 1000, size=48)
        })
        st.dataframe(example.head(12))
        st.stop()

    try:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        columns = df.columns.tolist()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.header("2) Select columns")
    date_col = st.selectbox(
        "Date column name",
        options=columns,
        index=columns.index('date') if 'date' in columns else 0
    )
    value_col = st.selectbox(
        "Value column name",
        options=columns,
        index=columns.index('value') if 'value' in columns else 0
    )

    metric_col = st.text_input("Metric column (optional)", value="metric")
    dims_raw = st.text_input("Dimension columns (comma-separated)", value="channel")
    horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=12)

    st.header("3) Weights & Periods")
    st.caption("Weights must sum to 1. Defaults to 0.5 / 0.5.")
    default_w_mm = st.number_input("Default weight — m/m", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    default_w_yoy = 1.0 - default_w_mm
    st.write(f"Default weight — y/y: **{default_w_yoy:.2f}**")

    # User inputs for forecast periods
    yoy_periods = st.number_input(
        "Number of months for y/y average",
        min_value=1,
        value=3,
        help="Number of months to average for the y/y momentum ratio."
    )
    mm_periods = st.number_input(
        "Number of years for m/m average",
        min_value=1,
        value=3,
        help="Number of years to average for the m/m seasonal ratio."
    )

    st.header("4) Scenarios")
    st.caption("Scenarios are computed off baseline: -5% and +5%.")

    debug_mode = st.checkbox("Enable debug panel (show calculation steps)")


if date_col not in df.columns or value_col not in df.columns:
    st.error("Selected date and value columns must exist in file.")
    st.stop()

dim_cols = [c.strip() for c in dims_raw.split(",") if c.strip()]
for c in dim_cols:
    if c not in df.columns:
        st.error(f"Dimension column '{c}' not found.")
        st.stop()

has_metric = metric_col in df.columns
_df = normalize_dates(df, date_col)

group_cols = dim_cols.copy()
if has_metric:
    group_cols.append(metric_col)

if len(group_cols) == 0:
    st.error("Please specify at least one dimension or a metric column to group by.")
    st.stop()

st.subheader("Data preview & coverage")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.dataframe(_df.head(20))
with col_b:
    st.write("Rows:", len(_df))
    st.write("Date range:", str(_df[date_col].min().date()) , "→", str(_df[date_col].max().date()))

st.subheader("Per-group weight overrides (optional)")
unique_groups = _df[group_cols].drop_duplicates().reset_index(drop=True)
unique_groups["weight_mm"] = default_w_mm
unique_groups["weight_yoy"] = 1.0 - unique_groups["weight_mm"]

edited = st.data_editor(unique_groups, num_rows="fixed", use_container_width=True, key="weights_editor")

if (edited[["weight_mm", "weight_yoy"]].sum(axis=1).round(6) != 1.0).any():
    st.warning("Some rows have weights that don't sum to 1. They will be normalized.")
    sums = edited[["weight_mm", "weight_yoy"]].sum(axis=1)
    edited["weight_mm"] = edited["weight_mm"] / sums
    edited["weight_yoy"] = edited["weight_yoy"] / sums

st.subheader("Forecast results")
all_out = []
all_debug = {}

for grp_vals, gdf in _df.groupby(group_cols):
    gdf = gdf.sort_values(date_col)
    if not isinstance(grp_vals, tuple):
        grp_vals = (grp_vals,)
    selector = pd.Series(True, index=edited.index)
    for col, val in zip(group_cols, grp_vals):
        selector &= edited[col] == val
    row = edited[selector]
    if row.empty:
        w_mm = default_w_mm
        w_yoy = 1.0 - w_mm
    else:
        w_mm = float(row.iloc[0]["weight_mm"])
        w_yoy = float(row.iloc[0]["weight_yoy"])

    out, debug_log = compute_group_forecast(
        gdf, date_col, value_col,
        horizon=int(horizon),
        w_mm=w_mm, w_yoy=w_yoy,
        yoy_periods=int(yoy_periods),
        mm_periods=int(mm_periods),
        debug=debug_mode
    )
    if out.empty:
        continue
    for col, val in zip(group_cols, grp_vals):
        out[col] = val
    all_out.append(out)
    if debug_mode:
        all_debug[grp_vals] = debug_log

if len(all_out) == 0:
    st.error("No forecasts could be generated. Check date/value coverage for each group.")
    st.stop()

fcst = pd.concat(all_out, ignore_index=True)
fcst = fcst[[*group_cols, "date", "mm_component", "yoy_component", "baseline", "downside_-5pct", "upside_+5pct"]]

st.dataframe(fcst.sort_values(group_cols + ["date"]).reset_index(drop=True), use_container_width=True)

csv_bytes = fcst.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download forecasts (CSV)",
    data=csv_bytes,
    file_name="scenario_forecasts.csv",
    mime="text/csv",
)

st.subheader("Quick chart")
if len(unique_groups) > 0:
    options = [tuple(r[group_cols].values) for _, r in unique_groups.iterrows()]
    sel = st.selectbox("Pick a group to visualize", options=options, format_func=lambda t: " | ".join(map(str, t)))

    mask = pd.Series(True, index=df.index)
    for col, val in zip(group_cols, sel):
        mask &= (df[col] == val)
    hist = df.loc[mask, [date_col, value_col]].sort_values(date_col)

    fmask = pd.Series(True, index=fcst.index)
    for col, val in zip(group_cols, sel):
        fmask &= (fcst[col] == val)

    fut = fcst.loc[fmask, ["date", "baseline", "downside_-5pct", "upside_+5pct"]].sort_values("date")

    left, right = st.columns(2)
    with left:
        st.caption("Actuals (history)")
        st.line_chart(hist.set_index(date_col))
    with right:
        st.caption("Forecast scenarios (12 months)")
        st.line_chart(fut.set_index("date"))

if debug_mode and len(all_debug) > 0:
    st.subheader("Debug panel — Calculation steps")
    sel_debug = st.selectbox("Pick a group to inspect", options=list(all_debug.keys()), format_func=lambda t: " | ".join(map(str, t)))
    if sel_debug in all_debug:
        st.text("\n".join(all_debug[sel_debug]))

st.success("Done. Forecast now cascades predicted months in both m/m and y/y components, with debug logging available.")