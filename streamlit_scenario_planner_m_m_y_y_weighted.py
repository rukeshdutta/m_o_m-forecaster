import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Scenario Planner (m/m + y/y Weighted)", layout="wide")

st.title("Scenario Planner — Weighted m/m & y/y (12‑month Forecast)")
st.write(
    """
Upload a **long-format** dataset with a date column, one **value** column, and one or more **dimension** columns (plus an optional metric column).
This app forecasts the next **12 months** for each (dimension[, metric]) group using a weighted blend of:

- **m/m seasonal factor**: average of the last 3 years' month-to-month ratio for the same calendar transition.
- **y/y momentum**: average of the **last 3 actual months** year-over-year ratios, applied to the value from the same month **one year prior**.

The baseline forecast is `weight_mm * m/m_projection + weight_yoy * y/y_projection`. Then we produce **-5%** and **+5%** scenarios around baseline.
"""
)

# =====================
# Helper Functions
# =====================

def _to_month_start(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)


def normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        st.warning("Some dates could not be parsed and were dropped.")
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp("M")
    # Use month end for storage, but we will also keep a month-start view for calc convenience
    return df


def last_three_months_yoy_avg(s: pd.Series) -> float:
    """Compute average YoY ratio over the last 3 actual months in a monthly indexed series.
    s is a Series indexed by month-end Timestamp with numeric values (actuals only).
    Returns np.nan if insufficient data.
    """
    if s.empty:
        return np.nan
    last_month = s.index.max()
    months = [last_month - relativedelta(months=i) for i in range(0, 3)]
    ratios = []
    for m in months:
        prev_year = m - relativedelta(years=1)
        if m in s.index and prev_year in s.index:
            denom = s.loc[prev_year]
            if pd.notna(s.loc[m]) and pd.notna(denom) and denom != 0:
                ratios.append(s.loc[m] / denom)
    if len(ratios) == 0:
        return np.nan
    return float(np.mean(ratios))


def seasonal_mm_avg_for_target(s: pd.Series, target_month: pd.Timestamp) -> float:
    """Average m/m ratio for the same calendar transition over up to the last 3 years.
    Example: target=2026-01 => use (2025-01/2024-12), (2024-01/2023-12), (2023-01/2022-12).
    s is actuals Series indexed by month-end Timestamp.
    """
    pairs = []
    for k in range(1, 4):  # last 3 years
        this_year = target_month - relativedelta(years=k)
        prev_month = this_year - relativedelta(months=1)
        if this_year in s.index and prev_month in s.index:
            num = s.loc[this_year]
            den = s.loc[prev_month]
            if pd.notna(num) and pd.notna(den) and den != 0:
                pairs.append(num / den)
    if len(pairs) == 0:
        return np.nan
    return float(np.mean(pairs))


def compute_group_forecast(group_df: pd.DataFrame, date_col: str, value_col: str, horizon: int,
                           w_mm: float, w_yoy: float) -> pd.DataFrame:
    """Compute 12-month forecast for a single group (one dimension/metric combo).
    group_df must contain actuals only.
    Returns a DataFrame with columns: [date, baseline, downside, upside, mm_component, yoy_component]
    """
    g = group_df[[date_col, value_col]].dropna().copy()
    g = g.sort_values(date_col)

    # reindex to monthly continuous index to simplify lookups
    idx = pd.period_range(g[date_col].min(), g[date_col].max(), freq="M").to_timestamp("M")
    s = g.set_index(date_col)[value_col].reindex(idx)

    last_actual = s.last_valid_index()
    if last_actual is None:
        return pd.DataFrame()

    yoy_avg = last_three_months_yoy_avg(s)

    forecasts = []
    values = s.copy()  # we'll append baseline forecasts for reference in m/m cascade

    for h in range(1, horizon + 1):
        t = last_actual + relativedelta(months=h)
        # Components
        mm_ratio = seasonal_mm_avg_for_target(s, t)
        mm_proj = np.nan
        if pd.notna(mm_ratio):
            prev_val = values.get(t - relativedelta(months=1), np.nan)
            if pd.notna(prev_val):
                mm_proj = prev_val * mm_ratio

        yoy_proj = np.nan
        if pd.notna(yoy_avg):
            same_month_prev_year = t - relativedelta(years=1)
            base = values.get(same_month_prev_year, np.nan)
            if pd.notna(base):
                yoy_proj = base * yoy_avg

        # Combine components
        if pd.isna(mm_proj) and pd.isna(yoy_proj):
            baseline = np.nan
        elif pd.isna(mm_proj):
            baseline = yoy_proj
        elif pd.isna(yoy_proj):
            baseline = mm_proj
        else:
            baseline = w_mm * mm_proj + w_yoy * yoy_proj

        # Cascade baseline so next month m/m can use it as prev
        values.loc[t] = baseline

        downside = baseline * 0.95 if pd.notna(baseline) else np.nan
        upside = baseline * 1.05 if pd.notna(baseline) else np.nan

        forecasts.append({
            "date": t,
            "mm_component": mm_proj,
            "yoy_component": yoy_proj,
            "baseline": baseline,
            "downside_-5pct": downside,
            "upside_+5pct": upside,
        })

    out = pd.DataFrame(forecasts)
    return out


# =====================
# Sidebar — Inputs
# =====================
with st.sidebar:
    st.header("1) Upload your data")
    file = st.file_uploader("CSV or Excel (long format)", type=["csv", "xlsx", "xls"])
    
    st.header("2) Select columns")
    date_col = st.text_input("Date column name", value="date")
    value_col = st.text_input("Value column name", value="value")
    metric_col = st.text_input("Metric column (optional)", value="metric")
    dims_raw = st.text_input("Dimension columns (comma-separated)", value="channel")
    horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=12)

    st.header("3) Weights (per group)")
    st.caption("Weights must sum to 1. Defaults to 0.5 / 0.5.")
    default_w_mm = st.number_input("Default weight — m/m", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    default_w_yoy = 1.0 - default_w_mm
    st.write(f"Default weight — y/y: **{default_w_yoy:.2f}**")

    st.header("4) Scenarios")
    st.caption("Scenarios are computed off baseline: -5% and +5%.")


if file is None:
    st.info("Upload a file to begin. A minimal example is shown below.")
    example = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=48, freq="MS"),
        "channel": np.random.choice(["Channel A", "Channel B", "Channel C"], size=48),
        "metric": "SALES",
        "value": np.random.randint(100, 1000, size=48)
    })
    st.dataframe(example.head(12))
    st.stop()

# Load data
try:
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Normalize date
if date_col not in df.columns:
    st.error(f"Date column '{date_col}' not found.")
    st.stop()

if value_col not in df.columns:
    st.error(f"Value column '{value_col}' not found.")
    st.stop()

# Parse dimension columns
dim_cols = [c.strip() for c in dims_raw.split(",") if c.strip()]
for c in dim_cols:
    if c not in df.columns:
        st.error(f"Dimension column '{c}' not found.")
        st.stop()

has_metric = metric_col in df.columns

# Prepare
_df = normalize_dates(df, date_col)

# Build group keys
group_cols = dim_cols.copy()
if has_metric:
    group_cols.append(metric_col)

if len(group_cols) == 0:
    st.error("Please specify at least one dimension or a metric column to group by.")
    st.stop()

# Show data summary
st.subheader("Data preview & coverage")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.dataframe(_df.head(20))
with col_b:
    st.write("**Rows:**", len(_df))
    st.write("**Date range:**", str(_df[date_col].min().date()) , "→", str(_df[date_col].max().date()))

# Per-group weights editor
st.subheader("Per-group weight overrides (optional)")
unique_groups = _df[group_cols].drop_duplicates().reset_index(drop=True)
unique_groups["weight_mm"] = default_w_mm
unique_groups["weight_yoy"] = 1.0 - unique_groups["weight_mm"]

edited = st.data_editor(unique_groups, num_rows="fixed", use_container_width=True, key="weights_editor")

# Validate weights
if (edited[["weight_mm", "weight_yoy"]].sum(axis=1).round(6) != 1.0).any():
    st.warning("Some rows have weights that don't sum to 1. They will be normalized.")
    sums = edited[["weight_mm", "weight_yoy"]].sum(axis=1)
    edited["weight_mm"] = edited["weight_mm"] / sums
    edited["weight_yoy"] = edited["weight_yoy"] / sums

# Forecast per group
st.subheader("Forecast results")
all_out = []

for grp_vals, gdf in _df.groupby(group_cols):
    gdf = gdf.sort_values(date_col)
    # find weights for this group
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

    out = compute_group_forecast(gdf, date_col, value_col, horizon=int(horizon), w_mm=w_mm, w_yoy=w_yoy)
    if out.empty:
        continue
    # attach group identifiers
    for col, val in zip(group_cols, grp_vals):
        out[col] = val
    all_out.append(out)

if len(all_out) == 0:
    st.error("No forecasts could be generated. Check date/value coverage for each group.")
    st.stop()

fcst = pd.concat(all_out, ignore_index=True)

# Order columns
fcst = fcst[[*group_cols, "date", "mm_component", "yoy_component", "baseline", "downside_-5pct", "upside_+5pct"]]

# Display
st.dataframe(fcst.sort_values(group_cols + ["date"]).reset_index(drop=True), use_container_width=True)

# Download
csv_bytes = fcst.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download forecasts (CSV)",
    data=csv_bytes,
    file_name="scenario_forecasts.csv",
    mime="text/csv",
)

# Simple chart for a selected group
st.subheader("Quick chart")
if len(unique_groups) > 0:
    # selection widget
    options = [tuple(r[group_cols].values) for _, r in unique_groups.iterrows()]
    sel = st.selectbox("Pick a group to visualize", options=options, format_func=lambda t: " | ".join(map(str, t)))
    mask = np.ones(len(_df), dtype=bool)
    for col, val in zip(group_cols, sel):
        mask &= (_df[col] == val).values
    hist = _df.loc[mask, [date_col, value_col]].sort_values(date_col)
    fmask = np.ones(len(fcst), dtype=bool)
    for col, val in zip(group_cols, sel):
        fmask &= (fcst[col] == val).values
    fut = fcst.loc[fmask, ["date", "baseline", "downside_-5pct", "upside_+5pct"]].sort_values("date")

    left, right = st.columns(2)
    with left:
        st.caption("Actuals (history)")
        st.line_chart(hist.set_index(date_col))
    with right:
        st.caption("Forecast scenarios (12 months)")
        st.line_chart(fut.set_index("date"))

st.success("Done. Adjust weights per group to explore different scenario shapes. Baseline is a weighted blend of m/m seasonal factor and recent y/y momentum; scenarios are ±5% around baseline.")
