import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml

from src.load_healthautoexport import load_payloads
from src.compute_phase1 import build_phase1_daily

st.set_page_config(page_title="Cardio-Protective Dashboard", layout="wide")
st.title("Cardio-Protective Dashboard — Phase 1")

# ----------------------------
# Targets
# ----------------------------
def load_targets(path="targets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["metrics"]

targets = load_targets()

def status(value: float, spec: dict):
    direction = spec.get("direction", "higher_better")
    g0, g1 = spec["green"]
    y0, y1 = spec["yellow"]
    r0, r1 = spec["red"]

    def in_band(v, lo, hi): return (v >= lo) and (v < hi)

    if direction == "range_best":
        if in_band(value, g0, g1): return "On target", "green"
        if (g0 - 1) <= value < g0 or g1 <= value <= (g1 + 1): return "Borderline", "orange"
        return "Off target", "red"

    if direction == "higher_better":
        if in_band(value, g0, g1): return "On target", "green"
        if in_band(value, y0, y1): return "Borderline", "orange"
        return "Off target", "red"

    if direction == "lower_better":
        if in_band(value, g0, g1): return "On target", "green"
        if in_band(value, y0, y1): return "Borderline", "orange"
        return "Off target", "red"

    return "Borderline", "orange"

def add_target_bands(fig, spec: dict):
    if not spec:
        return
    if spec.get("direction") == "range_best":
        fig.add_hrect(y0=spec["green"][0], y1=spec["green"][1], opacity=0.12, line_width=0)
    else:
        fig.add_hrect(y0=spec["green"][0], y1=spec["green"][1], opacity=0.10, line_width=0)
        fig.add_hrect(y0=spec["yellow"][0], y1=spec["yellow"][1], opacity=0.06, line_width=0)

def plot_line(df, col, title, spec_key=None, height=280):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines+markers", name=col))
    if spec_key and spec_key in targets:
        add_target_bands(fig, targets[spec_key])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=35, b=10), title=title, hovermode="x unified")
    return fig

# ----------------------------
# Load data
# ----------------------------
st.sidebar.header("Data")
st.sidebar.write("Upload HealthAutoExport-*.json into `/data` in Replit (NOT GitHub).")
try:
    payloads = load_payloads("data")
except Exception as e:
    st.error(f"Failed loading JSON files from /data: {e}")
    st.stop()

if not payloads:
    st.error("No HealthAutoExport JSON files found in /data.")
    st.stop()

df = build_phase1_daily(payloads)
if df.empty:
    st.error("Parsed payloads but produced an empty daily dataset. Likely metric key mismatch.")
    st.stop()

# Normalize types
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ----------------------------
# Time window
# ----------------------------
st.sidebar.header("Window")
window_days = st.sidebar.selectbox("Days", [7, 30, 90, 180, 365], index=1)
cut = df["date"].max() - pd.Timedelta(days=window_days)
dff = df[df["date"] >= cut].copy()

# ----------------------------
# Derived: rolling metrics & deltas
# ----------------------------
def rolling_mean(series, n=7, minp=4):
    return series.rolling(n, min_periods=minp).mean()

# HRV delta vs baseline (prior 30 days)
if "heart_rate_variability" in dff.columns:
    full = df[["date", "heart_rate_variability"]].dropna().copy()
    full["hrv7"] = rolling_mean(full["heart_rate_variability"], 7)
    # baseline: previous 30 days rolling mean
    full["baseline30"] = full["heart_rate_variability"].rolling(30, min_periods=14).mean()
    full["hrv_delta_pct"] = 100 * (full["hrv7"] - full["baseline30"]) / full["baseline30"]
    dff = dff.merge(full[["date", "hrv7", "hrv_delta_pct"]], on="date", how="left")

# RHR delta vs baseline
if "resting_heart_rate" in dff.columns:
    full = df[["date", "resting_heart_rate"]].dropna().copy()
    full["rhr7"] = rolling_mean(full["resting_heart_rate"], 7)
    full["baseline30_rhr"] = full["resting_heart_rate"].rolling(30, min_periods=14).mean()
    full["rhr_delta"] = full["rhr7"] - full["baseline30_rhr"]
    dff = dff.merge(full[["date", "rhr7", "rhr_delta"]], on="date", how="left")

# ----------------------------
# Scorecard (7-day rolling)
# ----------------------------
st.subheader("Scorecard (7-day rolling)")

def latest_val(col):
    s = dff[col].dropna()
    return float(s.iloc[-1]) if len(s) else None

tiles = st.columns(6)

# Zone2%
z2 = latest_val("zone2_pct") if "zone2_pct" in dff.columns else None
if z2 is not None:
    st0, _ = status(z2, targets["Zone2%"])
    tiles[0].metric("Zone 2 %", f"{z2:.1f}", st0)
else:
    tiles[0].metric("Zone 2 %", "—")

# HRR1
hrr1 = latest_val("hrr1") if "hrr1" in dff.columns else None
if hrr1 is not None:
    st1, _ = status(hrr1, targets["HRR1"])
    tiles[1].metric("HRR1", f"{hrr1:.1f}", st1)
else:
    tiles[1].metric("HRR1", "—")

# HRV 7d + delta
hrv7 = latest_val("hrv7") if "hrv7" in dff.columns else None
hrv_delta = latest_val("hrv_delta_pct") if "hrv_delta_pct" in dff.columns else None
if hrv7 is not None:
    label = f"{hrv7:.1f} ms"
    delta_txt = f"{hrv_delta:+.1f}%" if hrv_delta is not None else ""
    # simple flag: red if <= -10%
    status_txt = "On target"
    if hrv_delta is not None and hrv_delta <= -10:
        status_txt = "Off target"
    elif hrv_delta is not None and hrv_delta <= -5:
        status_txt = "Borderline"
    tiles[2].metric("HRV (7d)", label, f"{status_txt} {delta_txt}".strip())
else:
    tiles[2].metric("HRV (7d)", "—")

# RHR 7d + delta
rhr7 = latest_val("rhr7") if "rhr7" in dff.columns else None
rhr_delta = latest_val("rhr_delta") if "rhr_delta" in dff.columns else None
if rhr7 is not None:
    delta_txt = f"{rhr_delta:+.1f} bpm" if rhr_delta is not None else ""
    status_txt = "On target"
    if rhr_delta is not None and rhr_delta >= 3:
        status_txt = "Off target"
    elif rhr_delta is not None and rhr_delta >= 2:
        status_txt = "Borderline"
    tiles[3].metric("RHR (7d)", f"{rhr7:.1f}", f"{status_txt} {delta_txt}".strip())
else:
    tiles[3].metric("RHR (7d)", "—")

# Sleep score (derived)
sleep = latest_val("sleep_score_derived") if "sleep_score_derived" in dff.columns else None
if sleep is not None:
    st_s, _ = status(sleep, targets["SleepScore"])
    tiles[4].metric("Sleep score", f"{sleep:.0f}", st_s)
else:
    tiles[4].metric("Sleep score", "—")

# Alcohol
alc = latest_val("alcohol_consumption") if "alcohol_consumption" in dff.columns else None
if alc is not None:
    st_a, _ = status(alc, targets["AlcoholDaily"])
    tiles[5].metric("Alcohol (daily)", f"{alc:.1f}", st_a)
else:
    tiles[5].metric("Alcohol (daily)", "—")

# ----------------------------
# Zone 3 creep stress-test (simple + honest)
# ----------------------------
st.subheader("Zone 3 creep stress-test")
if "zone2_pct" in dff.columns:
    z2_series = dff.set_index("date")["zone2_pct"].dropna()
    z2_14 = z2_series.tail(14).mean() if len(z2_series) >= 7 else None
    z2_prev14 = z2_series.tail(28).head(14).mean() if len(z2_series) >= 28 else None

    if z2_14 is not None:
        msg = f"Zone 2 % (last 14d): **{z2_14:.1f}%**"
        if z2_prev14 is not None:
            delta = z2_14 - z2_prev14
            msg += f" | prior 14d: **{z2_prev14:.1f}%** | change: **{delta:+.1f}%**"
        st.write(msg)

        if z2_14 < 60 and (z2_prev14 is not None and (z2_14 - z2_prev14) < -3):
            st.error("Likely Zone 3 creep: Zone 2 share is dropping. Use a hard HR cap at 122 bpm on easy days.")
        elif z2_14 < 60:
            st.warning("Zone 2 share below target. You may be drifting into Zone 3. Tighten pacing / use HR alerts.")
        else:
            st.success("Zone distribution looks on track (Zone 2 is dominant).")
else:
    st.info("Zone 2 adherence not available yet (need workout HR buckets).")

# ----------------------------
# Charts
# ----------------------------
st.subheader("Trends")

c1, c2 = st.columns(2)

if "zone2_pct" in dff.columns:
    c1.plotly_chart(plot_line(dff, "zone2_pct", "Zone 2 % (daily)", "Zone2%"), use_container_width=True)
if "hrr1" in dff.columns:
    c2.plotly_chart(plot_line(dff, "hrr1", "HRR1 (daily mean)", "HRR1"), use_container_width=True)

c3, c4 = st.columns(2)
if "heart_rate_variability" in dff.columns:
    c3.plotly_chart(plot_line(dff, "heart_rate_variability", "HRV (daily)"), use_container_width=True)
if "resting_heart_rate" in dff.columns:
    c4.plotly_chart(plot_line(dff, "resting_heart_rate", "Resting HR (daily)", "RestingHR"), use_container_width=True)

c5, c6 = st.columns(2)
if "sleep_score_derived" in dff.columns:
    c5.plotly_chart(plot_line(dff, "sleep_score_derived", "Sleep Score (derived)", "SleepScore"), use_container_width=True)
if "breathing_disturbances" in dff.columns:
    c6.plotly_chart(plot_line(dff, "breathing_disturbances", "Breathing Disturbances (daily)", "BreathingDist"), use_container_width=True)

c7, c8 = st.columns(2)
if "respiratory_rate" in dff.columns:
    c7.plotly_chart(plot_line(dff, "respiratory_rate", "Respiratory Rate (daily)", "RespRate"), use_container_width=True)
if "mindful_minutes" in dff.columns:
    c8.plotly_chart(plot_line(dff, "mindful_minutes", "Mindful Minutes (daily)", "Mindful"), use_container_width=True)

if "alcohol_consumption" in dff.columns:
    st.plotly_chart(plot_line(dff, "alcohol_consumption", "Alcohol (daily)", "AlcoholDaily"), use_container_width=True)

with st.expander("Debug: daily dataset"):
    st.dataframe(dff.tail(50))

