
from pathlib import Path
import pandas as pd
import streamlit as st
from ui_theme import DATA, apply_theme, sidebar

st.set_page_config(page_title="GridGuard AI | Dashboard", page_icon="⚡", layout="wide")
apply_theme()
sidebar()

st.markdown('<div class="gg-eyebrow">WEST BENGAL GRID MONITORING</div>', unsafe_allow_html=True)
st.title("Electricity Demand Intelligence")
st.caption("Historical demand analytics, statistical demand-pressure screening and a clearly labelled demonstration forecast.")

st.warning("**Data limitation:** Electricity demand observations end on 28 April 2024. Values shown here are historical records or a historical demonstration—not live readings or current operational forecasts.")

path = DATA / "west_bengal_demand_weather_dataset.csv"
if not path.exists():
    st.error(f"Demand dataset not found: `{path}`")
    st.stop()

df = pd.read_csv(path, parse_dates=["date"])
for c in ["energy_MU", "peak_demand_MW"]:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce")
valid = df.dropna(subset=["date"]).sort_values("date")
observed = valid.dropna(subset=[c for c in ["energy_MU", "peak_demand_MW"] if c in valid.columns])
latest = observed.iloc[-1] if not observed.empty else None

st.markdown("### System overview")
a,b,c,d = st.columns(4)
a.metric("Demand records", f"{len(observed):,}")
b.metric("Latest observation", latest["date"].strftime("%d %b %Y") if latest is not None else "—")
c.metric("Daily energy", f'{latest["energy_MU"]:,.1f} MU' if latest is not None and pd.notna(latest.get("energy_MU")) else "—")
d.metric("Peak demand", f'{latest["peak_demand_MW"]:,.0f} MW' if latest is not None and pd.notna(latest.get("peak_demand_MW")) else "—")

left, right = st.columns([1.55,1])
with left:
    st.markdown("### Demand trend")
    metric = st.selectbox("Demand indicator", ["energy_MU","peak_demand_MW"], format_func=lambda x: "Daily energy (MU)" if x=="energy_MU" else "Peak demand (MW)")
    chart = valid[["date",metric]].dropna().set_index("date")
    st.line_chart(chart, height=290, color="#22a8ff")
with right:
    st.markdown("### Project modules")
    st.markdown('<div class="gg-panel"><b>Demand analysis</b><p class="gg-muted">Explore historical daily energy and peak-demand observations, date ranges and descriptive statistics.</p><span class="gg-pill">Historical records</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="gg-panel"><b>Risk assessment</b><p class="gg-muted">Review percentile-based demand pressure and independent historical weather context.</p><span class="gg-pill">Statistical screening</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="gg-panel"><b>Forecast demonstration</b><p class="gg-muted">Inspect the saved one-day-ahead model output from the last available demand date.</p><span class="gg-pill">Historical demo only</span></div>', unsafe_allow_html=True)

st.markdown("### Historical demand snapshot")
st.dataframe(valid[["date"] + [x for x in ["energy_MU","peak_demand_MW"] if x in valid]].tail(12), use_container_width=True, hide_index=True)
st.caption("Risk levels are percentile-based statistical categories—not official outage warnings, reserve-margin estimates or physical overload labels.")
