
import pandas as pd
import streamlit as st
from ui_theme import DATA, apply_theme, sidebar

st.set_page_config(page_title="Demand Analysis | GridGuard AI", page_icon="📈", layout="wide")
apply_theme(); sidebar()
st.markdown('<div class="gg-eyebrow">HISTORICAL ANALYTICS</div>', unsafe_allow_html=True)
st.title("Demand Analysis")
st.caption("Daily West Bengal electricity demand · 2013 to 28 April 2024")

path = DATA / "west_bengal_demand_weather_dataset.csv"
if not path.exists():
    st.error(f"Dataset not found: `{path}`"); st.stop()
df = pd.read_csv(path, parse_dates=["date"])
for col in ["energy_MU","peak_demand_MW"]:
    if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
metrics = [c for c in ["energy_MU","peak_demand_MW"] if c in df]
if not metrics: st.error("Expected demand columns were not found."); st.stop()

metric = st.selectbox("Demand metric", metrics, format_func=lambda x: "Daily energy (MU)" if x=="energy_MU" else "Peak demand (MW)")
min_date,max_date = df.date.min().date(),df.date.max().date()
x,y = st.columns(2)
start = x.date_input("From", min_value=min_date, max_value=max_date, value=min_date)
end = y.date_input("To", min_value=min_date, max_value=max_date, value=max_date)
if start > end: st.error("Start date must be on or before end date."); st.stop()
subset = df[(df.date.dt.date>=start)&(df.date.dt.date<=end)].dropna(subset=[metric]).copy()
st.markdown("### Selected period")
a,b,c,d = st.columns(4)
a.metric("Observations", f"{len(subset):,}")
b.metric("Average", f"{subset[metric].mean():,.2f}")
c.metric("Maximum", f"{subset[metric].max():,.2f}")
d.metric("Minimum", f"{subset[metric].min():,.2f}")
st.markdown("### Demand trend")
st.line_chart(subset.set_index("date")[[metric]], height=350, color="#22a8ff")
l,r = st.columns([1,1])
with l:
    st.markdown("### Descriptive statistics")
    st.dataframe(subset[metric].describe().to_frame("Value"), use_container_width=True)
with r:
    st.markdown("### Data coverage")
    st.markdown('<div class="gg-panel"><b>Source coverage</b><p class="gg-muted">Historical observations only. Missing calendar dates are retained as missing and are not interpolated.</p></div>', unsafe_allow_html=True)
    st.download_button("Download selected observations", subset.to_csv(index=False).encode("utf-8"), "gridguard_demand_selection.csv", "text/csv")
