
import pandas as pd
import streamlit as st
from ui_theme import DATA, apply_theme, sidebar

st.set_page_config(page_title="Risk Assessment | GridGuard AI", page_icon="🛡️", layout="wide")
apply_theme(); sidebar()
st.markdown('<div class="gg-eyebrow">STATISTICAL SCREENING</div>', unsafe_allow_html=True)
st.title("Demand Pressure & Weather Context")
st.caption("Demand pressure is the primary risk signal. Historical weather indicators remain separate context.")

path = DATA / "risk_assessment" / "historical_risk_assessment.csv"
if not path.exists(): st.error(f"Risk assessment file not found: `{path}`"); st.stop()
df = pd.read_csv(path, parse_dates=["date"])
if "date" not in df: st.error("Risk assessment data has no date column."); st.stop()
df = df.sort_values("date")
if "demand_pressure" not in df: st.error("Required demand_pressure column is missing."); st.stop()

counts = df.demand_pressure.fillna("unknown").astype(str).str.lower().value_counts()
a,b,c,d = st.columns(4)
a.metric("Days assessed", f"{len(df):,}")
b.metric("High / critical days", f"{int(counts.get('high',0)+counts.get('critical',0)):,}")
c.metric("Normal days", f"{int(counts.get('normal',0)):,}")
d.metric("Unknown days", f"{int(counts.get('unknown',0)):,}")

left,right = st.columns(2)
with left:
    st.markdown("### Demand-pressure distribution")
    st.bar_chart(counts, color="#1685f8")
with right:
    st.markdown("### Weather context distribution")
    wc = "weather_context" if "weather_context" in df else ("weather_risk" if "weather_risk" in df else None)
    if wc: st.bar_chart(df[wc].fillna("unknown").value_counts(), color="#28b8d9")
    else: st.info("Weather context category is not present in this output.")

indicators = [c for c in ["heat_risk","rainfall_risk","humidity_risk","wind_risk","pressure_drop_risk"] if c in df.columns]
if indicators:
    st.markdown("### Individual weather indicators")
    selected = st.selectbox("Indicator", indicators)
    st.bar_chart(df[selected].fillna("unknown").value_counts(), color="#28b8d9")

st.markdown("### Historical assessment records")
show = [c for c in ["date","energy_MU","peak_demand_MW","demand_pressure","weather_context","weather_risk","heat_risk","rainfall_risk","humidity_risk","wind_risk","pressure_drop_risk","risk_explanation"] if c in df]
st.dataframe(df[show].tail(100), use_container_width=True, hide_index=True)
st.download_button("Download risk assessment CSV", df.to_csv(index=False).encode("utf-8"), "gridguard_risk_assessment.csv", "text/csv")
st.warning("These are historical percentile-based screening categories. They are not outage predictions, official warnings, reserve-margin estimates or verified physical grid-overload labels.")
