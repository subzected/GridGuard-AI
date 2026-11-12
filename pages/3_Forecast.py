
import json
import streamlit as st

from ui_theme import DATA, apply_theme, sidebar


st.set_page_config(
    page_title="Forecast | GridGuard AI",
    page_icon="⚡",
    layout="wide"
)

apply_theme()
sidebar()

# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.markdown(
    '<div class="gg-eyebrow">MODEL OUTPUT</div>',
    unsafe_allow_html=True
)

st.title("One-Day-Ahead Demand Forecast")

st.caption(
    "Saved model output generated from the last available "
    "historical demand observations."
)

# --------------------------------------------------
# LOAD FORECAST JSON
# --------------------------------------------------

forecast_path = (
    DATA
    / "operational_forecasts"
    / "latest_demand_forecast.json"
)

if not forecast_path.exists():
    st.error(f"Forecast file not found: {forecast_path}")

    st.info(
        "Generate the forecast using:\n\n"
        "python -m ml.inference.daily_demand_forecast "
        "--allow-stale-history"
    )

    st.stop()

try:
    with forecast_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

except (json.JSONDecodeError, OSError) as error:
    st.error(f"Could not load forecast JSON: {error}")
    st.stop()


# --------------------------------------------------
# EXTRACT FORECAST DATA
# --------------------------------------------------

forecasts = data.get("forecasts", {})

energy = forecasts.get("energy_MU", {})
peak = forecasts.get("peak_demand_MW", {})

risk = data.get("risk_assessment", {})

# IMPORTANT:
# The actual JSON field is forecast_value.

energy_value = energy.get("forecast_value")
peak_value = peak.get("forecast_value")

if energy_value is None or peak_value is None:
    st.error(
        "Forecast values are missing from the JSON output. "
        "Expected forecasts.energy_MU.forecast_value and "
        "forecasts.peak_demand_MW.forecast_value."
    )
    st.stop()


# --------------------------------------------------
# DATA STATUS
# --------------------------------------------------

if data.get("data_status") == "stale_historical_demo":

    st.markdown(
        """
        <div class="gg-note">
            <b>HISTORICAL DEMONSTRATION</b><br>
            This is not a current operational forecast.
            Electricity observations end on 28 April 2024.
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# FORECAST METADATA
# --------------------------------------------------

st.markdown("### Forecast metadata")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Forecast date",
    data.get("forecast_date", "—")
)

col2.metric(
    "Last observed date",
    data.get("latest_observed_demand_date", "—")
)

col3.metric(
    "History age",
    f'{data.get("history_age_days", "—")} days'
)


# --------------------------------------------------
# DEMAND FORECAST VALUES
# --------------------------------------------------

st.markdown("### Predicted demand")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Daily energy forecast",
        value=f"{float(energy_value):,.3f} MU",
        delta=(
            f'{float(energy.get("predicted_change", 0)):+,.3f} MU'
        )
    )

    st.caption(
        f'Previous day actual: '
        f'{float(energy.get("previous_day_actual", 0)):,.3f} MU'
    )

    st.caption(
        f'Model: {energy.get("model_name", "—")}'
    )

    st.write(
        f'**Demand pressure:** '
        f'{energy.get("demand_pressure_category", "not available").upper()}'
    )


with col2:

    st.metric(
        label="Peak demand forecast",
        value=f"{float(peak_value):,.3f} MW",
        delta=(
            f'{float(peak.get("predicted_change", 0)):+,.3f} MW'
        )
    )

    st.caption(
        f'Previous day actual: '
        f'{float(peak.get("previous_day_actual", 0)):,.3f} MW'
    )

    st.caption(
        f'Model: {peak.get("model_name", "—")}'
    )

    st.write(
        f'**Demand pressure:** '
        f'{peak.get("demand_pressure_category", "not available").upper()}'
    )


# --------------------------------------------------
# RISK INTERPRETATION
# --------------------------------------------------

st.markdown("### Risk interpretation")

col1, col2 = st.columns(2)

col1.metric(
    "Primary demand risk",
    risk.get("primary_demand_risk_category", "not available").upper()
)

col2.metric(
    "Weather context",
    risk.get("weather_context_category", "not assessed").upper()
)

st.markdown(
    f'<div class="gg-panel">'
    f'<b>Interpretation</b>'
    f'<p class="gg-muted">{risk.get("interpretation", "No interpretation available.")}</p>'
    f'</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL LIMITATIONS
# --------------------------------------------------

st.markdown("### Model limitations")

for limitation in data.get("limitations", []):
    st.markdown(f"- {limitation}")


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

st.markdown("---")

st.download_button(
    label="Download forecast JSON",
    data=json.dumps(data, indent=2),
    file_name="gridguard_forecast.json",
    mime="application/json"
)

st.caption(
    f'Generated at: {data.get("generated_at_utc", "Not available")} UTC'
)
