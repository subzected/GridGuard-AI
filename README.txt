GRIDGUARD AI — STREAMLIT FRONTEND (DARK BLUE REDESIGN)

COPY
Copy app.py and ui_theme.py into F:\powergrid-ai\
Copy the pages folder into F:\powergrid-ai\pages\

EXPECTED STRUCTURE
F:\powergrid-ai\
  app.py
  ui_theme.py
  pages\
    1_Demand_Analysis.py
    2_Risk_Assessment.py
    3_Forecast.py
  data\processed\
    west_bengal_demand_weather_dataset.csv
    risk_assessment\historical_risk_assessment.csv
    operational_forecasts\latest_demand_forecast.json

INSTALL AND RUN (inside the active venv)
pip install streamlit
cd /d F:\powergrid-ai
streamlit run app.py

The frontend reads existing outputs; it does not retrain models.
It intentionally does not fabricate outage probabilities, live weather readings,
feeder-level monitoring, live maps or physical overload status.
Historical electricity observations end on 28 April 2024.
