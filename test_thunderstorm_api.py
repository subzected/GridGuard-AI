import requests
import json

URL = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 22.5726,
    "longitude": 88.3639,
    "hourly": "thunderstorm_probability",
    "forecast_days": 3,
    "timezone": "UTC"
}

response = requests.get(URL, params=params, timeout=30)

print("HTTP STATUS:", response.status_code)
print("REQUEST URL:", response.url)

response.raise_for_status()

data = response.json()

print("\nHOURLY RESPONSE:")
print(json.dumps(data.get("hourly", {}), indent=4))

print("\nHOURLY UNITS:")
print(json.dumps(data.get("hourly_units", {}), indent=4))

print("\nSELECTED MODEL:")
print(data.get("model", "Model field not provided"))

values = data.get("hourly", {}).get(
    "thunderstorm_probability"
)

if values is None:
    print("\nRESULT: Variable was not returned.")
else:
    available = [v for v in values if v is not None]

    print("\nTOTAL VALUES:", len(values))
    print("AVAILABLE VALUES:", len(available))
    print("MISSING VALUES:", len(values) - len(available))
    print("FIRST 10 VALUES:", values[:10])