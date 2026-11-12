from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IMD_INPUT = ROOT / "data" / "interim" / "imd_cleaned.csv"
OPEN_METEO_INPUT = ROOT / "data" / "interim" / "openmeteo_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "weather_features"

DATE = "snapshot_date"
LOCATION = "location_id"
WET_THRESHOLD_MM = 1.0


def _pick_column(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _prepare_source(df: pd.DataFrame, source: str):
    df = df.copy()

    required = [DATE, LOCATION]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")

    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    if df[DATE].dt.tz is not None:
        df[DATE] = df[DATE].dt.tz_localize(None)

    df[LOCATION] = df[LOCATION].astype("string").str.strip().str.lower()

    aliases = {
        "temperature": ["temperature_c", "temp_c"],
        "precipitation": ["precipitation_mm", "rain_mm", "rain"],
        "humidity": ["relative_humidity_pct", "humidity_pct"],
        "wind": ["wind_speed_ms", "wind_speed_kmh"],
        "pressure": ["pressure_hpa"],
    }
    selected = {}
    for key, candidates in aliases.items():
        column = _pick_column(df, candidates)
        if column:
            selected[key] = column
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "precipitation" not in selected:
        raise ValueError(
            f"{source}: expected precipitation_mm, rain_mm or rain."
        )

    duplicate = df.duplicated([LOCATION, DATE], keep=False)
    if duplicate.any():
        examples = df.loc[duplicate, [LOCATION, DATE]].head(8).to_dict("records")
        raise ValueError(
            f"{source}: duplicate location/date rows found; "
            f"resolve them before feature engineering. Examples: {examples}"
        )

    df = df.dropna(subset=[DATE, LOCATION])
    df = df.sort_values([LOCATION, DATE]).reset_index(drop=True)

    # Rolling row windows are only valid as calendar-day windows when dates
    # are continuous. Fail loudly rather than treating a multi-day gap as 1 day.
    for location_id, group in df.groupby(LOCATION, sort=False):
        date_deltas = group[DATE].diff().dropna()
        gaps = date_deltas[date_deltas != pd.Timedelta(days=1)]
        if not gaps.empty:
            bad_rows = group.loc[gaps.index, [DATE, LOCATION]].head(5)
            examples = bad_rows.to_dict("records")
            raise ValueError(
                f"{source}: non-daily date sequence for location "
                f"{location_id!r}. Rolling windows require consecutive "
                f"calendar days. Example rows: {examples}"
            )

    return df, selected


def _rolling_sum_previous(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=window).sum()


def _rolling_mean_previous(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=window).mean()


def _wet_days_previous(
    precipitation: pd.Series, window: int, threshold_mm: float
) -> pd.Series:
    wet = precipitation.ge(threshold_mm).where(precipitation.notna())
    return wet.shift(1).rolling(window, min_periods=window).sum()


def _consecutive_wet_days_before_today(
    precipitation: pd.Series, threshold_mm: float
) -> pd.Series:
    values = precipitation.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    streak = 0
    known = False

    for i, value in enumerate(values):
        # Feature at date D describes the streak ending on D-1.
        # No prior observation exists for the first row, so it remains NaN.
        result[i] = float(streak) if known else np.nan

        if np.isnan(value):
            streak = 0
            known = False
        elif value >= threshold_mm:
            if known:
                streak += 1
            else:
                # A wet observation after an unknown gap cannot establish
                # the full consecutive streak.
                streak = 1
                known = True
        else:
            # A known dry day resets the streak and establishes a valid zero.
            streak = 0
            known = True

    return pd.Series(result, index=precipitation.index)


def engineer_source_history(
    df: pd.DataFrame,
    source: str,
    wet_threshold_mm: float = WET_THRESHOLD_MM,
) -> pd.DataFrame:
    """
    Create causal daily features. Every feature on date D uses data strictly
    before D. Source-specific feature names prevent cross-source averaging.
    """
    df, columns = _prepare_source(df, source)
    prefix = source.lower().replace("-", "_").replace(" ", "_")

    result = df[[DATE, LOCATION]].copy()
    # Keep provenance and QC flags for auditing; these are not automatically
    # selected as model predictors.
    passthrough = [
        c for c in df.columns
        if c.endswith("_flag") or c in {"source_name", "source", "weather_grid_id"}
    ]
    for c in passthrough:
        result[c] = df[c].to_numpy()

    for location_id, indexes in df.groupby(LOCATION, sort=False).groups.items():
        group = df.loc[indexes].sort_values(DATE)
        idx = group.index
        rain = group[columns["precipitation"]]

        def add(name: str, values: pd.Series):
            result.loc[idx, f"{prefix}_{name}"] = values.reindex(idx).to_numpy()

        for window in (3, 7, 14):
            add(
                f"rainfall_{window}d_mm",
                _rolling_sum_previous(rain, window),
            )
            add(
                f"wet_days_{window}d",
                _wet_days_previous(rain, window, wet_threshold_mm),
            )

        add(
            "consecutive_wet_days",
            _consecutive_wet_days_before_today(rain, wet_threshold_mm),
        )

        if "temperature" in columns:
            temp = group[columns["temperature"]]
            add("temperature_mean_3d_c", _rolling_mean_previous(temp, 3))
            add("temperature_mean_7d_c", _rolling_mean_previous(temp, 7))
            # Yesterday minus four days ago = change over three days,
            # with today's observation excluded.
            add("temperature_change_3d_c", temp.shift(1) - temp.shift(4))

        if "humidity" in columns:
            add(
                "humidity_mean_7d_pct",
                _rolling_mean_previous(group[columns["humidity"]], 7),
            )

        if "wind" in columns:
            wind = group[columns["wind"]]
            add("wind_speed_mean_7d", _rolling_mean_previous(wind, 7))
            add(
                "wind_speed_max_7d",
                wind.shift(1).rolling(7, min_periods=7).max(),
            )

        if "pressure" in columns:
            pressure = group[columns["pressure"]]
            add(
                "pressure_mean_7d_hpa",
                _rolling_mean_previous(pressure, 7),
            )
            add(
                "pressure_change_1d_hpa",
                pressure.shift(1) - pressure.shift(2),
            )

    result = result.sort_values([LOCATION, DATE]).reset_index(drop=True)
    result.insert(2, "weather_source", source.upper())
    return result


def _summary(df: pd.DataFrame, source: str) -> dict:
    features = [
        c for c in df.columns
        if c not in {DATE, LOCATION, "weather_source"}
        and not c.endswith("_flag")
    ]
    return {
        "source": source,
        "rows": int(len(df)),
        "locations": int(df[LOCATION].nunique()),
        "date_min": df[DATE].min().strftime("%Y-%m-%d") if len(df) else None,
        "date_max": df[DATE].max().strftime("%Y-%m-%d") if len(df) else None,
        "feature_columns": features,
        "missing_by_feature": {c: int(df[c].isna().sum()) for c in features},
    }


def build_historical_weather_features() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (IMD_INPUT, OPEN_METEO_INPUT):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    imd = engineer_source_history(pd.read_csv(IMD_INPUT), "imd")
    om = engineer_source_history(pd.read_csv(OPEN_METEO_INPUT), "open_meteo")

    imd_path = OUTPUT_DIR / "imd_historical_features.csv"
    om_path = OUTPUT_DIR / "openmeteo_historical_features.csv"
    combined_path = OUTPUT_DIR / "historical_weather_features.csv"

    imd.to_csv(imd_path, index=False)
    om.to_csv(om_path, index=False)

    keys = [DATE, LOCATION]
    combined = imd.drop(columns=["weather_source"]).merge(
        om.drop(columns=["weather_source"]),
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_csv(combined_path, index=False)

    report = {
        "policy": {
            "rolling_windows": "Only observations strictly before each date",
            "rolling_min_periods": "Full window required",
            "rainfall_sources": "IMD and Open-Meteo remain separate",
            "missing_values": "Preserved; not replaced with zero",
            "duplicate_keys": "Raise error; never aggregate silently",
            "wet_day_threshold_mm": WET_THRESHOLD_MM,
        },
        "imd": _summary(imd, "IMD"),
        "open_meteo": _summary(om, "OPEN_METEO"),
        "combined": {
            "rows": int(len(combined)),
            "locations": int(combined[LOCATION].nunique()),
        },
        "outputs": {
            "imd": str(imd_path.relative_to(ROOT)),
            "open_meteo": str(om_path.relative_to(ROOT)),
            "combined": str(combined_path.relative_to(ROOT)),
        },
    }
    report_path = REPORT_DIR / "historical_feature_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Historical weather feature engineering completed.")
    print(f"IMD rows: {len(imd):,}")
    print(f"Open-Meteo rows: {len(om):,}")
    print(f"Combined rows: {len(combined):,}")
    print(f"Locations: {combined[LOCATION].nunique()}")
    print(f"Output: {combined_path}")
    print(f"Report: {report_path}")
    return combined


if __name__ == "__main__":
    build_historical_weather_features()
