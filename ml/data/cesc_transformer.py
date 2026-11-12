
from pathlib import Path
import pandas as pd


PLACEHOLDER_RE = r"^cesc"
LOCATION_MODE = "per_feeder"

# Automatically locate outage_events.csv in the current script directory
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "outage_events.csv"


def transform_cesc_events(path=CSV_PATH, location_mode=LOCATION_MODE):

    df = pd.read_csv(path)

    # Remove UTF-8 BOM from column names
    df.columns = df.columns.str.replace("^\ufeff", "", regex=True)

    # ---- 1. Drop summary/placeholder rows -------------------------------
    df = df[
        df["source_event_id"].astype(str).str.match(
            PLACEHOLDER_RE, na=False
        )
    ].copy()

    # ---- 2. Parse timestamps --------------------------------------------
    df["start_time"] = pd.to_datetime(
        df["start_time"], utc=True, errors="coerce"
    )

    df["end_time"] = pd.to_datetime(
        df["end_time"], utc=True, errors="coerce"
    )

    df = df.dropna(subset=["start_time", "end_time"]).copy()

    # ---- 3. Remove invalid durations ------------------------------------
    df = df[df["end_time"] > df["start_time"]].copy()

    # ---- 4. Derive location_id ------------------------------------------
    if location_mode == "per_feeder":
        df["location_id"] = (
            df["feeder"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    elif location_mode == "single":
        df["location_id"] = "CESC_KOLKATA"

    else:
        raise ValueError(
            "location_mode must be 'per_feeder' or 'single'"
        )

    # ---- 5. Mark verified -----------------------------------------------
    df["verified"] = True

    # ---- 6. Return expected schema --------------------------------------
    cols = [
        "location_id",
        "start_time",
        "end_time",
        "source",
        "verified"
    ]

    return df[cols].reset_index(drop=True)


if __name__ == "__main__":
    events = transform_cesc_events()
    print(f"Total valid CESC events: {len(events)}")
    print(events.head())
