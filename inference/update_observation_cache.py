import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENSOR_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sensor_catalog_v1.csv"
)

CACHE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "openaq_hourly_48h.csv"
)

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
HISTORY_HOURS = 48


load_dotenv(PROJECT_ROOT / ".env")

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

if not OPENAQ_API_KEY:
    raise ValueError("OPENAQ_API_KEY is missing.")


def create_session():
    retry_policy = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session = requests.Session()

    session.headers.update(
        {
            "X-API-Key": OPENAQ_API_KEY,
        }
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry_policy),
    )

    return session


def select_relevant_sensors():
    sensors = pd.read_csv(SENSOR_CATALOG_FILE)

    sensors["datetime_first_utc"] = pd.to_datetime(
        sensors["datetime_first_utc"],
        utc=True,
        errors="coerce",
    )

    sensors["datetime_last_utc"] = pd.to_datetime(
        sensors["datetime_last_utc"],
        utc=True,
        errors="coerce",
    )

    forecast_origin_ist = (
    pd.Timestamp.now(tz="Asia/Kolkata").floor("h")
    - pd.Timedelta(hours=1)
)

    forecast_origin_utc = (
    forecast_origin_ist.tz_convert("UTC")
)

    window_start_utc = (
        forecast_origin_utc
        - pd.Timedelta(hours=HISTORY_HOURS)
    )

    relevant = sensors[
        (sensors["datetime_first_utc"] <= forecast_origin_utc)
        & (sensors["datetime_last_utc"] >= window_start_utc)
    ].copy()

    return relevant, window_start_utc, forecast_origin_utc


def fetch_sensor_hours(
    session,
    sensor,
    datetime_from,
    datetime_to,
):
    response = session.get(
        (
            f"{OPENAQ_BASE_URL}"
            f"/sensors/{sensor.sensor_id}/hours"
        ),
        params={
            "datetime_from": datetime_from.isoformat(),
            "datetime_to": datetime_to.isoformat(),
            "limit": 100,
            "page": 1,
        },
        timeout=30,
    )

    response.raise_for_status()

    rows = []

    for result in response.json()["results"]:
        period = result.get("period") or {}
        datetime_from_data = (
            period.get("datetimeFrom") or {}
        )

        timestamp_utc = datetime_from_data.get("utc")
        value = result.get("value")

        if timestamp_utc is None or value is None:
            continue

        rows.append(
            {
                "location_id": sensor.location_id,
                "station_name": sensor.station_name,
                "sensor_id": sensor.sensor_id,
                "pollutant": sensor.pollutant,
                "unit": sensor.unit,
                "timestamp_utc": timestamp_utc,
                "value": value,
                "has_flags": (
                    result.get("flagInfo") or {}
                ).get("hasFlags", False),
            }
        )

    return rows


def update_observation_cache():
    relevant, window_start, forecast_origin = (
        select_relevant_sensors()
    )

    if CACHE_FILE.exists():
        old_cache = pd.read_csv(CACHE_FILE)

        old_cache["timestamp_utc"] = pd.to_datetime(
            old_cache["timestamp_utc"],
            utc=True,
        )
    else:
        old_cache = pd.DataFrame()

    new_rows = []
    failures = []

    session = create_session()

    for number, sensor in enumerate(
        relevant.itertuples(index=False),
        start=1,
    ):
        # On later runs, request only hours not already cached.
        fetch_start = window_start

        if not old_cache.empty:
            previous_sensor_rows = old_cache[
                old_cache["sensor_id"] == sensor.sensor_id
            ]

            if not previous_sensor_rows.empty:
                fetch_start = max(
                    window_start,
                    previous_sensor_rows[
                        "timestamp_utc"
                    ].max(),
                )

        try:
            rows = fetch_sensor_hours(
                session=session,
                sensor=sensor,
                datetime_from=fetch_start,
                datetime_to=forecast_origin,
            )

            new_rows.extend(rows)

        except requests.RequestException as error:
            failures.append(
                {
                    "sensor_id": sensor.sensor_id,
                    "error": str(error),
                }
            )

        if number % 20 == 0 or number == len(relevant):
            print(
                f"Processed {number}/{len(relevant)} sensors"
            )

    session.close()

    new_cache = pd.DataFrame(new_rows)

    if old_cache.empty:
        combined = new_cache
    elif new_cache.empty:
        combined = old_cache
    else:
        combined = pd.concat(
            [old_cache, new_cache],
            ignore_index=True,
        )

    if combined.empty:
        raise RuntimeError(
            "No hourly OpenAQ observations were retrieved."
        )

    combined["timestamp_utc"] = pd.to_datetime(
        combined["timestamp_utc"],
        utc=True,
    )

    combined = (
        combined[
            combined["timestamp_utc"].between(
                window_start,
                forecast_origin,
            )
        ]
        .drop_duplicates(
            ["sensor_id", "timestamp_utc"],
            keep="last",
        )
        .sort_values(
            ["timestamp_utc", "location_id", "pollutant"]
        )
        .reset_index(drop=True)
    )

    CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(CACHE_FILE, index=False)

    return combined, failures, window_start, forecast_origin


if __name__ == "__main__":
    cache, failures, window_start, forecast_origin = (
        update_observation_cache()
    )

    print("\nObservation cache updated.")
    print("Window:", window_start, "to", forecast_origin)
    print("Cached rows:", len(cache))
    print("Failed sensor requests:", len(failures))

    print("\nRows by pollutant:")

    print(
        cache["pollutant"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved:", CACHE_FILE)