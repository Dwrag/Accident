"""
generate_historical_data.py

Generates a REALISTIC SYNTHETIC historical accident dataset so the whole
pipeline (training + comparison + app) runs out of the box with no
external file needed.

*** IMPORTANT FOR YOUR PROJECT REPORT / VIVA ***
Replace this with your real dataset (e.g. state Traffic Police open data,
Kaggle "US Accidents", data.gov.in road accident records, etc). As long as
your CSV has columns compatible with utils.FEATURE_COLUMNS + a `severity`
target column, train_model.py will work unchanged. This file exists purely
so you have a runnable demo immediately.

Design: a handful of "hotspot" centres (busy junctions / highway
stretches) get denser, more severe accidents; distance from a hotspot,
night-time, bad weather, high speed limit and junctions all *causally*
push severity up, with noise added so it isn't a trivial rule for the
model to reverse-engineer.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ROAD_TYPES, WEATHER_CONDITIONS, derive_time_features  # noqa: E402

RNG = np.random.default_rng(42)
N_RECORDS = 6000

# Bounding box roughly covering Chennai, Tamil Nadu (swap for your city)
LAT_MIN, LAT_MAX = 12.90, 13.20
LON_MIN, LON_MAX = 80.10, 80.30

# A handful of known-dangerous hotspot centres within the box
HOTSPOTS = [
    (13.0827, 80.2707),  # central
    (13.0067, 80.2206),  # south junction
    (13.1500, 80.2101),  # north highway stretch
    (12.9716, 80.2200),  # OMR-like corridor
    (13.0475, 80.1734),  # west arterial road
]

ROAD_TYPE_NAMES = list(ROAD_TYPES.keys())
WEATHER_NAMES = list(WEATHER_CONDITIONS.keys())


def nearest_hotspot_km(lat, lon):
    from utils import haversine_km
    dists = [haversine_km(lat, lon, hs[0], hs[1]) for hs in HOTSPOTS]
    return min(dists)


def sample_location():
    """70% of records cluster near a random hotspot (gaussian), 30% uniform."""
    if RNG.random() < 0.7:
        hs = HOTSPOTS[RNG.integers(0, len(HOTSPOTS))]
        lat = np.clip(RNG.normal(hs[0], 0.012), LAT_MIN, LAT_MAX)
        lon = np.clip(RNG.normal(hs[1], 0.012), LON_MIN, LON_MAX)
    else:
        lat = RNG.uniform(LAT_MIN, LAT_MAX)
        lon = RNG.uniform(LON_MIN, LON_MAX)
    return lat, lon


def sample_datetime():
    # random datetime across a 3-year historical window
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-31")
    delta_days = (end - start).days
    day_offset = RNG.integers(0, delta_days)
    hour = RNG.integers(0, 24)
    minute = RNG.integers(0, 60)
    return start + pd.Timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute))


def sample_weather():
    # weighted towards clear/cloudy, rain more likely in monsoon-ish months handled loosely
    condition = RNG.choice(
        WEATHER_NAMES, p=[0.45, 0.25, 0.18, 0.06, 0.04, 0.02]
    )
    temperature = float(np.clip(RNG.normal(29, 4), 15, 42))
    precipitation = 0.0
    if condition == "Rain":
        precipitation = float(np.clip(RNG.exponential(6), 0, 60))
    elif condition == "Storm":
        precipitation = float(np.clip(RNG.exponential(15), 5, 100))
    wind_speed = float(np.clip(RNG.normal(12, 6), 0, 55))
    if condition in ("Fog",):
        visibility = float(np.clip(RNG.normal(1.5, 0.8), 0.1, 4))
    elif condition == "Storm":
        visibility = float(np.clip(RNG.normal(3, 1.5), 0.2, 6))
    elif condition == "Rain":
        visibility = float(np.clip(RNG.normal(5, 2), 1, 9))
    else:
        visibility = float(np.clip(RNG.normal(9, 2), 3, 12))
    return {
        "condition": condition, "temperature": temperature,
        "precipitation": precipitation, "wind_speed": wind_speed,
        "visibility": visibility,
    }


def generate():
    records = []
    for _ in range(N_RECORDS):
        lat, lon = sample_location()
        dt = sample_datetime()
        t = derive_time_features(dt)
        weather = sample_weather()

        road_type = RNG.choice(ROAD_TYPE_NAMES, p=[0.30, 0.40, 0.20, 0.10])
        speed_limit = {"Highway": 80, "Urban Road": 50, "Residential": 30, "Rural Road": 60}[road_type]
        speed_limit = int(np.clip(RNG.normal(speed_limit, 8), 20, 100))
        junction = int(RNG.random() < (0.55 if road_type in ("Urban Road", "Residential") else 0.25))

        dist_hotspot = nearest_hotspot_km(lat, lon)
        hotspot_factor = np.exp(-dist_hotspot / 3.0)  # closer -> near 1, far -> near 0

        # --- latent severity score, purely for LABEL generation ---
        score = 0.0
        score += 2.2 * hotspot_factor
        score += 1.3 * t["is_night"]
        score += 0.9 * t["is_weekend"]
        score += 0.015 * speed_limit
        score += 0.8 * junction
        score += {"Clear": 0, "Cloudy": 0.15, "Rain": 1.1, "Fog": 1.4, "Storm": 1.9, "Snow": 1.3}[weather["condition"]]
        score += 0.35 * (weather["precipitation"] / 20.0)
        score -= 0.12 * (weather["visibility"])
        score += 1.1 * t["traffic_density"]
        score += RNG.normal(0, 0.9)  # noise so it's learnable, not deterministic

        records.append({
            "latitude": lat, "longitude": lon,
            "date": dt.date().isoformat(), "hour": t["hour"],
            "day_of_week": t["day_of_week"], "month": t["month"],
            "is_weekend": t["is_weekend"], "is_night": t["is_night"],
            "road_type": road_type, "road_type_enc": ROAD_TYPES[road_type],
            "speed_limit": speed_limit, "junction": junction,
            "traffic_density": round(t["traffic_density"], 2),
            "weather_condition": weather["condition"],
            "weather_enc": WEATHER_CONDITIONS[weather["condition"]],
            "temperature": round(weather["temperature"], 1),
            "precipitation": round(weather["precipitation"], 1),
            "wind_speed": round(weather["wind_speed"], 1),
            "visibility": round(weather["visibility"], 1),
            "_severity_score": score,
        })

    df = pd.DataFrame(records)

    # convert continuous latent score -> 4-class severity using quantile-ish fixed cuts
    q1, q2, q3 = df["_severity_score"].quantile([0.45, 0.75, 0.93])
    def to_class(s):
        if s <= q1:
            return 0  # Low
        elif s <= q2:
            return 1  # Medium
        elif s <= q3:
            return 2  # High
        return 3      # Severe
    df["severity"] = df["_severity_score"].apply(to_class)
    df = df.drop(columns=["_severity_score"])
    return df


if __name__ == "__main__":
    df = generate()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_accidents.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} historical accident records -> {out_path}")
    print(df["severity"].value_counts(normalize=True).sort_index())
