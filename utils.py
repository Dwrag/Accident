

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------
# Fixed category encodings (must stay identical across train + inference)
# ----------------------------------------------------------------------
ROAD_TYPES = {"Highway": 0, "Urban Road": 1, "Residential": 2, "Rural Road": 3}
WEATHER_CONDITIONS = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Fog": 3, "Storm": 4, "Snow": 5}
SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Severe"}
SEVERITY_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

FEATURE_COLUMNS = [
    "latitude", "longitude", "hour", "day_of_week", "month",
    "is_weekend", "is_night", "road_type_enc", "speed_limit",
    "junction", "traffic_density", "weather_enc", "temperature",
    "precipitation", "wind_speed", "visibility",
]


def derive_time_features(dt):
    """dt: python datetime -> dict of time-derived features."""
    hour = dt.hour
    dow = dt.weekday()
    month = dt.month
    is_weekend = 1 if dow >= 5 else 0
    is_night = 1 if (hour >= 21 or hour < 6) else 0
    # crude but realistic default traffic density curve (rush hours busier)
    if hour in (8, 9, 18, 19):
        traffic_density = 0.9
    elif hour in (7, 10, 17, 20):
        traffic_density = 0.65
    elif 22 <= hour or hour <= 5:
        traffic_density = 0.15
    else:
        traffic_density = 0.4
    return {
        "hour": hour, "day_of_week": dow, "month": month,
        "is_weekend": is_weekend, "is_night": is_night,
        "traffic_density": traffic_density,
    }


def weather_code_to_condition(code):
    """Map Open-Meteo WMO weather codes -> our condition buckets."""
    if code in (0,):
        return "Clear"
    if code in (1, 2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
        return "Rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (95, 96, 99):
        return "Storm"
    return "Clear"


def fetch_live_weather(lat, lon, timeout=8):
    """
    Pulls current weather from Open-Meteo (free, no API key required).
    Returns dict with condition, temperature, precipitation, wind_speed,
    visibility (km). Raises on network failure so caller can decide fallback.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,weather_code,wind_speed_10m"
        "&hourly=visibility"
        "&timezone=auto"
    )
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    current = data.get("current", {})
    code = current.get("weather_code", 0)
    condition = weather_code_to_condition(code)

    # pull the visibility value closest to "now" from the hourly series
    visibility_km = 10.0
    try:
        hourly_times = data["hourly"]["time"]
        hourly_vis = data["hourly"]["visibility"]
        current_time = current.get("time")
        if current_time in hourly_times:
            idx = hourly_times.index(current_time)
        else:
            idx = 0
        visibility_km = round(hourly_vis[idx] / 1000.0, 1)
    except Exception:
        pass

    return {
        "condition": condition,
        "temperature": current.get("temperature_2m", 25.0),
        "precipitation": current.get("precipitation", 0.0),
        "wind_speed": current.get("wind_speed_10m", 5.0),
        "visibility": visibility_km,
    }


def build_feature_row(latitude, longitude, dt, road_type, speed_limit,
                       junction, weather, traffic_density_override=None):
    """
    Build a single-row feature dataframe in the exact column order the
    model expects, from raw/live inputs.

    weather: dict with keys condition, temperature, precipitation,
             wind_speed, visibility  (same shape fetch_live_weather returns)
    """
    t = derive_time_features(dt)
    traffic_density = (
        traffic_density_override if traffic_density_override is not None
        else t["traffic_density"]
    )

    row = {
        "latitude": latitude,
        "longitude": longitude,
        "hour": t["hour"],
        "day_of_week": t["day_of_week"],
        "month": t["month"],
        "is_weekend": t["is_weekend"],
        "is_night": t["is_night"],
        "road_type_enc": ROAD_TYPES.get(road_type, 1),
        "speed_limit": speed_limit,
        "junction": int(junction),
        "traffic_density": traffic_density,
        "weather_enc": WEATHER_CONDITIONS.get(weather["condition"], 0),
        "temperature": weather["temperature"],
        "precipitation": weather["precipitation"],
        "wind_speed": weather["wind_speed"],
        "visibility": weather["visibility"],
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c
