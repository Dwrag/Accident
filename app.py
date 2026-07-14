"""
app.py - Accident Risk & Hotspot Prediction (Streamlit)

Run:
    streamlit run app.py

Pipeline:
  1. Historical accident data is used to train + compare multiple ML
     algorithms (see train_model.py). The best one (by weighted F1) is
     loaded here automatically.
  2. The app grabs your browser's live GPS location (with manual
     override) and live weather for that exact spot (Open-Meteo, no
     API key needed).
  3. Those are turned into the same feature vector the model was
     trained on, and the model predicts a live accident-severity risk.
  4. A folium heatmap shows historical hotspots plus your live position.
  5. Below the map, a full dashboard shows overall accident statistics.
"""

import os
import json
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

try:
    from streamlit_js_eval import get_geolocation
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

from utils import (
    build_feature_row, fetch_live_weather, SEVERITY_LABELS,
    SEVERITY_COLORS, ROAD_TYPES, haversine_km,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")
DATA_PATH = os.path.join(BASE_DIR, "data", "historical_accidents.csv")

DEFAULT_LAT, DEFAULT_LON = 13.0827, 80.2707  # Chennai, used if geolocation unavailable

st.set_page_config(page_title="Accident Risk & Hotspot Predictor", page_icon="🚦", layout="wide")


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    comparison = pd.read_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"))
    fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
    fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    return {"model": model, "scaler": scaler, "meta": meta, "comparison": comparison, "fi": fi}


@st.cache_data
def load_historical_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def predict_risk(artifacts, feature_row):
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    needs_scaling = artifacts["meta"]["needs_scaling"]
    X = scaler.transform(feature_row) if needs_scaling else feature_row
    pred_class = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred_class, probs

st.sidebar.title("⚙️ Model")
artifacts = load_model_artifacts()

if artifacts is None:
    st.sidebar.warning("No trained model found yet.")
    if st.sidebar.button("Generate data + train models now"):
        with st.spinner("Generating historical data and training/comparing algorithms — this takes a minute..."):
            os.system(f"python {os.path.join(BASE_DIR, 'data', 'generate_historical_data.py')}")
            os.system(f"python {os.path.join(BASE_DIR, 'train_model.py')}")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.stop()
else:
    st.sidebar.success(f"Active model: **{artifacts['meta']['best_model_name']}**")
    st.sidebar.caption(f"Trained on {artifacts['meta']['trained_on_rows']} historical records")
    if st.sidebar.button("🔁 Retrain / refresh comparison"):
        with st.spinner("Retraining and re-comparing algorithms..."):
            os.system(f"python {os.path.join(BASE_DIR, 'train_model.py')}")
        st.cache_resource.clear()
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📊 Algorithm comparison")
st.sidebar.dataframe(
    artifacts["comparison"].round(3), hide_index=True, width='stretch'
)
st.sidebar.caption(
    "All candidate algorithms are trained on the same historical data and "
    "ranked by weighted F1-score on a held-out test set. The top row is "
    "automatically used for live predictions below."
)

st.title("🚦 Accident Risk & Hotspot Prediction")
st.caption(
    "Historical-data-trained ML model + your live location + live weather "
    "→ real-time accident severity risk and city hotspot map."
)
col_left, col_right = st.columns([1, 1.3])
with col_left:
    st.subheader("📍 Live location")

    lat, lon = None, None
    geo_source = "manual"

    if HAS_GEO:
        loc = get_geolocation()

        if loc and "coords" in loc:
            lat = loc["coords"]["latitude"]
            lon = loc["coords"]["longitude"]
            geo_source = "Browser GPS"
        else:
            st.warning("📍 Live Location is OFF")
            st.info("""
**Please enable your location.**

**📱 Android**
1. Turn ON **Location (GPS)**.
2. Open Chrome.
3. Tap **⋮ (three dots)** → **Settings** → **Site Settings** → **Location**.
4. Allow location access.
5. Reload the page.

**💻 Windows / Mac**
1. Turn ON Location Services.
2. Allow location permission in your browser.
3. Refresh the page.

If location still doesn't work, use **Manual Location Entry** below.
            """)
            if st.button("🔄 Retry Location"):
                st.rerun()
    else:
        st.info("Browser geolocation isn't available in this environment — use manual entry below.")

    use_manual = st.checkbox("Enter location manually instead", value=(lat is None))
    if use_manual or lat is None:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
        lon = c2.number_input("Longitude", value=DEFAULT_LON, format="%.6f")
        geo_source = "manual entry"

    st.caption(f"Using coordinates from: **{geo_source}** → ({lat:.5f}, {lon:.5f})")

    st.subheader("🌦️ Live weather")
    weather = None
    try:
        with st.spinner("Fetching live weather for this location..."):
            weather = fetch_live_weather(lat, lon)
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Condition", weather["condition"])
        wc2.metric("Temp (°C)", f"{weather['temperature']:.1f}")
        wc3.metric("Rain (mm)", f"{weather['precipitation']:.1f}")
        wc4.metric("Visibility (km)", f"{weather['visibility']:.1f}")
    except Exception as e:
        st.warning(f"Live weather fetch failed ({e}); enter conditions manually below.")
        cond = st.selectbox("Weather condition", ["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"])
        temp = st.slider("Temperature (°C)", -5, 45, 28)
        precip = st.slider("Precipitation (mm)", 0, 100, 0)
        wind = st.slider("Wind speed (km/h)", 0, 80, 10)
        vis = st.slider("Visibility (km)", 0.1, 12.0, 8.0)
        weather = {"condition": cond, "temperature": temp, "precipitation": precip,
                   "wind_speed": wind, "visibility": vis}

    st.subheader("🛣️ Road context")
    road_type = st.selectbox("Road type", list(ROAD_TYPES.keys()), index=1)
    speed_limit = st.slider("Speed limit (km/h)", 20, 100, 50)
    junction = st.checkbox("At / near a junction", value=False)
    override_traffic = st.checkbox("Manually set traffic density?", value=False)
    traffic_density = None
    if override_traffic:
        traffic_density = st.slider("Traffic density (0=empty, 1=gridlock)", 0.0, 1.0, 0.5)

    now = dt.datetime.now()
    st.caption(f"Prediction time: {now.strftime('%Y-%m-%d %H:%M')} ({'weekend' if now.weekday()>=5 else 'weekday'})")

    predict_clicked = st.button("🔮 Predict accident risk now", type="primary", width='stretch')

    if predict_clicked:
        feature_row = build_feature_row(
            latitude=lat, longitude=lon, dt=now, road_type=road_type,
            speed_limit=speed_limit, junction=junction, weather=weather,
            traffic_density_override=traffic_density,
        )
        pred_class, probs = predict_risk(artifacts, feature_row)
        label = SEVERITY_LABELS[pred_class]
        color = SEVERITY_COLORS[pred_class]

        st.markdown(
            f"""
            <div style="padding:1.2rem;border-radius:0.6rem;background:{color}22;
                        border:2px solid {color};text-align:center;">
                <h2 style="color:{color};margin:0;">Predicted risk: {label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probs is not None:
            prob_df = pd.DataFrame({
                "Severity": [SEVERITY_LABELS[i] for i in range(len(probs))],
                "Probability": probs,
            })
            st.bar_chart(prob_df.set_index("Severity"))

        hist = load_historical_data()
        if hist is not None:
            near = hist[haversine_km(lat, lon, hist["latitude"], hist["longitude"]) <= 2.0]
            st.info(
                f"📌 {len(near)} historical accident records within 2 km of this point "
                f"({(near['severity']>=2).mean()*100:.0f}% were High/Severe, if any found)."
                if len(near) > 0 else
                "📌 No historical accident records within 2 km of this point."
            )

        if artifacts["fi"] is not None:
            with st.expander("Why did the model predict this? (feature importance)"):
                st.dataframe(artifacts["fi"].head(8), hide_index=True, width='stretch')

with col_right:
    st.subheader("🗺️ Historical accident hotspot map")
    hist = load_historical_data()

    if hist is None:
        st.warning("No historical dataset found. Generate it from the sidebar first.")
    else:
        m = folium.Map(location=[lat or DEFAULT_LAT, lon or DEFAULT_LON], zoom_start=12, tiles="cartodbpositron")

        heat_points = hist[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_points, radius=12, blur=18, max_zoom=13).add_to(m)

        hist_binned = hist.copy()
        hist_binned["lat_bin"] = hist_binned["latitude"].round(2)
        hist_binned["lon_bin"] = hist_binned["longitude"].round(2)
        top_spots = (
            hist_binned.groupby(["lat_bin", "lon_bin"])
            .agg(count=("severity", "size"), avg_severity=("severity", "mean"))
            .reset_index()
            .sort_values("count", ascending=False)
            .head(8)
        )
        for _, row in top_spots.iterrows():
            folium.CircleMarker(
                location=[row["lat_bin"], row["lon_bin"]],
                radius=8 + row["count"] / 40,
                color="darkred", fill=True, fill_opacity=0.6,
                popup=f"{int(row['count'])} accidents, avg severity {row['avg_severity']:.1f}",
            ).add_to(m)

        if lat is not None and lon is not None:
            folium.Marker(
                location=[lat, lon],
                popup="Your live location",
                icon=folium.Icon(color="blue", icon="user"),
            ).add_to(m)

        st_folium(m, width=None, height=560, returned_objects=[])
        st.caption(
            "Red circles = top historical accident hotspots (bigger = more accidents). "
            "Heat layer = accident density. Blue pin = your live location."
        )

hist = load_historical_data()

if hist is not None:
    st.markdown("---")
    st.subheader("📊 Accident Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Accidents", len(hist))
    c2.metric("High Risk Accidents", len(hist[hist["severity"] >= 2]))
    c3.metric("Average Severity", f"{hist['severity'].mean():.2f}")

    st.markdown("---")
    st.subheader("📈 Accident Severity Distribution")
    severity_counts = hist["severity"].map(SEVERITY_LABELS).value_counts()
    st.bar_chart(severity_counts)

    st.markdown("---")
    st.subheader("🛣️ Road Type Distribution")
    road_labels = {v: k for k, v in ROAD_TYPES.items()}
    road_counts = hist["road_type_enc"].map(road_labels).value_counts()
    st.bar_chart(road_counts)

    st.markdown("---")
    st.subheader("🌦️ Weather Distribution")
    weather_labels = {0: "Clear", 1: "Cloudy", 2: "Rain", 3: "Fog", 4: "Storm", 5: "Snow"}
    weather_counts = hist["weather_enc"].map(weather_labels).value_counts()
    st.bar_chart(weather_counts)

    st.markdown("---")
    st.subheader("🔥 Top 10 Accident Hotspots")
    top10 = (
        hist.groupby(["latitude", "longitude"])
        .size()
        .reset_index(name="Accidents")
        .sort_values("Accidents", ascending=False)
        .head(10)
    )
    st.dataframe(top10, width='stretch', hide_index=True)